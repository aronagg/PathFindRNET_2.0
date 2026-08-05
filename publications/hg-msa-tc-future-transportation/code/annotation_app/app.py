"""Blind local Streamlit UI for independent trajectory annotation and adjudication."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - exercised only before optional install.
    raise SystemExit(
        "Streamlit is not installed. Install requirements-annotation.txt first."
    ) from exc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agreement import agreement_analysis, initialize_adjudication_database, save_adjudication  # noqa: E402
from approach_picker import approach_picker  # noqa: E402
from models import (
    CONFIDENCE_VALUES,
    MANEUVER_TYPES,
    SCENES,
    VALIDITY_VALUES,
    manual_maneuver_id,
    validate_blind_queue_columns,
)  # noqa: E402
from protocol import (  # noqa: E402
    freeze_protocol,
    load_yaml,
    normalize_approach_rows,
    require_frozen_protocol,
    validate_scene_guide_definition,
    write_yaml,
)
from rendering import (
    RENDERING_VERSION,
    extract_clip,
    render_scene_guide_from_yaml,
    render_trajectory,
)  # noqa: E402
from source_data import (  # noqa: E402
    load_full_polyline,
    read_video_frame,
    sha256_file,
    verify_video_checksum_on_demand,
)
from storage import (  # noqa: E402
    active_annotations,
    backup_database,
    initialize_database,
    mark_queue_event,
    queue_state,
    record_batch_operation,
    save_annotation,
    undo_last_annotation,
    update_queue_position,
)


APP_DIR = Path(__file__).resolve().parent
PUBLICATION_ROOT = APP_DIR.parents[1]
REPO_ROOT = APP_DIR.parents[3]
ANNOTATIONS = PUBLICATION_ROOT / "annotations"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--role",
        choices=[
            "protocol_pilot",
            "annotator_A",
            "annotator_B",
            "adjudicator",
            "protocol_designer",
        ],
        default="annotator_A",
    )
    parser.add_argument("--mode", choices=["single", "grid"], default="single")
    values, _ = parser.parse_known_args()
    return values


@st.cache_data(show_spinner=False)
def cached_polyline(
    path: str, track_id: int, checksum: str, frame_start: int, frame_end: int
) -> pd.DataFrame:
    verified_source(path, checksum)
    return load_full_polyline(
        REPO_ROOT, path, track_id, checksum, frame_start, frame_end, verify_checksum=False
    )


@st.cache_resource(show_spinner=False)
def verified_source(path: str, checksum: str) -> bool:
    if sha256_file(REPO_ROOT / path) != checksum:
        raise ValueError(f"Trajectory source checksum mismatch: {path}")
    return True


@st.cache_data(show_spinner=False)
def cached_frame(relative_video_path: str, frame_number: int):
    verified_video(relative_video_path)
    return read_video_frame(REPO_ROOT / relative_video_path, frame_number)


@st.cache_resource(show_spinner=False)
def verified_video(relative_video_path: str) -> str:
    return verify_video_checksum_on_demand(
        REPO_ROOT,
        relative_video_path,
        ANNOTATIONS / "provenance" / "video_access_checksums.jsonl",
    )


def approach_options(protocol: dict, scene: str) -> tuple[list[str], dict]:
    guide = next(row for row in protocol["scene_guides"] if row["scene_id"] == scene)
    return [str(row["id"]) for row in guide["approaches"]] + ["UNKNOWN"], guide


def human_form(prefix: str, approaches: list[str]) -> dict | None:
    with st.form(f"annotation_{prefix}", clear_on_submit=False):
        entry = st.selectbox("Entry approach", approaches, key=f"{prefix}_entry")
        exit_value = st.selectbox("Exit approach", approaches, key=f"{prefix}_exit")
        maneuver = st.selectbox("Coarse maneuver type", MANEUVER_TYPES, key=f"{prefix}_maneuver")
        validity = st.radio("Validity", VALIDITY_VALUES, horizontal=True, key=f"{prefix}_validity")
        confidence = st.radio(
            "Confidence", CONFIDENCE_VALUES, horizontal=True, key=f"{prefix}_confidence"
        )
        shortcut = st.text_input(
            "Keyboard command (entry exit type validity confidence)",
            placeholder="A B s v h",
            key=f"{prefix}_shortcut",
            help="Types: s/l/r/u/o/x. Validity: v/a/u. Confidence: h/m/l. Press Enter to submit.",
        )
        notes = st.text_input("Notes", key=f"{prefix}_notes")
        submitted = st.form_submit_button("Save and next", type="primary", width="stretch")
    if not submitted:
        return None
    if shortcut.strip():
        tokens = shortcut.split()
        if len(tokens) != 5:
            st.error("Keyboard command requires five tokens: entry exit type validity confidence.")
            return None
        entry, exit_value, type_key, validity_key, confidence_key = tokens
        type_map = {
            "s": "straight",
            "l": "left",
            "r": "right",
            "u": "u_turn",
            "o": "other",
            "x": "unknown",
        }
        validity_map = {"v": "valid", "a": "ambiguous", "u": "unusable"}
        confidence_map = {"h": "high", "m": "medium", "l": "low"}
        if (
            entry not in approaches
            or exit_value not in approaches
            or type_key not in type_map
            or validity_key not in validity_map
            or confidence_key not in confidence_map
        ):
            st.error("Keyboard command contains an invalid approach or shortcut.")
            return None
        maneuver, validity, confidence = (
            type_map[type_key],
            validity_map[validity_key],
            confidence_map[confidence_key],
        )
    return {
        "entry_approach": entry,
        "exit_approach": exit_value,
        "maneuver_type": maneuver,
        "validity": validity,
        "confidence": confidence,
        "notes": notes,
    }


def filtered_queue(
    queue: pd.DataFrame, active: pd.DataFrame, database: Path, annotator: str
) -> pd.DataFrame:
    active_ids = set(active["trajectory_id"]) if len(active) else set()
    queue = queue.copy()
    queue["annotation_status"] = "pending"
    import sqlite3

    with sqlite3.connect(database) as connection:
        events = pd.read_sql_query("SELECT trajectory_id, event_type FROM queue_events", connection)
    event_map = (
        events.groupby("trajectory_id")["event_type"].apply(set).to_dict() if len(events) else {}
    )
    queue.loc[
        queue["trajectory_id"].map(lambda value: "review" in event_map.get(value, set())),
        "annotation_status",
    ] = "review"
    queue.loc[
        queue["trajectory_id"].map(lambda value: "skip" in event_map.get(value, set())),
        "annotation_status",
    ] = "skipped"
    queue.loc[queue["trajectory_id"].isin(active_ids), "annotation_status"] = "completed"
    if len(active):
        validity = active.set_index("trajectory_id")["validity"].to_dict()
        queue.loc[queue["trajectory_id"].map(validity).eq("ambiguous"), "annotation_status"] = (
            "ambiguous"
        )
        queue.loc[queue["trajectory_id"].map(validity).eq("unusable"), "annotation_status"] = (
            "unusable"
        )
    selected = st.sidebar.multiselect(
        "Status filter",
        ["pending", "completed", "review", "skipped", "ambiguous", "unusable"],
        default=["pending"],
    )
    visible = queue[queue["annotation_status"].isin(selected)].copy()
    order = st.sidebar.selectbox(
        "Annotation order", ["frozen queue", "scene and recording", "trajectory ID"]
    )
    if order == "scene and recording":
        visible = visible.sort_values(
            ["scene_id", "recording_id", "queue_position"], kind="mergesort"
        )
    elif order == "trajectory ID":
        visible = visible.sort_values("trajectory_id", kind="mergesort")
    return visible.reset_index(drop=True)


def maybe_periodic_backup(database: Path, annotator: str, completed_count: int) -> None:
    if completed_count > 0 and completed_count % 250 == 0:
        backup_database(database, ANNOTATIONS / "backups" / annotator)


def controls(
    database: Path, annotator: str, protocol_version: str, row: pd.Series, position: int
) -> None:
    left, middle, right = st.columns(3)
    if left.button("Previous", width="stretch"):
        st.session_state.position = max(0, position - 1)
        st.rerun()
    if middle.button("Skip", width="stretch"):
        mark_queue_event(database, annotator, row["trajectory_id"], "skip")
        st.session_state.position = position
        st.rerun()
    if right.button("Review later", width="stretch"):
        mark_queue_event(database, annotator, row["trajectory_id"], "review")
        st.session_state.position = position
        st.rerun()
    if st.sidebar.button("Undo last annotation"):
        undo_last_annotation(database, annotator, protocol_version)
        st.rerun()


def display_trajectory(
    row: pd.Series,
    zoom: float,
    line_width: float,
    show_background: bool,
    margin_fraction: float = 0.08,
):
    polyline = cached_polyline(
        row["trajectory_source_path"],
        int(row["source_recording_track_id"]),
        row["trajectory_source_checksum"],
        int(row["frame_start"]),
        int(row["frame_end"]),
    )
    background = (
        cached_frame(row["source_recording_file"], int(polyline["frame"].iloc[len(polyline) // 2]))
        if show_background
        else None
    )
    figure = render_trajectory(
        polyline,
        background,
        zoom=zoom,
        line_width=line_width,
        margin_fraction=margin_fraction,
        show_background=show_background,
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    return polyline


def single_mode(queue: pd.DataFrame, database: Path, annotator: str, protocol: dict) -> None:
    active = active_annotations(database, annotator)
    visible = filtered_queue(queue, active, database, annotator)
    if visible.empty:
        st.info("No trajectories match the current filter.")
        return
    if "position" not in st.session_state:
        saved = queue_state(database, annotator)
        if saved:
            candidates = visible.index[
                visible["queue_position"] >= int(saved["last_position"])
            ].tolist()
            st.session_state.position = candidates[0] if candidates else 0
        else:
            st.session_state.position = 0
    position = min(int(st.session_state.position), len(visible) - 1)
    row = visible.iloc[position]
    st.session_state.position = position
    update_queue_position(database, annotator, row["queue_id"], int(row["queue_position"]))
    st.progress(len(active) / len(queue), text=f"{len(active):,} / {len(queue):,} completed")
    for scene, total in queue.groupby("scene_id").size().items():
        done = int((active["scene_id"] == scene).sum()) if len(active) else 0
        st.sidebar.caption(f"{scene}: {done:,}/{total:,}")
    st.caption(
        f"{row['scene_id']} | {row['recording_id']} | frames {row['frame_start']}-{row['frame_end']} | {row['number_of_points']} points"
    )
    zoom = st.sidebar.slider("Zoom", 1.0, 5.0, 1.0, 0.25)
    line_width = st.sidebar.slider("Line width", 1.0, 8.0, 2.5, 0.5)
    margin_fraction = st.sidebar.slider("Path margin", 0.0, 0.5, 0.08, 0.01)
    show_background = st.sidebar.toggle("Camera background", value=True)
    polyline = display_trajectory(row, zoom, line_width, show_background, margin_fraction)
    path_length = float(
        np.hypot(
            np.diff(polyline["cx"].to_numpy(dtype=float)),
            np.diff(polyline["cy"].to_numpy(dtype=float)),
        ).sum()
    )
    st.caption(f"Displayed camera-path length: {path_length:,.1f} px")
    approaches, guide = approach_options(protocol, row["scene_id"])
    guide_title = (
        "Draft scene guide"
        if str(protocol["protocol_version"]).startswith("draft-")
        else "Frozen scene guide"
    )
    with st.expander(guide_title, expanded=True):
        st.image(
            ANNOTATIONS / "protocol" / "scene_guides" / f"{row['scene_id']}_guide.png",
            width="stretch",
        )
        if guide.get("ambiguity_notes"):
            st.caption(guide["ambiguity_notes"])
    if row["video_available"] and st.button("Prepare optional source clip"):
        verified_video(row["source_recording_file"])
        clip = ANNOTATIONS / "cache" / "clips" / f"{row['trajectory_id'].replace(':', '_')}.mp4"
        extract_clip(
            REPO_ROOT / row["source_recording_file"],
            int(polyline["frame"].min()),
            int(polyline["frame"].max()),
            clip,
        )
        st.video(str(clip), autoplay=False)
    elif not row["video_available"]:
        st.warning("Video assistance is unavailable for this trajectory.")
    if row["video_available"]:
        with st.expander("Manual frame stepping"):
            frame_number = st.number_input(
                "Frame",
                min_value=int(row["frame_start"]),
                max_value=int(row["frame_end"]),
                value=int(row["frame_start"]),
                step=1,
            )
            st.image(
                cached_frame(row["source_recording_file"], int(frame_number)),
                width="stretch",
            )
    controls(database, annotator, protocol["protocol_version"], row, position)
    values = human_form(str(row["trajectory_id"]), approaches)
    if values is not None:
        save_annotation(
            database,
            row.to_dict(),
            values,
            annotator,
            protocol["protocol_version"],
            RENDERING_VERSION,
        )
        maybe_periodic_backup(database, annotator, len(active) + 1)
        st.session_state.position = position
        st.rerun()


def grid_mode(queue: pd.DataFrame, database: Path, annotator: str, protocol: dict) -> None:
    active = active_annotations(database, annotator)
    visible = filtered_queue(queue, active, database, annotator)
    if visible.empty:
        st.info("No trajectories match the current filter.")
        return
    recording = st.selectbox(
        "Scene/recording",
        visible.assign(key=visible["scene_id"] + " / " + visible["recording_id"])[
            "key"
        ].drop_duplicates(),
    )
    scene, recording_id = recording.split(" / ", 1)
    part = visible[(visible["scene_id"] == scene) & (visible["recording_id"] == recording_id)].head(
        12
    )
    selected = []
    columns = st.columns(3)
    for index, (_, row) in enumerate(part.iterrows()):
        with columns[index % 3]:
            st.caption(str(row["trajectory_id"]))
            display_trajectory(row, 1.0, 2.0, True)
            reviewed = st.checkbox("Reviewed", key=f"reviewed_{row['trajectory_id']}")
            chosen = st.checkbox(
                "Select for batch", key=f"selected_{row['trajectory_id']}", disabled=not reviewed
            )
            if reviewed and chosen:
                selected.append(row)
    approaches, _ = approach_options(protocol, scene)
    values = human_form(f"grid_{scene}_{recording_id}", approaches)
    if values is not None:
        if not selected:
            st.error("No manually reviewed cards were selected.")
            return
        batch_id = str(uuid.uuid4())
        selected_ids = [str(row["trajectory_id"]) for row in selected]
        for row in selected:
            save_annotation(
                database,
                row.to_dict(),
                values,
                annotator,
                protocol["protocol_version"],
                RENDERING_VERSION,
                batch_id=batch_id,
            )
        record_batch_operation(
            database, annotator, protocol["protocol_version"], batch_id, selected_ids
        )
        maybe_periodic_backup(database, annotator, len(active) + len(selected))
        st.success(f"Saved {len(selected)} manually selected trajectories in batch {batch_id}.")
        st.rerun()


def protocol_designer_mode() -> None:
    st.title("Manual scene-guide setup")
    st.warning(
        "Approaches must be placed manually. HG targets, endpoint groups, and cluster results are unavailable here."
    )
    scene = st.selectbox(
        "Scene",
        [p.stem for p in sorted((ANNOTATIONS / "protocol" / "scene_guides").glob("*.yaml"))],
    )
    path = ANNOTATIONS / "protocol" / "scene_guides" / f"{scene}.yaml"
    guide = load_yaml(path)
    frame_path = ANNOTATIONS / guide["representative_frame"]
    approach_rows = []
    for item in guide.get("approaches", []):
        label = item.get("label_position_normalized", {})
        arrow = item.get("arrow_end_normalized", {})
        approach_rows.append(
            {
                "id": item.get("id", ""),
                "human_readable_name": item.get("human_readable_name", ""),
                "label_x": label.get("x", 0.5),
                "label_y": label.get("y", 0.5),
                "arrow_x": arrow.get("x", 0.5),
                "arrow_y": arrow.get("y", 0.5),
                "region_x_min": item.get("region_normalized", {}).get("x_min"),
                "region_y_min": item.get("region_normalized", {}).get("y_min"),
                "region_x_max": item.get("region_normalized", {}).get("x_max"),
                "region_y_max": item.get("region_normalized", {}).get("y_max"),
                "region_role": item.get("region_role", "both"),
                "polygon_points": (
                    json.dumps(item["polygon_normalized"], separators=(",", ":"))
                    if item.get("polygon_normalized")
                    else None
                ),
            }
        )
    draft_key = f"scene_guide_rows::{scene}"
    version_key = f"scene_guide_table_version::{scene}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = approach_rows
    if version_key not in st.session_state:
        st.session_state[version_key] = 0
    st.subheader("Manual image placement")
    picker_columns = st.columns([2, 2, 2])
    with picker_columns[0]:
        picker_id = st.text_input("Approach ID", key=f"picker_id::{scene}").strip()
    with picker_columns[1]:
        picker_mode_label = st.segmented_control(
            "Geometry",
            ["Point", "Polygon"],
            default="Polygon",
            key=f"picker_mode::{scene}",
        )
    with picker_columns[2]:
        picker_role = st.selectbox(
            "Region role", ["entry", "exit", "both"], key=f"picker_role::{scene}"
        )
    existing_shapes = []
    for row in st.session_state[draft_key]:
        raw_id = row.get("id")
        if raw_id is None or pd.isna(raw_id) or not str(raw_id).strip():
            continue
        raw_role = row.get("region_role")
        region_role = raw_role if raw_role in {"entry", "exit", "both"} else "both"
        raw_polygon = row.get("polygon_points")
        if pd.notna(raw_polygon) and str(raw_polygon).strip():
            try:
                polygon_points = json.loads(str(raw_polygon))
            except json.JSONDecodeError:
                polygon_points = []
            if len(polygon_points) >= 3:
                existing_shapes.append(
                    {
                        "type": "polygon",
                        "approach_id": str(row["id"]),
                        "region_role": region_role,
                        "points": polygon_points,
                    }
                )
                continue
        region_fields = ["region_x_min", "region_y_min", "region_x_max", "region_y_max"]
        if all(pd.notna(row.get(field)) for field in region_fields):
            existing_shapes.append(
                {
                    "type": "rectangle",
                    "approach_id": str(row["id"]),
                    "region_role": region_role,
                    "x_min": float(row["region_x_min"]),
                    "y_min": float(row["region_y_min"]),
                    "x_max": float(row["region_x_max"]),
                    "y_max": float(row["region_y_max"]),
                }
            )
        else:
            point_x = row.get("arrow_x")
            point_y = row.get("arrow_y")
            if pd.isna(point_x) or pd.isna(point_y):
                continue
            existing_shapes.append(
                {
                    "type": "point",
                    "approach_id": str(row["id"]),
                    "region_role": region_role,
                    "x": float(point_x),
                    "y": float(point_y),
                }
            )
    selection = approach_picker(
        frame_path,
        "polygon" if picker_mode_label == "Polygon" else "point",
        picker_id,
        picker_role,
        existing_shapes,
        key=f"approach_picker::{scene}",
    )
    selection_nonce_key = f"approach_picker_nonce::{scene}"
    if selection and selection.get("nonce") != st.session_state.get(selection_nonce_key):
        st.session_state[selection_nonce_key] = selection["nonce"]
        selected_id = str(selection.get("approach_id", "")).strip()
        if not selected_id:
            st.warning("Enter one approach ID before placing a point or region.")
        else:
            rows = list(st.session_state[draft_key])
            row = next(
                (item for item in rows if str(item.get("id", "")).strip() == selected_id), None
            )
            if row is None:
                row = {"id": selected_id, "human_readable_name": ""}
                rows.append(row)
            if selection["type"] == "polygon":
                points = [
                    {"x": float(point["x"]), "y": float(point["y"])}
                    for point in selection["points"]
                ]
                center_x = sum(point["x"] for point in points) / len(points)
                center_y = sum(point["y"] for point in points) / len(points)
                row.update(
                    {
                        "label_x": center_x,
                        "label_y": center_y,
                        "arrow_x": center_x,
                        "arrow_y": center_y,
                        "region_x_min": None,
                        "region_y_min": None,
                        "region_x_max": None,
                        "region_y_max": None,
                        "region_role": selection.get("region_role", "both"),
                        "polygon_points": json.dumps(points, separators=(",", ":")),
                    }
                )
            elif selection["type"] == "rectangle":
                center_x = (float(selection["x_min"]) + float(selection["x_max"])) / 2
                center_y = (float(selection["y_min"]) + float(selection["y_max"])) / 2
                row.update(
                    {
                        "label_x": center_x,
                        "label_y": center_y,
                        "arrow_x": center_x,
                        "arrow_y": center_y,
                        "region_x_min": float(selection["x_min"]),
                        "region_y_min": float(selection["y_min"]),
                        "region_x_max": float(selection["x_max"]),
                        "region_y_max": float(selection["y_max"]),
                        "region_role": selection.get("region_role", "both"),
                        "polygon_points": None,
                    }
                )
            else:
                row.update(
                    {
                        "label_x": float(selection["x"]),
                        "label_y": float(selection["y"]),
                        "arrow_x": float(selection["x"]),
                        "arrow_y": float(selection["y"]),
                        "region_x_min": None,
                        "region_y_min": None,
                        "region_x_max": None,
                        "region_y_max": None,
                        "region_role": selection.get("region_role", "both"),
                        "polygon_points": None,
                    }
                )
            st.session_state[draft_key] = rows
            st.session_state[version_key] += 1
            st.rerun()
    st.caption(
        "These are manual visual guides only. They do not assign trajectories or infer maneuver labels."
    )
    approaches = pd.DataFrame(
        st.session_state[draft_key],
        columns=[
            "id",
            "human_readable_name",
            "label_x",
            "label_y",
            "arrow_x",
            "arrow_y",
            "region_x_min",
            "region_y_min",
            "region_x_max",
            "region_y_max",
            "region_role",
            "polygon_points",
        ],
    )
    edited = st.data_editor(
        approaches,
        num_rows="dynamic",
        width="stretch",
        key=f"approach_table::{scene}::{st.session_state[version_key]}",
    )
    st.session_state[draft_key] = edited.to_dict(orient="records")
    entries = st.text_input(
        "Valid entry IDs (comma separated)", ",".join(guide.get("valid_entry_approaches", []))
    )
    exits = st.text_input(
        "Valid exit IDs (comma separated)", ",".join(guide.get("valid_exit_approaches", []))
    )
    mapping_text = st.text_area(
        "Manual entry/exit to coarse-type mapping (YAML list)",
        yaml.safe_dump(guide.get("maneuver_type_mapping", []), sort_keys=False),
    )
    notes = st.text_area("Scene-specific ambiguity notes", guide.get("ambiguity_notes", ""))
    ready = st.checkbox(
        "I manually verified all approach labels and mappings; mark ready for freeze"
    )
    if st.button("Save scene guide", type="primary"):
        try:
            rows = normalize_approach_rows(edited.to_dict(orient="records"))
            mappings = yaml.safe_load(mapping_text) or []
            if not isinstance(mappings, list):
                raise ValueError("Maneuver mapping must be a YAML list.")
        except (ValueError, yaml.YAMLError) as exc:
            st.error(str(exc))
            return
        candidate = dict(guide)
        candidate.update(
            {
                "status": "ready_for_freeze" if ready else "draft_manual_configuration_required",
                "approaches": rows,
                "valid_entry_approaches": [
                    value.strip() for value in entries.split(",") if value.strip()
                ],
                "valid_exit_approaches": [
                    value.strip() for value in exits.split(",") if value.strip()
                ],
                "maneuver_type_mapping": mappings,
                "ambiguity_notes": notes,
            }
        )
        if ready:
            try:
                validate_scene_guide_definition(candidate, scene)
            except ValueError as exc:
                st.error(str(exc))
                return
        write_yaml(path, candidate)
        preview_path = render_scene_guide_from_yaml(path, ANNOTATIONS)
        st.session_state[draft_key] = [
            {
                "id": row["id"],
                "human_readable_name": row.get("human_readable_name", ""),
                "label_x": row["label_position_normalized"]["x"],
                "label_y": row["label_position_normalized"]["y"],
                "arrow_x": row["arrow_end_normalized"]["x"],
                "arrow_y": row["arrow_end_normalized"]["y"],
                "region_x_min": row.get("region_normalized", {}).get("x_min"),
                "region_y_min": row.get("region_normalized", {}).get("y_min"),
                "region_x_max": row.get("region_normalized", {}).get("x_max"),
                "region_y_max": row.get("region_normalized", {}).get("y_max"),
                "region_role": row.get("region_role", "both"),
                "polygon_points": (
                    json.dumps(row["polygon_normalized"], separators=(",", ":"))
                    if row.get("polygon_normalized")
                    else None
                ),
            }
            for row in rows
        ]
        st.success("Draft guide saved. Freeze separately after all five guides pass manual review.")
        st.image(preview_path, caption="Saved scene-guide overlay", width="stretch")
    else:
        preview_path = path.with_name(f"{scene}_guide.png")
        if preview_path.exists():
            st.image(preview_path, caption="Last saved scene-guide overlay", width="stretch")
    st.divider()
    freeze_confirmed = st.checkbox(
        "Freeze all five manually reviewed scene guides as annotation protocol v1"
    )
    if st.button("Freeze Protocol", disabled=not freeze_confirmed):
        frozen_path, checksum = freeze_protocol(
            ANNOTATIONS / "protocol",
            PUBLICATION_ROOT / "results" / "development" / "frozen_selection_manifest.json",
            ANNOTATIONS / "protocol" / "annotation_guideline.md",
        )
        st.success(
            f"Frozen {frozen_path.name}: {checksum}. Regenerate primary queues before annotation."
        )


def adjudicator_mode(protocol: dict) -> None:
    st.title("Independent-label adjudication")
    exports = ANNOTATIONS / "exports"
    a_path, b_path = (
        exports / "independent_test_annotator_A.csv",
        exports / "independent_test_annotator_B.csv",
    )
    if not a_path.exists() or not b_path.exists():
        st.info("Both immutable first-pass exports are required before adjudication.")
        return
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    disagreements = agreement_analysis(a, b)["disagreements"]
    database = ANNOTATIONS / "databases" / "adjudication.sqlite"
    initialize_adjudication_database(database, protocol["protocol_version"])
    import sqlite3

    with sqlite3.connect(database) as connection:
        resolved = {row[0] for row in connection.execute("SELECT trajectory_id FROM adjudications")}
    disagreements = disagreements[~disagreements["trajectory_id"].isin(resolved)].reset_index(
        drop=True
    )
    if disagreements.empty:
        st.success("No disagreements or low-confidence cases remain.")
        return
    index = st.number_input("Case", min_value=1, max_value=len(disagreements), value=1) - 1
    row = disagreements.iloc[int(index)]
    st.dataframe(
        pd.DataFrame({"Annotator A": row.filter(like="_A"), "Annotator B": row.filter(like="_B")})
    )
    queue_path = ANNOTATIONS / "queues" / "independent_test_annotator_A.csv"
    queue_row = pd.read_csv(queue_path).set_index("trajectory_id").loc[row["trajectory_id"]]
    display_trajectory(queue_row, 1.0, 2.5, True)
    approaches, _ = approach_options(protocol, row["scene_id"])
    values = human_form(f"adjudication_{row['trajectory_id']}", approaches)
    reason = st.text_input("Adjudication reason")
    if values is not None:
        consensus = {**values, "scene_id": row["scene_id"], "trajectory_id": row["trajectory_id"]}
        consensus["manual_maneuver_id"] = manual_maneuver_id(
            consensus["scene_id"], consensus["entry_approach"], consensus["exit_approach"]
        )
        raw_a = a.set_index("trajectory_id").loc[row["trajectory_id"]].to_dict()
        raw_b = b.set_index("trajectory_id").loc[row["trajectory_id"]].to_dict()
        raw_a["trajectory_id"] = row["trajectory_id"]
        raw_b["trajectory_id"] = row["trajectory_id"]
        save_adjudication(
            database, raw_a, raw_b, consensus, "adjudicator", protocol["protocol_version"], reason
        )
        st.rerun()


def main() -> None:
    args = arguments()
    st.set_page_config(page_title="Blind trajectory annotation", layout="wide")
    if args.role == "protocol_designer":
        protocol_designer_mode()
        return
    if args.role == "protocol_pilot":
        guides = [
            load_yaml(ANNOTATIONS / "protocol" / "scene_guides" / f"{scene}.yaml")
            for scene in SCENES
        ]
        draft_protocol = {
            "protocol_version": "draft-manual-annotation-protocol-pilot-only",
            "scene_guides": guides,
        }
        queue = pd.read_csv(
            ANNOTATIONS / "queues" / "protocol_pilot_queue.csv",
            keep_default_na=False,
        )
        validate_blind_queue_columns(tuple(queue.columns))
        database = ANNOTATIONS / "databases" / "protocol_pilot.sqlite"
        initialize_database(database, args.role, draft_protocol["protocol_version"], role="pilot")
        st.sidebar.header("Protocol pilot - excluded from test metrics")
        if any(not guide.get("approaches") for guide in guides):
            st.warning(
                "One or more manual scene guides have no approaches yet. "
                "Rendering remains available; configure approaches before pilot labeling."
            )
        mode = st.sidebar.radio("Mode", ["single", "grid"], index=0 if args.mode == "single" else 1)
        if mode == "single":
            single_mode(queue, database, args.role, draft_protocol)
        else:
            grid_mode(queue, database, args.role, draft_protocol)
        return
    protocol = require_frozen_protocol(ANNOTATIONS / "protocol")
    if args.role == "adjudicator":
        adjudicator_mode(protocol)
        return
    queue_path = ANNOTATIONS / "queues" / f"independent_test_{args.role}.csv"
    queue = pd.read_csv(queue_path, keep_default_na=False)
    validate_blind_queue_columns(tuple(queue.columns))
    if set(queue["queue_status"]) != {"available"}:
        raise PermissionError(
            "Primary queue remains locked pending protocol freeze and regeneration."
        )
    database = ANNOTATIONS / "databases" / f"{args.role}.sqlite"
    initialize_database(database, args.role, protocol["protocol_version"])
    st.sidebar.header(args.role)
    st.sidebar.caption(
        "Blind first-pass mode: no targets, clusters, suggestions, or other labels are loaded."
    )
    mode = st.sidebar.radio("Mode", ["single", "grid"], index=0 if args.mode == "single" else 1)
    if mode == "single":
        single_mode(queue, database, args.role, protocol)
    else:
        grid_mode(queue, database, args.role, protocol)


if __name__ == "__main__":
    main()
