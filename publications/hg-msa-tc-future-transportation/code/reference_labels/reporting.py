"""Inventories, sensitivity analysis, QA queues, and visual diagnostics."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as PolygonPatch

try:
    from .generator import (
        BOUNDARY_NEAR_THRESHOLDS_PX,
        SCENES,
        SPLITS,
        assign_reference_rows,
        build_scene_geometries,
        deterministic_write_csv,
    )
except ImportError:  # Direct script execution.
    from generator import (
        BOUNDARY_NEAR_THRESHOLDS_PX,
        SCENES,
        SPLITS,
        assign_reference_rows,
        build_scene_geometries,
        deterministic_write_csv,
    )


POLYGON_COLORS = {
    "A": "#00876c",
    "B": "#439981",
    "C": "#7eb196",
    "D": "#b7c9a8",
    "E": "#d43d51",
    "F": "#e66b5b",
    "G": "#ee9a70",
    "H": "#efc38b",
}


def build_legal_inventory(protocol: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for guide in protocol["scene_guides"]:
        for mapping in guide["maneuver_type_mapping"]:
            rows.append(
                {
                    "scene_id": guide["scene_id"],
                    "entry_polygon_id": mapping["entry"],
                    "exit_polygon_id": mapping["exit"],
                    "movement_id": (f"{guide['scene_id']}:{mapping['entry']}>{mapping['exit']}"),
                    "maneuver_type": mapping["maneuver_type"],
                    "legally_permitted": True,
                    "protocol_version": protocol["protocol_version"],
                }
            )
    return pd.DataFrame(rows)


def build_observed_inventory(labels: pd.DataFrame, legal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = labels[labels["reference_status"] == "valid"].copy()
    for split_name in (*SPLITS, "total"):
        split_valid = valid if split_name == "total" else valid[valid["split"] == split_name]
        for movement in legal.itertuples(index=False):
            scene_valid = split_valid[split_valid["scene_id"] == movement.scene_id]
            evidence = scene_valid[
                scene_valid["reference_movement_id"] == movement.movement_id
            ].sort_values("trajectory_id", kind="mergesort")
            denominator = len(scene_valid)
            recording_ids = sorted(evidence["recording_id"].astype(str).unique())
            rows.append(
                {
                    "scene_id": movement.scene_id,
                    "split": split_name,
                    "entry_polygon_id": movement.entry_polygon_id,
                    "exit_polygon_id": movement.exit_polygon_id,
                    "movement_id": movement.movement_id,
                    "maneuver_type": movement.maneuver_type,
                    "valid_assigned_count": len(evidence),
                    "percentage_of_valid_assigned": (
                        100.0 * len(evidence) / denominator if denominator else 0.0
                    ),
                    "observed": bool(len(evidence)),
                    "first_five_evidence_trajectory_ids": "|".join(
                        evidence["trajectory_id"].head(5).astype(str)
                    ),
                    "recording_coverage_count": len(recording_ids),
                    "recording_coverage_ids": "|".join(recording_ids),
                    "first_start_frame": (
                        int(evidence["start_frame"].min()) if len(evidence) else ""
                    ),
                    "last_end_frame": (int(evidence["end_frame"].max()) if len(evidence) else ""),
                    "protocol_version": movement.protocol_version,
                }
            )
    return pd.DataFrame(rows)


def _quality_row(scene: str, split_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    geometry_available = frame["finite_canonical_point_count"] > 0
    multiple = (frame["entry_match_count"] > 1) | (frame["exit_match_count"] > 1)
    row: dict[str, Any] = {
        "scene_id": scene,
        "split": split_name,
        "total_trajectories": len(frame),
        "valid_reference_labels": int((frame["reference_status"] == "valid").sum()),
        "valid_coverage_percentage": (
            100.0 * (frame["reference_status"] == "valid").mean() if len(frame) else 0.0
        ),
        "entry_no_polygon_count": int(
            (geometry_available & (frame["entry_match_count"] == 0)).sum()
        ),
        "exit_no_polygon_count": int((geometry_available & (frame["exit_match_count"] == 0)).sum()),
        "multiple_polygon_count": int(multiple.sum()),
        "illegal_mapping_count": int((frame["exclusion_reason"] == "mapping_not_legal").sum()),
        "missing_geometry_count": int(
            frame["exclusion_reason"]
            .isin(["missing_geometry", "invalid_endpoint", "source_mapping_error"])
            .sum()
        ),
    }
    minimum_boundary = frame[["entry_boundary_distance", "exit_boundary_distance"]].min(axis=1)
    for threshold in BOUNDARY_NEAR_THRESHOLDS_PX:
        row[f"boundary_near_{int(threshold)}px_count"] = int((minimum_boundary <= threshold).sum())
    return row


def build_quality_tables(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for scene in SCENES:
        scene_frame = labels[labels["scene_id"] == scene]
        for split_name in SPLITS:
            rows.append(
                _quality_row(scene, split_name, scene_frame[scene_frame["split"] == split_name])
            )
        rows.append(_quality_row(scene, "total", scene_frame))
    summary = pd.DataFrame(rows)
    reasons = (
        labels.assign(exclusion_reason=labels["exclusion_reason"].replace("", "valid"))
        .groupby(["scene_id", "split", "exclusion_reason"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    recording = (
        labels.assign(valid=labels["reference_status"].eq("valid"))
        .groupby(["scene_id", "split", "recording_id"], sort=False)
        .agg(
            total_trajectories=("trajectory_id", "size"),
            valid_reference_labels=("valid", "sum"),
            first_start_frame=("start_frame", "min"),
            last_end_frame=("end_frame", "max"),
        )
        .reset_index()
    )
    recording["valid_coverage_percentage"] = (
        100.0 * recording["valid_reference_labels"] / recording["total_trajectories"]
    )
    return summary, reasons, recording


def write_inventory_and_quality_reports(
    labels: pd.DataFrame,
    protocol: dict[str, Any],
    output_dir: Path,
    docs_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    legal = build_legal_inventory(protocol)
    observed = build_observed_inventory(labels, legal)
    quality, reasons, recording = build_quality_tables(labels)
    deterministic_write_csv(legal, output_dir / "legal_movement_inventory.csv")
    deterministic_write_csv(observed, output_dir / "observed_movement_inventory.csv")
    deterministic_write_csv(quality, output_dir / "coverage_summary.csv")
    deterministic_write_csv(reasons, output_dir / "exclusion_reason_summary.csv")
    deterministic_write_csv(recording, output_dir / "recording_coverage.csv")

    scene_inventory = []
    for scene in SCENES:
        scene_legal = legal[legal["scene_id"] == scene]
        total_observed = observed[
            (observed["scene_id"] == scene) & (observed["split"] == "total") & observed["observed"]
        ]
        test_observed = observed[
            (observed["scene_id"] == scene)
            & (observed["split"] == "independent_test")
            & observed["observed"]
        ]
        scene_inventory.append(
            {
                "scene_id": scene,
                "legal_movement_count": len(scene_legal),
                "observed_movement_count": len(total_observed),
                "observed_independent_test_movement_count": len(test_observed),
            }
        )
    scene_inventory_frame = pd.DataFrame(scene_inventory)
    unobserved = observed[(observed["split"] == "total") & ~observed["observed"]][
        ["scene_id", "movement_id", "maneuver_type"]
    ]
    illegal_count = int((labels["exclusion_reason"] == "mapping_not_legal").sum())
    inventory_report = (
        "# Legal and Observed Movement Inventory\n\n"
        "The legal inventory is defined only by the frozen human polygon protocol. "
        "Observed counts use valid deterministic endpoint-containment labels. No HG target "
        "or clustering output is compared here.\n\n"
        "## Scene Summary\n\n"
        + scene_inventory_frame.to_markdown(index=False)
        + "\n\n## Legal but Not Observed\n\n"
        + (unobserved.to_markdown(index=False) if len(unobserved) else "None.")
        + f"\n\n## Rejected Non-Legal Endpoint Pairs\n\nCount: **{illegal_count}**\n"
    )
    (output_dir / "movement_inventory_report.md").write_text(inventory_report, encoding="utf-8")

    movement_counts = (
        labels[labels["reference_status"] == "valid"]
        .groupby(["scene_id", "split", "reference_movement_id"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    quality_report = (
        "# Polygon Reference Label Quality Report\n\n"
        "Primary labels use the first and last finite point inside each canonical manifest "
        "interval and Shapely `covers`. Boundary-near diagnostics do not exclude labels unless "
        "an endpoint is covered by multiple polygons.\n\n"
        "## Scene and Split Coverage\n\n"
        + quality.to_markdown(index=False, floatfmt=".3f")
        + "\n\n## Exclusion Reasons\n\n"
        + reasons.to_markdown(index=False)
        + "\n\n## Per-Movement Counts\n\n"
        + movement_counts.to_markdown(index=False)
        + "\n\n## Recording-Level Coverage\n\n"
        + recording.to_markdown(index=False, floatfmt=".3f")
        + "\n"
    )
    (docs_dir / "polygon_reference_label_quality_report.md").write_text(
        quality_report, encoding="utf-8"
    )
    return legal, observed, quality


def _labels_as_endpoint_source(labels: pd.DataFrame) -> pd.DataFrame:
    frame = labels.copy()
    frame["source_recording_id"] = frame["recording_id"]
    frame["data_fingerprint"] = frame["trajectory_fingerprint"]
    frame["source_file_checksum_sha256"] = frame["source_checksum"]
    geometry_reasons = {"missing_geometry", "invalid_endpoint", "source_mapping_error"}
    frame["geometry_error"] = frame["exclusion_reason"].where(
        frame["exclusion_reason"].isin(geometry_reasons), ""
    )
    return frame


def run_sensitivity_analysis(
    labels: pd.DataFrame,
    publication_root: Path,
    protocol: dict[str, Any],
    output_path: Path,
    report_path: Path,
) -> pd.DataFrame:
    source = _labels_as_endpoint_source(labels)
    variants: list[tuple[str, pd.DataFrame]] = []
    primary_geometry = build_scene_geometries(publication_root, protocol)
    variants.append(
        (
            "median_first_last_3",
            assign_reference_rows(
                source,
                primary_geometry,
                protocol,
                ("entry_median3_x", "entry_median3_y"),
                ("exit_median3_x", "exit_median3_y"),
            ),
        )
    )
    variants.append(
        (
            "median_first_last_5",
            assign_reference_rows(
                source,
                primary_geometry,
                protocol,
                ("entry_median5_x", "entry_median5_y"),
                ("exit_median5_x", "exit_median5_y"),
            ),
        )
    )
    margin = float(protocol["sensitivity_polygon_buffer_px"])
    variants.append(
        (
            f"polygon_inward_{margin:g}px",
            assign_reference_rows(
                source,
                build_scene_geometries(publication_root, protocol, -margin),
                protocol,
            ),
        )
    )
    variants.append(
        (
            f"polygon_outward_{margin:g}px",
            assign_reference_rows(
                source,
                build_scene_geometries(publication_root, protocol, margin),
                protocol,
            ),
        )
    )
    primary = labels.set_index("trajectory_id", drop=False)
    comparison_rows = []
    primary_boundary = labels[["entry_boundary_distance", "exit_boundary_distance"]].min(axis=1)
    boundary_lookup = dict(zip(labels["trajectory_id"], primary_boundary, strict=True))
    for variant_name, variant in variants:
        variant_index = variant.set_index("trajectory_id", drop=False)
        for trajectory_id in primary.index:
            base = primary.loc[trajectory_id]
            changed = variant_index.loc[trajectory_id]
            comparison_rows.append(
                {
                    "scene_id": base["scene_id"],
                    "trajectory_id": trajectory_id,
                    "split": base["split"],
                    "variant": variant_name,
                    "primary_reference_status": base["reference_status"],
                    "variant_reference_status": changed["reference_status"],
                    "primary_movement_id": base["reference_movement_id"],
                    "variant_movement_id": changed["reference_movement_id"],
                    "primary_exclusion_reason": base["exclusion_reason"],
                    "variant_exclusion_reason": changed["exclusion_reason"],
                    "movement_label_changed": bool(
                        base["reference_movement_id"] != changed["reference_movement_id"]
                        or base["reference_status"] != changed["reference_status"]
                    ),
                    "status_changed": bool(base["reference_status"] != changed["reference_status"]),
                    "primary_boundary_distance_min": boundary_lookup[trajectory_id],
                    "primary_boundary_near_10px": bool(boundary_lookup[trajectory_id] <= 10),
                    "protocol_version": protocol["protocol_version"],
                    "protocol_hash": protocol["protocol_hash"],
                }
            )
    sensitivity = pd.DataFrame(comparison_rows)
    deterministic_write_csv(sensitivity, output_path)
    summary = (
        sensitivity.groupby(["scene_id", "variant"], sort=False)
        .agg(
            trajectories=("trajectory_id", "size"),
            changed_count=("movement_label_changed", "sum"),
            changed_percentage=("movement_label_changed", "mean"),
            status_changed_count=("status_changed", "sum"),
            boundary_near_count=("primary_boundary_near_10px", "sum"),
        )
        .reset_index()
    )
    summary["changed_percentage"] *= 100
    transitions = (
        sensitivity.groupby(
            ["variant", "primary_reference_status", "variant_reference_status"], sort=False
        )
        .size()
        .rename("count")
        .reset_index()
    )
    movement_stability = (
        sensitivity[sensitivity["primary_movement_id"] != ""]
        .groupby(["scene_id", "variant", "primary_movement_id"], sort=False)
        .agg(
            trajectories=("trajectory_id", "size"),
            stable_count=("movement_label_changed", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    movement_stability["stable_percentage"] = (
        100 * movement_stability["stable_count"] / movement_stability["trajectories"]
    )
    report = (
        "# Polygon Assignment Sensitivity Report\n\n"
        "The primary published reference remains the canonical first/last finite-point "
        "assignment. Variants are deterministic diagnostics and were not selected using "
        "clustering, HG targets, OD pseudo-labels, or EMAS. The polygon buffer magnitude is "
        f"{margin:g} camera pixels.\n\n"
        "## Scene-Level Stability\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## Status Transitions\n\n"
        + transitions.to_markdown(index=False)
        + "\n\n## Movement-Level Stability\n\n"
        + movement_stability.to_markdown(index=False, floatfmt=".4f")
        + "\n"
    )
    report_path.write_text(report, encoding="utf-8")
    return sensitivity


def _deterministic_rank(trajectory_id: str, protocol_hash: str) -> str:
    return hashlib.sha256(f"{protocol_hash}|{trajectory_id}".encode("utf-8")).hexdigest()


def build_qa_audit_queue(
    labels: pd.DataFrame, protocol: dict[str, Any], path: Path
) -> pd.DataFrame:
    reasons: dict[str, set[str]] = {}

    def add(frame: pd.DataFrame, reason: str) -> None:
        for trajectory_id in frame["trajectory_id"].astype(str):
            reasons.setdefault(trajectory_id, set()).add(reason)

    ranked = labels.assign(
        _rank=labels["trajectory_id"].map(
            lambda value: _deterministic_rank(str(value), protocol["protocol_hash"])
        )
    )
    for scene in SCENES:
        scene = str(scene)
        scene_frame = ranked[ranked["scene_id"] == scene]
        add(
            scene_frame[scene_frame["reference_status"] == "valid"].sort_values("_rank").head(20),
            "deterministic_valid_example",
        )
        no_polygon = scene_frame[
            scene_frame["exclusion_reason"].isin(["entry_no_polygon", "exit_no_polygon"])
        ].sort_values("_rank")
        add(no_polygon.head(100), "capped_no_polygon_example")
        boundary = scene_frame[
            scene_frame[["entry_boundary_distance", "exit_boundary_distance"]].min(axis=1) <= 3
        ].sort_values("_rank")
        add(boundary.head(100), "boundary_near_3px_example")
    add(
        labels[(labels["entry_match_count"] > 1) | (labels["exit_match_count"] > 1)],
        "all_multiple_polygon_cases",
    )
    add(
        labels[labels["exclusion_reason"] == "mapping_not_legal"],
        "all_illegal_mapping_cases",
    )
    selected = labels[labels["trajectory_id"].isin(reasons)].copy()
    selected["qa_reason"] = selected["trajectory_id"].map(
        lambda value: "|".join(sorted(reasons[str(value)]))
    )
    selected["qa_deterministic_rank"] = selected["trajectory_id"].map(
        lambda value: _deterministic_rank(str(value), protocol["protocol_hash"])
    )
    selected = selected.sort_values(
        ["scene_id", "qa_reason", "qa_deterministic_rank", "trajectory_id"],
        kind="mergesort",
    )
    deterministic_write_csv(selected, path)
    return selected


def _load_frame(publication_root: Path, guide: dict[str, Any]) -> np.ndarray:
    path = publication_root / "annotations" / guide["representative_frame"]
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot load representative frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_polygons(ax: Any, guide: dict[str, Any], width: int, height: int) -> None:
    for approach in guide["approaches"]:
        identifier = str(approach["id"])
        vertices = np.array(
            [
                [float(point["x"]) * width, float(point["y"]) * height]
                for point in approach["polygon_normalized"]
            ]
        )
        ax.add_patch(
            PolygonPatch(
                vertices,
                closed=True,
                fill=False,
                linewidth=1.8,
                edgecolor=POLYGON_COLORS[identifier],
            )
        )
        center = vertices.mean(axis=0)
        ax.text(
            center[0],
            center[1],
            identifier,
            color="white",
            weight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.75, "pad": 2},
        )


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_qc_figures(
    labels: pd.DataFrame,
    publication_root: Path,
    protocol: dict[str, Any],
    figures_root: Path,
) -> None:
    for guide in protocol["scene_guides"]:
        scene = str(guide["scene_id"])
        scene_dir = figures_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        source_guide = publication_root / "annotations/protocol/scene_guides" / f"{scene}_guide.png"
        shutil.copyfile(source_guide, scene_dir / "01_representative_frame_with_polygons.png")
        image = _load_frame(publication_root, guide)
        height, width = image.shape[:2]
        scene_labels = labels[labels["scene_id"] == scene]

        for prefix, filename, title in (
            ("entry", "02_entry_endpoints_by_polygon.png", "Entry endpoints"),
            ("exit", "03_exit_endpoints_by_polygon.png", "Exit endpoints"),
        ):
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.imshow(image)
            for identifier in ("A", "B", "C", "D") if prefix == "entry" else ("E", "F", "G", "H"):
                subset = scene_labels[scene_labels[f"{prefix}_polygon_id"] == identifier]
                ax.scatter(
                    subset[f"{prefix}_x"],
                    subset[f"{prefix}_y"],
                    s=3,
                    alpha=0.35,
                    color=POLYGON_COLORS[identifier],
                    label=identifier,
                )
            _draw_polygons(ax, guide, width, height)
            ax.set_title(f"{scene}: {title} by manual polygon")
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.axis("off")
            ax.legend(loc="upper right", ncol=4)
            _save_figure(fig, scene_dir / filename)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(image)
        entry_bad = scene_labels[scene_labels["entry_match_count"] != 1]
        exit_bad = scene_labels[scene_labels["exit_match_count"] != 1]
        ax.scatter(entry_bad["entry_x"], entry_bad["entry_y"], s=7, color="#ffbf00", label="entry")
        ax.scatter(
            exit_bad["exit_x"],
            exit_bad["exit_y"],
            s=9,
            facecolors="none",
            edgecolors="#d62728",
            label="exit",
        )
        _draw_polygons(ax, guide, width, height)
        ax.set_title(f"{scene}: unassigned or ambiguous endpoints")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        ax.legend(loc="upper right")
        _save_figure(fig, scene_dir / "04_unassigned_ambiguous_endpoints.png")

        fig, ax = plt.subplots(figsize=(12, 7))
        all_x = pd.concat([scene_labels["entry_x"], scene_labels["exit_x"]]).dropna()
        all_y = pd.concat([scene_labels["entry_y"], scene_labels["exit_y"]]).dropna()
        ax.imshow(image, alpha=0.35)
        ax.hist2d(all_x, all_y, bins=(80, 45), range=((0, width), (0, height)), cmap="magma")
        _draw_polygons(ax, guide, width, height)
        ax.set_title(f"{scene}: endpoint density with manual polygons")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        _save_figure(fig, scene_dir / "05_endpoint_density_with_polygons.png")

        matrix = np.zeros((4, 4), dtype=int)
        valid = scene_labels[scene_labels["reference_status"] == "valid"]
        entries = ["A", "B", "C", "D"]
        exits = ["E", "F", "G", "H"]
        for entry_index, entry in enumerate(entries):
            for exit_index, exit_id in enumerate(exits):
                matrix[entry_index, exit_index] = int(
                    (
                        (valid["entry_polygon_id"] == entry) & (valid["exit_polygon_id"] == exit_id)
                    ).sum()
                )
        fig, ax = plt.subplots(figsize=(7, 6))
        image_handle = ax.imshow(matrix, cmap="Blues")
        for row_index in range(4):
            for column_index in range(4):
                ax.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                )
        ax.set_xticks(range(4), exits)
        ax.set_yticks(range(4), entries)
        ax.set_xlabel("Exit polygon")
        ax.set_ylabel("Entry polygon")
        ax.set_title(f"{scene}: valid movement counts")
        fig.colorbar(image_handle, ax=ax, shrink=0.8)
        _save_figure(fig, scene_dir / "06_movement_count_matrix.png")


def write_qa_statement(path: Path, queue: pd.DataFrame) -> None:
    reason_counts = (
        queue["qa_reason"].value_counts().rename_axis("qa_reason").reset_index(name="count")
    )
    text = (
        "# Polygon Reference QA Audit Queue\n\n"
        "> Manual review of the QA queue is a quality-control step only. It does not define "
        "the reference labels and does not cherry-pick the scientific evaluation cohort.\n\n"
        f"Queue rows: **{len(queue)}**. No manual audit decisions were created.\n\n"
        + reason_counts.to_markdown(index=False)
        + "\n"
    )
    path.write_text(text, encoding="utf-8")
