"""Exhaustive deterministic polygon-rule reference-label generation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

try:
    from .protocol import SCENES, load_frozen_protocol, sha256_file
except ImportError:  # Direct script execution.
    from protocol import SCENES, load_frozen_protocol, sha256_file


LABEL_GENERATION_VERSION = "polygon-rule-label-generator-v1"
EXPECTED_TOTAL = 67_029
SPLITS = ("target_estimation", "model_selection", "independent_test")
BOUNDARY_NEAR_THRESHOLDS_PX = (1.0, 3.0, 5.0, 10.0)
OUTPUT_COLUMNS = (
    "scene_id",
    "trajectory_id",
    "split",
    "recording_id",
    "source_file",
    "start_frame",
    "end_frame",
    "entry_x",
    "entry_y",
    "exit_x",
    "exit_y",
    "entry_polygon_id",
    "exit_polygon_id",
    "entry_match_count",
    "exit_match_count",
    "entry_boundary_distance",
    "exit_boundary_distance",
    "entry_on_boundary",
    "exit_on_boundary",
    "reference_movement_id",
    "reference_maneuver_type",
    "reference_status",
    "exclusion_reason",
    "protocol_version",
    "protocol_hash",
    "trajectory_fingerprint",
    "source_checksum",
    "label_generation_version",
    "label_generation_timestamp_utc",
)
DIAGNOSTIC_COLUMNS = (
    "entry_matched_polygon_ids",
    "exit_matched_polygon_ids",
    "entry_nearest_polygon_id",
    "exit_nearest_polygon_id",
    "entry_nearest_polygon_distance",
    "exit_nearest_polygon_distance",
    "exclusion_reasons_all",
    "trajectory_source_path",
    "trajectory_source_checksum",
    "finite_canonical_point_count",
    "entry_median3_x",
    "entry_median3_y",
    "exit_median3_x",
    "exit_median3_y",
    "entry_median5_x",
    "entry_median5_y",
    "exit_median5_x",
    "exit_median5_y",
)


@dataclass(frozen=True)
class EndpointAssignment:
    matched_ids: tuple[str, ...]
    boundary_distance: float
    on_boundary: bool
    nearest_id: str
    nearest_distance: float


@dataclass(frozen=True)
class SceneGeometry:
    scene_id: str
    width: int
    height: int
    entries: dict[str, Polygon]
    exits: dict[str, Polygon]
    legal_mapping: dict[tuple[str, str], str]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_canonical_index(publication_root: Path) -> pd.DataFrame:
    manifest_path = publication_root / "data/manifests/trajectory_manifest.csv"
    split_path = publication_root / "data/splits/evaluation_split.csv"
    catalog_path = publication_root / "annotations/provenance/trajectory_source_catalog.csv"
    manifest_columns = [
        "scene_id",
        "trajectory_id",
        "source_file",
        "source_recording_id",
        "source_recording_file",
        "source_recording_track_id",
        "start_frame",
        "end_frame",
        "number_of_points",
        "data_fingerprint",
        "source_file_checksum_sha256",
    ]
    manifest = pd.read_csv(manifest_path, usecols=manifest_columns, keep_default_na=False)
    split = pd.read_csv(split_path, usecols=["trajectory_id", "split"], keep_default_na=False)
    catalog = pd.read_csv(catalog_path, keep_default_na=False)
    combined = manifest.merge(split, on="trajectory_id", how="left", validate="one_to_one")
    combined = combined.merge(
        catalog[
            [
                "scene_id",
                "recording_id",
                "trajectory_source_path",
                "trajectory_source_checksum",
            ]
        ],
        left_on=["scene_id", "source_recording_id"],
        right_on=["scene_id", "recording_id"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["recording_id"])
    if len(combined) != EXPECTED_TOTAL or not combined["trajectory_id"].is_unique:
        raise ValueError("Canonical cohort must contain exactly 67,029 unique trajectories.")
    if tuple(combined["scene_id"].drop_duplicates()) != SCENES:
        raise ValueError("Canonical cohort is not the fixed five-scene set.")
    if set(combined["split"]) != set(SPLITS):
        raise ValueError("Canonical cohort does not contain exactly the three frozen splits.")
    combined["_canonical_order"] = np.arange(len(combined), dtype=np.int64)
    return combined


def _read_selected_tracks(path: Path, track_ids: list[int]) -> pd.DataFrame:
    columns = ["track_id", "frame", "cx", "cy"]
    try:
        frame = pd.read_parquet(path, columns=columns, filters=[("track_id", "in", track_ids)])
    except (ValueError, TypeError, NotImplementedError):
        frame = pd.read_parquet(path, columns=columns)
    return frame[frame["track_id"].isin(track_ids)].copy()


def _endpoint_statistics(points: pd.DataFrame) -> dict[str, float | int]:
    points = points.sort_values(["frame"], kind="mergesort")
    coordinates = points[["cx", "cy"]].to_numpy(dtype=float)
    first3 = coordinates[:3]
    last3 = coordinates[-3:]
    first5 = coordinates[:5]
    last5 = coordinates[-5:]
    return {
        "entry_x": float(coordinates[0, 0]),
        "entry_y": float(coordinates[0, 1]),
        "exit_x": float(coordinates[-1, 0]),
        "exit_y": float(coordinates[-1, 1]),
        "entry_median3_x": float(np.median(first3[:, 0])),
        "entry_median3_y": float(np.median(first3[:, 1])),
        "exit_median3_x": float(np.median(last3[:, 0])),
        "exit_median3_y": float(np.median(last3[:, 1])),
        "entry_median5_x": float(np.median(first5[:, 0])),
        "entry_median5_y": float(np.median(first5[:, 1])),
        "exit_median5_x": float(np.median(last5[:, 0])),
        "exit_median5_y": float(np.median(last5[:, 1])),
        "finite_canonical_point_count": int(len(points)),
    }


def extract_canonical_endpoints(repo_root: Path, index: pd.DataFrame) -> pd.DataFrame:
    """Read each source shard once and extract canonical first/last finite points."""
    output = index.copy()
    endpoint_columns = [
        "entry_x",
        "entry_y",
        "exit_x",
        "exit_y",
        "entry_median3_x",
        "entry_median3_y",
        "exit_median3_x",
        "exit_median3_y",
        "entry_median5_x",
        "entry_median5_y",
        "exit_median5_x",
        "exit_median5_y",
    ]
    for column in endpoint_columns:
        output[column] = np.nan
    output["finite_canonical_point_count"] = 0
    output["geometry_error"] = ""

    grouped = output.groupby(["scene_id", "source_recording_id"], sort=False)
    for (_, _), row_indices in grouped.groups.items():
        metadata = output.loc[list(row_indices)]
        source_value = metadata["trajectory_source_path"].iloc[0]
        if not isinstance(source_value, str) or not source_value:
            output.loc[list(row_indices), "geometry_error"] = "source_mapping_error"
            continue
        path = repo_root / source_value
        if not path.exists():
            output.loc[list(row_indices), "geometry_error"] = "source_mapping_error"
            continue
        local_ids = metadata["source_recording_track_id"].astype(int).tolist()
        if len(local_ids) != len(set(local_ids)):
            output.loc[list(row_indices), "geometry_error"] = "source_mapping_error"
            continue
        try:
            source = _read_selected_tracks(path, local_ids)
        except Exception:
            output.loc[list(row_indices), "geometry_error"] = "source_mapping_error"
            continue
        source_groups = {int(key): value for key, value in source.groupby("track_id", sort=False)}
        for row_index, row in metadata.iterrows():
            track_id = int(row["source_recording_track_id"])
            track = source_groups.get(track_id)
            if track is None or track.empty:
                output.at[row_index, "geometry_error"] = "missing_geometry"
                continue
            canonical = track[
                (track["frame"] >= int(row["start_frame"]))
                & (track["frame"] <= int(row["end_frame"]))
            ]
            if canonical.empty:
                output.at[row_index, "geometry_error"] = "missing_geometry"
                continue
            finite = canonical[
                np.isfinite(canonical["cx"].to_numpy(dtype=float))
                & np.isfinite(canonical["cy"].to_numpy(dtype=float))
            ]
            if finite.empty:
                output.at[row_index, "geometry_error"] = "invalid_endpoint"
                continue
            statistics = _endpoint_statistics(finite)
            for column, value in statistics.items():
                output.at[row_index, column] = value
    return output.sort_values("_canonical_order", kind="mergesort").reset_index(drop=True)


def build_scene_geometries(
    publication_root: Path,
    protocol: dict[str, Any],
    polygon_buffer_px: float = 0.0,
) -> dict[str, SceneGeometry]:
    geometries: dict[str, SceneGeometry] = {}
    annotation_root = publication_root / "annotations"
    for guide in protocol["scene_guides"]:
        frame_path = annotation_root / guide["representative_frame"]
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Cannot read representative frame: {frame_path}")
        height, width = image.shape[:2]
        entries: dict[str, Polygon] = {}
        exits: dict[str, Polygon] = {}
        for approach in guide["approaches"]:
            polygon = Polygon(
                [
                    (float(point["x"]) * width, float(point["y"]) * height)
                    for point in approach["polygon_normalized"]
                ]
            )
            if polygon_buffer_px:
                polygon = polygon.buffer(float(polygon_buffer_px), join_style="mitre")
            if polygon.is_empty:
                polygon = Polygon()
            target = entries if approach["region_role"] == "entry" else exits
            target[str(approach["id"])] = polygon
        mapping = {
            (str(row["entry"]), str(row["exit"])): str(row["maneuver_type"])
            for row in guide["maneuver_type_mapping"]
        }
        geometries[str(guide["scene_id"])] = SceneGeometry(
            scene_id=str(guide["scene_id"]),
            width=width,
            height=height,
            entries=entries,
            exits=exits,
            legal_mapping=mapping,
        )
    return geometries


def assign_endpoint(
    x: float,
    y: float,
    polygons: dict[str, Polygon],
    boundary_tolerance_px: float,
) -> EndpointAssignment:
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Endpoint coordinates must be finite.")
    point = Point(float(x), float(y))
    matched = tuple(identifier for identifier, polygon in polygons.items() if polygon.covers(point))
    boundary_distances = {
        identifier: float(point.distance(polygon.boundary))
        for identifier, polygon in polygons.items()
        if not polygon.is_empty
    }
    nearest_distances = {
        identifier: float(point.distance(polygon))
        for identifier, polygon in polygons.items()
        if not polygon.is_empty
    }
    boundary_distance = min(boundary_distances.values(), default=math.nan)
    nearest_id = min(nearest_distances, key=lambda key: (nearest_distances[key], key))
    return EndpointAssignment(
        matched_ids=matched,
        boundary_distance=boundary_distance,
        on_boundary=bool(boundary_distance <= boundary_tolerance_px),
        nearest_id=nearest_id,
        nearest_distance=nearest_distances[nearest_id],
    )


def _status_and_reasons(
    geometry_error: str,
    entry: EndpointAssignment | None,
    exit_assignment: EndpointAssignment | None,
    legal: bool,
) -> tuple[str, str, str]:
    reasons: list[str] = []
    if geometry_error:
        reasons.append(geometry_error)
    elif entry is None or exit_assignment is None:
        reasons.append("invalid_endpoint")
    else:
        if len(entry.matched_ids) == 0:
            reasons.append("entry_no_polygon")
        elif len(entry.matched_ids) > 1:
            reasons.append("entry_multiple_polygons")
        if len(exit_assignment.matched_ids) == 0:
            reasons.append("exit_no_polygon")
        elif len(exit_assignment.matched_ids) > 1:
            reasons.append("exit_multiple_polygons")
        if not reasons and not legal:
            reasons.append("mapping_not_legal")
    if not reasons:
        return "valid", "", ""
    primary = reasons[0]
    if "multiple_polygons" in primary:
        status = "ambiguous"
    elif primary.endswith("no_polygon"):
        status = "unassigned"
    else:
        status = "excluded"
    return status, primary, "|".join(reasons)


def assign_reference_rows(
    endpoints: pd.DataFrame,
    geometries: dict[str, SceneGeometry],
    protocol: dict[str, Any],
    entry_columns: tuple[str, str] = ("entry_x", "entry_y"),
    exit_columns: tuple[str, str] = ("exit_x", "exit_y"),
) -> pd.DataFrame:
    boundary_tolerance = float(protocol["boundary_tolerance_px"])
    rows: list[dict[str, Any]] = []
    for row in endpoints.itertuples(index=False):
        payload = row._asdict()
        geometry = geometries[str(payload["scene_id"])]
        geometry_error = str(payload.get("geometry_error", ""))
        entry_assignment: EndpointAssignment | None = None
        exit_assignment: EndpointAssignment | None = None
        if not geometry_error:
            try:
                entry_assignment = assign_endpoint(
                    float(payload[entry_columns[0]]),
                    float(payload[entry_columns[1]]),
                    geometry.entries,
                    boundary_tolerance,
                )
                exit_assignment = assign_endpoint(
                    float(payload[exit_columns[0]]),
                    float(payload[exit_columns[1]]),
                    geometry.exits,
                    boundary_tolerance,
                )
            except (TypeError, ValueError):
                geometry_error = "invalid_endpoint"
        entry_id = (
            entry_assignment.matched_ids[0]
            if entry_assignment is not None and len(entry_assignment.matched_ids) == 1
            else ""
        )
        exit_id = (
            exit_assignment.matched_ids[0]
            if exit_assignment is not None and len(exit_assignment.matched_ids) == 1
            else ""
        )
        maneuver = geometry.legal_mapping.get((entry_id, exit_id), "")
        legal = bool(entry_id and exit_id and maneuver)
        status, reason, all_reasons = _status_and_reasons(
            geometry_error, entry_assignment, exit_assignment, legal
        )
        movement_id = f"{payload['scene_id']}:{entry_id}>{exit_id}" if status == "valid" else ""
        rows.append(
            {
                "scene_id": payload["scene_id"],
                "trajectory_id": payload["trajectory_id"],
                "split": payload["split"],
                "recording_id": payload["source_recording_id"],
                "source_file": payload["source_file"],
                "start_frame": int(payload["start_frame"]),
                "end_frame": int(payload["end_frame"]),
                "entry_x": payload[entry_columns[0]],
                "entry_y": payload[entry_columns[1]],
                "exit_x": payload[exit_columns[0]],
                "exit_y": payload[exit_columns[1]],
                "entry_polygon_id": entry_id,
                "exit_polygon_id": exit_id,
                "entry_match_count": (
                    len(entry_assignment.matched_ids) if entry_assignment is not None else 0
                ),
                "exit_match_count": (
                    len(exit_assignment.matched_ids) if exit_assignment is not None else 0
                ),
                "entry_boundary_distance": (
                    entry_assignment.boundary_distance if entry_assignment is not None else math.nan
                ),
                "exit_boundary_distance": (
                    exit_assignment.boundary_distance if exit_assignment is not None else math.nan
                ),
                "entry_on_boundary": (
                    entry_assignment.on_boundary if entry_assignment is not None else False
                ),
                "exit_on_boundary": (
                    exit_assignment.on_boundary if exit_assignment is not None else False
                ),
                "reference_movement_id": movement_id,
                "reference_maneuver_type": maneuver if status == "valid" else "",
                "reference_status": status,
                "exclusion_reason": reason,
                "protocol_version": protocol["protocol_version"],
                "protocol_hash": protocol["protocol_hash"],
                "trajectory_fingerprint": payload["data_fingerprint"],
                "source_checksum": payload["source_file_checksum_sha256"],
                "label_generation_version": LABEL_GENERATION_VERSION,
                "label_generation_timestamp_utc": protocol["label_generation_timestamp_utc"],
                "entry_matched_polygon_ids": (
                    "|".join(entry_assignment.matched_ids) if entry_assignment is not None else ""
                ),
                "exit_matched_polygon_ids": (
                    "|".join(exit_assignment.matched_ids) if exit_assignment is not None else ""
                ),
                "entry_nearest_polygon_id": (
                    entry_assignment.nearest_id if entry_assignment is not None else ""
                ),
                "exit_nearest_polygon_id": (
                    exit_assignment.nearest_id if exit_assignment is not None else ""
                ),
                "entry_nearest_polygon_distance": (
                    entry_assignment.nearest_distance if entry_assignment is not None else math.nan
                ),
                "exit_nearest_polygon_distance": (
                    exit_assignment.nearest_distance if exit_assignment is not None else math.nan
                ),
                "exclusion_reasons_all": all_reasons,
                "trajectory_source_path": payload["trajectory_source_path"],
                "trajectory_source_checksum": payload["trajectory_source_checksum"],
                "finite_canonical_point_count": int(payload["finite_canonical_point_count"]),
                "entry_median3_x": payload["entry_median3_x"],
                "entry_median3_y": payload["entry_median3_y"],
                "exit_median3_x": payload["exit_median3_x"],
                "exit_median3_y": payload["exit_median3_y"],
                "entry_median5_x": payload["entry_median5_x"],
                "entry_median5_y": payload["entry_median5_y"],
                "exit_median5_x": payload["exit_median5_x"],
                "exit_median5_y": payload["exit_median5_y"],
            }
        )
    return pd.DataFrame(rows, columns=[*OUTPUT_COLUMNS, *DIAGNOSTIC_COLUMNS])


def deterministic_write_csv(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")
    return sha256_file(path)


def write_reference_outputs(
    labels: pd.DataFrame,
    output_dir: Path,
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    if len(labels) != EXPECTED_TOTAL or not labels["trajectory_id"].is_unique:
        raise ValueError("Reference output must contain every canonical trajectory exactly once.")
    output_dir.mkdir(parents=True, exist_ok=True)
    publication_root = output_dir.parents[1]

    def portable_path(path: Path) -> str:
        try:
            return path.relative_to(publication_root).as_posix()
        except ValueError:
            return path.as_posix()

    files: dict[str, dict[str, Any]] = {}
    all_csv = output_dir / "polygon_rule_reference_labels_all.csv"
    files[all_csv.name] = {
        "path": portable_path(all_csv),
        "sha256": deterministic_write_csv(labels, all_csv),
        "rows": len(labels),
    }
    all_parquet = output_dir / "polygon_rule_reference_labels_all.parquet"
    labels.to_parquet(all_parquet, index=False, compression="zstd")
    files[all_parquet.name] = {
        "path": portable_path(all_parquet),
        "sha256": sha256_file(all_parquet),
        "rows": len(labels),
    }
    for split in SPLITS:
        split_frame = labels[labels["split"] == split].copy()
        path = output_dir / f"{split}_reference_labels.csv"
        files[path.name] = {
            "path": portable_path(path),
            "sha256": deterministic_write_csv(split_frame, path),
            "rows": len(split_frame),
        }
    manifest = {
        "label_generation_version": LABEL_GENERATION_VERSION,
        "row_count": len(labels),
        "trajectory_id_set_sha256": _sha256_text("\n".join(labels["trajectory_id"])),
        "input_files": {
            name: {"path": portable_path(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "output_files": files,
    }
    manifest_path = output_dir / "reference_output_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def generate_primary_reference_dataset(
    repo_root: Path,
    publication_root: Path,
    protocol_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol = load_frozen_protocol(protocol_path)
    index = load_canonical_index(publication_root)
    endpoints = extract_canonical_endpoints(repo_root, index)
    geometries = build_scene_geometries(publication_root, protocol)
    labels = assign_reference_rows(endpoints, geometries, protocol)
    input_paths = {
        "trajectory_manifest": publication_root / "data/manifests/trajectory_manifest.csv",
        "evaluation_split": publication_root / "data/splits/evaluation_split.csv",
        "source_catalog": publication_root / "annotations/provenance/trajectory_source_catalog.csv",
        "frozen_polygon_protocol": protocol_path,
        "frozen_model_selection_manifest": publication_root
        / "results/development/frozen_selection_manifest.json",
    }
    manifest = write_reference_outputs(
        labels,
        publication_root / "annotations/reference_labels",
        input_paths,
    )
    return labels, manifest
