from __future__ import annotations

import argparse
import fnmatch
import json
import math
import pickle
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None


SUPPORTED_TABLE_EXTS = {".csv", ".parquet", ".joblib", ".pkl", ".pickle", ".sqlite", ".db"}
PROXY_SAMPLE_ROWS_PER_FILE = 100_000
SCENE_HINTS = [
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
]

DETECTION_DIR_NAMES = {
    "02_yolov11x_detections",
    "02_yolo11x_detections",
    "02_detections",
}
TRACKING_DIR_NAMES = {
    "03_tracked_trajectories",
    "03_yolo_tracking_outputs",
    "03_tracking_outputs",
}
RAW_TRAJECTORY_DIR_NAMES = {
    "04_trajectories_raw",
    "raw_trajectories",
}
CLEANED_DIR_NAMES = {
    "04_cleaned_trajectories",
    "05_trajectories_filtered_filled_smoothed",
    "cleaned_trajectories",
}
HOMOGRAPHY_DIR_NAMES = {
    "05_map_based_geometric_correction",
    "06_homography_map_correction",
    "06_perspective_transform_tool",
    "07_perspective_transformation_tool",
}
RESULTS_DIR_NAMES = {
    "07_results_and_statistics",
    "09_metadata_and_statistics",
    "results",
    "reports",
}

ALIASES = {
    "scene": ["scene", "scene_id", "location", "site"],
    "source_file": ["source_file", "video_id", "video", "file", "input_file"],
    "frame": ["frame", "frame_id", "frame_number", "frame_idx"],
    "track_id": ["track_id", "id", "object_id", "obj_id", "objid"],
    "class_id": ["class_id", "cls", "class"],
    "class_name": ["class_name", "label", "category", "name"],
    "confidence": ["confidence", "conf", "score"],
    "x1": ["x1", "xmin", "left"],
    "y1": ["y1", "ymin", "top"],
    "x2": ["x2", "xmax", "right"],
    "y2": ["y2", "ymax", "bottom"],
    "cx": ["cx", "center_x", "x_center"],
    "cy": ["cy", "center_y", "y_center"],
    "x": ["x", "cx", "center_x", "x_image", "image_x"],
    "y": ["y", "cy", "center_y", "y_image", "image_y"],
    "w": ["w", "width", "bbox_width"],
    "h": ["h", "height", "bbox_height"],
    "map_x": ["map_x", "x_map", "x_corrected", "topview_x"],
    "map_y": ["map_y", "y_map", "y_corrected", "topview_y"],
    "vx": ["vx", "velocity_x"],
    "vy": ["vy", "velocity_y"],
    "ax": ["ax", "acceleration_x"],
    "ay": ["ay", "acceleration_y"],
    "is_filled": ["is_filled", "filled", "interpolated", "is_interpolated"],
}


@dataclass
class RunLog:
    new_root: Path
    old_root: Path | None
    out_dir: Path
    warnings: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    discovered: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def info(self, message: str) -> None:
        self.messages.append(message)
        print(message)

    def output(self, path: Path) -> None:
        self.outputs.append(path)
        print(f"Wrote {path}")


@dataclass
class ReleaseFiles:
    root: Path
    label: str
    detection_files: list[Path] = field(default_factory=list)
    tracked_observation_files: list[Path] = field(default_factory=list)
    raw_trajectory_files: list[Path] = field(default_factory=list)
    cleaned_trajectory_files: list[Path] = field(default_factory=list)
    cluster_files: list[Path] = field(default_factory=list)
    homography_files: list[Path] = field(default_factory=list)


@dataclass
class ReleaseMetrics:
    label: str
    root: Path
    metrics_per_scene: pd.DataFrame
    metrics_overall: pd.DataFrame
    class_distribution: pd.DataFrame
    track_length_summary: pd.DataFrame
    continuity_summary: pd.DataFrame
    gap_summary: pd.DataFrame
    cluster_outlier_summary: pd.DataFrame
    homography_summary: pd.DataFrame
    detail_vectors: dict[str, pd.DataFrame] = field(default_factory=dict)


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def normalize_part(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def find_col(df: pd.DataFrame, logical: str) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for alias in ALIASES.get(logical, [logical]):
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def numeric(df: pd.DataFrame, logical: str) -> pd.Series:
    col = find_col(df, logical)
    if col is None:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def boolish(df: pd.DataFrame, logical: str) -> pd.Series:
    col = find_col(df, logical)
    if col is None:
        return pd.Series([False] * len(df), index=df.index, dtype="bool")
    raw = df[col]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False)
    return raw.map(lambda v: str(v).strip().lower() in {"1", "true", "yes", "y"}).fillna(False)


def infer_scene(path: Path, root: Path) -> str:
    text = rel(path, root).replace("\\", "/").lower()
    normalized = text.replace("-", "_")
    for scene in sorted(SCENE_HINTS, key=len, reverse=True):
        if scene in normalized:
            return scene
    parts = [normalize_part(p) for p in Path(text).parts]
    for marker in ["interim", "processed", "raw", "data"]:
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                candidate = parts[idx + 1]
                if candidate and not candidate.startswith("."):
                    return candidate
    for part in parts:
        if "bellevue" in part:
            return part
    return "unknown_scene"


def infer_source(path: Path, root: Path, scene: str) -> str:
    stem = path.stem
    lower = stem.lower()
    for prefix in ["tracks_", "detections_", "trajectory_", "trajectories_"]:
        if lower.startswith(prefix):
            return stem[len(prefix) :]
    if lower in {"tracks", "trajectories", "trajectories_filtered_filled", "trajectory"}:
        return "scene_level"
    if lower.startswith(scene):
        return stem[len(scene) :].strip("_-") or "scene_level"
    return stem


def part_names(path: Path) -> set[str]:
    return {normalize_part(p) for p in path.parts}


def has_part(path: Path, names: set[str]) -> bool:
    return bool(part_names(path) & names)


def is_experiment_duplicate(path: Path) -> bool:
    parts = part_names(path)
    return bool(parts & {"rebuild_v1", "experiments", "audit", "feature_analysis", "feature_layers", "feature_variants"})


def should_skip_table(path: Path) -> bool:
    name = path.name.lower()
    if any(token in name for token in ["feature", "label", "trial", "methods_summary", "stats_", "manifest", "schema"]):
        return True
    if any(token in name for token in ["threshold", "audit_runs", "best_runs", "recommended_by_method"]):
        return True
    return False


def iter_supported_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".cursor"}
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() in SUPPORTED_TABLE_EXTS or path.suffix.lower() == ".json":
            out.append(path)
    return sorted(out)


def choose_non_aggregate_tracks(files: Iterable[Path], root: Path) -> list[Path]:
    by_scene_parent: dict[tuple[str, Path], list[Path]] = {}
    for path in files:
        scene = infer_scene(path, root)
        parent = path.parent
        by_scene_parent.setdefault((scene, parent), []).append(path)
    selected: list[Path] = []
    for (_, parent), group in by_scene_parent.items():
        per_source = [p for p in group if p.stem.lower().startswith("tracks_")]
        if per_source:
            selected.extend(per_source)
        else:
            selected.extend(group)
    return sorted(set(selected))


def discover_release_files(root: Path, label: str, scene_glob: str, log: RunLog) -> ReleaseFiles:
    files = iter_supported_files(root)
    release = ReleaseFiles(root=root, label=label)

    detection_candidates: list[Path] = []
    tracked_candidates: list[Path] = []
    raw_candidates: list[Path] = []
    cleaned_candidates: list[Path] = []
    cluster_candidates: list[Path] = []
    homography_candidates: list[Path] = []

    for path in files:
        scene = infer_scene(path, root)
        if not fnmatch.fnmatch(scene, scene_glob):
            continue
        suffix = path.suffix.lower()
        name = path.name.lower()
        parts = part_names(path)

        if suffix in SUPPORTED_TABLE_EXTS and (has_part(path, DETECTION_DIR_NAMES) or "detect" in name):
            metadata_or_summary = bool(parts & {"schemas", "07_results_and_statistics", "09_metadata_and_statistics"})
            if not should_skip_table(path) and not metadata_or_summary and "statistics" not in name and "summary" not in name:
                detection_candidates.append(path)

        if suffix in SUPPORTED_TABLE_EXTS and (
            has_part(path, TRACKING_DIR_NAMES)
            or "interim" in parts
            or name.startswith("tracks")
            or "tracking" in name
        ):
            if not should_skip_table(path) and not is_experiment_duplicate(path):
                tracked_candidates.append(path)

        if suffix in SUPPORTED_TABLE_EXTS and (
            has_part(path, RAW_TRAJECTORY_DIR_NAMES)
            or (("processed" in parts) and name == "trajectories.parquet")
        ):
            if not should_skip_table(path) and not is_experiment_duplicate(path):
                raw_candidates.append(path)

        if suffix in SUPPORTED_TABLE_EXTS and (
            has_part(path, CLEANED_DIR_NAMES)
            or "filtered_filled" in name
            or "cleaned" in name
            or "smoothed" in name
        ):
            if not should_skip_table(path) and not is_experiment_duplicate(path):
                cleaned_candidates.append(path)

        if suffix in {".csv", ".parquet", ".json"} and (
            "method_comparison" in parts
            or "optics" in name
            or "cluster" in name
            or "outlier" in name
            or name.startswith("methods_summary")
        ):
            if "visualizations" not in parts and "feature" not in name:
                cluster_candidates.append(path)

        if suffix in {".json", ".csv"} and (
            has_part(path, HOMOGRAPHY_DIR_NAMES)
            or "homography" in name
            or "reprojection" in name
            or "control_point" in name
            or "topview_transform" in name
        ):
            homography_candidates.append(path)

    release.detection_files = sorted(set(detection_candidates))
    release.tracked_observation_files = choose_non_aggregate_tracks(tracked_candidates, root)
    release.raw_trajectory_files = sorted(set(raw_candidates))
    release.cleaned_trajectory_files = sorted(set(cleaned_candidates))
    release.cluster_files = sorted(set(cluster_candidates))
    release.homography_files = sorted(set(homography_candidates))

    log.discovered[f"{label}_detection_files"] = len(release.detection_files)
    log.discovered[f"{label}_tracked_observation_files"] = len(release.tracked_observation_files)
    log.discovered[f"{label}_raw_trajectory_files"] = len(release.raw_trajectory_files)
    log.discovered[f"{label}_cleaned_trajectory_files"] = len(release.cleaned_trajectory_files)
    log.discovered[f"{label}_cluster_files"] = len(release.cluster_files)
    log.discovered[f"{label}_homography_files"] = len(release.homography_files)
    return release


def parquet_columns(path: Path) -> list[str] | None:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        return None


def parquet_num_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def read_parquet_sample(path: Path, max_rows: int) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        cols = wanted_columns(pf.schema.names)
        batches = []
        remaining = max_rows
        for batch in pf.iter_batches(batch_size=min(max_rows, 50_000), columns=cols):
            if remaining <= 0:
                break
            frame = batch.to_pandas()
            if len(frame) > remaining:
                frame = frame.head(remaining)
            batches.append(frame)
            remaining -= len(frame)
        if not batches:
            return pd.DataFrame(columns=cols)
        return pd.concat(batches, ignore_index=True)
    except Exception:
        return pd.read_parquet(path, columns=wanted_columns(parquet_columns(path) or []))


def wanted_columns(existing: Iterable[str]) -> list[str]:
    existing_list = list(existing)
    lower_to_real = {str(c).lower(): c for c in existing_list}
    wanted: list[str] = []
    for aliases in ALIASES.values():
        for alias in aliases:
            real = lower_to_real.get(alias.lower())
            if real is not None and real not in wanted:
                wanted.append(real)
    return wanted or existing_list


def read_pickle_like(path: Path) -> Any:
    if path.suffix.lower() == ".joblib":
        if joblib is None:
            raise RuntimeError("joblib is not installed")
        return joblib.load(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def objects_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, pd.DataFrame):
                return value
        try:
            return pd.DataFrame(payload)
        except Exception:
            return pd.DataFrame()
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for obj in payload:
            history = getattr(obj, "history", None)
            if history is not None:
                track_id = getattr(obj, "objID", getattr(obj, "track_id", None))
                for det in history:
                    rows.append(
                        {
                            "track_id": track_id,
                            "frame": getattr(det, "frameID", getattr(det, "frame", None)),
                            "x": getattr(det, "X", getattr(det, "x", None)),
                            "y": getattr(det, "Y", getattr(det, "y", None)),
                            "w": getattr(det, "Width", getattr(det, "w", None)),
                            "h": getattr(det, "Height", getattr(det, "h", None)),
                            "class_name": getattr(det, "label", getattr(det, "class_name", None)),
                            "confidence": getattr(det, "confidence", getattr(det, "conf", None)),
                        }
                    )
            elif isinstance(obj, dict):
                rows.append(obj)
        return pd.DataFrame(rows)
    return pd.DataFrame()


def read_sqlite(path: Path, log: RunLog) -> pd.DataFrame:
    try:
        con = sqlite3.connect(path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
        preferred = [
            t
            for t in tables
            if any(token in t.lower() for token in ["trajectory", "track", "detection", "object"])
        ]
        if not preferred and tables:
            preferred = [tables[0]]
        if not preferred:
            return pd.DataFrame()
        return pd.read_sql_query(f"SELECT * FROM {preferred[0]}", con)
    except Exception as exc:
        log.warn(f"Could not read sqlite file {path}: {exc}")
        return pd.DataFrame()
    finally:
        try:
            con.close()
        except Exception:
            pass


def read_table(path: Path, log: RunLog) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        if suffix == ".parquet":
            cols = parquet_columns(path)
            if cols:
                return pd.read_parquet(path, columns=wanted_columns(cols))
            return pd.read_parquet(path)
        if suffix in {".joblib", ".pkl", ".pickle"}:
            return objects_to_frame(read_pickle_like(path))
        if suffix in {".sqlite", ".db"}:
            return read_sqlite(path, log)
    except Exception as exc:
        log.warn(f"Could not read {path}: {exc}")
    return pd.DataFrame()


def normalize_detection_frame(df: pd.DataFrame, path: Path, root: Path) -> pd.DataFrame:
    scene = infer_scene(path, root)
    source = infer_source(path, root, scene)
    out = pd.DataFrame(index=df.index)
    out["scene"] = df[find_col(df, "scene")].map(clean_string) if find_col(df, "scene") else scene
    out["source_file"] = df[find_col(df, "source_file")].map(clean_string) if find_col(df, "source_file") else source
    out["frame"] = numeric(df, "frame")
    out["track_id"] = numeric(df, "track_id")
    out["class_id"] = numeric(df, "class_id")
    cls_name = find_col(df, "class_name")
    out["class_name"] = df[cls_name].map(clean_string) if cls_name else out["class_id"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    out["confidence"] = numeric(df, "confidence")
    out["cx"] = numeric(df, "cx")
    out["cy"] = numeric(df, "cy")
    out["x"] = numeric(df, "x")
    out["y"] = numeric(df, "y")
    out["w"] = numeric(df, "w")
    out["h"] = numeric(df, "h")
    if out["cx"].isna().all() and not out["x"].isna().all():
        out["cx"] = out["x"]
    if out["cy"].isna().all() and not out["y"].isna().all():
        out["cy"] = out["y"]
    return out


def normalize_trajectory_frame(df: pd.DataFrame, path: Path, root: Path) -> pd.DataFrame:
    scene = infer_scene(path, root)
    source = infer_source(path, root, scene)
    out = pd.DataFrame(index=df.index)
    out["scene"] = df[find_col(df, "scene")].map(clean_string) if find_col(df, "scene") else scene
    out["source_file"] = df[find_col(df, "source_file")].map(clean_string) if find_col(df, "source_file") else source
    out["frame"] = numeric(df, "frame")
    out["track_id"] = numeric(df, "track_id")
    out["x"] = numeric(df, "map_x")
    out["y"] = numeric(df, "map_y")
    if out["x"].isna().all():
        out["x"] = numeric(df, "x")
    if out["y"].isna().all():
        out["y"] = numeric(df, "y")
    out["vx"] = numeric(df, "vx")
    out["vy"] = numeric(df, "vy")
    out["ax"] = numeric(df, "ax")
    out["ay"] = numeric(df, "ay")
    out["class_id"] = numeric(df, "class_id")
    cls_name = find_col(df, "class_name")
    out["class_name"] = df[cls_name].map(clean_string) if cls_name else out["class_id"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    out["confidence"] = numeric(df, "confidence")
    out["is_filled"] = boolish(df, "is_filled")
    return out


def empty_scene_row(release: str, scene: str) -> dict[str, Any]:
    return {
        "release": release,
        "scene": scene,
        "detection_files": 0,
        "trajectory_files": 0,
        "detection_source_files": 0,
        "trajectory_source_files": 0,
        "source_files": 0,
        "total_detections": 0,
        "total_tracks": 0,
        "total_points": 0,
        "detections_per_source": np.nan,
        "tracks_per_source": np.nan,
        "average_confidence": np.nan,
        "median_confidence": np.nan,
        "confidence_p10": np.nan,
        "confidence_p25": np.nan,
        "confidence_p75": np.nan,
        "confidence_p90": np.nan,
        "mean_track_length_frames": np.nan,
        "median_track_length_frames": np.nan,
        "track_length_p25": np.nan,
        "track_length_p75": np.nan,
        "track_length_p90": np.nan,
        "track_length_p95": np.nan,
        "short_track_ratio": np.nan,
        "mean_detections_per_track": np.nan,
        "median_detections_per_track": np.nan,
        "mean_continuity_ratio": np.nan,
        "median_continuity_ratio": np.nan,
        "tracks_continuity_ge_0_90_ratio": np.nan,
        "missing_frame_gaps_per_track": np.nan,
        "mean_gap_length": np.nan,
        "max_gap_length": np.nan,
        "interpolation_demand_proxy": np.nan,
        "mean_class_consistency": np.nan,
        "median_class_consistency": np.nan,
        "trajectory_roughness": np.nan,
        "position_jump_threshold": np.nan,
        "position_jump_proxy_ratio": np.nan,
        "clustered_tracks": np.nan,
        "outlier_tracks": np.nan,
        "outlier_ratio": np.nan,
        "reachability_mean": np.nan,
        "reachability_median": np.nan,
        "reachability_min": np.nan,
        "reachability_max": np.nan,
        "number_of_clusters": np.nan,
        "cluster_size_mean": np.nan,
        "cluster_size_median": np.nan,
        "small_cluster_ratio": np.nan,
        "homography_control_points": np.nan,
        "reprojection_error_mean": np.nan,
        "reprojection_error_median": np.nan,
        "reprojection_error_rmse": np.nan,
        "homography_matrix_available": False,
    }


def summarize_detection_like(
    files: list[Path],
    root: Path,
    release: str,
    log: RunLog,
    proxy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scene_rows: dict[str, dict[str, Any]] = {}
    class_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []

    for idx, path in enumerate(files, start=1):
        log.info(f"[{release}] reading {proxy_name} {idx}/{len(files)}: {rel(path, root)}")
        bounded_proxy = "proxy" in proxy_name and path.suffix.lower() == ".parquet"
        exact_rows: int | None = None
        if bounded_proxy:
            exact_rows = parquet_num_rows(path)
            raw = read_parquet_sample(path, PROXY_SAMPLE_ROWS_PER_FILE)
        else:
            raw = read_table(path, log)
        if raw.empty:
            if bounded_proxy and exact_rows:
                scene = infer_scene(path, root)
                row = scene_rows.setdefault(scene, empty_scene_row(release, scene))
                row["detection_files"] += 1
                row["detection_source_files"] += 1
                row["source_files"] += 1
                row["total_detections"] += int(exact_rows)
            continue
        df = normalize_detection_frame(raw, path, root)
        scene = infer_scene(path, root)
        source = infer_source(path, root, scene)
        row = scene_rows.setdefault(scene, empty_scene_row(release, scene))
        row["detection_files"] += 1
        row["detection_source_files"] += 1
        row["source_files"] += 1
        row["total_detections"] += int(exact_rows if exact_rows is not None else len(df))
        tracks = df["track_id"].dropna()
        if not bounded_proxy:
            row["total_tracks"] += int(tracks.nunique()) if len(tracks) else 0

        conf = df["confidence"].dropna()
        if len(conf):
            class_rows.append(
                {
                    "release": release,
                    "scene": scene,
                    "class_id": "__confidence__",
                    "class_name": "__confidence__",
                    "count": len(conf),
                    "count_basis": "bounded_proxy_sample" if bounded_proxy else "full_input",
                    "mean_confidence": float(conf.mean()),
                    "median_confidence": float(conf.median()),
                    "p10": float(conf.quantile(0.10)),
                    "p25": float(conf.quantile(0.25)),
                    "p75": float(conf.quantile(0.75)),
                    "p90": float(conf.quantile(0.90)),
                }
            )

        if "class_name" in df.columns:
            work = df.copy()
            work["class_name"] = work["class_name"].replace("", "unknown")
            grouped = work.groupby(["scene", "class_id", "class_name"], dropna=False).agg(
                detection_count=("class_name", "size"),
                mean_confidence=("confidence", "mean"),
                median_confidence=("confidence", "median"),
            )
            for (scene_id, class_id, class_name), values in grouped.iterrows():
                class_rows.append(
                    {
                        "release": release,
                        "scene": scene_id,
                        "class_id": "" if pd.isna(class_id) else class_id,
                        "class_name": class_name,
                        "count": int(values["detection_count"]),
                        "count_basis": "bounded_proxy_sample" if bounded_proxy else "full_input",
                        "mean_confidence": values["mean_confidence"],
                        "median_confidence": values["median_confidence"],
                        "p10": np.nan,
                        "p25": np.nan,
                        "p75": np.nan,
                        "p90": np.nan,
                    }
                )

        if "track_id" in df.columns and df["track_id"].notna().any() and "class_name" in df.columns:
            work = df.loc[df["track_id"].notna(), ["scene", "track_id", "class_name"]].copy()
            counts = work.groupby(["scene", "track_id", "class_name"], dropna=False).size().rename("n").reset_index()
            totals = counts.groupby(["scene", "track_id"])["n"].sum().rename("total")
            dominant = counts.groupby(["scene", "track_id"])["n"].max().rename("dominant")
            cons = pd.concat([totals, dominant], axis=1).reset_index()
            cons["class_consistency"] = cons["dominant"] / cons["total"]
            for scene_id, scene_cons in cons.groupby("scene"):
                vals = pd.to_numeric(scene_cons["class_consistency"], errors="coerce").dropna()
                if len(vals):
                    consistency_rows.append(
                        {
                            "release": release,
                            "scene": scene_id,
                            "track_key": source,
                            "class_consistency": float(vals.mean()),
                            "class_consistency_median": float(vals.median()),
                            "track_count_weight": int(len(vals)),
                            "source_file": source,
                        }
                    )
        del raw, df

    scene_df = pd.DataFrame(scene_rows.values())
    class_df = pd.DataFrame(class_rows)
    consistency_df = pd.DataFrame(consistency_rows)

    if not scene_df.empty and not class_df.empty:
        conf_rows = class_df[class_df["class_name"] == "__confidence__"]
        for scene, group in conf_rows.groupby("scene"):
            idx = scene_df["scene"] == scene
            weights = pd.to_numeric(group["count"], errors="coerce").fillna(0)
            if weights.sum() > 0:
                scene_df.loc[idx, "average_confidence"] = np.average(pd.to_numeric(group["mean_confidence"], errors="coerce"), weights=weights)
            scene_df.loc[idx, "median_confidence"] = pd.to_numeric(group["median_confidence"], errors="coerce").median()
            for col, out_col in [("p10", "confidence_p10"), ("p25", "confidence_p25"), ("p75", "confidence_p75"), ("p90", "confidence_p90")]:
                scene_df.loc[idx, out_col] = pd.to_numeric(group[col], errors="coerce").median()
        class_df = class_df[class_df["class_name"] != "__confidence__"].copy()

    return scene_df, class_df, consistency_df


def summarize_trajectory_files(
    files: list[Path],
    root: Path,
    release: str,
    stage: str,
    min_short_track_frames: int,
    jump_percentile: float,
    log: RunLog,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scene_rows: dict[str, dict[str, Any]] = {}
    length_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    jump_rows: list[dict[str, Any]] = []

    for idx, path in enumerate(files, start=1):
        log.info(f"[{release}] reading {stage} trajectories {idx}/{len(files)}: {rel(path, root)}")
        raw = read_table(path, log)
        if raw.empty:
            continue
        df = normalize_trajectory_frame(raw, path, root)
        df = df.dropna(subset=["track_id", "frame"])
        if df.empty:
            continue
        scene = infer_scene(path, root)
        source = infer_source(path, root, scene)
        row = scene_rows.setdefault(scene, empty_scene_row(release, scene))
        row["trajectory_files"] += 1
        row["trajectory_source_files"] += 1
        row["source_files"] += 1
        row["total_points"] += int(len(df))

        frame_track = df[["track_id", "frame"]].drop_duplicates()
        grouped = frame_track.groupby("track_id")["frame"]
        track_stats = grouped.agg(["count", "min", "max"]).rename(columns={"count": "observed_frames", "min": "first_frame", "max": "last_frame"})
        track_stats["expected_frames"] = track_stats["last_frame"] - track_stats["first_frame"] + 1
        track_stats["missing_frames"] = (track_stats["expected_frames"] - track_stats["observed_frames"]).clip(lower=0)
        track_stats["continuity_ratio"] = track_stats["observed_frames"] / track_stats["expected_frames"].replace(0, np.nan)
        track_stats = track_stats.reset_index()
        track_stats["scene"] = scene
        track_stats["source_file"] = source
        track_stats["release"] = release
        track_stats["stage"] = stage
        row["total_tracks"] += int(len(track_stats))

        length_rows.extend(
            {
                "release": release,
                "scene": scene,
                "stage": stage,
                "source_file": source,
                "track_id": r["track_id"],
                "track_length_frames": int(r["observed_frames"]),
            }
            for _, r in track_stats.iterrows()
        )
        continuity_rows.extend(
            {
                "release": release,
                "scene": scene,
                "stage": stage,
                "source_file": source,
                "track_id": r["track_id"],
                "observed_frames": int(r["observed_frames"]),
                "expected_frames": int(r["expected_frames"]),
                "missing_frames": int(r["missing_frames"]),
                "continuity_ratio": float(r["continuity_ratio"]) if not pd.isna(r["continuity_ratio"]) else np.nan,
            }
            for _, r in track_stats.iterrows()
        )

        sorted_frames = frame_track.sort_values(["track_id", "frame"], kind="mergesort")
        sorted_frames["gap_length"] = sorted_frames.groupby("track_id")["frame"].diff() - 1
        gaps = sorted_frames[sorted_frames["gap_length"] > 0].copy()
        if not gaps.empty:
            gap_by_track = gaps.groupby("track_id")["gap_length"].agg(["count", "mean", "max", "sum"]).reset_index()
            gap_rows.extend(
                {
                    "release": release,
                    "scene": scene,
                    "stage": stage,
                    "source_file": source,
                    "track_id": r["track_id"],
                    "gap_count": int(r["count"]),
                    "mean_gap_length": float(r["mean"]),
                    "max_gap_length": float(r["max"]),
                    "missing_frames_from_gaps": float(r["sum"]),
                }
                for _, r in gap_by_track.iterrows()
            )

        if "is_filled" in df.columns:
            filled = int(df["is_filled"].sum())
            expected = int(track_stats["expected_frames"].sum())
            row["interpolation_demand_proxy"] = filled / expected if expected else np.nan

        if df["ax"].notna().any() and df["ay"].notna().any():
            rough = np.square(df["ax"].fillna(0.0)) + np.square(df["ay"].fillna(0.0))
            row["trajectory_roughness"] = float(rough.mean())
        elif df["x"].notna().any() and df["y"].notna().any():
            pos = df.dropna(subset=["x", "y"]).sort_values(["track_id", "frame"], kind="mergesort")
            ddx = pos.groupby("track_id")["x"].diff().groupby(pos["track_id"]).diff()
            ddy = pos.groupby("track_id")["y"].diff().groupby(pos["track_id"]).diff()
            rough = np.square(ddx) + np.square(ddy)
            row["trajectory_roughness"] = float(rough.mean()) if rough.notna().any() else np.nan

        if df["x"].notna().any() and df["y"].notna().any():
            pos = df.dropna(subset=["x", "y"]).sort_values(["track_id", "frame"], kind="mergesort")
            dx = pos.groupby("track_id")["x"].diff()
            dy = pos.groupby("track_id")["y"].diff()
            disp = np.sqrt(np.square(dx) + np.square(dy)).dropna()
            if len(disp):
                threshold = float(disp.quantile(jump_percentile / 100.0))
                ratio = float((disp > threshold).mean())
                jump_rows.append(
                    {
                        "release": release,
                        "scene": scene,
                        "stage": stage,
                        "source_file": source,
                        "jump_threshold": threshold,
                        "jump_ratio": ratio,
                        "displacement_count": int(len(disp)),
                    }
                )

    length_df = pd.DataFrame(length_rows)
    cont_df = pd.DataFrame(continuity_rows)
    gap_df = pd.DataFrame(gap_rows)
    jump_df = pd.DataFrame(jump_rows)
    scene_df = pd.DataFrame(scene_rows.values())

    if not scene_df.empty and not length_df.empty:
        for scene, group in length_df.groupby("scene"):
            idx = scene_df["scene"] == scene
            vals = pd.to_numeric(group["track_length_frames"], errors="coerce").dropna()
            if vals.empty:
                continue
            scene_df.loc[idx, "mean_track_length_frames"] = float(vals.mean())
            scene_df.loc[idx, "median_track_length_frames"] = float(vals.median())
            scene_df.loc[idx, "track_length_p25"] = float(vals.quantile(0.25))
            scene_df.loc[idx, "track_length_p75"] = float(vals.quantile(0.75))
            scene_df.loc[idx, "track_length_p90"] = float(vals.quantile(0.90))
            scene_df.loc[idx, "track_length_p95"] = float(vals.quantile(0.95))
            scene_df.loc[idx, "short_track_ratio"] = float((vals < min_short_track_frames).mean())
            scene_df.loc[idx, "mean_detections_per_track"] = float(vals.mean())
            scene_df.loc[idx, "median_detections_per_track"] = float(vals.median())

    if not scene_df.empty and not cont_df.empty:
        for scene, group in cont_df.groupby("scene"):
            idx = scene_df["scene"] == scene
            vals = pd.to_numeric(group["continuity_ratio"], errors="coerce").dropna()
            if vals.empty:
                continue
            scene_df.loc[idx, "mean_continuity_ratio"] = float(vals.mean())
            scene_df.loc[idx, "median_continuity_ratio"] = float(vals.median())
            scene_df.loc[idx, "tracks_continuity_ge_0_90_ratio"] = float((vals >= 0.90).mean())
            missing = pd.to_numeric(group["missing_frames"], errors="coerce").fillna(0).sum()
            expected = pd.to_numeric(group["expected_frames"], errors="coerce").fillna(0).sum()
            if expected:
                if scene_df.loc[idx, "interpolation_demand_proxy"].isna().all():
                    scene_df.loc[idx, "interpolation_demand_proxy"] = float(missing / expected)

    if not scene_df.empty and not gap_df.empty:
        for scene, group in gap_df.groupby("scene"):
            idx = scene_df["scene"] == scene
            tracks = scene_df.loc[idx, "total_tracks"].iloc[0]
            gap_count = pd.to_numeric(group["gap_count"], errors="coerce").fillna(0).sum()
            scene_df.loc[idx, "missing_frame_gaps_per_track"] = float(gap_count / tracks) if tracks else np.nan
            scene_df.loc[idx, "mean_gap_length"] = float(pd.to_numeric(group["mean_gap_length"], errors="coerce").mean())
            scene_df.loc[idx, "max_gap_length"] = float(pd.to_numeric(group["max_gap_length"], errors="coerce").max())

    if not scene_df.empty and not jump_df.empty:
        for scene, group in jump_df.groupby("scene"):
            idx = scene_df["scene"] == scene
            weights = pd.to_numeric(group["displacement_count"], errors="coerce").fillna(0)
            if weights.sum() > 0:
                scene_df.loc[idx, "position_jump_threshold"] = np.average(pd.to_numeric(group["jump_threshold"], errors="coerce"), weights=weights)
                scene_df.loc[idx, "position_jump_proxy_ratio"] = np.average(pd.to_numeric(group["jump_ratio"], errors="coerce"), weights=weights)

    return scene_df, length_df, cont_df, gap_df, jump_df


def summarize_cluster_files(files: list[Path], root: Path, release: str, log: RunLog) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in files:
        scene = infer_scene(path, root)
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            flat = {k: payload.get(k) for k in payload.keys() if isinstance(payload.get(k), (int, float, str, bool))}
            rows.append({"release": release, "scene": scene, "source_file": rel(path, root), **flat})
            continue
        if suffix not in {".csv", ".parquet"}:
            continue
        df = read_table(path, log)
        if df.empty:
            continue
        lower_cols = {c.lower(): c for c in df.columns.astype(str)}
        if path.name.lower().startswith("methods_summary") or "methods_summary" in path.name.lower():
            for _, row in df.iterrows():
                n_clusters = pd.to_numeric(pd.Series([row.get(lower_cols.get("n_clusters", "n_clusters"), np.nan)]), errors="coerce").iloc[0]
                n_outliers = pd.to_numeric(pd.Series([row.get(lower_cols.get("n_outliers", "n_outliers"), np.nan)]), errors="coerce").iloc[0]
                n_total = pd.to_numeric(pd.Series([row.get(lower_cols.get("n_total", "n_total"), np.nan)]), errors="coerce").iloc[0]
                pct = row.get(lower_cols.get("pct_outliers", "pct_outliers"), np.nan)
                rows.append(
                    {
                        "release": release,
                        "scene": scene,
                        "source_file": rel(path, root),
                        "method": row.get(lower_cols.get("method", "method"), ""),
                        "clustered_tracks": n_total - n_outliers if pd.notna(n_total) and pd.notna(n_outliers) else np.nan,
                        "outlier_tracks": n_outliers,
                        "outlier_ratio": float(pct) / 100.0 if pd.notna(pd.to_numeric(pd.Series([pct]), errors="coerce").iloc[0]) else np.nan,
                        "number_of_clusters": n_clusters,
                        "cluster_size_mean": np.nan,
                        "cluster_size_median": np.nan,
                        "small_cluster_ratio": np.nan,
                    }
                )
        elif "cluster_id" in lower_cols:
            cluster_col = lower_cols["cluster_id"]
            labels = pd.to_numeric(df[cluster_col], errors="coerce")
            valid = labels.dropna()
            outliers = int((valid < 0).sum())
            clustered = valid[valid >= 0]
            sizes = clustered.value_counts()
            rows.append(
                {
                    "release": release,
                    "scene": scene,
                    "source_file": rel(path, root),
                    "method": path.stem.replace("labels_", ""),
                    "clustered_tracks": int(len(clustered)),
                    "outlier_tracks": outliers,
                    "outlier_ratio": float(outliers / len(valid)) if len(valid) else np.nan,
                    "number_of_clusters": int(sizes.size),
                    "cluster_size_mean": float(sizes.mean()) if len(sizes) else np.nan,
                    "cluster_size_median": float(sizes.median()) if len(sizes) else np.nan,
                    "small_cluster_ratio": float((sizes < 10).mean()) if len(sizes) else np.nan,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "release",
                "scene",
                "source_file",
                "method",
                "clustered_tracks",
                "outlier_tracks",
                "outlier_ratio",
                "reachability_mean",
                "reachability_median",
                "reachability_min",
                "reachability_max",
                "number_of_clusters",
                "cluster_size_mean",
                "cluster_size_median",
                "small_cluster_ratio",
            ]
        )
    out = pd.DataFrame(rows)
    for col in ["reachability_mean", "reachability_median", "reachability_min", "reachability_max"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def summarize_homography_files(files: list[Path], root: Path, release: str, log: RunLog) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}

    def rec(scene: str) -> dict[str, Any]:
        return records.setdefault(
            scene,
            {
                "release": release,
                "scene": scene,
                "homography_files": 0,
                "control_point_files": 0,
                "homography_matrix_available": False,
                "control_point_count": np.nan,
                "reprojection_error_mean": np.nan,
                "reprojection_error_median": np.nan,
                "reprojection_error_rmse": np.nan,
                "reprojection_error_max": np.nan,
                "source_files": "",
            },
        )

    for path in files:
        scene = infer_scene(path, root)
        item = rec(scene)
        item["homography_files"] += 1
        item["source_files"] = ";".join(filter(None, [item["source_files"], rel(path, root)]))
        name = path.name.lower()
        if "control" in name:
            item["control_point_files"] += 1
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warn(f"Could not read homography JSON {path}: {exc}")
                continue
            text = json.dumps(payload).lower()
            if "homography" in text or "matrix" in text:
                item["homography_matrix_available"] = True
            for key in ["camera_points", "image_points", "source_points", "control_points"]:
                points = payload.get(key)
                if isinstance(points, list):
                    item["control_point_count"] = max(float(len(points)), 0.0)
            error_values: list[float] = []
            for key, value in payload.items():
                lk = key.lower()
                if "reprojection" in lk and "error" in lk:
                    if isinstance(value, list):
                        error_values.extend(float(v) for v in value if isinstance(v, (int, float)))
                    elif isinstance(value, (int, float)):
                        if "mean" in lk:
                            item["reprojection_error_mean"] = float(value)
                        elif "median" in lk:
                            item["reprojection_error_median"] = float(value)
                        elif "rmse" in lk:
                            item["reprojection_error_rmse"] = float(value)
                        elif "max" in lk:
                            item["reprojection_error_max"] = float(value)
            if error_values:
                arr = np.array(error_values, dtype=float)
                item["reprojection_error_mean"] = float(np.mean(arr))
                item["reprojection_error_median"] = float(np.median(arr))
                item["reprojection_error_rmse"] = float(np.sqrt(np.mean(np.square(arr))))
                item["reprojection_error_max"] = float(np.max(arr))
        elif path.suffix.lower() == ".csv":
            df = read_table(path, log)
            if "reprojection_error" in {str(c).lower() for c in df.columns}:
                col = [c for c in df.columns if str(c).lower() == "reprojection_error"][0]
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals):
                    item["reprojection_error_mean"] = float(vals.mean())
                    item["reprojection_error_median"] = float(vals.median())
                    item["reprojection_error_rmse"] = float(np.sqrt(np.mean(np.square(vals))))
                    item["reprojection_error_max"] = float(vals.max())
            if "control" in name and len(df):
                item["control_point_count"] = float(len(df))

    return pd.DataFrame(records.values())


def merge_scene_metrics(
    release: str,
    detection_scene: pd.DataFrame,
    trajectory_scene: pd.DataFrame,
    consistency: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    homography_summary: pd.DataFrame,
) -> pd.DataFrame:
    scenes: set[str] = set()
    for df in [detection_scene, trajectory_scene, consistency, cluster_summary, homography_summary]:
        if not df.empty and "scene" in df.columns:
            scenes.update(clean_string(v) for v in df["scene"].dropna().tolist() if clean_string(v))
    rows = [empty_scene_row(release, scene) for scene in sorted(scenes)]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame([empty_scene_row(release, "missing")])

    def overlay(source: pd.DataFrame, prefer_trajectory_counts: bool = False) -> None:
        if source.empty:
            return
        for _, row in source.iterrows():
            scene = row["scene"]
            idx = out["scene"] == scene
            for col in source.columns:
                if col in {"release", "scene"}:
                    continue
                if col in out.columns:
                    value = row[col]
                    if prefer_trajectory_counts and col in {
                        "total_tracks",
                        "total_points",
                        "trajectory_files",
                        "trajectory_source_files",
                    }:
                        out.loc[idx, col] = value
                    elif pd.notna(value) and value != "":
                        current = out.loc[idx, col].iloc[0]
                        if (pd.isna(current) if not isinstance(current, bool) else current is False) or current in [0, False]:
                            out.loc[idx, col] = value

    overlay(detection_scene)
    overlay(trajectory_scene, prefer_trajectory_counts=True)

    if not consistency.empty:
        for scene, group in consistency.groupby("scene"):
            idx = out["scene"] == scene
            vals = pd.to_numeric(group["class_consistency"], errors="coerce")
            weights = pd.to_numeric(group.get("track_count_weight", pd.Series([1] * len(group))), errors="coerce").fillna(1)
            mask = vals.notna() & (weights > 0)
            if mask.any():
                out.loc[idx, "mean_class_consistency"] = float(np.average(vals[mask], weights=weights[mask]))
            median_col = "class_consistency_median" if "class_consistency_median" in group.columns else "class_consistency"
            medians = pd.to_numeric(group[median_col], errors="coerce").dropna()
            if len(medians):
                out.loc[idx, "median_class_consistency"] = float(medians.median())

    if not cluster_summary.empty:
        grouped = cluster_summary.groupby("scene").agg(
            clustered_tracks=("clustered_tracks", "max"),
            outlier_tracks=("outlier_tracks", "max"),
            outlier_ratio=("outlier_ratio", "median"),
            reachability_mean=("reachability_mean", "median"),
            reachability_median=("reachability_median", "median"),
            reachability_min=("reachability_min", "median"),
            reachability_max=("reachability_max", "median"),
            number_of_clusters=("number_of_clusters", "median"),
            cluster_size_mean=("cluster_size_mean", "median"),
            cluster_size_median=("cluster_size_median", "median"),
            small_cluster_ratio=("small_cluster_ratio", "median"),
        )
        for scene, row in grouped.iterrows():
            idx = out["scene"] == scene
            for col, value in row.items():
                out.loc[idx, col] = value

    if not homography_summary.empty:
        for _, row in homography_summary.iterrows():
            scene = row["scene"]
            idx = out["scene"] == scene
            out.loc[idx, "homography_control_points"] = row.get("control_point_count", np.nan)
            out.loc[idx, "reprojection_error_mean"] = row.get("reprojection_error_mean", np.nan)
            out.loc[idx, "reprojection_error_median"] = row.get("reprojection_error_median", np.nan)
            out.loc[idx, "reprojection_error_rmse"] = row.get("reprojection_error_rmse", np.nan)
            out.loc[idx, "homography_matrix_available"] = bool(row.get("homography_matrix_available", False))

    empty_unknown = (
        (out["scene"] == "unknown_scene")
        & (pd.to_numeric(out["total_detections"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(out["total_tracks"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(out["total_points"], errors="coerce").fillna(0) == 0)
        & (~out["homography_matrix_available"].astype(bool))
    )
    out = out.loc[~empty_unknown].copy()
    out["detections_per_source"] = out.apply(
        lambda r: r["total_detections"] / r["detection_source_files"] if r["detection_source_files"] else np.nan,
        axis=1,
    )
    out["tracks_per_source"] = out.apply(
        lambda r: r["total_tracks"] / r["trajectory_source_files"] if r["trajectory_source_files"] else np.nan,
        axis=1,
    )
    return out


def summarize_overall(metrics: pd.DataFrame, release: str) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame([{"release": release, "scene_count": 0}])
    numeric_cols = [
        "total_detections",
        "total_tracks",
        "total_points",
        "detection_files",
        "trajectory_files",
        "detection_source_files",
        "trajectory_source_files",
        "source_files",
        "clustered_tracks",
        "outlier_tracks",
    ]
    weighted_cols = [
        "average_confidence",
        "median_confidence",
        "mean_track_length_frames",
        "median_track_length_frames",
        "track_length_p90",
        "track_length_p95",
        "short_track_ratio",
        "mean_continuity_ratio",
        "median_continuity_ratio",
        "tracks_continuity_ge_0_90_ratio",
        "interpolation_demand_proxy",
        "mean_class_consistency",
        "trajectory_roughness",
        "position_jump_proxy_ratio",
        "outlier_ratio",
        "number_of_clusters",
        "cluster_size_mean",
        "small_cluster_ratio",
        "reprojection_error_mean",
    ]
    row: dict[str, Any] = {"release": release, "scene_count": int(metrics["scene"].nunique())}
    for col in numeric_cols:
        row[col] = float(pd.to_numeric(metrics.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    for col in weighted_cols:
        vals = pd.to_numeric(metrics.get(col, pd.Series(dtype=float)), errors="coerce")
        weights = pd.to_numeric(metrics.get("total_tracks", pd.Series([1] * len(metrics))), errors="coerce").fillna(1)
        mask = vals.notna()
        row[col] = float(np.average(vals[mask], weights=weights[mask])) if mask.any() and weights[mask].sum() > 0 else np.nan
    row["detections_per_source"] = row["total_detections"] / row["detection_source_files"] if row["detection_source_files"] else np.nan
    row["tracks_per_source"] = row["total_tracks"] / row["trajectory_source_files"] if row["trajectory_source_files"] else np.nan
    return pd.DataFrame([row])


def analyze_release(files: ReleaseFiles, args: argparse.Namespace, log: RunLog) -> ReleaseMetrics:
    if files.detection_files:
        detection_scene, class_distribution, consistency = summarize_detection_like(
            files.detection_files, files.root, files.label, log, "detection"
        )
    elif files.tracked_observation_files and files.label.startswith("new_"):
        log.warn(
            f"{files.label}: separate detection exports not found; tracked observations are used as a detection-volume proxy. "
            f"Parquet row counts are exact; confidence/class summaries use up to {PROXY_SAMPLE_ROWS_PER_FILE} rows per file."
        )
        detection_scene, class_distribution, consistency = summarize_detection_like(
            files.tracked_observation_files, files.root, files.label, log, "tracked observation proxy"
        )
    else:
        if files.tracked_observation_files and files.label.startswith("old_"):
            log.warn(
                f"{files.label}: separate detection exports not found; old tracked-observation proxy reads are skipped to keep the comparison trajectory-focused."
            )
        detection_scene = pd.DataFrame()
        class_distribution = pd.DataFrame()
        consistency = pd.DataFrame()

    trajectory_files = files.cleaned_trajectory_files or files.raw_trajectory_files or files.tracked_observation_files
    stage = "cleaned" if files.cleaned_trajectory_files else ("raw" if files.raw_trajectory_files else "tracked")
    trajectory_scene, length_detail, continuity_detail, gap_detail, jump_detail = summarize_trajectory_files(
        trajectory_files,
        files.root,
        files.label,
        stage,
        args.min_short_track_frames,
        args.jump_percentile,
        log,
    )
    cluster_summary = summarize_cluster_files(files.cluster_files, files.root, files.label, log)
    homography_summary = summarize_homography_files(files.homography_files, files.root, files.label, log)
    per_scene = merge_scene_metrics(files.label, detection_scene, trajectory_scene, consistency, cluster_summary, homography_summary)
    overall = summarize_overall(per_scene, files.label)
    return ReleaseMetrics(
        label=files.label,
        root=files.root,
        metrics_per_scene=per_scene,
        metrics_overall=overall,
        class_distribution=class_distribution,
        track_length_summary=build_track_length_summary(length_detail),
        continuity_summary=build_continuity_summary(continuity_detail),
        gap_summary=build_gap_summary(gap_detail),
        cluster_outlier_summary=cluster_summary,
        homography_summary=homography_summary,
        detail_vectors={
            "track_lengths": length_detail,
            "continuity": continuity_detail,
            "gaps": gap_detail,
            "jumps": jump_detail,
            "class_consistency": consistency,
        },
    )


def build_track_length_summary(lengths: pd.DataFrame) -> pd.DataFrame:
    if lengths.empty:
        return pd.DataFrame(columns=["release", "scene", "stage", "track_count", "mean", "median", "p25", "p75", "p90", "p95"])
    rows = []
    for keys, group in lengths.groupby(["release", "scene", "stage"], dropna=False):
        vals = pd.to_numeric(group["track_length_frames"], errors="coerce").dropna()
        release, scene, stage = keys
        rows.append(
            {
                "release": release,
                "scene": scene,
                "stage": stage,
                "track_count": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
                "p25": float(vals.quantile(0.25)) if len(vals) else np.nan,
                "p75": float(vals.quantile(0.75)) if len(vals) else np.nan,
                "p90": float(vals.quantile(0.90)) if len(vals) else np.nan,
                "p95": float(vals.quantile(0.95)) if len(vals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_continuity_summary(continuity: pd.DataFrame) -> pd.DataFrame:
    if continuity.empty:
        return pd.DataFrame(columns=["release", "scene", "stage", "track_count", "mean", "median", "p10", "p90", "ge_0_90_ratio"])
    rows = []
    for keys, group in continuity.groupby(["release", "scene", "stage"], dropna=False):
        vals = pd.to_numeric(group["continuity_ratio"], errors="coerce").dropna()
        release, scene, stage = keys
        rows.append(
            {
                "release": release,
                "scene": scene,
                "stage": stage,
                "track_count": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
                "p10": float(vals.quantile(0.10)) if len(vals) else np.nan,
                "p90": float(vals.quantile(0.90)) if len(vals) else np.nan,
                "ge_0_90_ratio": float((vals >= 0.90).mean()) if len(vals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_gap_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    if gaps.empty:
        return pd.DataFrame(columns=["release", "scene", "stage", "tracks_with_gaps", "gap_count", "mean_gap_length", "max_gap_length"])
    rows = []
    for keys, group in gaps.groupby(["release", "scene", "stage"], dropna=False):
        release, scene, stage = keys
        rows.append(
            {
                "release": release,
                "scene": scene,
                "stage": stage,
                "tracks_with_gaps": int(group["track_id"].nunique()),
                "gap_count": int(pd.to_numeric(group["gap_count"], errors="coerce").fillna(0).sum()),
                "mean_gap_length": float(pd.to_numeric(group["mean_gap_length"], errors="coerce").mean()),
                "max_gap_length": float(pd.to_numeric(group["max_gap_length"], errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows)


def percent_change(old: Any, new: Any) -> float:
    old_v = pd.to_numeric(pd.Series([old]), errors="coerce").iloc[0]
    new_v = pd.to_numeric(pd.Series([new]), errors="coerce").iloc[0]
    if pd.isna(old_v) or old_v == 0 or pd.isna(new_v):
        return np.nan
    return float((new_v - old_v) / old_v * 100.0)


def build_comparison(new_overall: pd.DataFrame, old_overall: pd.DataFrame | None) -> pd.DataFrame:
    metrics = [
        ("total detections", "total_detections", "Higher detection/observation volume may indicate broader scene coverage, but not accuracy."),
        ("total tracks", "total_tracks", "Track count is an annotation-free volume indicator."),
        ("detections per source", "detections_per_source", "Normalizes detection/observation volume by input source count."),
        ("tracks per source", "tracks_per_source", "Normalizes track volume by input source count."),
        ("median track length", "median_track_length_frames", "Longer tracks can indicate better temporal persistence."),
        ("p90 track length", "track_length_p90", "Upper-tail trajectory persistence indicator."),
        ("short track ratio", "short_track_ratio", "Lower is generally preferable when short fragments are noise."),
        ("mean continuity ratio", "mean_continuity_ratio", "Higher continuity indicates fewer temporal gaps."),
        ("median continuity ratio", "median_continuity_ratio", "Robust continuity indicator."),
        ("interpolation demand proxy", "interpolation_demand_proxy", "Lower is generally preferable; high values indicate more missing-frame repair."),
        ("class consistency", "mean_class_consistency", "Higher means track-level class labels are more stable."),
        ("roughness", "trajectory_roughness", "Lower roughness can indicate smoother trajectories after processing."),
        ("outlier ratio", "outlier_ratio", "Cluster-level outlier behavior when cluster outputs are available."),
        ("cluster count", "number_of_clusters", "Annotation-free clustering structure indicator."),
        ("mean cluster size", "cluster_size_mean", "Cluster compactness/fragmentation proxy."),
    ]
    new_row = new_overall.iloc[0].to_dict() if new_overall is not None and not new_overall.empty else {}
    old_row = old_overall.iloc[0].to_dict() if old_overall is not None and not old_overall.empty else {}
    rows = []
    for name, col, interpretation in metrics:
        old_value = old_row.get(col, np.nan)
        new_value = new_row.get(col, np.nan)
        old_num = pd.to_numeric(pd.Series([old_value]), errors="coerce").iloc[0]
        new_num = pd.to_numeric(pd.Series([new_value]), errors="coerce").iloc[0]
        rows.append(
            {
                "metric": name,
                "old_yolov7_deepsort": old_value if pd.notna(old_num) else pd.NA,
                "new_yolo11x_bytetrack": new_value if pd.notna(new_num) else pd.NA,
                "absolute_change": new_num - old_num if pd.notna(old_num) and pd.notna(new_num) else pd.NA,
                "relative_change_percent": percent_change(old_value, new_value),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path, log: RunLog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.output(path)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows available._"
    work = df.head(max_rows).copy()
    work = work.replace({np.nan: ""})
    columns = [str(c) for c in work.columns]
    rows = [[clean_string(v) for v in row] for row in work.to_numpy().tolist()]
    widths = [len(c) for c in columns]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    lines = [fmt(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def safe_value(df: pd.DataFrame, col: str) -> Any:
    if df.empty or col not in df.columns:
        return np.nan
    return df.iloc[0].get(col, np.nan)


def write_paper_tables(
    out: Path,
    metrics_per_scene: pd.DataFrame,
    metrics_overall: pd.DataFrame,
    comparison: pd.DataFrame,
    class_distribution: pd.DataFrame,
    log: RunLog,
) -> None:
    lines = [
        "# Publication Tables",
        "",
        "These tables are generated from annotation-free indicators. They do not report ground-truth detection accuracy.",
        "",
        "## Overall Metrics",
        "",
        markdown_table(metrics_overall),
        "",
        "## Per-Scene Metrics",
        "",
        markdown_table(metrics_per_scene),
        "",
        "## Old-vs-New Comparison",
        "",
        markdown_table(comparison),
        "",
        "## Class Distribution",
        "",
        markdown_table(class_distribution[class_distribution.get("release", "") == "new_yolo11x_bytetrack"] if not class_distribution.empty else class_distribution),
        "",
    ]
    path = out / "paper_tables.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.output(path)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path, log: RunLog, color: str = "#315f72") -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        log.warn(f"Skipped figure {path.name}: missing {x}/{y}.")
        return
    work = df.copy()
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna(subset=[y])
    if work.empty:
        log.warn(f"Skipped figure {path.name}: no numeric data.")
        return
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.bar(work[x].astype(str), work[y], color=color)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.output(path)


def plot_hist(values: pd.Series, title: str, xlabel: str, path: Path, log: RunLog, color: str = "#4d7c54") -> None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        log.warn(f"Skipped figure {path.name}: no numeric values.")
        return
    upper = vals.quantile(0.99)
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.hist(vals.clip(upper=upper), bins=50, color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(f"{xlabel} (clipped at p99)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.output(path)


def generate_figures(
    out: Path,
    metrics_per_scene: pd.DataFrame,
    class_distribution: pd.DataFrame,
    length_detail: pd.DataFrame,
    continuity_detail: pd.DataFrame,
    comparison: pd.DataFrame,
    homography: pd.DataFrame,
    log: RunLog,
) -> None:
    new_scene = metrics_per_scene[metrics_per_scene["release"] == "new_yolo11x_bytetrack"] if "release" in metrics_per_scene else metrics_per_scene
    plot_bar(new_scene, "scene", "total_detections", "Detection or tracked-observation volume per scene", "Count", out / "fig_01_detections_per_scene.png", log)
    plot_bar(new_scene, "scene", "total_tracks", "Trajectory count per scene", "Track count", out / "fig_02_tracks_per_scene.png", log, "#4d7c54")
    if not length_detail.empty:
        plot_hist(length_detail[length_detail["release"] == "new_yolo11x_bytetrack"]["track_length_frames"], "Track length distribution", "Track length (frames)", out / "fig_03_track_length_distribution.png", log, "#936c3b")
    plot_bar(new_scene, "scene", "short_track_ratio", "Short-track ratio per scene", "Ratio", out / "fig_04_short_track_ratio_per_scene.png", log, "#8a4f62")
    if not continuity_detail.empty:
        plot_hist(continuity_detail[continuity_detail["release"] == "new_yolo11x_bytetrack"]["continuity_ratio"], "Continuity ratio distribution", "Continuity ratio", out / "fig_05_continuity_ratio_distribution.png", log, "#555d7a")
    plot_bar(new_scene, "scene", "interpolation_demand_proxy", "Gap/interpolation demand proxy per scene", "Ratio", out / "fig_06_gap_ratio_per_scene.png", log, "#6b6b6b")
    if not class_distribution.empty:
        cd = class_distribution[class_distribution["release"] == "new_yolo11x_bytetrack"].copy()
        if not cd.empty:
            grouped = cd.groupby("class_name", as_index=False)["count"].sum().sort_values("count", ascending=False).head(12)
            plot_bar(grouped, "class_name", "count", "Class distribution", "Count", out / "fig_07_class_distribution.png", log, "#315f72")
    if "average_confidence" in new_scene.columns:
        plot_bar(new_scene, "scene", "average_confidence", "Mean confidence per scene", "Confidence", out / "fig_08_confidence_distribution.png", log, "#4d7c54")
    plot_bar(new_scene, "scene", "trajectory_roughness", "Trajectory roughness per scene", "Mean squared acceleration / second derivative", out / "fig_09_roughness_per_scene.png", log, "#936c3b")
    if "outlier_ratio" in new_scene.columns and pd.to_numeric(new_scene["outlier_ratio"], errors="coerce").notna().any():
        plot_bar(new_scene, "scene", "outlier_ratio", "OPTICS/cluster outlier ratio per scene", "Ratio", out / "fig_10_outlier_ratio_per_scene.png", log, "#8a4f62")
    if comparison["old_yolov7_deepsort"].notna().any():
        comp = comparison.copy()
        comp["new_yolo11x_bytetrack"] = pd.to_numeric(comp["new_yolo11x_bytetrack"], errors="coerce")
        comp["old_yolov7_deepsort"] = pd.to_numeric(comp["old_yolov7_deepsort"], errors="coerce")
        comp = comp.dropna(subset=["new_yolo11x_bytetrack", "old_yolov7_deepsort"]).head(8)
        if not comp.empty:
            plt.figure(figsize=(10, 5.5))
            ax = plt.gca()
            x = np.arange(len(comp))
            ax.bar(x - 0.2, comp["old_yolov7_deepsort"], width=0.4, label="old YOLOv7 + DeepSORT", color="#8a4f62")
            ax.bar(x + 0.2, comp["new_yolo11x_bytetrack"], width=0.4, label="new YOLO11x + ByteTrack", color="#315f72")
            ax.set_xticks(x)
            ax.set_xticklabels(comp["metric"], rotation=30, ha="right")
            ax.set_title("Old-vs-new main annotation-free metrics")
            ax.legend()
            plt.tight_layout()
            path = out / "fig_11_old_vs_new_main_metrics.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            log.output(path)
    if not homography.empty and "reprojection_error_mean" in homography.columns:
        work = homography[homography["release"] == "new_yolo11x_bytetrack"].copy()
        if not work.empty:
            plot_bar(work, "scene", "reprojection_error_mean", "Homography reprojection error", "Mean error", out / "fig_12_homography_reprojection_error.png", log, "#555d7a")


def write_summary(
    out: Path,
    args: argparse.Namespace,
    log: RunLog,
    new_metrics: ReleaseMetrics,
    old_metrics: ReleaseMetrics | None,
    combined_scene: pd.DataFrame,
    combined_overall: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    old_available = old_metrics is not None and not old_metrics.metrics_per_scene.empty and old_metrics.metrics_per_scene["scene"].ne("missing").any()
    lines = [
        "# Publication Results Summary",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- New release root: `{Path(args.new_root)}`",
        f"- Old release root: `{Path(args.old_root) if args.old_root else 'not provided'}`",
        f"- Output directory: `{Path(args.out)}`",
        "",
        "## Discovered Input Files",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in sorted(log.discovered.items()))
    lines.extend(["", "## Missing Files / Caveats", ""])
    if log.missing:
        lines.extend(f"- {item}" for item in log.missing)
    else:
        lines.append("- No required input category was completely missing.")
    if log.warnings:
        lines.extend(f"- {warning}" for warning in log.warnings)
    lines.extend(["", "## Per-Scene Summary", "", markdown_table(combined_scene), ""])
    lines.extend(["## Overall Summary", "", markdown_table(combined_overall), ""])
    lines.extend(["## Old-vs-New Comparison Summary", ""])
    if old_available:
        lines.append(markdown_table(comparison))
    else:
        lines.append("Old-release files were not available or could not be summarized; comparison rows keep old values as `NA`.")
    lines.extend(
        [
            "",
            "## Interpretation for the Paper",
            "",
            "The generated tables report annotation-free indicators of dataset scale and trajectory usability. Detection volume, tracked-observation volume, track length, continuity, gap behavior, class consistency, and trajectory smoothness provide practical evidence about the processing pipeline without relying on manual labels.",
            "",
            "When old-release data are available, the comparison should be interpreted as a pipeline-level comparison between the previous YOLOv7 + DeepSORT release and the updated YOLO11x + Ultralytics/ByteTrack release. These indicators do not measure detector precision or recall on the user's dataset.",
            "",
            "## Limitations",
            "",
            "No manual annotation was used; therefore, detection accuracy is not measured directly on the dataset. The comparison is based on annotation-free trajectory quality indicators and published model benchmark context.",
            "",
        ]
    )
    path = out / "RESULTS_SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.output(path)


def write_paper_text(out: Path, new_overall: pd.DataFrame, comparison: pd.DataFrame, old_available: bool, log: RunLog) -> None:
    total_tracks = safe_value(new_overall, "total_tracks")
    total_obs = safe_value(new_overall, "total_detections")
    scenes = safe_value(new_overall, "scene_count")
    lines = [
        "# Paper Results Text",
        "",
        "The updated pipeline produced annotation-free dataset statistics for the Traffic Node Video Dataset 2.0 release. The analysis does not rely on manually annotated ground truth; instead, it evaluates practical trajectory usability through detection or tracked-observation volume, track persistence, continuity, interpolation demand, class consistency, trajectory roughness, clustering outlier behavior, and export coverage.",
        "",
        f"In the available new-release inputs, the script summarized `{scenes}` scenes, approximately `{total_obs}` detection/tracked-observation rows, and `{total_tracks}` trajectory tracks. These numbers should be reported as generated-output statistics, not as detector accuracy measures.",
        "",
    ]
    if old_available:
        lines.extend(
            [
                "Compared with the previous release, the old-vs-new table reports absolute and relative changes for the main annotation-free indicators. The results indicate how the updated YOLO11x + Ultralytics/ByteTrack pipeline changes dataset volume, temporal persistence, continuity, and smoothness relative to the YOLOv7 + DeepSORT based release.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The previous-release export was not sufficiently available for a complete numeric comparison in this run. The comparison table is therefore provided as a template with the new-release metrics filled and old-release fields marked as missing.",
                "",
            ]
        )
    lines.extend(
        [
            "The annotation-free indicators show the practical usability of the generated trajectories, but they should be interpreted conservatively. They do not establish ground-truth detection accuracy on this dataset.",
            "",
        ]
    )
    path = out / "PAPER_RESULTS_TEXT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.output(path)


def write_readme(out: Path, log: RunLog) -> None:
    lines = [
        "# Publication Result Generation",
        "",
        "This folder contains annotation-free publication tables and figures for the Traffic Node Video Dataset 2.0 paper.",
        "",
        "## Reproduce",
        "",
        "```bash",
        'python scripts/generate_publication_results.py --new-root "PATH_TO_TRAFFIC_NODE_VIDEO_DATASET_2_0" --out reports/publication_results',
        'python scripts/generate_publication_results.py --new-root "PATH_TO_TRAFFIC_NODE_VIDEO_DATASET_2_0" --old-root "PATH_TO_PREVIOUS_TRAFFIC_NODE_DATASET" --out reports/publication_results',
        "```",
        "",
        "Use the repository virtual environment if the system Python does not include pandas, pyarrow, matplotlib, and joblib:",
        "",
        "```powershell",
        '.\\.venv\\Scripts\\python.exe scripts\\generate_publication_results.py --new-root "TNVD2_UPLOAD_PACKAGE" --old-root "old_data" --out reports\\publication_results',
        "```",
        "",
        "## Scientific Note",
        "",
        "The script does not use manual annotations and does not report ground-truth detector accuracy. It reports annotation-free indicators of dataset scale, trajectory continuity, gap behavior, class consistency, smoothness, cluster/outlier behavior, and homography metadata availability.",
        "",
    ]
    path = out / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.output(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annotation-free publication results for TNVD2.")
    parser.add_argument("--new-root", required=True, help="Root of the Traffic Node Video Dataset 2.0 outputs")
    parser.add_argument("--old-root", default=None, help="Optional root of the previous YOLOv7 + DeepSORT release")
    parser.add_argument("--out", default="reports/publication_results", help="Output directory")
    parser.add_argument("--fps-default", type=float, default=30.0, help="Default FPS used for documentation/context")
    parser.add_argument("--scene-glob", default="*", help="Scene glob filter")
    parser.add_argument("--min-short-track-frames", type=int, default=30, help="Short-track threshold in frames")
    parser.add_argument("--jump-percentile", type=float, default=99.0, help="Percentile used for jump proxy threshold")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    new_root = Path(args.new_root)
    old_root = Path(args.old_root) if args.old_root else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(new_root=new_root, old_root=old_root, out_dir=out)

    if not new_root.exists():
        print(
            "New dataset root was not found. Pass the uploaded dataset root with --new-root, for example:\n"
            '  python scripts/generate_publication_results.py --new-root "Traffic_Node_Video_Dataset_2_0_2026" --out reports/publication_results'
        )
        return 2

    new_files = discover_release_files(new_root, "new_yolo11x_bytetrack", args.scene_glob, log)
    if not new_files.detection_files:
        log.missing.append("Separate YOLO11x detection export files were not found; tracked observations are used as the detection-volume proxy where available.")
    if not new_files.cleaned_trajectory_files and not new_files.raw_trajectory_files and not new_files.tracked_observation_files:
        log.missing.append("No new-release trajectory tables were found.")

    old_metrics: ReleaseMetrics | None = None
    if old_root and old_root.exists():
        old_files = discover_release_files(old_root, "old_yolov7_deepsort", args.scene_glob, log)
        if not old_files.cleaned_trajectory_files and not old_files.raw_trajectory_files and not old_files.tracked_observation_files:
            log.missing.append("Old-release root was provided but no old trajectory tables were found.")
        old_metrics = analyze_release(old_files, args, log)
    elif old_root:
        log.missing.append(f"Old-release root was provided but does not exist: {old_root}")
    else:
        log.missing.append("Old-release root was not provided.")

    new_metrics = analyze_release(new_files, args, log)

    combined_scene = pd.concat(
        [df for df in [new_metrics.metrics_per_scene, old_metrics.metrics_per_scene if old_metrics else None] if df is not None],
        ignore_index=True,
    )
    combined_overall = pd.concat(
        [df for df in [new_metrics.metrics_overall, old_metrics.metrics_overall if old_metrics else None] if df is not None],
        ignore_index=True,
    )
    combined_class = pd.concat(
        [df for df in [new_metrics.class_distribution, old_metrics.class_distribution if old_metrics else None] if df is not None],
        ignore_index=True,
    )
    combined_lengths = pd.concat(
        [df for df in [new_metrics.track_length_summary, old_metrics.track_length_summary if old_metrics else None] if df is not None],
        ignore_index=True,
    )
    combined_continuity = pd.concat(
        [df for df in [new_metrics.continuity_summary, old_metrics.continuity_summary if old_metrics else None] if df is not None],
        ignore_index=True,
    )
    combined_gaps = pd.concat(
        [df for df in [new_metrics.gap_summary, old_metrics.gap_summary if old_metrics else None] if df is not None],
        ignore_index=True,
    )
    combined_clusters = pd.concat(
        [df for df in [new_metrics.cluster_outlier_summary, old_metrics.cluster_outlier_summary if old_metrics else None] if df is not None],
        ignore_index=True,
    )
    combined_homography = pd.concat(
        [df for df in [new_metrics.homography_summary, old_metrics.homography_summary if old_metrics else None] if df is not None],
        ignore_index=True,
    )
    comparison = build_comparison(new_metrics.metrics_overall, old_metrics.metrics_overall if old_metrics else None)

    write_csv(combined_scene, out / "metrics_per_scene.csv", log)
    write_csv(combined_overall, out / "metrics_overall.csv", log)
    write_csv(combined_class, out / "class_distribution.csv", log)
    write_csv(combined_lengths, out / "track_length_summary.csv", log)
    write_csv(combined_continuity, out / "continuity_summary.csv", log)
    write_csv(combined_gaps, out / "gap_summary.csv", log)
    write_csv(combined_clusters, out / "cluster_outlier_summary.csv", log)
    write_csv(combined_homography, out / "homography_summary.csv", log)
    write_csv(comparison, out / "comparison_old_vs_new.csv", log)

    length_detail = pd.concat(
        [
            new_metrics.detail_vectors.get("track_lengths", pd.DataFrame()),
            old_metrics.detail_vectors.get("track_lengths", pd.DataFrame()) if old_metrics else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    continuity_detail = pd.concat(
        [
            new_metrics.detail_vectors.get("continuity", pd.DataFrame()),
            old_metrics.detail_vectors.get("continuity", pd.DataFrame()) if old_metrics else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    write_paper_tables(out, combined_scene, combined_overall, comparison, combined_class, log)
    generate_figures(out, combined_scene, combined_class, length_detail, continuity_detail, comparison, combined_homography, log)
    old_available = old_metrics is not None and comparison["old_yolov7_deepsort"].notna().any()
    write_summary(out, args, log, new_metrics, old_metrics, combined_scene, combined_overall, comparison)
    write_paper_text(out, new_metrics.metrics_overall, comparison, old_available, log)
    write_readme(out, log)

    log.info("")
    log.info("Publication result generation complete")
    log.info(f"- Output directory: {out}")
    log.info(f"- New scenes summarized: {len(new_metrics.metrics_per_scene)}")
    log.info(f"- Old release summarized: {bool(old_metrics is not None)}")
    log.info(f"- Warnings: {len(log.warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
