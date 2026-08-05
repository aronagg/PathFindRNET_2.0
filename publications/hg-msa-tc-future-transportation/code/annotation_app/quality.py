"""Real-data annotation preflight without clustering or label generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

try:
    from .models import SCENES
    from .source_data import sha256_file, write_annotation_access_log
except ImportError:  # Direct script execution.
    from models import SCENES
    from source_data import sha256_file, write_annotation_access_log


EXPECTED_TEST_COUNTS = {
    "bellevue_116th_ne12th": 964,
    "bellevue_150th_newport": 3924,
    "bellevue_150th_eastgate": 10755,
    "bellevue_150th_se38th": 3713,
    "bellevue_ne8th": 8037,
}


def validate_queue_sources(repo_root: Path, queue: pd.DataFrame, access_log: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scene, source_path, checksum), part in queue.groupby(
        ["scene_id", "trajectory_source_path", "trajectory_source_checksum"], sort=True
    ):
        path = repo_root / str(source_path)
        actual_checksum = sha256_file(path)
        if actual_checksum != checksum:
            raise ValueError(f"Source checksum mismatch: {path}")
        wanted = set(part["source_recording_track_id"].astype(int))
        table = ds.dataset(path, format="parquet").to_table(
            columns=["track_id", "frame", "cx", "cy"],
            filter=ds.field("track_id").isin(sorted(wanted)),
        )
        frame = table.to_pandas()
        finite = np.isfinite(frame[["cx", "cy"]].to_numpy(dtype=float)).all(axis=1)
        grouped = (
            frame.assign(_finite=finite)
            .groupby("track_id")
            .agg(
                actual_points=("frame", "size"),
                actual_frame_start=("frame", "min"),
                actual_frame_end=("frame", "max"),
                finite_coordinates=("_finite", "all"),
            )
        )
        for item in part.itertuples(index=False):
            source_track = int(item.source_recording_track_id)
            status = grouped.loc[source_track] if source_track in grouped.index else None
            rows.append(
                {
                    "scene_id": scene,
                    "trajectory_id": item.trajectory_id,
                    "recording_id": item.recording_id,
                    "source_track_id": source_track,
                    "full_polyline_found": status is not None and int(status.actual_points) >= 2,
                    "actual_points": int(status.actual_points) if status is not None else 0,
                    "expected_frame_start": int(item.frame_start),
                    "actual_frame_start": int(status.actual_frame_start)
                    if status is not None
                    else None,
                    "expected_frame_end": int(item.frame_end),
                    "actual_frame_end": int(status.actual_frame_end)
                    if status is not None
                    else None,
                    "finite_coordinates": bool(status.finite_coordinates)
                    if status is not None
                    else False,
                    "frame_range_matches": bool(
                        status is not None
                        and int(status.actual_frame_start) == int(item.frame_start)
                        and int(status.actual_frame_end) == int(item.frame_end)
                    ),
                    "frame_range_covered": bool(
                        status is not None
                        and int(status.actual_frame_start) <= int(item.frame_start)
                        and int(status.actual_frame_end) >= int(item.frame_end)
                    ),
                    "video_available": bool(item.video_available),
                    "source_checksum_matches": True,
                }
            )
        write_annotation_access_log(
            access_log,
            "validate_full_polylines",
            scene,
            str(source_path),
            len(part),
            actual_checksum,
        )
    result = pd.DataFrame(rows)
    if len(result) != len(queue) or result["trajectory_id"].duplicated().any():
        raise ValueError("Preflight result does not map one-to-one to the queue.")
    return result


def summarize_preflight(result: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "total": int(len(result)),
        "all_renderable": bool(
            (
                result["full_polyline_found"]
                & result["finite_coordinates"]
                & result["frame_range_covered"]
                & result["video_available"]
            ).all()
        ),
        "full_polyline_missing": int((~result["full_polyline_found"]).sum()),
        "nonfinite": int((~result["finite_coordinates"]).sum()),
        "frame_range_mismatch": int((~result["frame_range_matches"]).sum()),
        "frame_range_not_covered": int((~result["frame_range_covered"]).sum()),
        "video_missing": int((~result["video_available"]).sum()),
        "scene_counts": {str(k): int(v) for k, v in result.groupby("scene_id").size().items()},
    }
    if tuple(summary["scene_counts"].keys()) != tuple(sorted(SCENES)):
        raise ValueError("Preflight does not contain all five scenes.")
    return summary


def write_preflight(result: pd.DataFrame, csv_path: Path, json_path: Path) -> dict[str, Any]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False, lineterminator="\n")
    summary = summarize_preflight(result)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
