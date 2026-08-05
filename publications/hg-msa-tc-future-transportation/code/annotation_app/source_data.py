"""Read-only source mapping and full-polyline access for human annotation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from .models import BLIND_QUEUE_COLUMNS, SCENES, validate_blind_queue_columns
except ImportError:  # Direct script execution.
    from models import BLIND_QUEUE_COLUMNS, SCENES, validate_blind_queue_columns


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_shard_path(repo_root: Path, scene: str, recording_id: str) -> Path:
    return repo_root / "data" / "interim" / scene / f"tracks_{recording_id}.parquet"


def load_manifest_and_split(manifest_path: Path, split_path: Path) -> pd.DataFrame:
    manifest_columns = [
        "scene_id",
        "trajectory_id",
        "original_track_id",
        "source_recording_id",
        "source_recording_file",
        "source_recording_track_id",
        "start_frame",
        "end_frame",
        "number_of_points",
        "source_file_checksum_sha256",
    ]
    split_columns = ["trajectory_id", "split"]
    manifest = pd.read_csv(manifest_path, usecols=manifest_columns, keep_default_na=False)
    split = pd.read_csv(split_path, usecols=split_columns, keep_default_na=False)
    combined = manifest.merge(split, on="trajectory_id", how="inner", validate="one_to_one")
    if len(combined) != len(manifest):
        raise ValueError("Manifest and split do not map one-to-one.")
    if tuple(combined["scene_id"].drop_duplicates()) != SCENES:
        raise ValueError("Annotation source is not the fixed five-scene cohort.")
    if not combined["trajectory_id"].is_unique:
        raise ValueError("Duplicate trajectory IDs in annotation source index.")
    return combined


def build_source_catalog(repo_root: Path, index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    recordings = index[
        ["scene_id", "source_recording_id", "source_recording_file"]
    ].drop_duplicates()
    for row in recordings.itertuples(index=False):
        shard = source_shard_path(repo_root, row.scene_id, row.source_recording_id)
        video = repo_root / row.source_recording_file
        if not shard.exists():
            raise FileNotFoundError(shard)
        if not video.exists():
            raise FileNotFoundError(video)
        parquet = pq.ParquetFile(shard)
        required = {"frame", "track_id", "cx", "cy"}
        if not required.issubset(parquet.schema.names):
            raise ValueError(f"Missing full-polyline fields in {shard}")
        rows.append(
            {
                "scene_id": row.scene_id,
                "recording_id": row.source_recording_id,
                "trajectory_source_path": shard.relative_to(repo_root).as_posix(),
                "trajectory_source_checksum": sha256_file(shard),
                "trajectory_source_size_bytes": shard.stat().st_size,
                "trajectory_source_rows": parquet.metadata.num_rows,
                "trajectory_source_row_groups": parquet.metadata.num_row_groups,
                "source_recording_file": Path(row.source_recording_file).as_posix(),
                "video_exists": True,
                "video_size_bytes": video.stat().st_size,
                "video_checksum_policy": "sha256_on_demand_before_clip_or_frame_use",
            }
        )
    catalog = pd.DataFrame(rows)
    if catalog.duplicated(["scene_id", "recording_id"]).any():
        raise ValueError("Duplicate recording rows in source catalog.")
    return catalog.sort_values(["scene_id", "recording_id"], kind="mergesort")


def attach_sources(index: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    attached = index.merge(
        catalog[
            [
                "scene_id",
                "recording_id",
                "trajectory_source_path",
                "trajectory_source_checksum",
                "video_exists",
            ]
        ],
        left_on=["scene_id", "source_recording_id"],
        right_on=["scene_id", "recording_id"],
        how="left",
        validate="many_to_one",
    )
    if attached["trajectory_source_path"].isna().any():
        raise ValueError("Some canonical trajectories have no source shard mapping.")
    attached["video_available"] = attached["video_exists"].astype(bool)
    return attached.drop(columns=["recording_id"])


def load_full_polyline(
    repo_root: Path,
    trajectory_source_path: str,
    local_track_id: int,
    expected_checksum: str,
    frame_start: int | None = None,
    frame_end: int | None = None,
    verify_checksum: bool = True,
) -> pd.DataFrame:
    path = repo_root / trajectory_source_path
    if verify_checksum and sha256_file(path) != expected_checksum:
        raise ValueError(f"Trajectory source checksum mismatch: {path}")
    frame = pd.read_parquet(
        path,
        columns=["track_id", "frame", "cx", "cy"],
        filters=[("track_id", "=", int(local_track_id))],
    )
    frame = frame[frame["track_id"] == int(local_track_id)].copy()
    if frame_start is not None:
        frame = frame[frame["frame"] >= int(frame_start)]
    if frame_end is not None:
        frame = frame[frame["frame"] <= int(frame_end)]
    if len(frame) < 2:
        raise ValueError(
            f"Full canonical-range polyline unavailable for local track {local_track_id} in {path}"
        )
    if not np.isfinite(frame[["cx", "cy"]].to_numpy(dtype=float)).all():
        raise ValueError("Polyline contains non-finite camera coordinates.")
    return frame.sort_values("frame", kind="mergesort").reset_index(drop=True)


def read_video_frame(video_path: Path, frame_number: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open source video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_number)))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Cannot read frame {frame_number} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def write_annotation_access_log(
    path: Path,
    action: str,
    scene: str,
    input_path: str,
    row_count: int,
    checksum: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "access_purpose": "human_annotation_preflight",
        "action": action,
        "scene": scene,
        "input_path": input_path,
        "row_count": int(row_count),
        "checksum_sha256": checksum,
        "timestamp_utc": utc_timestamp(),
        "scientific_test_execution": False,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def verify_video_checksum_on_demand(
    repo_root: Path, relative_video_path: str, ledger_path: Path
) -> str:
    """Hash a source video before use and append annotation-only provenance."""
    video_path = repo_root / relative_video_path
    checksum = sha256_file(video_path)
    known = None
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("input_path") == relative_video_path:
                known = record.get("checksum_sha256")
    if known is not None and known != checksum:
        raise ValueError(f"Source video checksum mismatch: {video_path}")
    write_annotation_access_log(
        ledger_path,
        "verify_source_video_before_rendering",
        Path(relative_video_path).parent.name,
        relative_video_path,
        1,
        checksum,
    )
    return checksum


def queue_ui_record(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    output = {column: payload[column] for column in BLIND_QUEUE_COLUMNS}
    validate_blind_queue_columns(tuple(output))
    return output
