"""Build the publication trajectory manifest without modifying source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


PROTOCOL_VERSION = "future-transportation-evaluation-v1"
SCENES = (
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
)
FPS = 30.0
FEATURE_RELATIVE_PATH = Path("feature_analysis/features_trimmed_frame_disp_norm.parquet")
REQUIRED_FEATURE_COLUMNS = (
    "track_id",
    "frame_start",
    "frame_end",
    "len_frames",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "path_length",
    "displacement",
    "straightness",
)
MANIFEST_COLUMNS = (
    "manifest_protocol_version",
    "scene_id",
    "trajectory_id",
    "original_track_id",
    "source_file",
    "source_row",
    "source_object_identifier",
    "source_recording_id",
    "source_recording_file",
    "source_recording_track_id",
    "source_recording_start_time",
    "source_provenance_method",
    "object_class",
    "object_class_status",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "number_of_points",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "validity_under_existing_preprocessing_pipeline",
    "filtering_status",
    "data_fingerprint",
    "exact_duplicate_count",
    "exact_duplicate_group",
    "possible_near_duplicate",
    "near_duplicate_candidate_count",
    "near_duplicate_group",
    "source_file_checksum_sha256",
)
TIMESTAMP_PATTERN = re.compile(r"__(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class RecordingRange:
    scene_id: str
    recording_id: str
    shard_path: Path
    raw_video_path: Path
    start_time: datetime
    id_offset: int
    global_min_track_id: int
    global_max_track_id: int


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_output_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "publications/hg-msa-tc-future-transportation/data/manifests/trajectory_manifest.csv"
    )


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_bounds(path: Path, column: str) -> tuple[int, int]:
    parquet = pq.ParquetFile(path)
    column_index = parquet.schema_arrow.names.index(column)
    minima: list[int] = []
    maxima: list[int] = []
    for row_group_index in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(row_group_index).column(column_index).statistics
        if statistics is not None and statistics.has_min_max:
            minima.append(int(statistics.min))
            maxima.append(int(statistics.max))
    if not minima:
        values = pd.read_parquet(path, columns=[column])[column]
        if values.empty:
            raise ValueError(f"No {column} values in {path}")
        return int(values.min()), int(values.max())
    return min(minima), max(maxima)


def _recording_timestamp(recording_id: str) -> datetime:
    match = TIMESTAMP_PATTERN.search(recording_id)
    if match is None:
        raise ValueError(f"Recording timestamp is not encoded in {recording_id!r}")
    return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")


def build_recording_ranges(repo_root: Path, scene_id: str) -> list[RecordingRange]:
    interim_dir = repo_root / "data/interim" / scene_id
    shards = sorted(interim_dir.glob("tracks_*.parquet"), key=lambda path: path.name)
    if not shards:
        raise FileNotFoundError(f"No per-recording track shards found in {interim_dir}")

    ranges: list[RecordingRange] = []
    offset = 0
    for shard_path in shards:
        local_min, local_max = _column_bounds(shard_path, "track_id")
        recording_id = shard_path.stem.removeprefix("tracks_")
        raw_video_path = repo_root / "data/raw" / scene_id / f"{recording_id}.mp4"
        if not raw_video_path.exists():
            raise FileNotFoundError(
                f"Expected raw video for {shard_path.name} was not found: {raw_video_path}"
            )
        ranges.append(
            RecordingRange(
                scene_id=scene_id,
                recording_id=recording_id,
                shard_path=shard_path,
                raw_video_path=raw_video_path,
                start_time=_recording_timestamp(recording_id),
                id_offset=offset,
                global_min_track_id=local_min + offset,
                global_max_track_id=local_max + offset,
            )
        )
        # This reproduces scripts/merge_tracks.py. Dropping negative IDs does not
        # change the maximum and therefore does not change subsequent offsets.
        offset += local_max + 1
    return ranges


def map_recording(track_id: int, ranges: Iterable[RecordingRange]) -> RecordingRange:
    matches = [
        item
        for item in ranges
        if item.global_min_track_id <= track_id <= item.global_max_track_id
    ]
    if len(matches) != 1:
        names = [item.recording_id for item in matches]
        raise ValueError(
            f"Merged track ID {track_id} maps to {len(matches)} recordings: {names}"
        )
    return matches[0]


def _stable_number(value: Any) -> str | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, bool)):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        return None
    return format(number, ".12g")


def _hash_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def trajectory_fingerprint(row: pd.Series) -> str:
    payload = {
        "scene_id": row["scene_id"],
        "start_frame": _stable_number(row["start_frame"]),
        "end_frame": _stable_number(row["end_frame"]),
        "number_of_points": _stable_number(row["number_of_points"]),
        "start_x": _stable_number(row["start_x"]),
        "start_y": _stable_number(row["start_y"]),
        "end_x": _stable_number(row["end_x"]),
        "end_y": _stable_number(row["end_y"]),
        "path_length": _stable_number(row["path_length"]),
        "displacement": _stable_number(row["displacement"]),
        "straightness": _stable_number(row["straightness"]),
    }
    return _hash_payload(payload)


def near_duplicate_signature(row: pd.Series) -> str:
    def quantize(value: Any, width: float) -> int | None:
        if value is None or pd.isna(value):
            return None
        return int(round(float(value) / width))

    payload = {
        "scene_id": row["scene_id"],
        "start_frame_q5": quantize(row["start_frame"], 5.0),
        "end_frame_q5": quantize(row["end_frame"], 5.0),
        "number_of_points_q5": quantize(row["number_of_points"], 5.0),
        "start_x_q2": quantize(row["start_x"], 2.0),
        "start_y_q2": quantize(row["start_y"], 2.0),
        "end_x_q2": quantize(row["end_x"], 2.0),
        "end_y_q2": quantize(row["end_y"], 2.0),
        "path_length_q2": quantize(row["path_length"], 2.0),
    }
    return _hash_payload(payload)


def build_manifest(repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scene_order, scene_id in enumerate(SCENES):
        feature_path = repo_root / "data/processed" / scene_id / FEATURE_RELATIVE_PATH
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        features = pd.read_parquet(feature_path)
        missing = [column for column in REQUIRED_FEATURE_COLUMNS if column not in features.columns]
        if missing:
            raise ValueError(f"{feature_path} is missing columns: {missing}")
        if features["track_id"].isna().any() or features["track_id"].duplicated().any():
            raise ValueError(f"{feature_path} does not have unique, non-null track_id values")

        checksum = sha256_file(feature_path)
        recording_ranges = build_recording_ranges(repo_root, scene_id)
        for source_row, feature in features.reset_index(drop=True).iterrows():
            original_track_id = int(feature["track_id"])
            recording = map_recording(original_track_id, recording_ranges)
            local_track_id = original_track_id - recording.id_offset
            start_frame = int(feature["frame_start"])
            end_frame = int(feature["frame_end"])
            start_time = recording.start_time + timedelta(seconds=start_frame / FPS)
            end_time = recording.start_time + timedelta(seconds=end_frame / FPS)
            rows.append(
                {
                    "_scene_order": scene_order,
                    "_recording_order": recording.start_time.isoformat(timespec="seconds"),
                    "manifest_protocol_version": PROTOCOL_VERSION,
                    "scene_id": scene_id,
                    "trajectory_id": f"{scene_id}:{original_track_id}",
                    "original_track_id": original_track_id,
                    "source_file": relative_path(feature_path, repo_root),
                    "source_row": int(source_row),
                    "source_object_identifier": str(original_track_id),
                    "source_recording_id": recording.recording_id,
                    "source_recording_file": relative_path(recording.raw_video_path, repo_root),
                    "source_recording_track_id": local_track_id,
                    "source_recording_start_time": recording.start_time.isoformat(timespec="seconds"),
                    "source_provenance_method": "reconstructed_from_merge_tracks_id_offsets_v1",
                    "object_class": "",
                    "object_class_status": "not_retained_after_vehicle_class_filter",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time.isoformat(timespec="milliseconds"),
                    "end_time": end_time.isoformat(timespec="milliseconds"),
                    "number_of_points": int(feature["len_frames"]),
                    "start_x": float(feature["start_x"]),
                    "start_y": float(feature["start_y"]),
                    "end_x": float(feature["end_x"]),
                    "end_y": float(feature["end_y"]),
                    "path_length": float(feature["path_length"]),
                    "displacement": float(feature["displacement"]),
                    "straightness": float(feature["straightness"]),
                    "validity_under_existing_preprocessing_pipeline": True,
                    "filtering_status": "included_in_features_trimmed_frame_disp_norm",
                    "source_file_checksum_sha256": checksum,
                }
            )

    manifest = pd.DataFrame(rows)
    manifest["data_fingerprint"] = manifest.apply(trajectory_fingerprint, axis=1)
    exact_counts = manifest.groupby("data_fingerprint")["trajectory_id"].transform("size")
    manifest["exact_duplicate_count"] = exact_counts.astype(int)
    manifest["exact_duplicate_group"] = manifest["data_fingerprint"].where(exact_counts > 1, "")

    manifest["_near_signature"] = manifest.apply(near_duplicate_signature, axis=1)
    near_counts = manifest.groupby("_near_signature")["trajectory_id"].transform("size")
    exact_distinct = manifest.groupby("_near_signature")["data_fingerprint"].transform("nunique")
    possible_near = (near_counts > 1) & (exact_distinct > 1)
    manifest["possible_near_duplicate"] = possible_near
    manifest["near_duplicate_candidate_count"] = near_counts.where(possible_near, 0).astype(int)
    manifest["near_duplicate_group"] = manifest["_near_signature"].where(possible_near, "")

    manifest = manifest.sort_values(
        [
            "_scene_order",
            "_recording_order",
            "start_frame",
            "end_frame",
            "original_track_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    return manifest.loc[:, MANIFEST_COLUMNS]


def write_manifest(manifest: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, lineterminator="\n", float_format="%.12g")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve() if args.output else default_output_path(repo_root)
    manifest = build_manifest(repo_root)
    write_manifest(manifest, output)
    print(f"Wrote {len(manifest):,} trajectories to {output}")
    for scene_id, count in manifest.groupby("scene_id", sort=False).size().items():
        print(f"  {scene_id}: {count:,}")
    print(f"Exact duplicate rows: {(manifest['exact_duplicate_count'] > 1).sum():,}")
    print(f"Possible near-duplicate rows: {manifest['possible_near_duplicate'].sum():,}")


if __name__ == "__main__":
    main()
