"""Data isolation, hashing, and protocol guards for split-aware HG-MSA-TC."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


PHASE_SPLITS = {
    "target": "target_estimation",
    "select": "model_selection",
    "test": "independent_test",
}
FEATURE_COLUMNS = ("start_x", "start_y", "end_x", "end_y")
ANNOTATION_MARKERS = (
    "annotation",
    "manual_label",
    "maneuver_label",
    "ground_truth",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(
        path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    )


def write_yaml_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(
        path,
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False, width=100),
    )


def reject_annotation_inputs(paths: Iterable[str | Path], phase: str) -> None:
    if phase not in {"target", "select"}:
        return
    rejected = []
    for raw_path in paths:
        normalized = str(raw_path).replace("\\", "/").lower()
        if any(marker in normalized for marker in ANNOTATION_MARKERS):
            rejected.append(str(raw_path))
    if rejected:
        raise PermissionError(
            f"{phase} phase cannot receive annotation inputs: {rejected}"
        )


def assert_phase_rows(frame: pd.DataFrame, phase: str) -> None:
    if phase not in PHASE_SPLITS:
        raise ValueError(f"Unknown phase: {phase}")
    if "split" not in frame.columns:
        raise ValueError("Split-aware input is missing the split column.")
    observed = set(frame["split"].dropna().astype(str).unique())
    expected = {PHASE_SPLITS[phase]}
    if observed != expected:
        raise PermissionError(
            f"{phase} phase expected {sorted(expected)}, observed {sorted(observed)}"
        )


def load_split_membership(
    split_path: Path, phase: str, scenes: tuple[str, ...]
) -> pd.DataFrame:
    """Read split control metadata and retain only the phase-authorized index."""
    if phase not in PHASE_SPLITS:
        raise ValueError(f"Unknown phase: {phase}")
    required = [
        "scene_id",
        "trajectory_id",
        "split",
        "source_recording_id",
        "data_fingerprint",
    ]
    selected_chunks = []
    allowed = PHASE_SPLITS[phase]
    for chunk in pd.read_csv(
        split_path, usecols=required, dtype=str, chunksize=100_000
    ):
        selected = chunk[chunk["split"] == allowed]
        if len(selected):
            selected_chunks.append(selected)
    if not selected_chunks:
        raise RuntimeError(f"No rows found for split {allowed} in {split_path}")
    membership = pd.concat(selected_chunks, ignore_index=True)
    assert_phase_rows(membership, phase)
    observed_scenes = tuple(membership["scene_id"].drop_duplicates())
    if observed_scenes != scenes:
        raise ValueError(
            f"Unexpected scene order/content for {phase}: {observed_scenes}"
        )
    if not membership["trajectory_id"].is_unique:
        raise ValueError(f"Duplicate trajectory IDs in {phase} membership.")
    return membership


def load_manifest_metadata(
    manifest_path: Path, membership: pd.DataFrame
) -> pd.DataFrame:
    required = [
        "scene_id",
        "trajectory_id",
        "original_track_id",
        "source_file",
        "source_row",
        "source_recording_id",
        "source_file_checksum_sha256",
    ]
    allowed_ids = set(membership["trajectory_id"])
    selected_chunks = []
    for chunk in pd.read_csv(
        manifest_path, usecols=required, dtype={"trajectory_id": str}, chunksize=100_000
    ):
        selected = chunk[chunk["trajectory_id"].isin(allowed_ids)]
        if len(selected):
            selected_chunks.append(selected)
    metadata = pd.concat(selected_chunks, ignore_index=True)
    metadata = metadata.merge(
        membership[["trajectory_id", "split", "data_fingerprint"]],
        on="trajectory_id",
        how="inner",
        validate="one_to_one",
    )
    if len(metadata) != len(membership):
        raise ValueError(
            f"Manifest join returned {len(metadata)} rows for {len(membership)} memberships."
        )
    return metadata


class AccessLogger:
    def __init__(self, path: Path, phase: str) -> None:
        self.path = path
        self.phase = phase
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        scene: str,
        input_path: Path,
        observed_splits: Iterable[str],
        row_count: int,
        checksum: str,
        input_kind: str = "feature_cohort",
    ) -> None:
        self.records.append(
            {
                "phase": self.phase,
                "scene": scene,
                "input_path": input_path.as_posix(),
                "input_kind": input_kind,
                "allowed_split": PHASE_SPLITS[self.phase],
                "observed_split_values": sorted(set(observed_splits)),
                "row_count": int(row_count),
                "checksum_sha256": checksum,
                "timestamp_utc": utc_timestamp(),
            }
        )

    def flush(self) -> None:
        existing: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("phase") != self.phase:
                        existing.append(record)
        records = existing + self.records
        write_text_atomic(
            self.path,
            "".join(
                json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
                for record in records
            ),
        )


def load_scene_features(
    repo_root: Path,
    scene_metadata: pd.DataFrame,
    phase: str,
    access_logger: AccessLogger | None = None,
) -> pd.DataFrame:
    assert_phase_rows(scene_metadata, phase)
    if scene_metadata["scene_id"].nunique() != 1:
        raise ValueError("Feature loading requires exactly one scene.")
    source_files = scene_metadata["source_file"].drop_duplicates().tolist()
    if len(source_files) != 1:
        raise ValueError(f"Expected one feature file, found {source_files}")
    source_path = repo_root / source_files[0]
    expected_checksum = scene_metadata[
        "source_file_checksum_sha256"
    ].drop_duplicates().tolist()
    if len(expected_checksum) != 1:
        raise ValueError("Inconsistent source file checksums in manifest metadata.")
    actual_checksum = sha256_file(source_path)
    if actual_checksum != expected_checksum[0]:
        raise ValueError(f"Feature checksum changed: {source_path}")

    track_ids = scene_metadata["original_track_id"].astype(int).tolist()
    columns = ["track_id", *FEATURE_COLUMNS]
    features = pd.read_parquet(
        source_path,
        columns=columns,
        filters=[("track_id", "in", track_ids)],
    )
    features = features[features["track_id"].isin(track_ids)].copy()
    if features["track_id"].duplicated().any():
        raise ValueError(f"Duplicate track IDs returned from {source_path}")
    metadata = scene_metadata.copy()
    metadata["original_track_id"] = metadata["original_track_id"].astype(int)
    joined = metadata.merge(
        features,
        left_on="original_track_id",
        right_on="track_id",
        how="left",
        validate="one_to_one",
    )
    if joined[list(FEATURE_COLUMNS)].isna().any().any():
        missing = joined.loc[
            joined[list(FEATURE_COLUMNS)].isna().any(axis=1), "trajectory_id"
        ].tolist()
        raise ValueError(f"Missing feature rows for trajectory IDs: {missing[:10]}")
    if len(joined) != len(scene_metadata):
        raise ValueError("Feature join changed the authorized cohort size.")
    assert_phase_rows(joined, phase)
    if access_logger is not None:
        access_logger.add(
            scene=str(joined["scene_id"].iloc[0]),
            input_path=source_path.relative_to(repo_root),
            observed_splits=joined["split"].unique(),
            row_count=len(joined),
            checksum=actual_checksum,
        )
    return joined.sort_values("trajectory_id", kind="mergesort").reset_index(
        drop=True
    )


def validate_scene_video_field(config_path: Path, expected_scene: str) -> None:
    payload = load_yaml(config_path)
    if str(payload.get("scene")) != expected_scene:
        raise ValueError(
            f"Scene config declares {payload.get('scene')!r}, expected {expected_scene!r}."
        )
    video = payload.get("video")
    if not video:
        return
    normalized_expected = re.sub(r"[^a-z0-9]", "", expected_scene.lower())
    normalized_video = re.sub(r"[^a-z0-9]", "", str(video).lower())
    aliases = {
        "bellevue150thnewport": ("150thnewport",),
        "bellevue150theastgate": ("150theastgate",),
        "bellevue150thse38th": ("150thse38th",),
        "bellevue116thne12th": ("116thne12th",),
        "bellevuene8th": ("ne8th",),
    }
    expected_tokens = aliases.get(normalized_expected, (normalized_expected,))
    if not any(token in normalized_video for token in expected_tokens):
        raise ValueError(
            f"Scene config {config_path} points to a video from another scene: {video}"
        )


def validate_frozen_payload(payload: dict[str, Any]) -> str:
    recorded_hash = payload.get("complete_frozen_configuration_sha256")
    if not recorded_hash:
        raise ValueError("Frozen payload has no complete configuration hash.")
    unhashed = dict(payload)
    unhashed.pop("complete_frozen_configuration_sha256", None)
    computed_hash = canonical_sha256(unhashed)
    if computed_hash != recorded_hash:
        raise ValueError(
            f"Frozen configuration hash mismatch: {computed_hash} != {recorded_hash}"
        )
    return computed_hash


def require_test_unlock(
    unlock_path: Path,
    frozen_hash: str,
    confirmation: str | None,
    required_confirmation: str,
) -> None:
    if not unlock_path.exists():
        raise PermissionError(
            f"Independent test remains locked; missing unlock file: {unlock_path}"
        )
    unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    if unlock.get("frozen_protocol_sha256") != frozen_hash:
        raise PermissionError("Unlock file does not match the frozen protocol hash.")
    if confirmation != required_confirmation:
        raise PermissionError("Explicit independent-test confirmation is missing.")
