"""Annotation enums, schemas, and blind UI data contracts."""

from __future__ import annotations

from typing import Any


SCENES = (
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
)
SPLITS = ("target_estimation", "model_selection", "independent_test")
MANEUVER_TYPES = ("straight", "left", "right", "u_turn", "other", "unknown")
VALIDITY_VALUES = ("valid", "ambiguous", "unusable")
CONFIDENCE_VALUES = ("high", "medium", "low")
ROLES = (
    "protocol_pilot",
    "annotator_A",
    "annotator_B",
    "adjudicator",
    "protocol_designer",
)

RAW_ANNOTATION_COLUMNS = (
    "annotation_id",
    "scene_id",
    "trajectory_id",
    "split",
    "recording_id",
    "entry_approach",
    "exit_approach",
    "manual_maneuver_id",
    "maneuver_type",
    "validity",
    "confidence",
    "annotator_id",
    "annotation_timestamp_utc",
    "protocol_version",
    "trajectory_source_checksum",
    "rendering_version",
    "notes",
    "revision_number",
    "supersedes_annotation_id",
)

# This is the only real queue data exposed to first-pass UI code. It intentionally
# contains no target, cluster, pseudo-label, model, metric, class-frequency, or label
# fields from any other annotator.
BLIND_QUEUE_COLUMNS = (
    "queue_id",
    "queue_type",
    "queue_status",
    "annotator_id",
    "queue_position",
    "scene_id",
    "trajectory_id",
    "split",
    "recording_id",
    "source_recording_file",
    "source_recording_track_id",
    "frame_start",
    "frame_end",
    "number_of_points",
    "trajectory_source_path",
    "trajectory_source_checksum",
    "video_available",
)

FORBIDDEN_BLIND_COLUMNS = (
    "hg_target",
    "hg_estimated_target",
    "entry_region",
    "exit_region",
    "od_pseudo_label",
    "cluster_id",
    "cluster_label",
    "selected_algorithm",
    "target_error",
    "emas_hg",
    "quick_score",
    "suggested_label",
    "rare_movement",
    "class_frequency",
    "consensus_label",
)


def manual_maneuver_id(scene_id: str, entry_approach: str, exit_approach: str) -> str:
    scene = scene_id.strip()
    entry = entry_approach.strip()
    exit_value = exit_approach.strip()
    if not scene or not entry or not exit_value:
        raise ValueError("Scene, entry approach, and exit approach are required.")
    return f"{scene}:{entry}>{exit_value}"


def validate_annotation_values(record: dict[str, Any]) -> None:
    if record.get("maneuver_type") not in MANEUVER_TYPES:
        raise ValueError(f"Invalid maneuver_type: {record.get('maneuver_type')}")
    if record.get("validity") not in VALIDITY_VALUES:
        raise ValueError(f"Invalid validity: {record.get('validity')}")
    if record.get("confidence") not in CONFIDENCE_VALUES:
        raise ValueError(f"Invalid confidence: {record.get('confidence')}")
    if "rare_movement" in record:
        raise ValueError("rare_movement is derived and cannot be a human input.")
    expected = manual_maneuver_id(
        str(record["scene_id"]),
        str(record["entry_approach"]),
        str(record["exit_approach"]),
    )
    if record.get("manual_maneuver_id") not in {None, "", expected}:
        raise ValueError("manual_maneuver_id does not match human entry/exit choices.")


def validate_blind_queue_columns(columns: list[str] | tuple[str, ...]) -> None:
    normalized = {str(column).lower() for column in columns}
    forbidden = sorted(normalized.intersection(FORBIDDEN_BLIND_COLUMNS))
    if forbidden:
        raise PermissionError(f"Blind queue contains forbidden columns: {forbidden}")
    unexpected = sorted(normalized.difference(BLIND_QUEUE_COLUMNS))
    if unexpected:
        raise PermissionError(f"Blind queue contains unexpected columns: {unexpected}")
