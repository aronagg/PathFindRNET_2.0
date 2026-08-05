"""Deterministic recording-stratified blind annotation queue builders."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .models import BLIND_QUEUE_COLUMNS, SCENES, validate_blind_queue_columns
except ImportError:  # Direct script execution.
    from models import BLIND_QUEUE_COLUMNS, SCENES, validate_blind_queue_columns


PILOT_SEED = 20260805
ANNOTATOR_SEEDS = {"annotator_A": 20260811, "annotator_B": 20260823}


def _stable_key(seed: int, trajectory_id: str) -> str:
    return hashlib.sha256(f"{seed}:{trajectory_id}".encode("utf-8")).hexdigest()


def _round_robin_recordings(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    groups: dict[str, list[pd.Series]] = {}
    recording_ids = frame["source_recording_id"].drop_duplicates().tolist()
    recording_ids = sorted(
        recording_ids,
        key=lambda value: _stable_key(seed, str(value)),
    )
    for recording_id in recording_ids:
        part = frame[frame["source_recording_id"] == recording_id].copy()
        part["_key"] = part["trajectory_id"].map(lambda value: _stable_key(seed, str(value)))
        groups[str(recording_id)] = [
            row for _, row in part.sort_values("_key", kind="mergesort").iterrows()
        ]
    output = []
    while any(groups.values()):
        for recording_id in recording_ids:
            key = str(recording_id)
            if groups[key]:
                output.append(groups[key].pop(0))
    return pd.DataFrame(output).drop(columns=["_key"], errors="ignore")


def recording_stratified_pilot(
    source_index: pd.DataFrame,
    per_scene: int = 100,
    seed: int = PILOT_SEED,
) -> pd.DataFrame:
    rows = []
    for scene_index, scene in enumerate(SCENES):
        scene_rows = source_index[
            (source_index["scene_id"] == scene) & (source_index["split"] == "target_estimation")
        ].copy()
        ordered = _round_robin_recordings(scene_rows, seed + scene_index * 1000)
        if len(ordered) < per_scene:
            raise ValueError(f"Not enough target-estimation rows for pilot in {scene}")
        rows.append(ordered.iloc[:per_scene])
    pilot = pd.concat(rows, ignore_index=True)
    pilot.insert(0, "queue_position", np.arange(1, len(pilot) + 1))
    pilot.insert(0, "annotator_id", "protocol_pilot")
    pilot.insert(0, "queue_status", "available_for_protocol_development")
    pilot.insert(0, "queue_type", "protocol_pilot")
    pilot.insert(0, "queue_id", "protocol_pilot_v1")
    return _to_blind_queue(pilot)


def primary_annotator_queue(
    source_index: pd.DataFrame,
    annotator_id: str,
    protocol_frozen: bool,
) -> pd.DataFrame:
    if annotator_id not in ANNOTATOR_SEEDS:
        raise ValueError(f"Unknown independent annotator: {annotator_id}")
    rows = []
    for scene_index, scene in enumerate(SCENES):
        scene_rows = source_index[
            (source_index["scene_id"] == scene) & (source_index["split"] == "independent_test")
        ].copy()
        rows.append(
            _round_robin_recordings(scene_rows, ANNOTATOR_SEEDS[annotator_id] + scene_index * 1000)
        )
    queue = pd.concat(rows, ignore_index=True)
    queue.insert(0, "queue_position", np.arange(1, len(queue) + 1))
    queue.insert(0, "annotator_id", annotator_id)
    queue.insert(
        0,
        "queue_status",
        "available" if protocol_frozen else "locked_pending_protocol_freeze",
    )
    queue.insert(0, "queue_type", "independent_test_primary")
    queue.insert(0, "queue_id", f"independent_test_{annotator_id}_v1")
    return _to_blind_queue(queue)


def _to_blind_queue(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "source_recording_id": "recording_id",
            "start_frame": "frame_start",
            "end_frame": "frame_end",
        }
    ).copy()
    columns = list(BLIND_QUEUE_COLUMNS)
    missing = sorted(set(columns).difference(renamed.columns))
    if missing:
        raise ValueError(f"Queue source is missing columns: {missing}")
    queue = renamed[columns].copy()
    validate_blind_queue_columns(tuple(queue.columns))
    if queue["trajectory_id"].duplicated().any():
        raise ValueError("Queue contains duplicate trajectory IDs.")
    return queue


def write_queue(frame: pd.DataFrame, path: Path) -> None:
    validate_blind_queue_columns(tuple(frame.columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def validate_primary_queues(
    queue_a: pd.DataFrame,
    queue_b: pd.DataFrame,
    expected_counts: dict[str, int],
) -> None:
    for queue in (queue_a, queue_b):
        if set(queue["split"]) != {"independent_test"}:
            raise ValueError("Primary queue contains non-test rows.")
        if len(queue) != sum(expected_counts.values()):
            raise ValueError("Primary queue has an unexpected total size.")
        observed = queue.groupby("scene_id").size().to_dict()
        if observed != expected_counts:
            raise ValueError(f"Primary queue scene counts differ: {observed}")
        if not queue["trajectory_id"].is_unique:
            raise ValueError("Primary queue trajectory IDs are not unique.")
    if set(queue_a["trajectory_id"]) != set(queue_b["trajectory_id"]):
        raise ValueError("Annotator queues do not contain the same cohort.")
    if queue_a["trajectory_id"].tolist() == queue_b["trajectory_id"].tolist():
        raise ValueError("Annotator queue orders must differ.")
