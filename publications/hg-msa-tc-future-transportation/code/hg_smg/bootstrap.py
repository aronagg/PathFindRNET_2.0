"""Recording-aware bootstrap primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hierarchical_recording_indices(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """Sample recordings and trajectories with replacement at frozen sizes."""
    if "recording_id" not in frame.columns:
        raise ValueError("Hierarchical bootstrap requires recording_id")
    recordings = sorted(frame["recording_id"].astype(str).unique())
    if not recordings:
        raise ValueError("No recordings available for hierarchical bootstrap")
    groups = {
        recording: np.flatnonzero(frame["recording_id"].astype(str).to_numpy() == recording)
        for recording in recordings
    }
    rng = np.random.default_rng(int(seed))
    selected_recordings = rng.choice(recordings, size=len(recordings), replace=True)
    output: list[np.ndarray] = []
    for recording in selected_recordings:
        indices = groups[str(recording)]
        output.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(output).astype(np.int64)


def percentile_interval(values: np.ndarray, level: float) -> tuple[int, int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Percentile interval requires finite bootstrap targets")
    alpha = (1.0 - float(level)) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
    return int(np.floor(lower)), int(np.ceil(upper))


def smaller_tie_mode(values: np.ndarray) -> int:
    unique, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(unique[np.flatnonzero(counts == counts.max())[0]])


def shannon_entropy_bits(values: np.ndarray) -> float:
    _, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))
