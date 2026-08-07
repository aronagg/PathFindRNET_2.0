"""Prior-Constrained Model Selection (PCMS)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def interval_distance(cluster_count: int, lower: int, upper: int) -> int:
    count = int(cluster_count)
    if count < int(lower):
        return int(lower) - count
    if count > int(upper):
        return count - int(upper)
    return 0


def _finite_key(value: object, missing: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    return number if np.isfinite(number) else missing


def _candidate_key(row: pd.Series, method: str) -> tuple[object, ...]:
    base: list[object] = [int(row["interval_distance"])]
    if method in {"hdbscan", "optics"}:
        base.append(_finite_key(row.get("pct_outliers"), np.inf))
    base.extend(
        [
            -_finite_key(row.get("silhouette_clustered_only"), -np.inf),
            _finite_key(row.get("davies_bouldin_clustered_only"), np.inf),
            -_finite_key(row.get("calinski_harabasz_clustered_only"), -np.inf),
            _finite_key(row.get("largest_cluster_ratio"), np.inf),
            str(row["params_json"]),
        ]
    )
    return tuple(base)


def select_pcms_candidates(
    candidates: pd.DataFrame,
    intervals: pd.DataFrame,
    strategy_id: str = "A5",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one existing model-selection candidate per scene and method."""
    required = {"scene", "method", "params_json", "n_clusters"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")
    interval_lookup = intervals.set_index("scene")[["interval_lower", "interval_upper"]]
    work = candidates.copy()
    work["interval_lower"] = work["scene"].map(interval_lookup["interval_lower"])
    work["interval_upper"] = work["scene"].map(interval_lookup["interval_upper"])
    if work[["interval_lower", "interval_upper"]].isna().any().any():
        raise ValueError("Missing UATP interval for a model-selection candidate")
    work["interval_distance"] = [
        interval_distance(count, lower, upper)
        for count, lower, upper in zip(
            work["n_clusters"], work["interval_lower"], work["interval_upper"], strict=True
        )
    ]
    work["pcms_strategy_id"] = strategy_id
    selected_indices: list[int] = []
    keys: dict[int, str] = {}
    for (scene, method), group in work.groupby(["scene", "method"], sort=False):
        indexed = [(index, _candidate_key(row, str(method))) for index, row in group.iterrows()]
        selected_index, selected_key = min(indexed, key=lambda item: item[1])
        selected_indices.append(int(selected_index))
        keys[int(selected_index)] = json.dumps(selected_key, ensure_ascii=True)
    work["selected_pcms"] = work.index.isin(selected_indices)
    work["pcms_selection_key_json"] = [
        keys.get(int(index), "") for index in work.index
    ]
    selected = work[work["selected_pcms"]].copy()
    return work.reset_index(drop=True), selected.reset_index(drop=True)


def select_point_target_candidates(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    strategy_id: str,
) -> pd.DataFrame:
    """Apply the frozen point-error-first tie order for A2/A3/A10."""
    target_map = targets.set_index("scene")["point_target"]
    work = candidates.copy()
    work["point_target"] = work["scene"].map(target_map)
    work["point_target_error"] = (work["n_clusters"] - work["point_target"]).abs()
    selected = []
    for (scene, method), group in work.groupby(["scene", "method"], sort=False):
        def key(item: tuple[int, pd.Series]) -> tuple[object, ...]:
            index, row = item
            values: list[object] = [int(row["point_target_error"])]
            if method in {"hdbscan", "optics"}:
                values.append(_finite_key(row.get("pct_outliers"), np.inf))
            values.extend(
                [
                    -_finite_key(row.get("silhouette_clustered_only"), -np.inf),
                    _finite_key(row.get("davies_bouldin_clustered_only"), np.inf),
                    -_finite_key(row.get("calinski_harabasz_clustered_only"), -np.inf),
                    _finite_key(row.get("largest_cluster_ratio"), np.inf),
                    str(row["params_json"]),
                ]
            )
            return tuple(values)

        selected.append(min(group.iterrows(), key=key)[0])
    work["selected"] = work.index.isin(selected)
    work["selection_strategy"] = strategy_id
    return work[work["selected"]].reset_index(drop=True)
