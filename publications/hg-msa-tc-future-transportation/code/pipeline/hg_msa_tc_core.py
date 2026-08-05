"""Core HG-MSA-TC calculations used by the split-aware publication runner.

The formulas and candidate grids are extracted from the isolated FoV implementation
in ``run_hg_msa_tc_five_scene_pipeline.py``. Data access and protocol enforcement live
in ``split_aware_io.py`` and ``run_split_aware_hg_msa_tc.py``.
"""

from __future__ import annotations

import json
import time
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


FEATURE_COLUMNS = ("start_x", "start_y", "end_x", "end_y")
METHODS = ("kmeans", "hdbscan", "optics")
IMPLEMENTATION_VERSION = "split-aware-hg-msa-tc-v1"


def isotropic_normalize(
    values: np.ndarray, parameters: dict[str, float] | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply shared-scale normalization, fitting parameters only when omitted."""
    values = np.asarray(values, dtype=np.float64)
    if parameters is None:
        x_values = values[:, [0, 2]]
        y_values = values[:, [1, 3]]
        x_min = float(np.nanmin(x_values))
        y_min = float(np.nanmin(y_values))
        x_range = max(float(np.nanmax(x_values) - x_min), 1e-12)
        y_range = max(float(np.nanmax(y_values) - y_min), 1e-12)
        parameters = {
            "x_min": x_min,
            "y_min": y_min,
            "x_range": x_range,
            "y_range": y_range,
            "shared_scale": max(x_range, y_range),
        }
    required = {"x_min", "y_min", "shared_scale"}
    missing = required.difference(parameters)
    if missing:
        raise ValueError(f"Missing normalization parameters: {sorted(missing)}")
    scale = max(float(parameters["shared_scale"]), 1e-12)
    out = values.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - float(parameters["x_min"])) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - float(parameters["y_min"])) / scale
    return out, {key: float(value) for key, value in parameters.items()}


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(homography, dtype=np.float64)
    homogeneous = np.hstack(
        [points, np.ones((len(points), 1), dtype=np.float64)]
    ) @ matrix.T
    denominator = homogeneous[:, 2:3]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("Homography produced a near-zero homogeneous scale.")
    return homogeneous[:, :2] / denominator


def transform_camera_endpoints(
    frame: pd.DataFrame, homography: np.ndarray
) -> pd.DataFrame:
    start = apply_homography(frame[["start_x", "start_y"]].to_numpy(), homography)
    end = apply_homography(frame[["end_x", "end_y"]].to_numpy(), homography)
    return pd.DataFrame(
        {
            "trajectory_id": frame["trajectory_id"].astype(str).to_numpy(),
            "recording_id": frame["source_recording_id"].astype(str).to_numpy(),
            "start_x_topview": start[:, 0],
            "start_y_topview": start[:, 1],
            "end_x_topview": end[:, 0],
            "end_y_topview": end[:, 1],
        }
    )


def _endpoint_features(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    delta = points - center
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    radius = np.linalg.norm(delta, axis=1)
    radius_norm = radius / max(float(np.nanmedian(radius)), 1e-12)
    radius_norm = np.clip(radius_norm, 0, 3) / 3.0
    return np.column_stack([np.cos(angle), np.sin(angle), 0.25 * radius_norm])


def _region_metrics(
    features: np.ndarray, labels: np.ndarray, metric_indices: np.ndarray
) -> tuple[float, float]:
    selected_features = features[metric_indices]
    selected_labels = labels[metric_indices]
    if len(np.unique(selected_labels)) < 2:
        return np.nan, np.nan
    try:
        silhouette = float(silhouette_score(selected_features, selected_labels))
    except Exception:
        silhouette = np.nan
    try:
        davies_bouldin = float(
            davies_bouldin_score(selected_features, selected_labels)
        )
    except Exception:
        davies_bouldin = np.nan
    return silhouette, davies_bouldin


def estimate_endpoint_regions(
    points: np.ndarray,
    role: str,
    scene: str,
    seed: int,
    region_counts: list[int],
    metric_sample_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Select entry or exit region count and retain every evaluated candidate."""
    center = np.median(points, axis=0)
    features = _endpoint_features(points, center)
    if len(features) > metric_sample_size:
        rng = np.random.default_rng(seed)
        metric_indices = np.sort(
            rng.choice(len(features), size=metric_sample_size, replace=False)
        )
    else:
        metric_indices = np.arange(len(features))

    rows: list[dict[str, Any]] = []
    labels_by_count: dict[int, np.ndarray] = {}
    for region_count in region_counts:
        labels = KMeans(
            n_clusters=int(region_count),
            n_init=10,
            random_state=seed,
            algorithm="lloyd",
        ).fit_predict(features)
        silhouette, davies_bouldin = _region_metrics(
            features, labels, metric_indices
        )
        rows.append(
            {
                "scene": scene,
                "endpoint_role": role,
                "n_regions": int(region_count),
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "n_trajectories": int(len(points)),
                "metric_sample_size": int(len(metric_indices)),
                "random_seed": int(seed),
            }
        )
        labels_by_count[int(region_count)] = labels.astype(int)

    candidates = pd.DataFrame(rows)
    candidates["_silhouette_key"] = candidates["silhouette"].fillna(-np.inf)
    candidates["_db_key"] = candidates["davies_bouldin"].fillna(np.inf)
    selected = candidates.sort_values(
        ["_silhouette_key", "_db_key", "n_regions"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    selected_count = int(selected["n_regions"])
    candidates["selected"] = candidates["n_regions"] == selected_count
    return (
        labels_by_count[selected_count],
        candidates.drop(columns=["_silhouette_key", "_db_key"]),
    )


def estimate_hg_target(
    scene: str,
    endpoints: pd.DataFrame,
    seed: int,
    region_counts: list[int],
    support_thresholds: list[float],
    region_metric_sample_size: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate the observed maneuver target from top-view endpoint geometry."""
    start_points = endpoints[
        ["start_x_topview", "start_y_topview"]
    ].to_numpy(dtype=np.float64)
    end_points = endpoints[["end_x_topview", "end_y_topview"]].to_numpy(
        dtype=np.float64
    )
    entry_labels, entry_candidates = estimate_endpoint_regions(
        start_points,
        "entry",
        scene,
        seed,
        region_counts,
        region_metric_sample_size,
    )
    exit_labels, exit_candidates = estimate_endpoint_regions(
        end_points,
        "exit",
        scene,
        seed + 17,
        region_counts,
        region_metric_sample_size,
    )
    region_candidates = pd.concat(
        [entry_candidates, exit_candidates], ignore_index=True
    )
    selected_entry = int(
        entry_candidates.loc[entry_candidates["selected"], "n_regions"].iloc[0]
    )
    selected_exit = int(
        exit_candidates.loc[exit_candidates["selected"], "n_regions"].iloc[0]
    )

    od_pairs = pd.DataFrame(
        {
            "trajectory_id": endpoints["trajectory_id"].astype(str),
            "entry_region": entry_labels,
            "exit_region": exit_labels,
        }
    )
    od_pairs["od_pair"] = (
        od_pairs["entry_region"].astype(str)
        + "->"
        + od_pairs["exit_region"].astype(str)
    )
    od_counts = (
        od_pairs["od_pair"]
        .value_counts()
        .rename_axis("od_pair")
        .reset_index(name="count")
    )
    od_counts["share"] = od_counts["count"] / len(od_pairs)

    threshold_rows: list[dict[str, Any]] = []
    for threshold in support_thresholds:
        valid = od_counts[od_counts["share"] >= float(threshold)]
        threshold_rows.append(
            {
                "scene": scene,
                "support_threshold": float(threshold),
                "support_threshold_percent": 100.0 * float(threshold),
                "support_threshold_absolute_count": int(
                    np.ceil(float(threshold) * len(od_pairs))
                ),
                "n_entry_regions": selected_entry,
                "n_exit_regions": selected_exit,
                "hg_estimated_target": int(len(valid)),
                "od_coverage": float(valid["count"].sum() / len(od_pairs)),
                "smallest_valid_od_pair_share": (
                    float(valid["share"].min()) if len(valid) else np.nan
                ),
                "n_trajectories": int(len(od_pairs)),
            }
        )
    threshold_candidates = pd.DataFrame(threshold_rows)
    targets = threshold_candidates["hg_estimated_target"].tolist()
    instability: list[float] = []
    for index, target in enumerate(targets):
        neighbor_differences = []
        if index > 0:
            neighbor_differences.append(abs(target - targets[index - 1]))
        if index < len(targets) - 1:
            neighbor_differences.append(abs(target - targets[index + 1]))
        instability.append(
            float(np.mean(neighbor_differences)) if neighbor_differences else 0.0
        )
    threshold_candidates["target_local_instability"] = instability

    eligible = threshold_candidates[
        (threshold_candidates["od_coverage"] >= 0.90)
        & (threshold_candidates["hg_estimated_target"] >= 2)
    ].copy()
    if eligible.empty:
        eligible = threshold_candidates[
            (threshold_candidates["od_coverage"] >= 0.80)
            & (threshold_candidates["hg_estimated_target"] >= 2)
        ].copy()
    if eligible.empty:
        eligible = threshold_candidates.copy()
    eligible["_threshold_distance"] = (
        eligible["support_threshold"] - 0.005
    ).abs()
    selected = eligible.sort_values(
        [
            "target_local_instability",
            "_threshold_distance",
            "od_coverage",
            "support_threshold",
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).iloc[0]
    threshold_candidates["selected"] = (
        threshold_candidates["support_threshold"]
        == float(selected["support_threshold"])
    )
    summary = {
        "scene": scene,
        "n_entry_regions": selected_entry,
        "n_exit_regions": selected_exit,
        "support_threshold": float(selected["support_threshold"]),
        "support_threshold_percent": float(
            selected["support_threshold_percent"]
        ),
        "support_threshold_absolute_count": int(
            selected["support_threshold_absolute_count"]
        ),
        "hg_estimated_target": int(selected["hg_estimated_target"]),
        "od_coverage": float(selected["od_coverage"]),
        "n_trajectories": int(len(endpoints)),
        "sampling_applied": False,
        "random_seed": int(seed),
        "implementation_version": IMPLEMENTATION_VERSION,
        "target_selection_rule": (
            "prefer coverage>=0.90 and target>=2; minimize adjacent-threshold "
            "target instability; tie by distance to 0.5% support, higher "
            "coverage, then lower threshold"
        ),
    }
    return summary, threshold_candidates, region_candidates, od_counts


def safe_cluster_metrics(
    features: np.ndarray, labels: np.ndarray, seed: int, metric_sample_size: int
) -> dict[str, float]:
    clustered_indices = np.flatnonzero(labels >= 0)
    if len(clustered_indices) > metric_sample_size:
        rng = np.random.default_rng(seed)
        clustered_indices = np.sort(
            rng.choice(
                clustered_indices, size=metric_sample_size, replace=False
            )
        )
    selected_features = features[clustered_indices]
    selected_labels = labels[clustered_indices]
    unique = np.unique(selected_labels)
    empty = {
        "silhouette_clustered_only": np.nan,
        "davies_bouldin_clustered_only": np.nan,
        "calinski_harabasz_clustered_only": np.nan,
    }
    if len(unique) < 2 or len(selected_labels) <= len(unique):
        return empty
    output: dict[str, float] = {}
    try:
        output["silhouette_clustered_only"] = float(
            silhouette_score(selected_features, selected_labels)
        )
    except Exception:
        output["silhouette_clustered_only"] = np.nan
    try:
        output["davies_bouldin_clustered_only"] = float(
            davies_bouldin_score(selected_features, selected_labels)
        )
    except Exception:
        output["davies_bouldin_clustered_only"] = np.nan
    try:
        output["calinski_harabasz_clustered_only"] = float(
            calinski_harabasz_score(selected_features, selected_labels)
        )
    except Exception:
        output["calinski_harabasz_clustered_only"] = np.nan
    return output


def quick_score(metrics: dict[str, Any]) -> float:
    alignment = 1.0 / (1.0 + float(metrics["cluster_count_error"]))
    outlier = 1.0 - np.clip(float(metrics["pct_outliers"]) / 100.0, 0, 1)
    largest = metrics.get("largest_cluster_ratio")
    balance = 1.0 - float(largest) if pd.notna(largest) else 0.5
    silhouette = metrics.get("silhouette_clustered_only")
    silhouette_term = (
        (float(silhouette) + 1.0) / 2.0 if pd.notna(silhouette) else 0.5
    )
    davies = metrics.get("davies_bouldin_clustered_only")
    db_term = (
        1.0 / (1.0 + float(davies))
        if pd.notna(davies) and float(davies) >= 0
        else 0.5
    )
    return float(
        0.55 * alignment
        + 0.20 * outlier
        + 0.10 * np.clip(balance, 0, 1)
        + 0.10 * np.clip(silhouette_term, 0, 1)
        + 0.05 * db_term
    )


def emas_hg(metrics: dict[str, Any]) -> float:
    expected = max(float(metrics["hg_estimated_target"]), 1.0)
    target_term = np.clip(
        1.0 - float(metrics["cluster_count_error"]) / expected, 0, 1
    )
    outlier_term = np.clip(1.0 - float(metrics["pct_outliers"]) / 100.0, 0, 1)
    largest = metrics.get("largest_cluster_ratio")
    balance = np.clip(1.0 - float(largest), 0, 1) if pd.notna(largest) else 0.5
    silhouette = metrics.get("silhouette_clustered_only")
    silhouette_term = (
        np.clip((float(silhouette) + 1.0) / 2.0, 0, 1)
        if pd.notna(silhouette)
        else 0.5
    )
    davies = metrics.get("davies_bouldin_clustered_only")
    db_term = (
        1.0 / (1.0 + float(davies))
        if pd.notna(davies) and float(davies) >= 0
        else 0.5
    )
    return float(
        0.50 * target_term
        + 0.20 * outlier_term
        + 0.10 * balance
        + 0.10 * silhouette_term
        + 0.10 * db_term
    )


def label_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    hg_target: int,
    fit_time_seconds: float,
    seed: int,
    metric_sample_size: int,
) -> dict[str, Any]:
    clustered = labels[labels >= 0]
    counts = pd.Series(clustered).value_counts()
    cluster_count = int(len(np.unique(clustered)))
    metrics: dict[str, Any] = {
        "n_total": int(len(labels)),
        "n_clusters": cluster_count,
        "cluster_count_error": abs(cluster_count - int(hg_target)),
        "n_outliers": int((labels == -1).sum()),
        "pct_outliers": float(100.0 * (labels == -1).sum() / len(labels)),
        "largest_cluster_ratio": (
            float(counts.max() / len(clustered)) if len(clustered) else np.nan
        ),
        "fit_time_s": float(fit_time_seconds),
        "hg_estimated_target": int(hg_target),
    }
    metrics.update(
        safe_cluster_metrics(features, labels, seed, metric_sample_size)
    )
    metrics["quick_score"] = quick_score(metrics)
    metrics["EMAS_HG"] = emas_hg(metrics)
    return metrics


def optics_eps_grid(
    features: np.ndarray,
    seed: int,
    quantiles: list[float],
    sample_size: int,
) -> list[float]:
    sample = features
    if len(features) > sample_size:
        rng = np.random.default_rng(seed)
        sample = features[
            np.sort(rng.choice(len(features), size=sample_size, replace=False))
        ]
    neighbor_count = min(21, max(2, len(sample) - 1))
    distances, _ = NearestNeighbors(n_neighbors=neighbor_count).fit(
        sample
    ).kneighbors(sample)
    values = np.quantile(distances[:, -1], quantiles)
    return sorted({max(float(value), 1e-6) for value in values})


def candidate_grid(
    method: str, features: np.ndarray, seed: int, config: dict[str, Any]
) -> list[dict[str, Any]]:
    grids = config["candidate_grids"]
    if method == "kmeans":
        return [
            {
                "n_clusters": int(cluster_count),
                "n_init": int(grids["kmeans"]["n_init"]),
                "max_iter": int(grids["kmeans"]["max_iter"]),
            }
            for cluster_count in grids["kmeans"]["n_clusters"]
        ]
    if method == "hdbscan":
        return [
            {
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
            }
            for min_cluster_size in grids["hdbscan"]["min_cluster_size"]
            for min_samples in grids["hdbscan"]["min_samples"]
            if int(min_cluster_size) < len(features)
        ]
    if method == "optics":
        eps_values = optics_eps_grid(
            features,
            seed,
            [float(value) for value in grids["optics"]["max_eps_quantiles"]],
            int(grids["optics"]["quantile_sample_size"]),
        )
        return [
            {
                "min_samples": int(min_samples),
                "xi": float(xi),
                "max_eps": float(max_eps),
            }
            for min_samples in grids["optics"]["min_samples"]
            for xi in grids["optics"]["xi"]
            for max_eps in eps_values
            if int(min_samples) < len(features)
        ]
    raise ValueError(f"Unsupported clustering method: {method}")


def fit_predict(
    method: str, features: np.ndarray, parameters: dict[str, Any], seed: int
) -> np.ndarray:
    if method == "kmeans":
        return KMeans(
            n_clusters=int(parameters["n_clusters"]),
            n_init=int(parameters["n_init"]),
            max_iter=int(parameters["max_iter"]),
            random_state=int(seed),
            algorithm="lloyd",
        ).fit_predict(features).astype(int)
    if method == "hdbscan":
        return hdbscan.HDBSCAN(
            min_cluster_size=int(parameters["min_cluster_size"]),
            min_samples=int(parameters["min_samples"]),
            metric="euclidean",
            core_dist_n_jobs=1,
        ).fit_predict(features).astype(int)
    if method == "optics":
        return OPTICS(
            min_samples=int(parameters["min_samples"]),
            xi=float(parameters["xi"]),
            max_eps=float(parameters["max_eps"]),
        ).fit_predict(features).astype(int)
    raise ValueError(f"Unsupported clustering method: {method}")


def run_candidate_trials(
    scene: str,
    method: str,
    features: np.ndarray,
    hg_target: int,
    seed: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grid = candidate_grid(method, features, seed, config)
    for trial_index, parameters in enumerate(grid, start=1):
        fit_seed = int(seed + trial_index)
        started = time.perf_counter()
        try:
            labels = fit_predict(method, features, parameters, fit_seed)
            metrics = label_statistics(
                features,
                labels,
                hg_target,
                time.perf_counter() - started,
                fit_seed,
                int(config["metrics"]["cluster_metric_sample_size"]),
            )
            error = ""
        except Exception as exc:
            metrics = {
                "n_total": int(len(features)),
                "n_clusters": 0,
                "cluster_count_error": int(hg_target),
                "n_outliers": np.nan,
                "pct_outliers": np.nan,
                "largest_cluster_ratio": np.nan,
                "fit_time_s": float(time.perf_counter() - started),
                "hg_estimated_target": int(hg_target),
                "silhouette_clustered_only": np.nan,
                "davies_bouldin_clustered_only": np.nan,
                "calinski_harabasz_clustered_only": np.nan,
                "quick_score": np.nan,
                "EMAS_HG": np.nan,
            }
            error = repr(exc)
        rows.append(
            {
                "scene": scene,
                "method": method,
                "trial_index": int(trial_index),
                "fit_random_seed": fit_seed,
                "params_json": json.dumps(parameters, sort_keys=True),
                **metrics,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def _finite(value: Any, fallback: float, invert: bool = False) -> float:
    if pd.isna(value):
        return fallback
    numeric = float(value)
    return -numeric if invert else numeric


def untargeted_selection_key(row: pd.Series) -> tuple[Any, ...]:
    return (
        _finite(row["silhouette_clustered_only"], 1.0, invert=True),
        _finite(row["davies_bouldin_clustered_only"], np.inf),
        _finite(row["calinski_harabasz_clustered_only"], np.inf, invert=True),
        _finite(row["largest_cluster_ratio"], 1.0),
        str(row["params_json"]),
    )


def expected_selection_key(row: pd.Series, method: str) -> tuple[Any, ...]:
    key: list[Any] = [_finite(row["cluster_count_error"], np.inf)]
    if method in {"hdbscan", "optics"}:
        key.append(_finite(row["pct_outliers"], 100.0))
    key.extend(
        [
            _finite(row["silhouette_clustered_only"], 1.0, invert=True),
            _finite(row["davies_bouldin_clustered_only"], np.inf),
            _finite(
                row["calinski_harabasz_clustered_only"], np.inf, invert=True
            ),
            _finite(row["largest_cluster_ratio"], 1.0),
            str(row["params_json"]),
        ]
    )
    return tuple(key)


def select_candidate(
    trials: pd.DataFrame, strategy: str, method: str
) -> pd.Series:
    valid = trials[trials["error"] == ""].copy()
    if valid.empty:
        raise RuntimeError(f"No successful {method} candidates are available.")
    if strategy == "untargeted_selection":
        keys = valid.apply(untargeted_selection_key, axis=1)
    elif strategy == "hg_expected_aware_selection":
        keys = valid.apply(lambda row: expected_selection_key(row, method), axis=1)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
    selected_index = min(keys.index, key=lambda index: keys.loc[index])
    selected = valid.loc[selected_index].copy()
    serialized_key = []
    for value in keys.loc[selected_index]:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            serialized_key.append("Infinity" if value > 0 else "-Infinity")
        elif isinstance(value, np.generic):
            serialized_key.append(value.item())
        else:
            serialized_key.append(value)
    selected["selection_key_json"] = json.dumps(serialized_key)
    return selected
