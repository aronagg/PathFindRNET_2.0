"""Canonical frozen HG target estimator used by the publication workflow.

This module reproduces ``split-aware-hg-msa-tc-v1``. It intentionally preserves
the original heuristic endpoint-region and support-threshold choices. Diagnostic
callers may inspect intermediate fits, but they must not replace frozen outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


IMPLEMENTATION_VERSION = "split-aware-hg-msa-tc-v1"
CANONICAL_MODULE_VERSION = "canonical-hg-target-estimator-v1"


@dataclass(frozen=True)
class EndpointRegionFit:
    """All deterministic endpoint-region candidates and the frozen selection."""

    role: str
    center: np.ndarray
    features: np.ndarray
    metric_indices: np.ndarray
    labels_by_count: dict[int, np.ndarray]
    models_by_count: dict[int, KMeans]
    candidates: pd.DataFrame
    selected_count: int

    @property
    def labels(self) -> np.ndarray:
        return self.labels_by_count[self.selected_count]

    @property
    def selected_model(self) -> KMeans:
        return self.models_by_count[self.selected_count]

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Assign new endpoints using the target-split center and selected KMeans fit."""
        return self.selected_model.predict(endpoint_features(points, self.center)).astype(int)


@dataclass(frozen=True)
class TargetEstimateResult:
    """Frozen outputs plus diagnostic-only intermediate assignments and fits."""

    summary: dict[str, Any]
    threshold_candidates: pd.DataFrame
    region_candidates: pd.DataFrame
    od_counts: pd.DataFrame
    endpoint_assignments: pd.DataFrame
    entry_fit: EndpointRegionFit
    exit_fit: EndpointRegionFit


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Transform finite 2-D points with a 3x3 camera-to-top-view homography."""
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
    """Transform the frozen camera-space start/end feature columns."""
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


def endpoint_features(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Return the frozen circular-angle plus weak clipped-radius representation."""
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    delta = points - center
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    radius = np.linalg.norm(delta, axis=1)
    radius_norm = radius / max(float(np.nanmedian(radius)), 1e-12)
    radius_norm = np.clip(radius_norm, 0, 3) / 3.0
    return np.column_stack([np.cos(angle), np.sin(angle), 0.25 * radius_norm])


def region_metrics(
    features: np.ndarray, labels: np.ndarray, metric_indices: np.ndarray
) -> tuple[float, float]:
    """Compute the exact internal metrics used for endpoint-region selection."""
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


def estimate_endpoint_region_fit(
    points: np.ndarray,
    role: str,
    scene: str,
    seed: int,
    region_counts: list[int],
    metric_sample_size: int,
) -> EndpointRegionFit:
    """Fit every frozen K candidate and select by silhouette, DB, then lower K."""
    points = np.asarray(points, dtype=np.float64)
    center = np.median(points, axis=0)
    features = endpoint_features(points, center)
    if len(features) > metric_sample_size:
        rng = np.random.default_rng(seed)
        metric_indices = np.sort(
            rng.choice(len(features), size=metric_sample_size, replace=False)
        )
    else:
        metric_indices = np.arange(len(features))

    rows: list[dict[str, Any]] = []
    labels_by_count: dict[int, np.ndarray] = {}
    models_by_count: dict[int, KMeans] = {}
    for raw_count in region_counts:
        region_count = int(raw_count)
        model = KMeans(
            n_clusters=region_count,
            n_init=10,
            random_state=seed,
            algorithm="lloyd",
        )
        labels = model.fit_predict(features)
        silhouette, davies_bouldin = region_metrics(
            features, labels, metric_indices
        )
        rows.append(
            {
                "scene": scene,
                "endpoint_role": role,
                "n_regions": region_count,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "n_trajectories": int(len(points)),
                "metric_sample_size": int(len(metric_indices)),
                "random_seed": int(seed),
            }
        )
        labels_by_count[region_count] = labels.astype(int)
        models_by_count[region_count] = model

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
    candidates = candidates.drop(columns=["_silhouette_key", "_db_key"])
    return EndpointRegionFit(
        role=role,
        center=center,
        features=features,
        metric_indices=metric_indices,
        labels_by_count=labels_by_count,
        models_by_count=models_by_count,
        candidates=candidates,
        selected_count=selected_count,
    )


def estimate_endpoint_regions(
    points: np.ndarray,
    role: str,
    scene: str,
    seed: int,
    region_counts: list[int],
    metric_sample_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compatibility API used by the frozen split-aware runner."""
    fit = estimate_endpoint_region_fit(
        points, role, scene, seed, region_counts, metric_sample_size
    )
    return fit.labels, fit.candidates


def build_od_support(
    trajectory_ids: pd.Series,
    entry_labels: np.ndarray,
    exit_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create trajectory-level OD assignments and observed-pair support counts."""
    assignments = pd.DataFrame(
        {
            "trajectory_id": trajectory_ids.astype(str),
            "entry_region": entry_labels.astype(int),
            "exit_region": exit_labels.astype(int),
        }
    )
    assignments["od_pair"] = (
        assignments["entry_region"].astype(str)
        + "->"
        + assignments["exit_region"].astype(str)
    )
    counts = (
        assignments["od_pair"]
        .value_counts()
        .rename_axis("od_pair")
        .reset_index(name="count")
    )
    counts["share"] = counts["count"] / len(assignments)
    return assignments, counts


def build_threshold_candidates(
    scene: str,
    od_counts: pd.DataFrame,
    n_trajectories: int,
    selected_entry: int,
    selected_exit: int,
    support_thresholds: list[float],
) -> pd.DataFrame:
    """Evaluate the frozen support grid and adjacent-target instability."""
    rows: list[dict[str, Any]] = []
    for raw_threshold in support_thresholds:
        threshold = float(raw_threshold)
        valid = od_counts[od_counts["share"] >= threshold]
        rows.append(
            {
                "scene": scene,
                "support_threshold": threshold,
                "support_threshold_percent": 100.0 * threshold,
                "support_threshold_absolute_count": int(
                    np.ceil(threshold * n_trajectories)
                ),
                "n_entry_regions": int(selected_entry),
                "n_exit_regions": int(selected_exit),
                "hg_estimated_target": int(len(valid)),
                "od_coverage": float(valid["count"].sum() / n_trajectories),
                "smallest_valid_od_pair_share": (
                    float(valid["share"].min()) if len(valid) else np.nan
                ),
                "n_trajectories": int(n_trajectories),
            }
        )
    candidates = pd.DataFrame(rows)
    targets = candidates["hg_estimated_target"].tolist()
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
    candidates["target_local_instability"] = instability
    return candidates


def select_threshold_candidate(candidates: pd.DataFrame) -> pd.Series:
    """Apply the exact frozen data-dependent threshold heuristic."""
    eligible = candidates[
        (candidates["od_coverage"] >= 0.90)
        & (candidates["hg_estimated_target"] >= 2)
    ].copy()
    if eligible.empty:
        eligible = candidates[
            (candidates["od_coverage"] >= 0.80)
            & (candidates["hg_estimated_target"] >= 2)
        ].copy()
    if eligible.empty:
        eligible = candidates.copy()
    eligible["_threshold_distance"] = (
        eligible["support_threshold"] - 0.005
    ).abs()
    return eligible.sort_values(
        [
            "target_local_instability",
            "_threshold_distance",
            "od_coverage",
            "support_threshold",
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).iloc[0]


def estimate_hg_target_detailed(
    scene: str,
    endpoints: pd.DataFrame,
    seed: int,
    region_counts: list[int],
    support_thresholds: list[float],
    region_metric_sample_size: int,
) -> TargetEstimateResult:
    """Reproduce the frozen target and retain diagnostic-only intermediate state."""
    start_points = endpoints[
        ["start_x_topview", "start_y_topview"]
    ].to_numpy(dtype=np.float64)
    end_points = endpoints[["end_x_topview", "end_y_topview"]].to_numpy(
        dtype=np.float64
    )
    entry_fit = estimate_endpoint_region_fit(
        start_points,
        "entry",
        scene,
        seed,
        region_counts,
        region_metric_sample_size,
    )
    exit_fit = estimate_endpoint_region_fit(
        end_points,
        "exit",
        scene,
        seed + 17,
        region_counts,
        region_metric_sample_size,
    )
    region_candidates = pd.concat(
        [entry_fit.candidates, exit_fit.candidates], ignore_index=True
    )
    assignments, od_counts = build_od_support(
        endpoints["trajectory_id"], entry_fit.labels, exit_fit.labels
    )
    threshold_candidates = build_threshold_candidates(
        scene,
        od_counts,
        len(assignments),
        entry_fit.selected_count,
        exit_fit.selected_count,
        support_thresholds,
    )
    selected = select_threshold_candidate(threshold_candidates)
    threshold_candidates["selected"] = (
        threshold_candidates["support_threshold"]
        == float(selected["support_threshold"])
    )
    assignments["supported_at_selected_threshold"] = assignments["od_pair"].isin(
        set(
            od_counts.loc[
                od_counts["share"] >= float(selected["support_threshold"]), "od_pair"
            ]
        )
    )
    summary = {
        "scene": scene,
        "n_entry_regions": entry_fit.selected_count,
        "n_exit_regions": exit_fit.selected_count,
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
    return TargetEstimateResult(
        summary=summary,
        threshold_candidates=threshold_candidates,
        region_candidates=region_candidates,
        od_counts=od_counts,
        endpoint_assignments=assignments,
        entry_fit=entry_fit,
        exit_fit=exit_fit,
    )


def estimate_hg_target(
    scene: str,
    endpoints: pd.DataFrame,
    seed: int,
    region_counts: list[int],
    support_thresholds: list[float],
    region_metric_sample_size: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compatibility API returning the original four frozen output objects."""
    result = estimate_hg_target_detailed(
        scene,
        endpoints,
        seed,
        region_counts,
        support_thresholds,
        region_metric_sample_size,
    )
    return (
        result.summary,
        result.threshold_candidates,
        result.region_candidates,
        result.od_counts,
    )
