import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS


def optics_cluster(
    entry_exit_points: np.ndarray, min_samples: int = 30, xi: float = 0.05, max_eps: float = np.inf
):
    """Run OPTICS on 4D [x_entry, y_entry, x_exit, y_exit] points.

    Parameters
    ----------
    entry_exit_points : np.ndarray
        4D array of [x_entry, y_entry, x_exit, y_exit]
    min_samples : int
        Minimum samples in neighborhood for core point
    xi : float
        Minimum steepness for cluster extraction (lower = more clusters)
    max_eps : float
        Maximum distance between two samples for one to be in neighborhood of the other

    Returns
    -------
    labels : np.ndarray
        Cluster labels (-1 indicates outliers/noise)
    model : OPTICS
        Fitted OPTICS model with reachability info
    """
    model = OPTICS(min_samples=min_samples, xi=xi, max_eps=max_eps)
    labels = model.fit_predict(entry_exit_points)
    return labels, model


def get_outlier_stats(labels: np.ndarray, track_ids: np.ndarray | None = None) -> dict:
    """Compute statistics about outliers (samples with label -1).

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels from OPTICS (-1 = outlier)
    track_ids : np.ndarray | None
        Optional track IDs corresponding to each sample

    Returns
    -------
    dict
        Statistics including count, percentage, and optionally track IDs
    """
    outlier_mask = labels == -1
    n_outliers = outlier_mask.sum()
    n_total = len(labels)
    pct_outliers = 100.0 * n_outliers / n_total if n_total > 0 else 0.0

    stats = {
        "n_outliers": int(n_outliers),
        "n_total": int(n_total),
        "pct_outliers": float(pct_outliers),
        "n_clusters": int(labels[labels >= 0].max() + 1) if (labels >= 0).any() else 0,
    }

    if track_ids is not None:
        stats["outlier_track_ids"] = track_ids[outlier_mask].tolist()

    return stats


def analyze_outliers(
    entry_exit_points: np.ndarray,
    labels: np.ndarray,
    model: OPTICS,
    track_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Analyze outlier samples in detail using OPTICS reachability.

    Parameters
    ----------
    entry_exit_points : np.ndarray
        Original 4D entry-exit coordinates
    labels : np.ndarray
        Cluster labels from OPTICS
    model : OPTICS
        Fitted OPTICS model
    track_ids : np.ndarray | None
        Optional track IDs

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier analysis including reachability distances
    """
    outlier_mask = labels == -1
    outlier_indices = np.where(outlier_mask)[0]

    if len(outlier_indices) == 0:
        # No outliers found
        cols = ["index", "x_entry", "y_entry", "x_exit", "y_exit", "reachability"]
        if track_ids is not None:
            cols.insert(0, "track_id")
        return pd.DataFrame(columns=cols)

    # Extract outlier data
    outlier_coords = entry_exit_points[outlier_indices]
    reachability = model.reachability_[outlier_indices]

    # Replace inf values with a large finite number for better handling
    # inf means the point is completely unreachable from any cluster
    max_finite_reach = (
        reachability[~np.isinf(reachability)].max() if np.any(~np.isinf(reachability)) else 1.0
    )
    reachability_clean = np.where(np.isinf(reachability), max_finite_reach * 2.0, reachability)

    data = {
        "index": outlier_indices,
        "x_entry": outlier_coords[:, 0],
        "y_entry": outlier_coords[:, 1],
        "x_exit": outlier_coords[:, 2],
        "y_exit": outlier_coords[:, 3],
        "reachability": reachability_clean,
        "is_unreachable": np.isinf(reachability),  # Flag true outliers
    }

    if track_ids is not None:
        data["track_id"] = track_ids[outlier_indices]
        # Reorder to put track_id first
        data = {"track_id": data.pop("track_id"), **data}

    df = pd.DataFrame(data)
    # Sort by reachability (highest = most outlier-like)
    df = df.sort_values("reachability", ascending=False)

    return df
