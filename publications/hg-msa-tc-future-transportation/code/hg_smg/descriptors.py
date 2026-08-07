"""Circular geometry and endpoint-adjacent heading descriptors."""

from __future__ import annotations

import numpy as np


TAU = 2.0 * np.pi
NUMERICAL_FLOOR = 8.0 * np.finfo(np.float64).eps


def wrap_angle(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values + np.pi) % TAU - np.pi


def circular_distance(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.abs(wrap_angle(np.asarray(a) - np.asarray(b)))


def circular_l1_median(values: np.ndarray) -> float:
    """Return the observed circular L1 median with the frozen tie-break.

    Candidate objectives are evaluated in O(n log n) using prefix sums over an
    unwrapped duplicate sequence. The smallest wrapped observed angle wins ties.
    """
    angles = np.sort(wrap_angle(np.asarray(values, dtype=np.float64)))
    angles = angles[np.isfinite(angles)]
    if not len(angles):
        return float("nan")
    if len(angles) == 1:
        return float(angles[0])
    extended = np.concatenate([angles - TAU, angles, angles + TAU])
    prefix = np.concatenate([[0.0], np.cumsum(extended)])
    objectives = np.empty(len(angles), dtype=np.float64)
    n_values = len(angles)
    for index, candidate in enumerate(angles):
        left = int(np.searchsorted(extended, candidate - np.pi, side="left"))
        right = left + n_values
        middle = int(np.searchsorted(extended, candidate, side="right"))
        middle = min(max(middle, left), right)
        lower = candidate * (middle - left) - (prefix[middle] - prefix[left])
        upper = (prefix[right] - prefix[middle]) - candidate * (right - middle)
        objectives[index] = lower + upper
    minimum = float(np.min(objectives))
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(minimum))
    return float(angles[np.flatnonzero(objectives <= minimum + tolerance)[0]])


def circular_quantile_radius(values: np.ndarray, center: float, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite) or not np.isfinite(center):
        return float("nan")
    return float(np.quantile(circular_distance(finite, center), quantile, method="linear"))


def circular_mad(values: np.ndarray, center: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite) or not np.isfinite(center):
        return float("nan")
    return float(np.median(circular_distance(finite, center)))


def directed_heading(points: np.ndarray, role: str, window_points: int) -> float:
    """Compute the frozen endpoint-adjacent directed heading."""
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < int(window_points):
        return float("nan")
    if role == "entry":
        start, end = points[0], points[int(window_points) - 1]
    elif role == "exit":
        start, end = points[-int(window_points)], points[-1]
    else:
        raise ValueError(f"Unknown role: {role}")
    delta = end - start
    if not np.isfinite(delta).all() or np.linalg.norm(delta) <= 0.0:
        return float("nan")
    return float(np.arctan2(delta[1], delta[0]))


def _bootstrap_centers_unwrapped(samples: np.ndarray, anchor: float) -> np.ndarray:
    """Vectorized exact L1 centers when samples occupy one open semicircle."""
    delta = wrap_angle(samples - anchor)
    if np.ptp(delta) >= np.pi:
        return np.asarray([circular_l1_median(row) for row in samples])
    return wrap_angle(anchor + np.median(delta, axis=1))


def bootstrap_circular_radii(
    values: np.ndarray,
    sample_indices: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[float, float]:
    """Calculate nested self-consistency radii from fixed bootstrap samples."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        return {float(q): float("nan") for q in quantiles}
    samples = values[sample_indices]
    anchor = circular_l1_median(values)
    centers = _bootstrap_centers_unwrapped(samples, anchor)
    residuals = circular_distance(samples, centers[:, None])
    output: dict[float, float] = {}
    for raw_q in quantiles:
        q = float(raw_q)
        per_replicate = np.quantile(residuals, q, axis=1, method="linear")
        output[q] = max(
            float(np.quantile(per_replicate, q, method="linear")),
            NUMERICAL_FLOOR,
        )
    return output


def bootstrap_bearing_radii(
    points: np.ndarray,
    role_center: np.ndarray,
    sample_indices: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[float, float]:
    """Bootstrap the robust-centroid bearing and endpoint-bearing residuals."""
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(role_center, dtype=np.float64)
    if len(points) < 2 or not np.isfinite(points).all():
        return {float(q): float("nan") for q in quantiles}
    sampled = points[sample_indices]
    centroids = np.median(sampled, axis=1)
    bootstrap_centers = np.arctan2(
        centroids[:, 1] - center[1], centroids[:, 0] - center[0]
    )
    point_angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    residuals = circular_distance(point_angles[sample_indices], bootstrap_centers[:, None])
    output: dict[float, float] = {}
    for raw_q in quantiles:
        q = float(raw_q)
        per_replicate = np.quantile(residuals, q, axis=1, method="linear")
        output[q] = max(
            float(np.quantile(per_replicate, q, method="linear")),
            NUMERICAL_FLOOR,
        )
    return output


def jensen_shannon_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Base-2 Jensen-Shannon distance for normalized finite profiles."""
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1 or np.any(p < 0) or np.any(q < 0):
        return float("nan")
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))
