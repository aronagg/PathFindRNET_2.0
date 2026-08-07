"""Canonical, read-only reproduction of the frozen scene homographies.

The historical implementation called ``cv2.findHomography`` with RANSAC and a
10 px destination-space reprojection threshold. OpenCV defaults were used for
the iteration limit and confidence. These defaults are passed explicitly here
so the publication description and executable implementation are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


HISTORICAL_RANSAC_THRESHOLD_PX = 10.0
HISTORICAL_MAX_ITERS = 2000
HISTORICAL_CONFIDENCE = 0.995
IMPLEMENTATION_VERSION = "homography-quality-v1"


@dataclass(frozen=True)
class QualityGate:
    quality_class: str
    passes: bool
    reasons: tuple[str, ...]


def normalize_homography(matrix: np.ndarray) -> np.ndarray:
    """Normalize a projective matrix to H[2, 2] = 1."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 homography, got {value.shape}.")
    if not np.isfinite(value).all() or abs(value[2, 2]) < 1e-12:
        raise ValueError("Homography cannot be normalized by H[2, 2].")
    return value / value[2, 2]


def apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a homography to finite 2-D points with denominator validation."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Points must have shape (n, 2).")
    if not np.isfinite(values).all():
        raise ValueError("Homography inputs must be finite.")
    homogeneous = np.column_stack([values, np.ones(len(values))]) @ normalize_homography(
        matrix
    ).T
    denominator = homogeneous[:, 2]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("Homography produced a near-zero homogeneous denominator.")
    return homogeneous[:, :2] / denominator[:, None]


def estimate_historical_homography(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    *,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run the exact historical RANSAC call, including its fallback."""
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Source and destination points must have matching (n, 2) shapes.")
    if len(source) < 4:
        raise ValueError("At least four point correspondences are required.")
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("Calibration correspondences must be finite.")
    if rng_seed is not None:
        cv2.setRNGSeed(int(rng_seed) & 0x7FFFFFFF)
    matrix, mask = cv2.findHomography(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=HISTORICAL_RANSAC_THRESHOLD_PX,
        maxIters=HISTORICAL_MAX_ITERS,
        confidence=HISTORICAL_CONFIDENCE,
    )
    method = "ransac_10px"
    if matrix is None or mask is None or int(mask.sum()) < 4:
        matrix, mask = cv2.findHomography(source, destination, method=0)
        method = "least_squares_fallback"
    if matrix is None or mask is None:
        raise RuntimeError("cv2.findHomography failed.")
    return normalize_homography(matrix), mask.reshape(-1).astype(np.int8), method


def reprojection_statistics(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    matrix: np.ndarray,
    inlier_mask: np.ndarray,
    source_image_size: tuple[int, int],
    destination_image_size: tuple[int, int],
) -> dict[str, float]:
    """Calculate forward, inverse, and normalized calibration diagnostics."""
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    mask = np.asarray(inlier_mask, dtype=bool)
    forward_projected = apply_homography(source, matrix)
    inverse_projected = apply_homography(destination, np.linalg.inv(matrix))
    forward = np.linalg.norm(forward_projected - destination, axis=1)
    inverse = np.linalg.norm(inverse_projected - source, axis=1)
    source_diagonal = float(np.hypot(*source_image_size))
    destination_diagonal = float(np.hypot(*destination_image_size))
    symmetric_normalized = forward / destination_diagonal + inverse / source_diagonal
    inlier = forward[mask]
    return {
        "forward_mean_error_px": float(np.mean(forward)),
        "forward_median_error_px": float(np.median(forward)),
        "forward_rmse_error_px": float(np.sqrt(np.mean(forward**2))),
        "forward_max_error_px": float(np.max(forward)),
        "forward_p90_error_px": float(np.percentile(forward, 90)),
        "forward_p95_error_px": float(np.percentile(forward, 95)),
        "normalized_rmse_fraction_destination_diagonal": float(
            np.sqrt(np.mean(forward**2)) / destination_diagonal
        ),
        "normalized_p95_fraction_destination_diagonal": float(
            np.percentile(forward, 95) / destination_diagonal
        ),
        "inverse_mean_error_px": float(np.mean(inverse)),
        "inverse_median_error_px": float(np.median(inverse)),
        "inverse_rmse_error_px": float(np.sqrt(np.mean(inverse**2))),
        "symmetric_normalized_mean": float(np.mean(symmetric_normalized)),
        "symmetric_normalized_p95": float(np.percentile(symmetric_normalized, 95)),
        "inlier_mean_error_px": float(np.mean(inlier)) if len(inlier) else float("nan"),
        "inlier_median_error_px": float(np.median(inlier)) if len(inlier) else float("nan"),
        "inlier_max_error_px": float(np.max(inlier)) if len(inlier) else float("nan"),
        "inlier_count": int(mask.sum()),
        "inlier_fraction": float(mask.mean()),
    }


def convex_hull_coverage(
    source_points: np.ndarray,
    source_image_size: tuple[int, int],
    endpoint_points: np.ndarray,
) -> dict[str, Any]:
    """Measure calibration-point coverage and endpoint extrapolation."""
    source = np.asarray(source_points, dtype=np.float32)
    endpoints = np.asarray(endpoint_points, dtype=np.float64)
    hull = cv2.convexHull(source).reshape(-1, 2)
    width, height = source_image_size
    hull_area_fraction = float(cv2.contourArea(hull.astype(np.float32)) / (width * height))
    inside = np.array(
        [cv2.pointPolygonTest(hull.astype(np.float32), tuple(point), False) >= 0 for point in endpoints],
        dtype=bool,
    )
    grid_cells = set()
    for x_value, y_value in source:
        x_cell = min(3, max(0, int(4 * x_value / width)))
        y_cell = min(3, max(0, int(4 * y_value / height)))
        grid_cells.add((x_cell, y_cell))
    return {
        "source_calibration_hull_area_fraction": hull_area_fraction,
        "source_calibration_grid_coverage_fraction": len(grid_cells) / 16.0,
        "endpoint_inside_calibration_hull_fraction": float(inside.mean()),
        "endpoint_extrapolation_fraction": float(1.0 - inside.mean()),
        "calibration_hull": hull,
        "endpoint_inside_hull": inside,
    }


def normalized_dlt_condition(
    source_points: np.ndarray, destination_points: np.ndarray
) -> dict[str, float]:
    """Return scale-aware DLT conditioning diagnostics without fitting a new H."""

    def normalize(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        center = np.mean(values, axis=0)
        shifted = values - center
        mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
        scale = np.sqrt(2.0) / mean_distance if mean_distance > 0 else 1.0
        return shifted * scale

    source = normalize(source_points)
    destination = normalize(destination_points)
    rows = []
    for (x_value, y_value), (u_value, v_value) in zip(
        source, destination, strict=True
    ):
        rows.extend(
            [
                [-x_value, -y_value, -1, 0, 0, 0, u_value * x_value, u_value * y_value, u_value],
                [0, 0, 0, -x_value, -y_value, -1, v_value * x_value, v_value * y_value, v_value],
            ]
        )
    singular_values = np.linalg.svd(np.asarray(rows, dtype=np.float64), compute_uv=False)
    return {
        "normalized_dlt_effective_condition": float(singular_values[0] / singular_values[-2]),
        "normalized_dlt_nullspace_gap": float(singular_values[-2] / singular_values[-1]),
    }


def classify_homography_quality(metrics: dict[str, float]) -> QualityGate:
    """Apply a priori engineering thresholds independent of clustering outcomes."""
    normalized_rmse = metrics["normalized_rmse_fraction_destination_diagonal"]
    normalized_p95 = metrics["normalized_p95_fraction_destination_diagonal"]
    inlier_fraction = metrics["inlier_fraction"]
    hull_fraction = metrics["source_calibration_hull_area_fraction"]
    extrapolation = metrics["endpoint_extrapolation_fraction"]

    tiers = (
        ("good", 0.010, 0.020, 0.60, 0.20, 0.95),
        ("acceptable", 0.020, 0.040, 0.40, 0.10, 0.98),
        ("acceptable_with_caution", 0.030, 0.060, 0.30, 0.05, 0.995),
    )
    for name, rmse_limit, p95_limit, inlier_min, hull_min, extrapolation_max in tiers:
        if (
            normalized_rmse <= rmse_limit
            and normalized_p95 <= p95_limit
            and inlier_fraction >= inlier_min
            and hull_fraction >= hull_min
            and extrapolation <= extrapolation_max
        ):
            reasons = []
            if extrapolation > 0.90:
                reasons.append("most trajectory endpoints require extrapolation beyond the calibration hull")
            if normalized_rmse > 0.01:
                reasons.append("normalized RMSE exceeds the good-class threshold")
            return QualityGate(name, True, tuple(reasons))
    reasons = []
    if normalized_rmse > 0.03:
        reasons.append("normalized RMSE exceeds 3% of destination diagonal")
    if normalized_p95 > 0.06:
        reasons.append("P95 error exceeds 6% of destination diagonal")
    if inlier_fraction < 0.30:
        reasons.append("RANSAC inlier fraction is below 30%")
    if hull_fraction < 0.05:
        reasons.append("calibration hull covers less than 5% of the camera image")
    if extrapolation > 0.995:
        reasons.append("more than 99.5% of target-estimation endpoints require extrapolation")
    return QualityGate("poor", False, tuple(reasons) or ("combined gate criteria failed",))


def align_labels_and_change_rate(
    reference_labels: np.ndarray, candidate_labels: np.ndarray
) -> tuple[np.ndarray, float]:
    """Align arbitrary cluster IDs by maximum overlap and return change rate."""
    reference = np.asarray(reference_labels, dtype=int)
    candidate = np.asarray(candidate_labels, dtype=int)
    reference_ids = np.unique(reference)
    candidate_ids = np.unique(candidate)
    contingency = np.zeros((len(reference_ids), len(candidate_ids)), dtype=np.int64)
    for row, reference_id in enumerate(reference_ids):
        for column, candidate_id in enumerate(candidate_ids):
            contingency[row, column] = int(
                np.sum((reference == reference_id) & (candidate == candidate_id))
            )
    row_indices, column_indices = linear_sum_assignment(-contingency)
    mapping = {
        int(candidate_ids[column]): int(reference_ids[row])
        for row, column in zip(row_indices, column_indices, strict=True)
    }
    unmatched_base = int(reference_ids.max()) + 1 if len(reference_ids) else 0
    aligned = np.array(
        [mapping.get(int(value), unmatched_base + int(value)) for value in candidate],
        dtype=int,
    )
    return aligned, float(np.mean(aligned != reference))
