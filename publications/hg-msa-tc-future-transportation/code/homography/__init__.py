"""Homography calibration and uncertainty analysis for the publication workflow."""

from .calibration import (
    HISTORICAL_CONFIDENCE,
    HISTORICAL_MAX_ITERS,
    HISTORICAL_RANSAC_THRESHOLD_PX,
    apply_homography,
    classify_homography_quality,
    estimate_historical_homography,
    reprojection_statistics,
)

__all__ = [
    "HISTORICAL_CONFIDENCE",
    "HISTORICAL_MAX_ITERS",
    "HISTORICAL_RANSAC_THRESHOLD_PX",
    "apply_homography",
    "classify_homography_quality",
    "estimate_historical_homography",
    "reprojection_statistics",
]
