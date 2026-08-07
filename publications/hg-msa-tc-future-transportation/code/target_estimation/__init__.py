"""Frozen homography-guided target-estimation implementation and diagnostics."""

from .hg_target_estimator import (
    CANONICAL_MODULE_VERSION,
    IMPLEMENTATION_VERSION,
    EndpointRegionFit,
    TargetEstimateResult,
    apply_homography,
    endpoint_features,
    estimate_endpoint_region_fit,
    estimate_endpoint_regions,
    estimate_hg_target,
    estimate_hg_target_detailed,
    transform_camera_endpoints,
)

__all__ = [
    "CANONICAL_MODULE_VERSION",
    "IMPLEMENTATION_VERSION",
    "EndpointRegionFit",
    "TargetEstimateResult",
    "apply_homography",
    "endpoint_features",
    "estimate_endpoint_region_fit",
    "estimate_endpoint_regions",
    "estimate_hg_target",
    "estimate_hg_target_detailed",
    "transform_camera_endpoints",
]
