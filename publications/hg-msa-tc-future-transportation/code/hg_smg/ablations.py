"""Frozen HG-SMG ablation and sensitivity definitions."""

from __future__ import annotations

from .sac import SACVariant


def executable_sac_variants() -> tuple[SACVariant, ...]:
    """Return only variants fully identified by protocol v1."""
    return (
        SACVariant("A5", "heading_camera_w5_{role}"),
        SACVariant("A6", "heading_camera_w5_{role}", use_heading=False),
        SACVariant("A7", "heading_camera_w5_{role}", use_bearing=False),
        SACVariant("A9", "heading_topview_w5_{role}"),
        SACVariant("sensitivity_heading_w3", "heading_camera_w3_{role}"),
        SACVariant("sensitivity_heading_w7", "heading_camera_w7_{role}"),
        SACVariant(
            "sensitivity_self_consistency_q090",
            "heading_camera_w5_{role}",
            self_consistency_quantile=0.90,
        ),
        SACVariant(
            "sensitivity_self_consistency_q0975",
            "heading_camera_w5_{role}",
            self_consistency_quantile=0.975,
        ),
    )


def a8_diagnostic_variant() -> SACVariant:
    return SACVariant(
        "A8",
        "heading_camera_w5_{role}",
        require_od_compatibility=True,
    )


def primary_variant() -> SACVariant:
    return executable_sac_variants()[0]
