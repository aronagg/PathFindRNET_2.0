"""Semantic Approach Consolidation (SAC)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .descriptors import (
    NUMERICAL_FLOOR,
    bootstrap_bearing_radii,
    bootstrap_circular_radii,
    circular_distance,
    circular_l1_median,
    circular_mad,
    jensen_shannon_distance,
    wrap_angle,
)
from .provenance import derive_seed


class ProtocolNotIdentifiableError(RuntimeError):
    """Raised when a preregistered diagnostic lacks an executable threshold."""


@dataclass(frozen=True)
class SACVariant:
    variant_id: str
    heading_column: str
    self_consistency_quantile: float = 0.95
    use_bearing: bool = True
    use_heading: bool = True
    require_od_compatibility: bool = False


@dataclass(frozen=True)
class SACResult:
    variant_id: str
    descriptors: pd.DataFrame
    pairwise: pd.DataFrame
    merge_trace: pd.DataFrame
    assignments: pd.DataFrame
    supernodes: pd.DataFrame


def _od_profiles(
    entry_labels: np.ndarray, exit_labels: np.ndarray, role: str
) -> dict[int, np.ndarray]:
    if role == "entry":
        row_labels, column_labels = entry_labels, exit_labels
    else:
        row_labels, column_labels = exit_labels, entry_labels
    columns = sorted(int(value) for value in np.unique(column_labels))
    profiles: dict[int, np.ndarray] = {}
    for region in sorted(int(value) for value in np.unique(row_labels)):
        mask = row_labels == region
        counts = np.asarray([(column_labels[mask] == value).sum() for value in columns])
        profiles[region] = counts.astype(np.float64)
    return profiles


def _region_descriptor(
    scene: str,
    role: str,
    region: int,
    points: np.ndarray,
    headings: np.ndarray,
    role_center: np.ndarray,
    bootstrap_replicates: int,
    quantile: float,
    variant_id: str,
    seed_context: str,
) -> dict[str, Any]:
    n_rows = len(points)
    seed = derive_seed(
        scene, "SAC", role, f"{seed_context}:{variant_id}:{region}"
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, n_rows, size=(int(bootstrap_replicates), n_rows), endpoint=False
    )
    finite_points = np.isfinite(points).all(axis=1)
    finite_headings = np.isfinite(headings)
    centroid = (
        np.median(points[finite_points], axis=0)
        if finite_points.any()
        else np.asarray([np.nan, np.nan])
    )
    bearing = (
        float(np.arctan2(centroid[1] - role_center[1], centroid[0] - role_center[0]))
        if np.isfinite(centroid).all()
        else float("nan")
    )
    point_bearings = np.arctan2(
        points[:, 1] - role_center[1], points[:, 0] - role_center[0]
    )
    heading = circular_l1_median(headings[finite_headings])
    bearing_radii = (
        bootstrap_bearing_radii(
            points, role_center, sample_indices, (float(quantile),)
        )
        if finite_points.all() and n_rows >= 2
        else {float(quantile): float("nan")}
    )
    if finite_headings.all() and n_rows >= 2:
        heading_radii = bootstrap_circular_radii(
            headings, sample_indices, (float(quantile),)
        )
    else:
        valid_values = headings[finite_headings]
        if len(valid_values) >= 2:
            heading_seed = derive_seed(
                scene,
                "SAC",
                role,
                f"{seed_context}:{variant_id}:{region}:valid-heading",
            )
            heading_rng = np.random.default_rng(heading_seed)
            heading_indices = heading_rng.integers(
                0,
                len(valid_values),
                size=(int(bootstrap_replicates), len(valid_values)),
                endpoint=False,
            )
            heading_radii = bootstrap_circular_radii(
                valid_values, heading_indices, (float(quantile),)
            )
        else:
            heading_radii = {float(quantile): float("nan")}
    return {
        "scene": scene,
        "variant_id": variant_id,
        "role": role,
        "micro_region": int(region),
        "n_support": int(n_rows),
        "n_finite_bearings": int(finite_points.sum()),
        "n_finite_headings": int(finite_headings.sum()),
        "centroid_x_topview": float(centroid[0]),
        "centroid_y_topview": float(centroid[1]),
        "role_center_x_topview": float(role_center[0]),
        "role_center_y_topview": float(role_center[1]),
        "bearing": bearing,
        "heading": heading,
        "bearing_circular_mad": circular_mad(point_bearings[finite_points], bearing),
        "heading_circular_mad": circular_mad(headings[finite_headings], heading),
        "bearing_radius": float(bearing_radii[float(quantile)]),
        "heading_radius": float(heading_radii[float(quantile)]),
        "self_consistency_quantile": float(quantile),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(seed),
        "invalid_bearing_descriptor": bool(finite_points.sum() < 2),
        "invalid_heading_descriptor": bool(finite_headings.sum() < 2),
        "zero_support_region": bool(n_rows == 0),
    }


def _pairwise_table(
    descriptors: pd.DataFrame,
    profiles: dict[int, np.ndarray],
    variant: SACVariant,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    records = {int(row.micro_region): row for row in descriptors.itertuples()}
    regions = sorted(records)
    for left_index, left_region in enumerate(regions):
        for right_region in regions[left_index + 1 :]:
            left = records[left_region]
            right = records[right_region]
            bearing_tolerance = max(
                float(left.bearing_radius) + float(right.bearing_radius),
                NUMERICAL_FLOOR,
            )
            heading_tolerance = max(
                float(left.heading_radius) + float(right.heading_radius),
                NUMERICAL_FLOOR,
            )
            bearing_separation = float(circular_distance(left.bearing, right.bearing))
            heading_separation = float(circular_distance(left.heading, right.heading))
            bearing_ratio = bearing_separation / bearing_tolerance
            heading_ratio = heading_separation / heading_tolerance
            invalid = False
            components: list[float] = []
            if variant.use_bearing:
                invalid = invalid or bool(
                    left.invalid_bearing_descriptor
                    or right.invalid_bearing_descriptor
                    or not np.isfinite(bearing_ratio)
                )
                components.append(bearing_ratio)
            if variant.use_heading:
                invalid = invalid or bool(
                    left.invalid_heading_descriptor
                    or right.invalid_heading_descriptor
                    or not np.isfinite(heading_ratio)
                )
                components.append(heading_ratio)
            distance = float("inf") if invalid or not components else float(max(components))
            js_distance = jensen_shannon_distance(
                profiles[left_region], profiles[right_region]
            )
            if variant.require_od_compatibility:
                raise ProtocolNotIdentifiableError(
                    "A8 preregisters JSD compatibility but freezes no JSD threshold."
                )
            rows.append(
                {
                    "scene": left.scene,
                    "variant_id": variant.variant_id,
                    "role": left.role,
                    "left_micro_region": left_region,
                    "right_micro_region": right_region,
                    "bearing_separation": bearing_separation,
                    "heading_separation": heading_separation,
                    "combined_bearing_radius": bearing_tolerance,
                    "combined_heading_radius": heading_tolerance,
                    "bearing_ratio": bearing_ratio,
                    "heading_ratio": heading_ratio,
                    "primary_distance": distance,
                    "compatible": bool(distance <= 1.0),
                    "od_profile_js_distance_diagnostic": js_distance,
                }
            )
    return pd.DataFrame(rows)


def _complete_link_merge(
    scene: str, role: str, variant_id: str, pairwise: pd.DataFrame, regions: list[int]
) -> tuple[list[tuple[int, ...]], pd.DataFrame]:
    distances = {
        tuple(sorted((int(row.left_micro_region), int(row.right_micro_region)))): float(
            row.primary_distance
        )
        for row in pairwise.itertuples()
    }
    clusters: list[tuple[int, ...]] = [(region,) for region in sorted(regions)]
    trace: list[dict[str, Any]] = []
    step = 0
    while len(clusters) > 1:
        candidates = []
        for left_index, left in enumerate(clusters):
            for right in clusters[left_index + 1 :]:
                values = [
                    0.0 if a == b else distances[tuple(sorted((a, b)))]
                    for a in left
                    for b in right
                ]
                complete_distance = float(max(values))
                if complete_distance <= 1.0:
                    candidates.append((complete_distance, left, right))
        if not candidates:
            break
        distance, left, right = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        merged = tuple(sorted((*left, *right)))
        step += 1
        trace.append(
            {
                "scene": scene,
                "variant_id": variant_id,
                "role": role,
                "merge_step": step,
                "left_members": ";".join(map(str, left)),
                "right_members": ";".join(map(str, right)),
                "merged_members": ";".join(map(str, merged)),
                "complete_link_distance": distance,
            }
        )
        clusters = [cluster for cluster in clusters if cluster not in {left, right}]
        clusters.append(merged)
        clusters.sort()
    return clusters, pd.DataFrame(trace)


def _ordered_supernodes(
    role: str,
    clusters: list[tuple[int, ...]],
    descriptors: pd.DataFrame,
) -> list[tuple[int, ...]]:
    lookup = descriptors.set_index("micro_region")

    def key(cluster: tuple[int, ...]) -> tuple[float, float, int]:
        weights = lookup.loc[list(cluster), "n_support"].to_numpy(dtype=np.float64)
        bearings = lookup.loc[list(cluster), "bearing"].to_numpy(dtype=np.float64)
        headings = lookup.loc[list(cluster), "heading"].to_numpy(dtype=np.float64)
        bearing = circular_l1_median(np.repeat(bearings, np.maximum(weights.astype(int), 1)))
        heading = circular_l1_median(np.repeat(headings, np.maximum(weights.astype(int), 1)))
        return (float(wrap_angle(bearing)), float(wrap_angle(heading)), min(cluster))

    return sorted(clusters, key=key)


def run_sac(
    scene: str,
    frame: pd.DataFrame,
    entry_labels: np.ndarray,
    exit_labels: np.ndarray,
    entry_center: np.ndarray,
    exit_center: np.ndarray,
    variant: SACVariant,
    bootstrap_replicates: int = 500,
    seed_context: str = "full",
) -> SACResult:
    """Run deterministic same-role SAC for one EMD result."""
    work = frame.reset_index(drop=True).copy()
    work["entry_region"] = np.asarray(entry_labels, dtype=int)
    work["exit_region"] = np.asarray(exit_labels, dtype=int)
    descriptor_tables: list[pd.DataFrame] = []
    pair_tables: list[pd.DataFrame] = []
    trace_tables: list[pd.DataFrame] = []
    assignment_tables: list[pd.DataFrame] = []
    supernode_rows: list[dict[str, Any]] = []
    for role, point_columns, labels, role_center in (
        ("entry", ["start_x_topview", "start_y_topview"], entry_labels, entry_center),
        ("exit", ["end_x_topview", "end_y_topview"], exit_labels, exit_center),
    ):
        descriptors = []
        for region in sorted(int(value) for value in np.unique(labels)):
            mask = np.asarray(labels) == region
            descriptors.append(
                _region_descriptor(
                    scene,
                    role,
                    region,
                    work.loc[mask, point_columns].to_numpy(dtype=np.float64),
                    work.loc[
                        mask, variant.heading_column.format(role=role)
                    ].to_numpy(dtype=np.float64),
                    role_center,
                    bootstrap_replicates,
                    variant.self_consistency_quantile,
                    variant.variant_id,
                    seed_context,
                )
            )
        descriptor_frame = pd.DataFrame(descriptors)
        profiles = _od_profiles(entry_labels, exit_labels, role)
        pairwise = _pairwise_table(descriptor_frame, profiles, variant)
        clusters, trace = _complete_link_merge(
            scene,
            role,
            variant.variant_id,
            pairwise,
            sorted(int(value) for value in np.unique(labels)),
        )
        ordered = _ordered_supernodes(role, clusters, descriptor_frame)
        mapping: dict[int, str] = {}
        for index, members in enumerate(ordered):
            supernode_id = f"{role}:S{index:02d}"
            for member in members:
                mapping[member] = supernode_id
            selected = descriptor_frame[
                descriptor_frame["micro_region"].isin(members)
            ]
            supernode_rows.append(
                {
                    "scene": scene,
                    "variant_id": variant.variant_id,
                    "role": role,
                    "supernode_id": supernode_id,
                    "member_micro_regions": ";".join(map(str, members)),
                    "n_micro_regions": len(members),
                    "n_support": int(selected["n_support"].sum()),
                    "bearing": circular_l1_median(selected["bearing"].to_numpy()),
                    "heading": circular_l1_median(selected["heading"].to_numpy()),
                    "contains_invalid_descriptor": bool(
                        selected[
                            ["invalid_bearing_descriptor", "invalid_heading_descriptor"]
                        ].any(axis=None)
                    ),
                }
            )
        assignment_tables.append(
            pd.DataFrame(
                {
                    "scene": scene,
                    "variant_id": variant.variant_id,
                    "role": role,
                    "micro_region": sorted(mapping),
                    "supernode_id": [mapping[value] for value in sorted(mapping)],
                }
            )
        )
        descriptor_tables.append(descriptor_frame)
        pair_tables.append(pairwise)
        trace_tables.append(trace)
    return SACResult(
        variant_id=variant.variant_id,
        descriptors=pd.concat(descriptor_tables, ignore_index=True),
        pairwise=pd.concat(pair_tables, ignore_index=True),
        merge_trace=pd.concat(trace_tables, ignore_index=True),
        assignments=pd.concat(assignment_tables, ignore_index=True),
        supernodes=pd.DataFrame(supernode_rows),
    )
