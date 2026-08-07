"""Semantic Maneuver Graph construction and support-threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from target_estimation.hg_target_estimator import (
    build_threshold_candidates,
    select_threshold_candidate,
)

from .sac import SACResult


@dataclass(frozen=True)
class SMGResult:
    summary: dict[str, Any]
    edges: pd.DataFrame
    threshold_candidates: pd.DataFrame
    trajectory_assignments: pd.DataFrame


def _region_maps(sac: SACResult) -> tuple[dict[int, str], dict[int, str]]:
    entry = sac.assignments[sac.assignments["role"] == "entry"]
    exit_rows = sac.assignments[sac.assignments["role"] == "exit"]
    return (
        dict(zip(entry["micro_region"].astype(int), entry["supernode_id"], strict=True)),
        dict(
            zip(
                exit_rows["micro_region"].astype(int),
                exit_rows["supernode_id"],
                strict=True,
            )
        ),
    )


def run_smg(
    scene: str,
    trajectory_ids: pd.Series,
    entry_labels: np.ndarray,
    exit_labels: np.ndarray,
    sac: SACResult,
    support_thresholds: list[float],
) -> SMGResult:
    """Aggregate micro-OD counts over SAC supernodes and select theta."""
    entry_map, exit_map = _region_maps(sac)
    assignments = pd.DataFrame(
        {
            "trajectory_id": trajectory_ids.astype(str).to_numpy(),
            "entry_region": np.asarray(entry_labels, dtype=int),
            "exit_region": np.asarray(exit_labels, dtype=int),
        }
    )
    assignments["entry_supernode"] = assignments["entry_region"].map(entry_map)
    assignments["exit_supernode"] = assignments["exit_region"].map(exit_map)
    valid = assignments[
        assignments["entry_supernode"].notna()
        & assignments["exit_supernode"].notna()
    ].copy()
    if valid.empty:
        raise RuntimeError(f"{scene}: SMG has zero valid trajectories")
    valid["od_pair"] = valid["entry_supernode"] + ">" + valid["exit_supernode"]
    counts = (
        valid.groupby(
            ["entry_supernode", "exit_supernode", "od_pair"],
            sort=True,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )
    denominator = int(len(valid))
    counts["share"] = counts["count"] / denominator
    threshold_candidates = build_threshold_candidates(
        scene,
        counts[["od_pair", "count", "share"]],
        denominator,
        sac.supernodes.loc[sac.supernodes["role"] == "entry", "supernode_id"].nunique(),
        sac.supernodes.loc[sac.supernodes["role"] == "exit", "supernode_id"].nunique(),
        support_thresholds,
    ).rename(columns={"hg_estimated_target": "smg_target"})
    selector = threshold_candidates.rename(columns={"smg_target": "hg_estimated_target"})
    selected = select_threshold_candidate(selector)
    threshold = float(selected["support_threshold"])
    threshold_candidates["selected"] = np.isclose(
        threshold_candidates["support_threshold"], threshold, atol=0.0, rtol=0.0
    )
    counts.insert(0, "scene", scene)
    counts.insert(1, "variant_id", sac.variant_id)
    counts["supported_at_selected_threshold"] = counts["share"] >= threshold
    assignments["valid_supernode_assignment"] = (
        assignments["entry_supernode"].notna()
        & assignments["exit_supernode"].notna()
    )
    summary = {
        "scene": scene,
        "variant_id": sac.variant_id,
        "n_entry_supernodes": int(
            sac.supernodes.loc[sac.supernodes["role"] == "entry", "supernode_id"].nunique()
        ),
        "n_exit_supernodes": int(
            sac.supernodes.loc[sac.supernodes["role"] == "exit", "supernode_id"].nunique()
        ),
        "support_threshold": threshold,
        "support_threshold_percent": 100.0 * threshold,
        "smg_target": int(selected["hg_estimated_target"]),
        "supported_trajectory_coverage": float(selected["od_coverage"]),
        "valid_assignment_denominator": denominator,
        "missing_geometry_count": int(len(assignments) - denominator),
        "supported_edges": int((counts["share"] >= threshold).sum()),
        "unsupported_edges": int((counts["share"] < threshold).sum()),
        "zero_supported_edges": bool((counts["share"] >= threshold).sum() == 0),
    }
    threshold_candidates.insert(1, "variant_id", sac.variant_id)
    return SMGResult(summary, counts, threshold_candidates, assignments)


def sac_mapped_supported_micro_target(
    micro_od_counts: pd.DataFrame,
    original_threshold: float,
    sac: SACResult,
) -> int:
    """A2 target: map originally supported micro-OD pairs without reaggregation."""
    entry_map, exit_map = _region_maps(sac)
    supported = micro_od_counts[micro_od_counts["share"] >= float(original_threshold)].copy()
    parsed = supported["od_pair"].str.split("->", expand=True).astype(int)
    supported["entry_supernode"] = parsed[0].map(entry_map)
    supported["exit_supernode"] = parsed[1].map(exit_map)
    return int(
        supported[["entry_supernode", "exit_supernode"]]
        .dropna()
        .drop_duplicates()
        .shape[0]
    )
