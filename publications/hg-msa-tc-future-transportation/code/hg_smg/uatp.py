"""Uncertainty-Aware Target Prior (UATP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from target_estimation.hg_target_estimator import estimate_hg_target_detailed

from .bootstrap import (
    hierarchical_recording_indices,
    percentile_interval,
    shannon_entropy_bits,
    smaller_tie_mode,
)
from .provenance import derive_seed
from .sac import SACVariant, run_sac
from .smg import run_smg


@dataclass(frozen=True)
class UATPResult:
    runs: pd.DataFrame
    summary: pd.DataFrame


def _endpoint_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "trajectory_id",
            "recording_id",
            "start_x_topview",
            "start_y_topview",
            "end_x_topview",
            "end_y_topview",
        ]
    ].copy()


def run_uatp_scene(
    scene: str,
    frame: pd.DataFrame,
    variants: tuple[SACVariant, ...],
    region_counts: list[int],
    support_thresholds: list[float],
    metric_sample_size: int,
    bootstrap_replicates: int = 500,
    sac_bootstrap_replicates: int = 500,
    n_jobs: int = 1,
) -> UATPResult:
    """Rerun EMD->SAC->SMG for each hierarchical bootstrap replicate."""
    def run_replicate(replicate: int) -> list[dict[str, Any]]:
        replicate_rows: list[dict[str, Any]] = []
        with threadpool_limits(limits=1):
            sample_seed = derive_seed(scene, "UATP", "all", replicate)
            sampled_indices = hierarchical_recording_indices(frame, sample_seed)
            sampled = frame.iloc[sampled_indices].reset_index(drop=True).copy()
            sampled["trajectory_id"] = [
                f"{value}#uatp{replicate}:{index}"
                for index, value in enumerate(sampled["trajectory_id"].astype(str))
            ]
            emd_seed = derive_seed(scene, "UATP-EMD", "all", replicate)
            emd = estimate_hg_target_detailed(
                scene,
                _endpoint_frame(sampled),
                emd_seed,
                region_counts,
                support_thresholds,
                metric_sample_size,
            )
            for variant in variants:
                sac = run_sac(
                    scene,
                    sampled,
                    emd.entry_fit.labels,
                    emd.exit_fit.labels,
                    emd.entry_fit.center,
                    emd.exit_fit.center,
                    variant,
                    bootstrap_replicates=sac_bootstrap_replicates,
                    seed_context=f"uatp:{replicate}",
                )
                smg = run_smg(
                    scene,
                    sampled["trajectory_id"],
                    emd.entry_fit.labels,
                    emd.exit_fit.labels,
                    sac,
                    support_thresholds,
                )
                replicate_rows.append(
                    {
                        "scene": scene,
                        "variant_id": variant.variant_id,
                        "replicate": replicate,
                        "bootstrap_seed": sample_seed,
                        "emd_seed": emd_seed,
                        "sampled_trajectory_count": int(len(sampled)),
                        "sampled_recording_count": int(sampled["recording_id"].nunique()),
                        "emd_entry_regions": int(emd.entry_fit.selected_count),
                        "emd_exit_regions": int(emd.exit_fit.selected_count),
                        "entry_supernodes": int(smg.summary["n_entry_supernodes"]),
                        "exit_supernodes": int(smg.summary["n_exit_supernodes"]),
                        "support_threshold": float(smg.summary["support_threshold"]),
                        "smg_target": int(smg.summary["smg_target"]),
                        "coverage": float(smg.summary["supported_trajectory_coverage"]),
                        "invalid_replicate": False,
                        "failure_reason": "",
                    }
                )
        return replicate_rows

    nested = Parallel(n_jobs=int(n_jobs), backend="loky", batch_size="auto")(
        delayed(run_replicate)(replicate)
        for replicate in range(int(bootstrap_replicates))
    )
    rows = [row for group in nested for row in group]
    runs = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for variant_id, group in runs.groupby("variant_id", sort=False):
        values = group["smg_target"].to_numpy(dtype=int)
        counts = pd.Series(values).value_counts()
        base = {
            "scene": scene,
            "variant_id": variant_id,
            "bootstrap_replicates": int(len(group)),
            "mode": smaller_tie_mode(values),
            "median": float(np.median(values)),
            "entropy_bits": shannon_entropy_bits(values),
            "probability_full_split_target": np.nan,
            "failed_replicates": int(group["invalid_replicate"].sum()),
            "distinct_targets": int(len(counts)),
            "target_histogram_json": counts.sort_index().to_json(),
        }
        for level in (0.80, 0.90, 0.95):
            lower, upper = percentile_interval(values, level)
            suffix = str(int(100 * level))
            base[f"interval_{suffix}_lower"] = lower
            base[f"interval_{suffix}_upper"] = upper
            base[f"interval_{suffix}_width"] = upper - lower
        summary_rows.append(base)
    return UATPResult(runs, pd.DataFrame(summary_rows))
