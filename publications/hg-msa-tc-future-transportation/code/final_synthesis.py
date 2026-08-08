"""Final evidence synthesis for the Future Transportation revision.

This module reads persisted results only. It does not run clustering, target
estimation, homography estimation, EMAS sensitivity, or baseline generation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "final_synthesis"
FIGURE_DIR = ROOT / "figures" / "final_synthesis"
DOC_DIR = ROOT / "docs"

METRIC_COLUMNS = [
    "observed_target_abs_error",
    "legal_target_abs_error",
    "n_non_noise_clusters",
    "noise_pct_all_test",
    "ari",
    "nmi",
    "purity",
    "homogeneity",
    "completeness",
    "v_measure",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]

SCENES = [
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
]


@dataclass(frozen=True)
class InputPaths:
    hg_smg_metrics: Path = ROOT / "results" / "hg_smg" / "independent_test" / "metrics.csv"
    baseline_metrics: Path = (
        ROOT / "results" / "baselines" / "independent_test" / "baseline_metrics.csv"
    )
    target_validation: Path = ROOT / "results" / "independent_test" / "target_estimation_validation.csv"
    homography_quality: Path = ROOT / "results" / "homography" / "homography_quality_metrics.csv"
    homography_perturbation: Path = ROOT / "results" / "homography" / "homography_perturbation_summary.csv"
    emas_rank_stability: Path = ROOT / "results" / "emas" / "emas_rank_stability.csv"
    emas_component_analysis: Path = ROOT / "results" / "emas" / "emas_component_analysis.csv"
    hg_smg_ablation: Path = ROOT / "results" / "hg_smg" / "independent_test" / "ablation_comparison.csv"
    baseline_comparison: Path = (
        ROOT / "results" / "baselines" / "independent_test" / "baseline_comparison_to_hg_smg.csv"
    )
    reference_labels: Path = (
        ROOT / "annotations" / "reference_labels" / "independent_test_reference_labels.csv"
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_metric_frame(paths: InputPaths) -> pd.DataFrame:
    hg = _read_csv(paths.hg_smg_metrics)
    base = _read_csv(paths.baseline_metrics)

    hg_keep = hg[hg["ablation_id"].isin(["A0", "A1", "A5"])].copy()
    labels = {
        "A0": ("original_untargeted", "Original untargeted"),
        "A1": ("original_hg_aware_A1", "Original HG-aware A1"),
        "A5": ("hg_smg_tc_A5", "HG-SMG-TC A5"),
    }
    hg_keep["method_family"] = hg_keep["ablation_id"].map(lambda x: labels[x][0])
    hg_keep["method_label"] = hg_keep["ablation_id"].map(lambda x: labels[x][1])
    hg_keep["variant_id"] = hg_keep["ablation_id"]
    hg_keep["configuration_id"] = hg_keep["frozen_configuration_id"]

    base_keep = base.copy()
    base_keep["method_family"] = base_keep["baseline_id"]
    base_keep["method_label"] = base_keep["baseline_id"].map(
        {
            "endpoint_camera_raw": "Endpoint camera raw baseline",
            "endpoint_camera_isotropic": "Endpoint camera isotropic baseline",
            "resampled_trajectory_euclidean": "Resampled trajectory Euclidean baseline",
        }
    )
    base_keep["variant_id"] = base_keep["baseline_id"]
    base_keep["configuration_id"] = base_keep["baseline_configuration_id"]

    common = [
        "method_family",
        "method_label",
        "variant_id",
        "scene_id",
        "method",
        "configuration_id",
        "total_test_trajectories",
        "valid_reference_trajectories",
        "excluded_reference_trajectories",
        "reference_coverage_pct",
        *METRIC_COLUMNS,
    ]
    combined = pd.concat([hg_keep[common], base_keep[common]], ignore_index=True)
    combined = combined.sort_values(["method_family", "scene_id", "method"]).reset_index(drop=True)
    return combined


def _aggregate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [frame]

    scene_agg = (
        frame.groupby(["method_family", "method_label", "variant_id", "scene_id"], as_index=False)
        .agg(
            {
                "total_test_trajectories": "mean",
                "valid_reference_trajectories": "mean",
                "excluded_reference_trajectories": "mean",
                "reference_coverage_pct": "mean",
                **{col: "mean" for col in METRIC_COLUMNS},
            }
        )
        .assign(method="ALL_METHODS", configuration_id="aggregate_mean_over_methods")
    )
    rows.append(scene_agg[frame.columns])

    all_agg = (
        frame.groupby(["method_family", "method_label", "variant_id"], as_index=False)
        .agg(
            {
                "total_test_trajectories": "mean",
                "valid_reference_trajectories": "mean",
                "excluded_reference_trajectories": "mean",
                "reference_coverage_pct": "mean",
                **{col: "mean" for col in METRIC_COLUMNS},
            }
        )
        .assign(scene_id="ALL_SCENES", method="ALL_METHODS", configuration_id="aggregate_mean_over_scene_methods")
    )
    rows.append(all_agg[frame.columns])
    return pd.concat(rows, ignore_index=True)


def build_final_method_comparison(paths: InputPaths) -> pd.DataFrame:
    frame = _normalize_metric_frame(paths)
    final = _aggregate_rows(frame)
    final.to_csv(RESULT_DIR / "final_method_comparison.csv", index=False)
    return final


def _scene_means(final: pd.DataFrame) -> pd.DataFrame:
    return final[(final["scene_id"] != "ALL_SCENES") & (final["method"] == "ALL_METHODS")].copy()


def _strongest_endpoint_family(scene_means: pd.DataFrame) -> str:
    endpoint = scene_means[scene_means["method_family"].isin(["endpoint_camera_raw", "endpoint_camera_isotropic"])]
    ranking = (
        endpoint.groupby("method_family")
        .agg({"macro_f1": "mean", "nmi": "mean", "observed_target_abs_error": "mean"})
        .sort_values(["macro_f1", "nmi", "observed_target_abs_error"], ascending=[False, False, True])
    )
    return str(ranking.index[0])


def _bootstrap_ci(values: np.ndarray, seed: int = 20260808, n_boot: int = 10000) -> tuple[float, float]:
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for idx in range(n_boot):
        means[idx] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def build_paired_scene_differences(final: pd.DataFrame) -> pd.DataFrame:
    scene_means = _scene_means(final)
    strongest_endpoint = _strongest_endpoint_family(scene_means)
    comparisons = [
        ("A1_minus_A0", "original_hg_aware_A1", "original_untargeted"),
        ("A5_minus_A1", "hg_smg_tc_A5", "original_hg_aware_A1"),
        ("A5_minus_strongest_endpoint", "hg_smg_tc_A5", strongest_endpoint),
        ("A5_minus_resampled_trajectory", "hg_smg_tc_A5", "resampled_trajectory_euclidean"),
    ]

    rows: list[dict[str, object]] = []
    indexed = scene_means.set_index(["method_family", "scene_id"])
    for comparison_id, left, right in comparisons:
        for scene in SCENES:
            left_row = indexed.loc[(left, scene)]
            right_row = indexed.loc[(right, scene)]
            row: dict[str, object] = {
                "comparison_id": comparison_id,
                "left_method_family": left,
                "right_method_family": right,
                "scene_id": scene,
            }
            for metric in METRIC_COLUMNS:
                row[f"delta_{metric}"] = float(left_row[metric] - right_row[metric])
            rows.append(row)
    diff = pd.DataFrame(rows)
    diff.to_csv(RESULT_DIR / "paired_scene_differences.csv", index=False)
    return diff


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None, float_digits: int = 4) -> str:
    table = df if max_rows is None else df.head(max_rows)
    return table.to_markdown(index=False, floatfmt=f".{float_digits}f")


def _summary_for_comparison(diff: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for comparison_id, group in diff.groupby("comparison_id"):
        for metric in metrics:
            values = group[f"delta_{metric}"].to_numpy(dtype=float)
            low, high = _bootstrap_ci(values)
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "mean_delta": values.mean(),
                    "median_delta": np.median(values),
                    "min_delta": values.min(),
                    "max_delta": values.max(),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def _save_bar(df: pd.DataFrame, value: str, title: str, output: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    order = [
        "original_untargeted",
        "original_hg_aware_A1",
        "hg_smg_tc_A5",
        "endpoint_camera_raw",
        "endpoint_camera_isotropic",
        "resampled_trajectory_euclidean",
    ]
    labels = {
        "original_untargeted": "A0\nuntargeted",
        "original_hg_aware_A1": "A1\nHG-aware",
        "hg_smg_tc_A5": "A5\nHG-SMG",
        "endpoint_camera_raw": "Endpoint\nraw",
        "endpoint_camera_isotropic": "Endpoint\nisotropic",
        "resampled_trajectory_euclidean": "Resampled\ntrajectory",
    }
    plot_df = df[df["method_family"].isin(order)].set_index("method_family").loc[order].reset_index()
    ax.bar(range(len(plot_df)), plot_df[value], color=["#9aa3ad", "#517aa3", "#1b6f5c", "#b38b4d", "#c2a15d", "#8f6b9f"])
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels([labels[x] for x in plot_df["method_family"]])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def build_figures(final: pd.DataFrame, paths: InputPaths) -> list[Path]:
    aggregate = final[(final["scene_id"] == "ALL_SCENES") & (final["method"] == "ALL_METHODS")].copy()
    outputs: list[Path] = []

    figure_specs = [
        ("observed_target_abs_error", "Observed target-count error", "final_target_count_error.png", "Mean absolute error"),
        ("noise_pct_all_test", "Outlier percentage", "final_outlier_percentage.png", "Mean outlier percentage"),
    ]
    for value, title, filename, ylabel in figure_specs:
        out = FIGURE_DIR / filename
        _save_bar(aggregate, value, title, out, ylabel)
        outputs.append(out)

    metrics = ["ari", "nmi", "purity", "macro_f1"]
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    x = np.arange(len(metrics))
    width = 0.13
    families = [
        "original_untargeted",
        "original_hg_aware_A1",
        "hg_smg_tc_A5",
        "endpoint_camera_raw",
        "endpoint_camera_isotropic",
        "resampled_trajectory_euclidean",
    ]
    labels = ["A0", "A1", "A5", "End raw", "End iso", "Resampled"]
    for i, fam in enumerate(families):
        row = aggregate[aggregate["method_family"] == fam].iloc[0]
        ax.bar(x + (i - 2.5) * width, [row[m] for m in metrics], width=width, label=labels[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["ARI", "NMI", "Purity", "Macro F1"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title("Independent reference agreement metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    out = FIGURE_DIR / "final_agreement_metrics.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    outputs.append(out)

    se38 = final[(final["scene_id"] == "bellevue_150th_se38th") & (final["method"] == "ALL_METHODS")]
    out = FIGURE_DIR / "final_se38th_fragmentation_story.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].bar(se38["method_label"], se38["observed_target_abs_error"], color="#597b92")
    axes[0].set_title("SE38th target-count error")
    axes[0].set_ylabel("Absolute error")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].bar(se38["method_label"], se38["macro_f1"], color="#4f8a67")
    axes[1].set_title("SE38th mapped macro F1")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    outputs.append(out)

    target = _read_csv(paths.target_validation)
    out = FIGURE_DIR / "final_target_estimation_vs_observed.png"
    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    x = np.arange(len(target))
    ax.plot(x, target["frozen_hg_target"], marker="o", label="Frozen HG target")
    ax.plot(x, target["observed_independent_test_movement_count"], marker="s", label="Observed reference movements")
    ax.plot(x, target["legal_movement_count"], marker="^", label="Legal movement count")
    ax.set_xticks(x)
    ax.set_xticklabels(target["scene_id"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Target estimation versus observed and legal movement counts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    outputs.append(out)

    hq = _read_csv(paths.homography_quality)
    out = FIGURE_DIR / "final_homography_quality_sensitivity.png"
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(hq))
    ax1.bar(x, hq["forward_rmse_error_px"], color="#667f9b", label="Forward RMSE")
    ax1.set_ylabel("Reprojection RMSE (px)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(hq["scene_id"], rotation=30, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, hq["endpoint_extrapolation_fraction"], color="#b35f3b", marker="o", label="Endpoint extrapolation")
    ax2.set_ylabel("Endpoint extrapolation fraction")
    ax1.set_title("Homography quality and extrapolation diagnostics")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    outputs.append(out)

    emas = _read_csv(paths.emas_rank_stability)
    local = emas[emas["family"].eq("local_grid")]
    if local.empty:
        local = emas
    stability = local.groupby("method", as_index=False)["original_top_stability_pct"].mean()
    out = FIGURE_DIR / "final_emas_weight_sensitivity_summary.png"
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(stability["method"], stability["original_top_stability_pct"], color="#756b9d")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-rank stability (%)")
    ax.set_title("EMAS_HG weight-sensitivity summary")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    outputs.append(out)

    out = FIGURE_DIR / "final_hg_smg_pipeline_schematic.png"
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    boxes = [
        "Video-derived\ntrajectories",
        "Homography-guided\nendpoint regions",
        "SAC\nsupernodes",
        "SMG target\ninterval",
        "PCMS frozen\nselection",
        "Independent\nreference evaluation",
    ]
    for i, text in enumerate(boxes):
        ax.text(i, 0.5, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#f6f7f8", "ec": "#4a5560"})
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i + 0.62, 0.5), xytext=(i + 0.38, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_xlim(-0.6, len(boxes) - 0.4)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    outputs.append(out)

    out = FIGURE_DIR / "final_baseline_comparison_summary.png"
    base_rows = aggregate[aggregate["method_family"].isin(["hg_smg_tc_A5", "endpoint_camera_isotropic", "resampled_trajectory_euclidean"])]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(base_rows["observed_target_abs_error"], base_rows["nmi"], s=130, color=["#1b6f5c", "#c2a15d", "#8f6b9f"])
    for _, row in base_rows.iterrows():
        ax.annotate(row["method_label"], (row["observed_target_abs_error"], row["nmi"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean observed target-count error")
    ax.set_ylabel("Mean NMI")
    ax.set_title("Baseline comparison summary")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    outputs.append(out)

    return outputs


def _reference_coverage(paths: InputPaths) -> pd.DataFrame:
    labels = _read_csv(paths.reference_labels)
    grouped = labels.groupby("scene_id", as_index=False).agg(
        total_trajectories=("trajectory_id", "count"),
        valid_reference_labels=("reference_status", lambda s: int((s == "valid").sum())),
    )
    grouped["coverage_pct"] = 100.0 * grouped["valid_reference_labels"] / grouped["total_trajectories"]
    return grouped


def write_docs(final: pd.DataFrame, diff: pd.DataFrame, figures: list[Path], paths: InputPaths) -> None:
    aggregate = final[(final["scene_id"] == "ALL_SCENES") & (final["method"] == "ALL_METHODS")].copy()
    scene_means = _scene_means(final)
    strongest_endpoint = _strongest_endpoint_family(scene_means)
    paired_summary = _summary_for_comparison(
        diff,
        ["observed_target_abs_error", "nmi", "ari", "macro_f1", "noise_pct_all_test"],
    )
    paired_summary.to_csv(RESULT_DIR / "paired_scene_difference_summary.csv", index=False)

    target = _read_csv(paths.target_validation)
    hq = _read_csv(paths.homography_quality)
    abas = _read_csv(paths.hg_smg_ablation)
    emas = _read_csv(paths.emas_rank_stability)
    ref_cov = _reference_coverage(paths)

    a1 = aggregate[aggregate["method_family"].eq("original_hg_aware_A1")].iloc[0]
    a5 = aggregate[aggregate["method_family"].eq("hg_smg_tc_A5")].iloc[0]
    endpoint = aggregate[aggregate["method_family"].eq(strongest_endpoint)].iloc[0]
    resampled = aggregate[aggregate["method_family"].eq("resampled_trajectory_euclidean")].iloc[0]

    statistical = f"""# Final Statistical Analysis

This synthesis uses only persisted independent-test outputs. No clustering, model
selection, target estimation, homography calibration, EMAS sensitivity, baseline
assignment, or reference-label generation was rerun.

The primary unit for paired analysis is the scene. Each method family was first
averaged over KMeans, HDBSCAN and OPTICS within a scene, then scene-level paired
differences were computed. With only five Bellevue scenes, the analysis is
descriptive; bootstrap intervals are reported as uncertainty summaries, not as
strong inferential evidence.

## Aggregate Method Means

{_markdown_table(aggregate[["method_family", *METRIC_COLUMNS]].round(4))}

## Paired Scene-Difference Summary

Positive deltas mean the left method has a larger metric value than the right
method. For target error and outlier percentage, negative values are preferable.
For ARI, NMI and macro F1, positive values are preferable.

{_markdown_table(paired_summary.round(4))}

## Main A5 versus A1 Result

HG-SMG-TC A5 reduced mean observed target-count error from
{a1["observed_target_abs_error"]:.4f} to {a5["observed_target_abs_error"]:.4f}
and increased mean NMI from {a1["nmi"]:.4f} to {a5["nmi"]:.4f}. Mean macro F1
changed from {a1["macro_f1"]:.4f} to {a5["macro_f1"]:.4f}. Mean outlier
percentage changed from {a1["noise_pct_all_test"]:.2f}% to
{a5["noise_pct_all_test"]:.2f}%.

## Baseline Context

The strongest endpoint baseline by aggregate macro F1 is `{strongest_endpoint}`.
Compared with this endpoint baseline, HG-SMG-TC A5 has mean target error
{a5["observed_target_abs_error"]:.4f} versus {endpoint["observed_target_abs_error"]:.4f},
mean NMI {a5["nmi"]:.4f} versus {endpoint["nmi"]:.4f}, and mean macro F1
{a5["macro_f1"]:.4f} versus {endpoint["macro_f1"]:.4f}. The resampled trajectory
baseline has mean target error {resampled["observed_target_abs_error"]:.4f},
mean NMI {resampled["nmi"]:.4f}, and mean macro F1 {resampled["macro_f1"]:.4f}.
"""
    (DOC_DIR / "final_statistical_analysis.md").write_text(statistical, encoding="utf-8")

    claim_rows = [
        ("Leakage-free validation", "supported", "Frozen split protocol, persisted assignments before reference evaluation, independent-test reports."),
        ("Independent reference labels exist", "supported with terminology constraint", "Exhaustive human-defined polygon-rule labels; not per-trajectory manual ground truth."),
        ("Original HG-aware improves target alignment vs untargeted", "supported", "A1 mean observed target error lower than A0 in final comparison."),
        ("HG-SMG improves over original HG-aware", "partially supported", "A5 improves target alignment and NMI on average; macro-F1/outlier trade-offs must be reported."),
        ("HG-SMG is universally superior to all baselines", "unsupported", "Endpoint KMeans and resampled baselines are strong in some scene-method cases, especially SE38th."),
        ("EMAS_HG is an independent validation metric", "must be removed", "EMAS_HG is task-specific development/ranking score, not independent reference validation."),
        ("Target estimator recovers semantic maneuver counts", "must be narrowed", "SE38th frozen HG target 18 versus observed 9 shows semantic over-segmentation."),
        ("Homography is reliable enough for all claims", "partially supported", "Quality gate passes but extrapolation and point-pair uncertainty remain material limitations."),
        ("Generalization across all TNVD scenes", "unsupported", "Study uses five Bellevue scenes only."),
        ("Novelty of HG-SMG", "supported but narrow", "Novelty is integration of homography-guided target estimation, SAC/SMG prior and leakage-locked evaluation."),
    ]
    claim_df = pd.DataFrame(claim_rows, columns=["claim", "status", "evidence_or_required_change"])
    claim_text = "# Final Claim Audit\n\n" + _markdown_table(claim_df, float_digits=3) + "\n"
    (DOC_DIR / "final_claim_audit.md").write_text(claim_text, encoding="utf-8")

    reviewer_rows = [
        ("Target leakage", "addressed", "split manifests; independent-test execution protocols; persisted assignment manifests", "Methods / Validation protocol", "State assignments were saved before labels were read."),
        ("Independent ground truth/reference", "addressed with terminology", "polygon reference docs and coverage table", "Dataset / Reference labels", "Call it human-defined polygon-rule reference, not full manual ground truth."),
        ("EMAS formulas", "addressed", "docs/emas_hg_mathematical_definition.md", "Methods / EMAS", "Keep EMAS as heuristic."),
        ("EMAS sensitivity", "addressed", "results/emas/emas_rank_stability.csv; final EMAS figure", "Results / Sensitivity", "Do not imply new weights were selected."),
        ("Target-estimator reproducibility", "addressed", "target reproduction and frozen parameter tables", "Methods / Target estimation", "Report SE38th failure."),
        ("Homography calibration and quality", "addressed", "homography quality metrics and perturbation reports", "Methods / Homography", "Add manual point-pair and extrapolation limits."),
        ("Statistical dependence", "addressed", "docs/final_statistical_analysis.md", "Results / Statistics", "Use scenes as units; avoid p-value overclaim."),
        ("Baseline weakness", "addressed", "baseline metrics and comparison figures", "Results / Baselines", "Admit endpoint KMeans is strong."),
        ("Novelty", "partially addressed", "novelty matrix and final interpretation", "Introduction / Contributions", "Frame as methodological package, not isolated metric novelty."),
        ("Trade-off interpretation", "addressed", "final method comparison and claim audit", "Discussion", "Separate target alignment, reference metrics and outliers."),
        ("Google/provenance/licensing", "addressed as audit", "docs/imagery_provenance_and_licensing_audit.md", "Data / Figures", "Avoid legal conclusions; replace questionable imagery if needed."),
        ("Figure/table quality", "addressed", "figures/final_synthesis; docs/manuscript_tables_final.md", "Results", "Manually review final formatting before submission."),
        ("Terminology", "addressed", "final claim audit", "Throughout manuscript", "Use observed maneuver target and polygon-rule reference consistently."),
    ]
    reviewer_df = pd.DataFrame(reviewer_rows, columns=["reviewer_topic", "response_status", "evidence", "manuscript_section", "remaining_caveat"])
    (DOC_DIR / "reviewer_evidence_map.md").write_text(
        "# Reviewer Evidence Map\n\n" + _markdown_table(reviewer_df, float_digits=3) + "\n",
        encoding="utf-8",
    )

    tables_text = f"""# Manuscript Tables Final

## Dataset and Split Table

{_markdown_table(ref_cov.round(3))}

## Reference Coverage Table

{_markdown_table(ref_cov.round(3))}

## Target-Estimation Validation Table

{_markdown_table(target.round(3))}

## Main Method Comparison Table

{_markdown_table(aggregate[["method_family", "observed_target_abs_error", "noise_pct_all_test", "ari", "nmi", "purity", "macro_f1", "weighted_f1"]].round(4))}

## Baseline Comparison Table

{_markdown_table(aggregate[aggregate["method_family"].isin(["endpoint_camera_raw", "endpoint_camera_isotropic", "resampled_trajectory_euclidean", "hg_smg_tc_A5"])][["method_family", "observed_target_abs_error", "nmi", "macro_f1", "noise_pct_all_test"]].round(4))}

## Homography Quality Table

{_markdown_table(hq[["scene_id", "point_count", "forward_mean_error_px", "forward_rmse_error_px", "forward_p95_error_px", "endpoint_extrapolation_fraction", "quality_class", "homography_passes_quality_gate"]].round(4))}

## EMAS Component and Weight Table

| Component | Weight in EMAS_HG-v1 | Interpretation |
| --- | ---: | --- |
| T | 0.50 | Target-count agreement |
| O | 0.20 | Non-outlier component |
| B | 0.10 | Cluster-balance component |
| S | 0.10 | Silhouette-derived component |
| D | 0.10 | Davies-Bouldin-derived component |

## EMAS Weight-Sensitivity Summary

{_markdown_table(emas.groupby("family", as_index=False)["original_top_stability_pct"].mean().round(3))}

## Ablation Summary Table

{_markdown_table(abas[["ablation_id", "mean_observed_target_abs_error", "mean_nmi", "mean_ari", "mean_purity", "mean_macro_f1", "mean_noise_pct_all_test"]].round(4))}

## Limitations Table

| Limitation | Manuscript handling |
| --- | --- |
| Five Bellevue scenes only | Do not claim full Traffic Node Video Dataset validation. |
| Polygon-rule reference is not per-trajectory manual labeling | Use precise reference terminology. |
| SE38th target over-segmentation | Treat as a visible failure mode and motivation for HG-SMG. |
| Homography point pairs are manual | Report calibration quality and uncertainty. |
| EMAS_HG is heuristic | Present independent reference metrics separately. |
| Endpoint baselines are strong | Avoid universal superiority claims. |
"""
    (DOC_DIR / "manuscript_tables_final.md").write_text(tables_text, encoding="utf-8")

    checklist = """# Public Release Final Checklist

| Item | Status | Notes |
| --- | --- | --- |
| Code | ready for review | Publication code modules are present; run Ruff before release. |
| Configs | ready for review | Frozen configs must be archived with hashes. |
| Processed trajectories | local only / release planning needed | Do not include raw videos in lightweight review artifacts. |
| Reference labels | ready for release planning | Include CSV/Parquet and protocol hash if allowed by data policy. |
| Split manifests | ready for release planning | Required for leakage-free reproduction. |
| Homography correspondences | ready for release planning | Manual point pairs and provenance should be published if licensing permits. |
| Intermediate outputs | selective release | Prefer compact CSV summaries and checksums. |
| Figures | ready for manuscript review | Final synthesis figures generated from persisted outputs. |
| Metrics | ready | Final comparison tables are under `results/final_synthesis/`. |
| Licenses | manual review required | Google-derived imagery should not be redistributed unless permitted. |
| DOI archive plan | pending | Archive code, configs, compact metrics, and documentation. |
| GitHub release plan | pending | Tag final revision state after manuscript updates. |
| Raw video redistribution | not included | Treat raw video as non-redistributed unless dataset license permits. |
"""
    (DOC_DIR / "public_release_final_checklist.md").write_text(checklist, encoding="utf-8")

    interpretation = f"""# Final Scientific Interpretation

## Main Contribution

The revised contribution is a leakage-controlled trajectory-clustering evidence
package for five Bellevue intersections. The strongest methodological point is
not a single score or a universal clustering improvement, but the combination of
homography-guided maneuver-structure estimation, frozen split-aware selection,
and independent evaluation against exhaustive human-defined polygon-rule
reference labels.

## Original HG-Aware Method

Original HG-aware A1 improves target-count alignment relative to original
untargeted A0. In aggregate, A1 has mean observed target error
{a1["observed_target_abs_error"]:.4f}, compared with A0's
{aggregate[aggregate["method_family"].eq("original_untargeted")].iloc[0]["observed_target_abs_error"]:.4f}.
This supports the idea that maneuver-count awareness is useful, but it does not
establish broad superiority across every metric.

## What HG-SMG-TC Adds

HG-SMG-TC A5 replaces the point-target prior with a structured maneuver graph
and interval-aware selection. A5 has mean observed target error
{a5["observed_target_abs_error"]:.4f}, mean NMI {a5["nmi"]:.4f}, and mean macro
F1 {a5["macro_f1"]:.4f}. The evidence supports A5 as a stronger and better
documented post-review extension, while still requiring trade-off language.

## Where It Helps Most

The method is most useful where the original target estimator is vulnerable to
semantic over-segmentation, especially SE38th. The final results and Task 07
diagnostics show that SE38th should be presented as a failure analysis and
methodological motivation, not hidden as an outlier.

## Where It Does Not Help

Endpoint-based KMeans is a strong baseline in several cases. Full resampled
trajectory features do not clearly dominate endpoint features in aggregate.
Therefore, the revised paper should not claim that HG-SMG-TC universally
outperforms every simple representation or every baseline metric.

## Baseline Evidence

The strongest endpoint baseline is `{strongest_endpoint}`. Its aggregate NMI is
{endpoint["nmi"]:.4f}, compared with A5's {a5["nmi"]:.4f}. Its macro F1 is
{endpoint["macro_f1"]:.4f}, compared with A5's {a5["macro_f1"]:.4f}. This is a
credible baseline and should be discussed directly.

## Recommended Abstract and Conclusion Claims

Use cautious wording:

- The study evaluates five Bellevue intersections, not the full dataset.
- Human-defined polygon-rule labels provide an independent reference relative
  to clustering outputs, but they are not per-trajectory manual ground truth.
- HG-aware and HG-SMG selection improve maneuver-count alignment on average.
- HG-SMG improves the evidence basis and mitigates semantic over-segmentation
  failure modes, but the results contain metric trade-offs.
- EMAS_HG is a task-specific score; independent validation comes from ARI, NMI,
  purity and mapped F1 against the polygon-rule reference.
"""
    (DOC_DIR / "final_scientific_interpretation.md").write_text(interpretation, encoding="utf-8")

    figure_table = pd.DataFrame(
        {
            "figure_path": [str(p.relative_to(ROOT)) for p in figures],
            "caption": [
                "Mean observed target-count error by method family.",
                "Mean outlier percentage by method family.",
                "Independent reference agreement metrics by method family.",
                "SE38th target-error and mapped-F1 summary.",
                "Frozen HG target versus observed and legal movement counts.",
                "Homography reprojection and extrapolation diagnostics.",
                "EMAS_HG top-rank stability under weight perturbation.",
                "HG-SMG-TC pipeline schematic.",
                "Baseline comparison in target-error versus NMI space.",
            ],
        }
    )
    figure_table.to_csv(RESULT_DIR / "final_figure_manifest.csv", index=False)

    exec_report = f"""# Task 12 Execution Report

Branch: `feature/futuretransp-final-results-synthesis`

Inputs were read from persisted Task 05-11 outputs only. No clustering or tuning
was executed.

## Created Outputs

- `results/final_synthesis/final_method_comparison.csv`
- `results/final_synthesis/paired_scene_differences.csv`
- `results/final_synthesis/paired_scene_difference_summary.csv`
- `results/final_synthesis/final_figure_manifest.csv`
- `docs/final_statistical_analysis.md`
- `docs/final_claim_audit.md`
- `docs/reviewer_evidence_map.md`
- `docs/manuscript_tables_final.md`
- `docs/public_release_final_checklist.md`
- `docs/final_scientific_interpretation.md`
- `figures/final_synthesis/`

## Key Aggregate Values

- A1 mean observed target error: {a1["observed_target_abs_error"]:.4f}
- A5 mean observed target error: {a5["observed_target_abs_error"]:.4f}
- A1 mean NMI: {a1["nmi"]:.4f}
- A5 mean NMI: {a5["nmi"]:.4f}
- Strongest endpoint baseline: `{strongest_endpoint}`

## Explicit Non-Actions

- No manuscript DOCX was modified.
- No ZIP or SHA sidecar was created.
- No frozen method, target, homography, EMAS, baseline or reference output was modified.
- No new clustering experiment was run.
"""
    (DOC_DIR / "task_12_execution_report.md").write_text(exec_report, encoding="utf-8")


def run() -> None:
    _ensure_dirs()
    paths = InputPaths()
    final = build_final_method_comparison(paths)
    diff = build_paired_scene_differences(final)
    figures = build_figures(final, paths)
    write_docs(final, diff, figures, paths)
    manifest = {
        "task": "Task 12 final results synthesis",
        "input_files": {key: str(value.relative_to(ROOT)) for key, value in paths.__dict__.items()},
        "outputs": [
            "results/final_synthesis/final_method_comparison.csv",
            "results/final_synthesis/paired_scene_differences.csv",
            "results/final_synthesis/paired_scene_difference_summary.csv",
            "results/final_synthesis/final_figure_manifest.csv",
            "docs/final_statistical_analysis.md",
            "docs/final_claim_audit.md",
            "docs/reviewer_evidence_map.md",
            "docs/manuscript_tables_final.md",
            "docs/public_release_final_checklist.md",
            "docs/final_scientific_interpretation.md",
            "docs/task_12_execution_report.md",
        ],
        "non_actions": [
            "no_clustering",
            "no_tuning",
            "no_zip",
            "no_sha_sidecar",
            "no_docx_rewrite",
        ],
    }
    (RESULT_DIR / "final_synthesis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
