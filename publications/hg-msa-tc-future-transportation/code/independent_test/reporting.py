"""Generate conservative independent-test statistical and interpretation reports."""

from __future__ import annotations

import pandas as pd

from .protocol import SCENES, Paths


def _strategy_summary(paired: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "observed_target_abs_error": "lower",
        "legal_target_abs_error": "lower",
        "ari": "higher",
        "nmi": "higher",
        "purity": "higher",
        "macro_f1": "higher",
        "weighted_f1": "higher",
        "noise_pct_all_test": "lower",
        "silhouette_clustered_only": "higher",
        "davies_bouldin_clustered_only": "lower",
        "EMAS_HG": "higher",
    }
    rows = []
    for metric, favorable in definitions.items():
        values = paired[f"delta_{metric}_hg_minus_untargeted"]
        improvement = values < 0 if favorable == "lower" else values > 0
        worsened = values > 0 if favorable == "lower" else values < 0
        rows.append(
            {
                "metric": metric,
                "favorable_direction": favorable,
                "mean_hg_minus_untargeted": values.mean(),
                "median_hg_minus_untargeted": values.median(),
                "improved_cases": int(improvement.sum()),
                "unchanged_cases": int((values.abs() < 1e-12).sum()),
                "worsened_cases": int(worsened.sum()),
                "descriptive_cases": len(values),
            }
        )
    return pd.DataFrame(rows)


def write_statistical_report(paths: Paths) -> None:
    paired = pd.read_csv(paths.results / "paired_strategy_differences.csv")
    bootstrap = pd.read_csv(paths.results / "scene_level_bootstrap_ci.csv")
    summary = _strategy_summary(paired)
    text = (
        "# Independent-Test Statistical Analysis\n\n"
        "The experimental unit is the **scene**. The five Bellevue scenes are the "
        "independent scene-level units; the three methods are repeated analyses within "
        "each scene. The 15 method-scene rows are not interpreted as 15 independent "
        "intersections, and trajectories are not pooled as study replicates.\n\n"
        "Primary inference is descriptive because `n = 5` scenes. The intervals below "
        "are fixed-seed nonparametric bootstrap intervals over the five scene-level "
        "paired differences, separately by method. They quantify scene variation but "
        "should not be treated as strong population-level inference.\n\n"
        "## Descriptive Paired Summary\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## Method-Within-Scene Paired Differences\n\n"
        + paired.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## Scene-Level Bootstrap Intervals\n\n"
        + bootstrap.to_markdown(index=False, floatfmt=".4f")
        + "\n"
    )
    (paths.docs / "independent_test_statistical_analysis.md").write_text(text, encoding="utf-8")


def write_sensitivity_report(paths: Paths) -> None:
    sensitivity = pd.read_csv(paths.results / "reference_sensitivity_evaluation.csv")
    variants = sensitivity[sensitivity["reference_variant"] != "primary"]
    metric_summary = (
        variants.groupby("reference_variant")
        .agg(
            min_coverage_pct=("valid_reference_coverage_pct", "min"),
            max_coverage_pct=("valid_reference_coverage_pct", "max"),
            mean_abs_delta_ari=("delta_ari_vs_primary", lambda s: s.abs().mean()),
            mean_abs_delta_nmi=("delta_nmi_vs_primary", lambda s: s.abs().mean()),
            mean_abs_delta_purity=("delta_purity_vs_primary", lambda s: s.abs().mean()),
            mean_abs_delta_macro_f1=("delta_macro_f1_vs_primary", lambda s: s.abs().mean()),
        )
        .reset_index()
    )
    ranking_rows = []
    for variant, frame in sensitivity.groupby("reference_variant", sort=False):
        unique = frame.drop_duplicates(["scene_id", "method"])
        for metric in ("ari", "nmi", "purity", "macro_f1"):
            delta = unique[f"hg_minus_untargeted_{metric}"]
            ranking_rows.append(
                {
                    "reference_variant": variant,
                    "metric": metric,
                    "hg_aware_higher": int((delta > 1e-12).sum()),
                    "equal": int((delta.abs() <= 1e-12).sum()),
                    "untargeted_higher": int((delta < -1e-12).sum()),
                }
            )
    ranking = pd.DataFrame(ranking_rows)
    ne8th = variants[variants["scene_id"] == "bellevue_ne8th"]
    ne8th_summary = (
        ne8th.groupby("reference_variant")
        .agg(
            valid_coverage_pct=("valid_reference_coverage_pct", "first"),
            mean_abs_delta_ari=("delta_ari_vs_primary", lambda s: s.abs().mean()),
            mean_abs_delta_macro_f1=("delta_macro_f1_vs_primary", lambda s: s.abs().mean()),
        )
        .reset_index()
    )
    text = (
        "# Reference Sensitivity Evaluation\n\n"
        "The canonical first/last-point, unbuffered polygon reference remains primary. "
        "These analyses do not overwrite primary metrics and were not used to alter "
        "targets, configurations, or interpretation thresholds.\n\n"
        "## Metric Changes from Primary Reference\n\n"
        + metric_summary.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## Strategy Ranking Counts\n\n"
        + ranking.to_markdown(index=False)
        + "\n\n## NE8th Diagnostic\n\n"
        + ne8th_summary.to_markdown(index=False, floatfmt=".4f")
        + "\n\nPolygon buffering changes coverage more strongly than the 3/5-point endpoint "
        "variants, especially for NE8th. Therefore conclusions that change only under "
        "buffering must be presented as boundary-sensitive.\n"
    )
    (paths.docs / "reference_sensitivity_evaluation_report.md").write_text(text, encoding="utf-8")


def write_scientific_interpretation(paths: Paths) -> None:
    metrics = pd.read_csv(paths.results / "independent_test_metrics.csv")
    paired = pd.read_csv(paths.results / "paired_strategy_differences.csv")
    targets = pd.read_csv(paths.results / "target_estimation_validation.csv")
    summary = _strategy_summary(paired).set_index("metric")
    scene_rows = []
    for scene in SCENES:
        frame = paired[paired["scene_id"] == scene]
        scene_rows.append(
            {
                "scene_id": scene,
                "mean_delta_observed_error": frame[
                    "delta_observed_target_abs_error_hg_minus_untargeted"
                ].mean(),
                "mean_delta_ari": frame["delta_ari_hg_minus_untargeted"].mean(),
                "mean_delta_nmi": frame["delta_nmi_hg_minus_untargeted"].mean(),
                "mean_delta_macro_f1": frame["delta_macro_f1_hg_minus_untargeted"].mean(),
                "mean_delta_outlier_pct": frame[
                    "delta_noise_pct_all_test_hg_minus_untargeted"
                ].mean(),
            }
        )
    scene_summary = pd.DataFrame(scene_rows)
    se38 = targets[targets["scene_id"] == "bellevue_150th_se38th"].iloc[0]

    def statement(metric: str, lower: bool = False) -> str:
        row = summary.loc[metric]
        direction = "lower" if lower else "higher"
        return (
            f"HG-aware was favorable in {int(row.improved_cases)}/15 cases, equal in "
            f"{int(row.unchanged_cases)}/15, and unfavorable in "
            f"{int(row.worsened_cases)}/15 ({direction} is favorable)."
        )

    text = f"""# Independent-Test Scientific Interpretation

## Direct answers

1. **Does HG-aware selection produce counts closer to independently observed movements?**  
   {statement("observed_target_abs_error", lower=True)} The result must be read by scene and method because the frozen HG target is not identical to the observed reference count.

2. **Does it improve independent agreement?**  
   ARI: {statement("ari")} NMI: {statement("nmi")} Purity: {statement("purity")} Mapped macro F1: {statement("macro_f1")} These metrics are independent of EMAS_HG but still share the general endpoint-based maneuver-identification setting.

3. **Does it reduce outliers?**  
   {statement("noise_pct_all_test", lower=True)} KMeans has no noise by construction; density-method changes should be discussed separately.

4. **Does it worsen compactness or separation?**  
   Silhouette: {statement("silhouette_clustered_only")} Davies-Bouldin: {statement("davies_bouldin_clustered_only", lower=True)} A target-alignment gain may therefore coexist with weaker internal separation.

5. **General superiority or trade-off?**  
   The evidence supports a trade-off analysis, not an unconditional superiority claim. HG-aware selection is designed to favor the frozen geometric target, while independent reference agreement and internal compactness may move in either direction.

6. **Which scenes support the method?**  
   Scenes with negative mean observed-count-error change and non-negative independent metric changes provide the clearest support. The complete scene table below must accompany any claim.

7. **Which scenes limit it?**  
   Any scene with increased observed-count error or reduced ARI/NMI/macro F1 limits generalization. These results are retained rather than averaged away.

8. **How severe is the SE38th target-estimation failure?**  
   The frozen target is **{int(se38.frozen_hg_target)}**, whereas the independent reference observes **{int(se38.observed_independent_test_movement_count)}** movements and the legal mapping contains 12. The absolute error of **{int(se38.hg_target_abs_error_vs_observed)}** is substantial and is a central limitation. It is not corrected post hoc.

9. **How boundary-sensitive are conclusions?**  
   The dedicated sensitivity report shows the exact ranking changes. Endpoint medians are generally less disruptive than 3-pixel polygon buffers; NE8th requires particular caution.

10. **Manuscript claim disposition.**  
    Retain the leakage-controlled separation of target estimation, model selection, and independent evaluation. Retain only scene-qualified claims about count alignment and outlier behavior. Narrow claims of clustering-quality improvement to the metrics and scenes that support them. Remove any claim that EMAS_HG is independent validation or that HG-aware selection universally dominates untargeted selection.

## Scene-Level Trade-offs

{scene_summary.to_markdown(index=False, floatfmt=".4f")}

## Full Independent Metrics

{metrics.to_markdown(index=False, floatfmt=".4f")}

## Interpretation limits

There are five scene-level units from one city subset, the reference is human-defined and rule-based rather than per-trajectory manual ground truth, and polygon boundaries influence coverage. Hungarian mapping is one-to-one; unmatched clusters remain unmatched and unmatched reference movements receive zero recall. Noise is retained for partition metrics but is not mapped to a movement class. EMAS_HG is reported only as the frozen task-specific development/ranking score.
"""
    (paths.docs / "independent_test_scientific_interpretation.md").write_text(
        text, encoding="utf-8"
    )


def write_all_reports(paths: Paths) -> None:
    write_statistical_report(paths)
    write_sensitivity_report(paths)
    write_scientific_interpretation(paths)
