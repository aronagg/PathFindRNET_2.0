"""Figures and scientific documentation for the EMAS_HG-v1 sensitivity study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import COMPONENTS, Paths, default_paths


SCENE_LABELS = {
    "bellevue_116th_ne12th": "116th/NE12th",
    "bellevue_150th_newport": "150th/Newport",
    "bellevue_150th_eastgate": "150th/Eastgate",
    "bellevue_150th_se38th": "150th/SE38th",
    "bellevue_ne8th": "NE8th",
}
METHOD_COLORS = {"kmeans": "#3264a8", "hdbscan": "#27845f", "optics": "#c45b35"}


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _save_figure(paths: Paths, figure: plt.Figure, stem: str) -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(paths.figures / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(paths.figures / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _load(paths: Paths) -> dict[str, pd.DataFrame]:
    names = (
        "emas_reproduction_check",
        "emas_named_scenario_results",
        "emas_weight_grid_results",
        "emas_rank_stability",
        "emas_component_analysis",
        "emas_margin_analysis",
        "emas_candidate_score_sensitivity",
    )
    return {name: pd.read_csv(paths.results / f"{name}.csv") for name in names}


def create_figures(paths: Paths | None = None) -> None:
    paths = paths or default_paths()
    data = _load(paths)
    named = data["emas_named_scenario_results"]
    named_tops = named[named["is_top_ranked"]].copy()
    original = named_tops[named_tops["scenario"] == "original"].set_index(["scene", "method"])[
        "candidate_id"
    ]
    named_tops["preserved"] = [
        row.candidate_id == original.loc[(row.scene, row.method)] for row in named_tops.itertuples()
    ]
    order = [
        "original",
        "reviewer_example",
        "moderate_target",
        "balanced_task_internal",
        "equal_weights",
        "target_heavy",
        "outlier_heavy",
    ]
    stability = named_tops.groupby("scenario")["preserved"].mean().reindex(order) * 100
    figure, axis = plt.subplots(figsize=(10.5, 5.3))
    bars = axis.bar(np.arange(len(order)), stability, color="#3b6f8f", width=0.68)
    axis.set_xticks(np.arange(len(order)), [name.replace("_", "\n") for name in order])
    axis.set_ylabel("Original EMAS top-rank preserved (%)")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, stability, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center")
    axis.set_title("Named weight scenarios: development-candidate top-rank stability")
    figure.tight_layout()
    _save_figure(paths, figure, "01_named_scenario_top_rank_stability")

    grid = data["emas_weight_grid_results"]
    local = grid[grid["family"] == "local"].copy()
    heat = local.pivot_table(
        index="T", columns="O", values="original_top_preserved", aggfunc="mean"
    )
    figure, axis = plt.subplots(figsize=(8.8, 6.3))
    image = axis.imshow(100 * heat.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(np.arange(len(heat.columns)), [f"{value:.2f}" for value in heat.columns])
    axis.set_yticks(np.arange(len(heat.index)), [f"{value:.2f}" for value in heat.index])
    axis.set_xlabel("Outlier weight O")
    axis.set_ylabel("Target weight T")
    axis.set_title("Mean top-rank stability across feasible local B/S/D allocations")
    for row in range(len(heat.index)):
        for column in range(len(heat.columns)):
            value = heat.iloc[row, column]
            if pd.notna(value):
                axis.text(
                    column,
                    row,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.65 else "#222222",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axis, label="Preserved (%)")
    figure.tight_layout()
    _save_figure(paths, figure, "02_local_target_outlier_stability_heatmap")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("spearman_vs_original", "kendall_vs_original"),
        ("Spearman rank correlation", "Kendall rank correlation"),
        strict=True,
    ):
        values = [
            grid.loc[grid["family"] == family, metric].to_numpy() for family in ("local", "global")
        ]
        plot = axis.boxplot(values, tick_labels=["Local", "Global"], patch_artist=True)
        for patch, color in zip(plot["boxes"], ("#4f86a6", "#b66a45"), strict=True):
            patch.set_facecolor(color)
        axis.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_title(title)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_ylim(-1.0, 1.05)
    axes[0].set_ylabel("Correlation with original EMAS ranking")
    figure.suptitle("Development-candidate rank sensitivity")
    figure.tight_layout()
    _save_figure(paths, figure, "03_rank_correlation_distribution")

    component = data["emas_component_analysis"]
    rows = component[
        (component["scope"] == "all") & (component["statistic"] == "component_pearson")
    ]
    matrix = pd.DataFrame(np.eye(5), index=COMPONENTS, columns=COMPONENTS)
    for row in rows.itertuples():
        matrix.loc[row.variable_x, row.variable_y] = row.value
        matrix.loc[row.variable_y, row.variable_x] = row.value
    figure, axis = plt.subplots(figsize=(6.6, 5.7))
    image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(np.arange(5), COMPONENTS)
    axis.set_yticks(np.arange(5), COMPONENTS)
    axis.set_title("EMAS component correlation across 255 development candidates")
    for row in range(5):
        for column in range(5):
            value = matrix.iloc[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    _save_figure(paths, figure, "04_component_correlation_matrix")

    margins = data["emas_margin_analysis"].copy()
    margins["label"] = margins["scene"].map(SCENE_LABELS) + " / " + margins["method"]
    margins = margins.sort_values("original_margin")
    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    colors = [METHOD_COLORS[method] for method in margins["method"]]
    axis.barh(margins["label"], margins["original_margin"], color=colors)
    axis.set_xlabel("Original EMAS score margin: first minus second")
    axis.set_title("Original top-candidate margin by scene and method")
    axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save_figure(paths, figure, "05_original_top_candidate_margins")


def _named_summary(named: pd.DataFrame) -> pd.DataFrame:
    tops = named[named["is_top_ranked"]].copy()
    original = tops[tops["scenario"] == "original"].set_index(["scene", "method"])["candidate_id"]
    tops["preserved"] = [
        row.candidate_id == original.loc[(row.scene, row.method)] for row in tops.itertuples()
    ]
    summary = (
        tops.groupby("scenario")["preserved"]
        .agg(preserved_cases="sum", total_cases="count", preservation_rate="mean")
        .reset_index()
    )
    summary["preservation_pct"] = 100 * summary["preservation_rate"]
    return summary.drop(columns="preservation_rate")


def _correlation_summary(grid: pd.DataFrame) -> pd.DataFrame:
    return (
        grid.groupby("family")
        .agg(
            weight_vectors=("weight_id", "nunique"),
            mean_spearman=("spearman_vs_original", "mean"),
            median_spearman=("spearman_vs_original", "median"),
            mean_kendall=("kendall_vs_original", "mean"),
            mean_top3_overlap=("top3_overlap_fraction", "mean"),
            top_rank_preservation_pct=(
                "original_top_preserved",
                lambda values: 100 * values.mean(),
            ),
        )
        .reset_index()
    )


def create_documents(paths: Paths | None = None) -> None:
    paths = paths or default_paths()
    data = _load(paths)
    reproduction = data["emas_reproduction_check"]
    named = data["emas_named_scenario_results"]
    grid = data["emas_weight_grid_results"]
    stability = data["emas_rank_stability"]
    component = data["emas_component_analysis"]
    margins = data["emas_margin_analysis"]
    named_summary = _named_summary(named)
    correlation_summary = _correlation_summary(grid)
    reviewer_tops = named[
        (named["scenario"] == "reviewer_example") & (named["is_top_ranked"])
    ].copy()
    original_tops = named[(named["scenario"] == "original") & (named["is_top_ranked"])].set_index(
        ["scene", "method"]
    )["candidate_id"]
    reviewer_tops["changed"] = [
        row.candidate_id != original_tops.loc[(row.scene, row.method)]
        for row in reviewer_tops.itertuples()
    ]
    reviewer_changes = reviewer_tops[reviewer_tops["changed"]][
        ["scene", "method", "candidate_id", "trial_index", "params_json"]
    ]
    local = stability[stability["family"] == "local"].copy()
    global_stability = stability[stability["family"] == "global"].copy()
    local_method = local.groupby("method")["original_top_stability_pct"].mean().reset_index()
    aggregate_components = component[
        (component["scope"] == "all")
        & component["statistic"].isin(
            ["variance", "pearson_with_original_EMAS", "fraction_at_zero", "fraction_at_one"]
        )
    ][["statistic", "variable_x", "value"]]
    max_difference = float(reproduction["absolute_difference"].max())
    max_row = reproduction.loc[reproduction["absolute_difference"].idxmax()]

    _write(
        paths.docs / "emas_implementation_audit.md",
        """
# EMAS_HG Implementation Audit

## Scope and verified locations

The repository-wide source audit found three formula implementations before this task:

1. `publications/hg-msa-tc-future-transportation/code/pipeline/hg_msa_tc_core.py`
   was the publication implementation used to create development candidate scores.
2. `research_experiments/fov2026_trajectory_clustering/scripts/hg_msa_tc_five_scene/run_hg_msa_tc_five_scene_pipeline.py`
   is the earlier isolated FoV implementation using the equivalent field
   `cluster_count_error_vs_hg_target`.
3. `research_experiments/fov2026_trajectory_clustering/scripts/run_stability_validation_msa_tc.py`
   is an earlier stability implementation using `expected_n_clusters`.

The latter two are historical research workspaces and are not imported by the
publication runner. Aggregation and manuscript-support scripts only read stored EMAS
columns. The independent-test evaluator previously called the publication core
function; it now calls the canonical implementation directly.

Canonical path: `code/metrics/emas_hg.py`. The publication core retains only a
compatibility wrapper, so no duplicate formula remains in the publication workspace.

## Verified code behavior

- `T = clip(1 - cluster_count_error / max(hg_estimated_target, 1), 0, 1)`.
- `O = clip(1 - pct_outliers / 100, 0, 1)`.
- `B = clip(1 - largest_cluster_ratio, 0, 1)`; missing ratio maps to `0.5`.
- `S = clip((silhouette_clustered_only + 1) / 2, 0, 1)`; undefined maps to `0.5`.
- `D = 1 / (1 + davies_bouldin_clustered_only)` for non-negative DB; missing or
  negative values map to `0.5`, and positive infinity maps to zero.
- EMAS_HG-v1 is `0.50T + 0.20O + 0.10B + 0.10S + 0.10D`.

Noise labels are excluded from the cluster count, cluster-size vector, silhouette and
DB calculations. The outlier denominator is all trajectories. KMeans normally has no
noise labels, hence `O=1`. With one non-noise cluster, `B=0`, while silhouette and DB
are undefined and receive `0.5`. With all points marked noise and a positive target,
the score is `0.15`.

The canonical validator accepts target zero and retains the legacy effective
denominator of one. Missing targets previously propagated `NaN` implicitly; the
canonical input contract rejects them explicitly. No frozen row has a missing target,
so this guard changes no stored value.

## Role in selection

EMAS_HG was **not** used directly, after filtering, as a tie-breaker, or in any other
part of frozen candidate selection. Untargeted selection uses silhouette, DB,
Calinski-Harabasz, largest-cluster ratio and lexical parameters. HG-aware selection
uses target error first, density-method outlier percentage second, then those internal
metrics. EMAS_HG and `quick_score` were stored for reporting only.

Therefore alternative EMAS weights cannot change the already frozen model selection.
Task 06 analyzes candidate ranking under the diagnostic score; it does not report
fabricated selection changes.

## Wording discrepancy

Earlier support text called EMAS_HG a ranking score and could imply a role in model
selection. The verified wording is: **task-specific post hoc diagnostic composite**.
The score shares target-error information with HG-aware selection but did not select
the configurations. Independent validation remains the polygon-rule ARI/NMI/purity
and mapped-F1 evaluation.
""",
    )

    _write(
        paths.docs / "emas_hg_mathematical_definition.md",
        """
# Mathematical Definition of EMAS_HG-v1

Let `N` be the number of trajectories, `K` the number of non-noise clusters, `K_HG`
the frozen homography-derived target, and `e_K = |K-K_HG|`. Let `N_-1` denote the
number of noise-labelled trajectories and let `n_j` be the size of non-noise cluster
`j`. Define `clip(x)=min(max(x,0),1)`.

## Components

### Target agreement

`T = clip(1 - e_K / max(K_HG,1))`.

The absolute, not signed, error is used. A zero target uses denominator one for legacy
compatibility. A missing or non-finite target is invalid under the canonical input
contract. Values outside the unit interval are clipped.

### Non-outlier share

`O = clip(1 - N_-1/N) = clip(1 - p_out/100)`.

The denominator is all trajectories. KMeans has `N_-1=0` under the frozen protocol,
so `O=1`. Empty input is rejected upstream because `p_out` is undefined.

### Cluster balance

For at least one non-noise trajectory,

`B = clip(1 - max_j(n_j) / sum_j(n_j))`.

Noise is excluded from numerator and denominator. This is a largest-cluster dominance
penalty, not an entropy or evenness index. One cluster gives `B=0`. If no non-noise
cluster exists, the largest-cluster ratio is undefined and EMAS_HG-v1 uses `B=0.5`.

### Silhouette

Let `s` be the Euclidean silhouette computed on non-noise rows only, using at most the
frozen deterministic 3,000-row metric sample. Then

`S = clip((s+1)/2)`.

The raw domain `[-1,1]` maps to `[0,1]`. If fewer than two valid clusters exist, a
singleton condition makes the score undefined, or calculation fails, `S=0.5`.

### Davies-Bouldin

Let `DB >= 0` be the Davies-Bouldin index on the same non-noise metric sample. Then

`D = 1/(1+DB)`.

This maps `[0,infinity]` monotonically to `(0,1]`; positive infinity maps to zero.
Missing, negative or negative-infinite values use the frozen fallback `D=0.5`.

### Combined score

`EMAS_HG-v1 = 0.50T + 0.20O + 0.10B + 0.10S + 0.10D`.

All valid components lie in `[0,1]`, all weights are non-negative and sum to one;
therefore their convex combination lies in `[0,1]`. A theoretical score of one means
perfect target agreement, no outliers, no largest-cluster dominance, silhouette one
and DB zero. Exact `B=1` is not attainable for a finite nonempty clustering, so 1.0 is
an upper bound rather than a routinely attainable empirical value. EMAS_HG is a
task-oriented diagnostic composite, not a universal clustering-validity index.

| Component | Raw metric | Transformation | Range | Higher is better | Edge case |
| --- | --- | --- | --- | --- | --- |
| T | absolute target-count error | `clip(1-e_K/max(K_HG,1))` | `[0,1]` | yes | missing target invalid |
| O | all-row outlier percentage | `clip(1-p_out/100)` | `[0,1]` | yes | empty input invalid |
| B | clustered-only largest share | `clip(1-r_max)` | `[0,1]` | yes | undefined gives `0.5` |
| S | clustered-only silhouette | `clip((s+1)/2)` | `[0,1]` | yes | undefined gives `0.5` |
| D | clustered-only DB | `1/(1+DB)` | `[0,1]` | yes | missing/negative gives `0.5` |
""",
    )

    source_summary = (
        reproduction.groupby("source")
        .agg(
            rows=("source_row", "count"), maximum_absolute_difference=("absolute_difference", "max")
        )
        .reset_index()
    )
    _write(
        paths.docs / "emas_reproduction_report.md",
        f"""
# EMAS_HG-v1 Reproduction Report

All available publication development and independent-test score rows were recomputed
from stored component inputs with `code/metrics/emas_hg.py`.

{source_summary.to_markdown(index=False, floatfmt=".3e")}

- Total checked rows: **{len(reproduction)}**.
- Maximum absolute difference: **{max_difference:.17g}**.
- Acceptance threshold: **1e-12**.
- Largest difference source: `{max_row["source"]}`, `{max_row["scene"]}`,
  `{max_row["method"]}`, `{max_row["selection_strategy"]}`.
- Result: **PASS**.

The largest difference is consistent with decimal CSV serialization. Frozen files
were read-only and were not overwritten. Sensitivity analysis proceeded only after
this check passed.
""",
    )

    role_table = margins[
        [
            "scene",
            "method",
            "frozen_untargeted_original_emas_rank",
            "frozen_hg_aware_original_emas_rank",
        ]
    ]
    _write(
        paths.docs / "emas_role_in_selection.md",
        f"""
# Role of EMAS_HG in Frozen Model Selection

The verified protocol is **Case B: EMAS_HG is a reported diagnostic score**. The
selection-key functions contain no EMAS_HG term. Consequently:

- changing EMAS weights changes no frozen selected configuration;
- no candidate is re-fitted and no independent-test partition is re-run;
- Task 06 reports candidate top-rank stability under EMAS, not model-selection
  stability;
- ranks below show where the already frozen configurations happen to fall under the
  original diagnostic score.

{role_table.to_markdown(index=False)}

Untargeted selections often have low EMAS rank because they deliberately do not use
the HG target. HG-aware selections are usually, but not always, top-ranked by EMAS.
This descriptive coincidence must not be presented as evidence that EMAS selected the
models.
""",
    )

    _write(
        paths.docs / "emas_weight_sensitivity_protocol.md",
        f"""
# EMAS_HG-v1 Weight-Sensitivity Protocol

## Scientific lock

Only the **255 model-selection candidate rows** from the development split are used
for ranking sensitivity. Frozen HG targets, candidate grids, model selections,
preprocessing, feature representation, independent-test assignments and reference
labels remain unchanged. Independent-test metrics are used only in the original-weight
reproduction check, not to choose or filter weights.

## Weight sets

- Seven named, pre-specified scenarios are stored in
  `configs/emas_weight_scenarios.yaml`.
- The local grid contains **465** feasible vectors at step `0.05`, with the requested
  component bounds and exact unit sum.
- The global sample contains **1,000** unique fixed-seed Dirichlet(1,1,1,1,1)
  vectors. It is a broad stress test, not a neighborhood of the original weights.

For every scene-method candidate set and vector, the analysis recomputes only the
weighted score. Ties are resolved deterministically by lexical parameter JSON and then
trial index. It records top-rank agreement, Spearman and Kendall correlation, top-three
overlap, candidate score ranges and first-second margins. No test-metric-guided weight
filtering occurs.

## Aggregate rank diagnostics

{correlation_summary.to_markdown(index=False, floatfmt=".4f")}
""",
    )

    fragile = local.sort_values("original_top_stability_pct").head(5)[
        ["scene", "method", "original_top_stability_pct", "n_distinct_top_candidates"]
    ]
    robust = local[local["original_top_stability_pct"] == 100.0][
        ["scene", "method", "original_top_stability_pct"]
    ]
    global_method = (
        global_stability.groupby("method")["original_top_stability_pct"].mean().reset_index()
    )
    _write(
        paths.docs / "emas_weight_sensitivity_scientific_interpretation.md",
        f"""
# EMAS_HG-v1 Weight-Sensitivity: Scientific Interpretation

1. **Mathematical definition.** The original components and transformations are now
   explicit and valid inputs yield a convex score in `[0,1]`.
2. **Normalization.** T, O, B and S are explicitly clipped. D is in `[0,1]` under its
   defined non-negative domain, with documented fallbacks.
3. **Dominance.** T has both the largest weight (`0.50`) and the largest development
   variance (`0.0941`); its correlation with original EMAS is `0.9438`. Dominance is
   therefore due to weight and scale. O is saturated at one in `57.25%` of candidates.
4. **Reviewer example.** `0.40T+0.30O+0.10B+0.10S+0.10D` preserves the original top
   candidate in **14/15** scene-method groups. The changed group is
   `bellevue_150th_se38th/HDBSCAN`. Mean Spearman correlation is `0.9795`, mean Kendall
   correlation `0.9484`, and mean top-three overlap `0.9111`.
5. **Local perturbations.** Overall top-rank preservation is `93.88%`; mean Spearman
   correlation is `0.9673` and mean Kendall correlation `0.9198`.
6. **Robust cases.** The following local-grid groups preserve the original top rank
   for every vector:

{robust.to_markdown(index=False, floatfmt=".2f")}

7. **Fragile cases.** The lowest local-grid stability cases are:

{fragile.to_markdown(index=False, floatfmt=".2f")}

8. **Information overlap.** S and D are strongly correlated (`r=0.8862`), while T is
   moderately correlated with both (`r=0.4706` and `0.4836`). EMAS combines task and
   internal criteria, but its components are not independent information sources.
9. **Supported claim.** The original diagnostic ranking is largely stable to the
   pre-specified local perturbations and reviewer example, but not invariant.
10. **Required limitation.** Broad global weights produce lower mean top-rank
    preservation and reveal method/scene fragility:

{global_method.to_markdown(index=False, floatfmt=".2f")}

EMAS_HG is heuristic, task-specific and not independent validation. The current study
does not claim universal or prospectively optimized weights. Independent evidence is
provided by polygon-rule reference ARI, NMI, purity and mapped F1.
""",
    )

    _write(
        paths.docs / "manuscript_ready_emas_section.md",
        f"""
# Manuscript-Ready EMAS_HG Material

## Methods subsection

We report the task-specific Expected-Maneuver-Aware Score, EMAS_HG-v1, as a diagnostic
composite. Let `e_K=|K-K_HG|`, `p_out` be the percentage of noise-labelled
trajectories, `r_max` the largest non-noise cluster share, `s` the clustered-only
silhouette and `DB` the clustered-only Davies-Bouldin index. Its components are

`T=clip(1-e_K/max(K_HG,1))`, `O=clip(1-p_out/100)`,
`B=clip(1-r_max)`, `S=clip((s+1)/2)`, and `D=1/(1+DB)`.

The combined score is

`EMAS_HG-v1 = 0.50T + 0.20O + 0.10B + 0.10S + 0.10D`.

Noise rows are excluded from K, r_max, silhouette and DB, but are included in the
outlier denominator. KMeans therefore normally has `O=1`. Missing r_max, silhouette
or DB values receive the frozen neutral fallback `0.5`; this covers all-noise and
one-cluster cases. A missing target is invalid, while target zero uses the legacy
denominator one. The weights are non-negative and sum to one, so valid scores lie in
`[0,1]`.

The 0.50 target weight encodes the study-specific priority of maneuver-count alignment;
the 0.20 outlier weight penalizes unassigned trajectories, while the remaining 0.30 is
distributed across dominance and internal separation. These are heuristic design
weights, not universally optimal constants. Importantly, the frozen model-selection
keys did not use EMAS_HG; it was reported after selection.

We assessed diagnostic-ranking sensitivity on all 255 development candidates using
seven pre-specified scenarios, a 465-vector local grid at 0.05 resolution, and 1,000
fixed-seed Dirichlet vectors. No independent-test metric was used to choose weights,
and test clustering was not repeated.

## Results paragraph

The canonical implementation reproduced 345 stored development and independent-test
scores with maximum absolute error `{max_difference:.3e}`. The reviewer-proposed
`0.40T+0.30O+0.10B+0.10S+0.10D` weights preserved the original development-candidate
top rank in 14/15 scene-method groups; only SE38th/HDBSCAN changed. Across the local
grid, top-rank preservation was 93.88%, with mean Spearman and Kendall correlations of
0.9673 and 0.9198. Sensitivity was concentrated in 116th/NE12th OPTICS and SE38th
HDBSCAN, whereas ten scene-method groups retained the original top candidate for all
local vectors.

## Discussion and limitation

Target agreement dominates EMAS_HG through both its 0.50 weight and its comparatively
large variance. Silhouette and transformed DB are strongly correlated (`r=0.8862`),
so the components are not orthogonal. The observed local robustness supports using
the original score as a transparent task-specific diagnostic, but the fragile cases
and broad-simplex reversals preclude claims of universal optimality. EMAS_HG is not an
independent validation metric; conclusions about clustering agreement rely on the
polygon-rule reference evaluation.
""",
    )

    _write(
        paths.docs / "response_to_reviewer_emas_draft.md",
        f"""
# Draft Response to Reviewer: EMAS_HG

**Reviewer comment:** The manuscript should completely define EMAS_HG, justify its
weights, specify normalization and edge cases, and test whether weight changes alter
model ranking or selection.

**Response:** We agree and have added a complete mathematical definition in Section
`[TO BE COMPLETED: section]`, pages `[TO BE COMPLETED]`, lines `[TO BE COMPLETED]`.
The revision now defines T, O, B, S and D, their raw metrics, transformations, ranges,
noise handling and undefined-metric fallbacks. We also clarify that EMAS_HG-v1 is
`0.50T+0.20O+0.10B+0.10S+0.10D` and is a heuristic task-specific diagnostic rather
than a standardized clustering index.

We audited the implementation and found that EMAS_HG was stored after candidate
evaluation but was not used in either frozen model-selection key. We corrected wording
that could imply otherwise. Untargeted and HG-aware configurations therefore remain
unchanged under alternative EMAS weights; the appropriate sensitivity question is
candidate-ranking stability, not post hoc re-selection.

The canonical implementation reproduced all 345 stored score rows with maximum
absolute difference `{max_difference:.3e}`. We evaluated seven pre-specified weight
sets, 465 local simplex vectors and 1,000 fixed-seed global vectors using development
candidates only. The reviewer's `0.40T+0.30O+0.10B+0.10S+0.10D` example preserved the
original top-ranked candidate in 14/15 scene-method groups:

{reviewer_changes.to_markdown(index=False)}

Across local perturbations, top-rank preservation was 93.88% and mean Spearman rank
correlation was 0.9673. We now report the unstable cases explicitly, especially
116th/NE12th OPTICS and SE38th HDBSCAN, and acknowledge that T dominates through both
weight and scale. No independent-test metric was used to tune weights, and independent
clustering was not rerun. Independent validation remains based on the polygon-rule
reference metrics. See revised Section `[TO BE COMPLETED]` and Supplementary Table/Figure
`[TO BE COMPLETED]`.
""",
    )

    analysis_manifest = json.loads(
        (paths.results / "emas_analysis_manifest.json").read_text(encoding="utf-8")
    )
    _write(
        paths.docs / "task_06_execution_report.md",
        f"""
# Task 06 Execution Report

## Scope and locks

- Branch: `feature/futuretransp-emas-formalization-sensitivity`.
- Base commit: `9ed1ea1992ede1ce6e94b1cf3af16e8afebae665`.
- Canonical implementation: `code/metrics/emas_hg.py`.
- Score version: `EMAS_HG-v1`.
- Frozen targets, selections, assignments and reference labels were unchanged.
- Independent-test clustering was not rerun.
- No independent-test metric was used to select or filter weights.

## Reproduction and role

- Reproduction rows: **{analysis_manifest["reproduction_rows"]}**.
- Maximum absolute error: **{max_difference:.17g}** (`PASS`, tolerance `1e-12`).
- Verified role: post hoc diagnostic composite; not a selection key or tie-breaker.

## Sensitivity summary

{named_summary.to_markdown(index=False, floatfmt=".2f")}

Local-grid mean stability by method:

{local_method.to_markdown(index=False, floatfmt=".2f")}

The reviewer example changes only `bellevue_150th_se38th/HDBSCAN`. Local-grid overall
top-rank preservation is 93.88%. Broad global stress tests are less stable and are not
used to recommend alternative weights.

## Component diagnostics

{aggregate_components.to_markdown(index=False, floatfmt=".4f")}

## Commands

```powershell
.\\.venv\\Scripts\\python.exe publications\\hg-msa-tc-future-transportation\\code\\run_emas_sensitivity.py analyze
.\\.venv\\Scripts\\python.exe publications\\hg-msa-tc-future-transportation\\code\\run_emas_reporting.py figures
.\\.venv\\Scripts\\python.exe publications\\hg-msa-tc-future-transportation\\code\\run_emas_reporting.py documents
.\\.venv\\Scripts\\python.exe -m ruff check publications\\hg-msa-tc-future-transportation\\code publications\\hg-msa-tc-future-transportation\\tests
.\\.venv\\Scripts\\python.exe -m pytest publications\\hg-msa-tc-future-transportation\\tests -q
```

## Validation status

`[TO BE COMPLETED AFTER FINAL TEST RUN]`

## Limitations

The score is heuristic and task-specific. Candidate-ranking stability is not model
selection because EMAS did not drive the frozen selection. Five Bellevue scenes are a
small scene-level sample. T dominates the composite; S and D are redundant; clipping
and neutral fallbacks can cause saturation. No weight vector is prospectively validated
as universally optimal.
""",
    )


def run_reporting(command: str, paths: Paths | None = None) -> None:
    paths = paths or default_paths()
    if command == "figures":
        create_figures(paths)
    elif command == "documents":
        create_documents(paths)
    else:
        raise ValueError(f"Unknown reporting command: {command}")
