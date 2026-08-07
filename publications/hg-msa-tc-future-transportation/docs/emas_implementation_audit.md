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
