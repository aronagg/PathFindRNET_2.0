# HG Target Estimator Implementation Audit

## Scope and scientific lock

This audit reconstructs the already frozen `split-aware-hg-msa-tc-v1` target
estimator. It does not define a replacement estimator and does not modify any frozen
target, endpoint-region count, support threshold, selected clustering configuration,
or independent-test assignment.

## Verified implementation locations

| Role | Path | Verified behavior |
| --- | --- | --- |
| Canonical implementation | `code/target_estimation/hg_target_estimator.py` | Single source for homography, endpoint features, region KMeans, OD support, and threshold selection. |
| Compatibility API | `code/pipeline/hg_msa_tc_core.py` | Delegates the original public functions to the canonical module without changing outputs. |
| Frozen development caller | `code/pipeline/run_split_aware_hg_msa_tc.py::target_phase` | Loads only `target_estimation`, transforms endpoints, adds scene seed offsets, and persists target artifacts. |
| Split/checksum guard | `code/pipeline/split_aware_io.py` | Verifies split membership, one-to-one manifest joins, source hashes, finite feature rows, and deterministic trajectory ordering. |
| Task 07 analysis | `code/target_estimation/analysis.py` | Reproduces first; human reference and persisted test assignments are loaded only in a later diagnostic stage. |

A historical isolated implementation remains at
`research_experiments/fov2026_trajectory_clustering/scripts/hg_msa_tc_five_scene/run_hg_msa_tc_five_scene_pipeline.py`.
It was a source for the publication implementation but is not called by the frozen
publication runner. The historical code contains an unused joint-center variable,
did not sort the metric subsample indices, and had a shorter threshold tie-break key.
The authoritative behavior is the frozen publication implementation reproduced here.

## Exact verified behavior

1. The camera start and end feature endpoints are transformed separately by the
   frozen camera-to-top-view homography.
2. Entry and exit points use separate coordinate-wise median centers. There is no
   single shared scene center in the executed code.
3. Each point is represented by `cos(angle)`, `sin(angle)`, and a 0.25-weighted,
   median-normalized radius clipped at three times the median radius.
4. KMeans evaluates K in `[3, 4, 5, 6, 7, 8]`, with `n_init=10`, Lloyd updates, and
   deterministic scene/role seeds.
5. Endpoint K maximizes silhouette, then minimizes Davies-Bouldin, then prefers lower K.
6. OD support is the trajectory fraction for each observed entry-region/exit-region pair.
7. A pair is supported when `share >= threshold`; zero-support combinations are absent.
8. The threshold grid is `[0.1%, 0.25%, 0.5%, 1%, 2%]`. The selected threshold minimizes
   adjacent-grid target instability among candidates with at least 90% OD coverage and
   at least two pairs (80% fallback), then prefers closeness to 0.5%, higher coverage,
   and finally the lower threshold.

## Frozen scene outputs

| scene                   |   chosen_entry_regions |   chosen_exit_regions |   support_threshold_percent |   minimum_required_support_count |   frozen_target |   entry_random_seed |   exit_random_seed |
|:------------------------|-----------------------:|----------------------:|----------------------------:|---------------------------------:|----------------:|--------------------:|-------------------:|
| bellevue_116th_ne12th   |                      4 |                     4 |                      0.2500 |                                2 |              10 |            20260702 |           20260719 |
| bellevue_150th_newport  |                      4 |                     4 |                      0.1000 |                                3 |              12 |            20261702 |           20261719 |
| bellevue_150th_eastgate |                      4 |                     4 |                      0.5000 |                               44 |               9 |            20262702 |           20262719 |
| bellevue_150th_se38th   |                      7 |                     3 |                      0.2500 |                                7 |              18 |            20263702 |           20263719 |
| bellevue_ne8th          |                      4 |                     4 |                      0.5000 |                               28 |               9 |            20264702 |           20264719 |

The different final scene thresholds were not entered manually per scene. They are
data-dependent outputs of the same frozen heuristic over one global threshold grid.
Repository evidence does not support describing these weights, K bounds, or threshold
rules as theoretically optimal; they are heuristic design choices frozen before the
independent-test evaluation.

## Edge cases and failure behavior

- Source feature NaNs are rejected by the split-aware loader before estimation.
- A homography denominator with absolute value below `1e-12` raises an error.
- Empty cohorts, fewer rows than candidate K, or all-invalid input are not assigned a
  fallback target; the estimator fails instead of fabricating a result.
- KMeans endpoint grouping has no noise label. Every valid endpoint is assigned to one region.
- Numeric KMeans labels have no semantic ordering; only their deterministic partitions matter.
- Silhouette/DB failures are represented as NaN and ranked as worst.
- The support denominator is every authorized target-estimation trajectory in the scene.

## Intended versus verified manuscript role

The verified target is a count of supported geometric OD submodes. It may approximate
semantic maneuvers when one endpoint region corresponds to one physical approach, but
the code does not enforce that correspondence. Any manuscript wording that calls it a
guaranteed semantic or legal maneuver count must be narrowed.
