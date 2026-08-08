# Revised Methods

## Dataset and Splits

The experiments use five Bellevue intersections from the Traffic Node Video
Dataset:

- `bellevue_116th_ne12th`
- `bellevue_150th_newport`
- `bellevue_150th_eastgate`
- `bellevue_150th_se38th`
- `bellevue_ne8th`

The study does not claim full eight-scene dataset validation. The canonical
trajectory cohort is divided into target-estimation, model-selection and locked
independent-test splits. HG targets and HG-SMG priors are estimated on the
target-estimation split. Clustering configurations are selected on the
model-selection split. The independent-test split is used only after the frozen
assignments are persisted.

## Polygon-Rule Reference Labels

Independent-test evaluation uses exhaustive human-defined polygon-rule-based
reference labels. For each scene, four entry polygons, four exit polygons and
twelve legal entry-exit mappings were manually defined. A trajectory receives a
valid reference label only when its canonical entry endpoint falls in exactly one
entry polygon, its canonical exit endpoint falls in exactly one exit polygon, and
the entry-exit pair is included in the legal mapping. Ambiguous, unassigned,
illegal-mapping and missing-geometry cases remain in the output with explicit
status codes. These labels are independent of clustering outputs, HG targets,
automatic OD assignments and EMAS scores.

## Homography Layer

For a camera-space point `p = [x, y, 1]^T` and homography matrix `H`, the
top-view point is computed as `p' ~ H p` with

`x' = h_1^T p / h_3^T p`

`y' = h_2^T p / h_3^T p`.

The frozen implementation uses OpenCV `findHomography` with RANSAC, a 10 px
threshold, 2,000 maximum iterations and confidence 0.995. Homography is used as
a support mechanism for endpoint-structure estimation, not as the final
clustering feature space. The Task-08 quality gate is a diagnostic engineering
screening step. It is not a prospective external validation criterion and does
not remove scenes post hoc.

## Endpoint Micro-Mode Discovery

The original homography-guided estimator transforms trajectory endpoints into
top-view space, computes endpoint-region structure, builds supported automatic
origin-destination pairs, and counts the supported pairs as `K_HG`. The target is
therefore an automatically estimated observed geometric-maneuver target. It is
not guaranteed to equal a semantic or legal maneuver count. In SE38th, the
frozen target is 18 while the independent observed semantic movement count is 9,
showing semantic over-segmentation.

## Semantic Approach Consolidation

HG-SMG-TC uses the frozen endpoint micro-regions as inputs. Same-role regions are
consolidated by semantic approach consolidation (SAC). For two micro-regions
`r` and `s`, the primary preregistered distance is

`D(r,s) = max(d_bearing / R_bearing, d_heading / R_heading)`.

Here `d_bearing` and `d_heading` are circular distances, and the denominators are
bootstrap self-consistency radii. Regions are compatible when `D <= 1`.
Complete-link consolidation forms entry and exit supernodes without using
reference labels.

## Semantic Maneuver Graph

For entry supernode `u` and exit supernode `v`, micro-OD support is aggregated as

`n_tilde_uv = sum_{i in u} sum_{j in v} n_ij`.

The normalized support is

`q_tilde_uv = n_tilde_uv / N`,

where `N` is the frozen valid-assignment denominator. Supported semantic
superedges form the semantic maneuver graph and define the full-split `K_SMG`.

## Uncertainty-Aware Target Prior

The uncertainty-aware target prior (UATP) reruns the frozen
`EMD -> SAC -> SMG` sequence under 500 recording-aware hierarchical bootstrap
replicates on the target-estimation split. The primary prior is the 90%
percentile interval `[Q0.05, Q0.95]`, integerized by flooring the lower endpoint
and ceiling the upper endpoint.

## Prior-Constrained Model Selection

Prior-constrained model selection (PCMS) uses existing model-selection candidate
grids only. It does not introduce new candidate parameters. For a candidate with
cluster count `K` and prior interval `[L,U]`, interval distance is zero if
`K in [L,U]`, `L-K` if `K<L`, and `K-U` if `K>U`. KMeans, HDBSCAN and OPTICS use
the preregistered tie-break order. EMAS_HG remains diagnostic and is not the
independent validation metric.

## EMAS_HG-v1

Let `K` be the number of non-noise clusters, `K_HG` the frozen homography-derived
target, `e_K = |K-K_HG|`, `N` the number of trajectories, `N_-1` the number of
noise-labelled trajectories, and `n_j` the size of non-noise cluster `j`.

`T = clip(1 - e_K / max(K_HG,1))`

`O = clip(1 - N_-1/N)`

`B = clip(1 - max_j(n_j) / sum_j(n_j))`

`S = clip((silhouette + 1)/2)`

`D = 1/(1 + Davies-Bouldin)`

The diagnostic score is

`EMAS_HG-v1 = 0.50T + 0.20O + 0.10B + 0.10S + 0.10D`.

It is task-oriented and heuristic. Independent validation is based on
polygon-rule reference ARI, NMI, purity, homogeneity, completeness, V-measure,
mapped precision, recall and F1.
