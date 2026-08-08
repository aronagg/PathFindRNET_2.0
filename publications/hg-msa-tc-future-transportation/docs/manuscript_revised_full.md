# From Geometric Endpoint Micro-Modes to Semantic Maneuver Graphs: Homography-Guided Vehicle Trajectory Clustering at Complex Urban Intersections

Author list: [TO BE COMPLETED]

## Abstract

This study presents Homography-Guided Semantic Maneuver Graph Trajectory
Clustering (HG-SMG-TC), a leakage-controlled unsupervised workflow for clustering
vehicle trajectories from fixed traffic video sensors at complex urban
intersections. The method is evaluated on five Bellevue intersections from the
Traffic Node Video Dataset. The workflow separates three roles that are often
conflated in trajectory clustering: geometric scene-structure estimation,
unsupervised model selection, and independent reference evaluation. Homography is
used as a support layer for endpoint micro-mode discovery and maneuver-structure
estimation, while camera isotropic shared-scale coordinates remain the main
clustering representation. Endpoint micro-regions are consolidated into semantic
approach supernodes, aggregated into a semantic maneuver graph, and converted
into an uncertainty-aware maneuver-count prior for prior-constrained model
selection.

The revised evaluation uses frozen target-estimation, model-selection and
independent-test splits. Independent-test assessment is performed only after
cluster assignments are persisted, using exhaustive human-defined
polygon-rule-based reference labels. Against the original untargeted selection,
the original HG-aware selection improves mean observed target-count error from
3.2000 to 2.5333. The HG-SMG-TC extension further reduces this error to 2.1333
and slightly increases mean NMI from 0.8376 to 0.8386 relative to the original
HG-aware method, while mean macro F1 changes from 0.6838 to 0.6723 and mean
outlier percentage changes from 9.94% to 10.34%. Baseline comparisons show that
simple endpoint KMeans can be strong, so the results are interpreted as
methodological trade-offs rather than universal dominance. The main contribution
is a reproducible, split-aware and reference-isolated framework for connecting
homography-derived scene structure with unsupervised trajectory-clustering model
selection.

## Keywords

Vehicle trajectory clustering; traffic video analytics; unsupervised learning;
homography; maneuver graph; model selection; reference-label leakage.

## 1. Introduction

Fixed traffic cameras provide large volumes of vehicle trajectory data for
traffic monitoring, autonomous-technology evaluation, and AI-based intersection
scene understanding. A recurring task is to group observed trajectories into
movement patterns such as left-turn, through and right-turn maneuvers. This task
is difficult because video trajectories are affected by perspective distortion,
occlusion, incomplete tracks, lane-level variation, detector noise and
scene-specific geometry.

Unsupervised trajectory clustering is attractive because many intersections do
not have dense manual trajectory labels. However, unsupervised clustering also
creates a model-selection problem: different algorithms and hyperparameters can
produce plausible but incompatible partitions. Internal clustering metrics alone
can prefer compact partitions that do not align with maneuver structure. A
method that uses scene structure must therefore be evaluated carefully to avoid
target leakage.

The revised manuscript centers on Homography-Guided Semantic Maneuver Graph
Trajectory Clustering (HG-SMG-TC). The method uses homography-transformed
endpoints to estimate endpoint micro-modes, consolidates compatible same-role
micro-modes into semantic approach supernodes, builds a semantic maneuver graph
from observed origin-destination support, and converts this structure into an
uncertainty-aware maneuver-count prior for model selection. The extension is not
presented as a universally superior clustering algorithm. It is presented as a
reproducible framework for separating geometric prior construction from
independent reference evaluation.

The contributions are:

1. A leakage-controlled split-aware evaluation protocol for trajectory clustering.
2. Exhaustive human-defined polygon-rule-based reference labels for independent
   evaluation.
3. A homography-guided endpoint micro-mode discovery layer.
4. Semantic approach consolidation and semantic maneuver graph construction.
5. An uncertainty-aware maneuver-count prior and prior-constrained model
   selection.
6. Independent-test and baseline evaluation showing method trade-offs.
7. A reproducibility package including formulas, frozen protocols, calibration
   diagnostics, sensitivity analyses and public-release planning.

## 2. Related Work

Vehicle trajectory clustering has been studied for video, GPS and UAV-derived
trajectory data. Prior work covers endpoint-based origin-destination extraction,
shape-based similarity, density-based clustering, graph-based lane and movement
structure, and multi-criteria cluster ranking. This manuscript does not claim
novelty for trajectory clustering in general or for endpoint-based OD discovery
alone.

Recent unsupervised endpoint-region methods identify persistent entry and exit
regions from vehicle trajectories and use them for turning-movement counts
[Rathore et al., 2026]. This directly overlaps with the endpoint-region
discovery component. HG-SMG-TC differs by using endpoint micro-modes as an
intermediate geometric layer that is consolidated into same-role semantic
supernodes and then used as an uncertainty-aware prior for clustering model
selection.

Graph-based and density-based methods have been used to infer lane-level road
structure and movement geometry from trajectories [Uduwaragoda et al., 2013;
Wang et al., 2017; Yuan et al., 2024; Wan et al., 2025]. These studies show that
geometric lane modes and semantic road approaches are distinct representational
levels. HG-SMG-TC does not reconstruct lane centerlines or a routable road
network. Its contribution is narrower: it consolidates endpoint micro-modes into
a semantic maneuver graph for clustering guidance.

Trajectory clustering evaluation is sensitive to the reference definition.
OD-derived references are useful but can blur the distinction between clustering,
classification and manual labeling [Rezaie and Saunier, 2021]. The revised
evaluation uses human-defined polygon-rule-based reference labels generated from
predefined entry and exit polygons. These labels are independent of cluster
assignments and model selection, but they are not fully manual per-trajectory
ground truth.

Multi-criteria ranking has been used for video trajectory clustering [Sekh et
al., 2020]. EMAS_HG in this work is a task-specific diagnostic score, not a
standard clustering-validity index and not the independent validation criterion.

## 3. Methods

### 3.1 Dataset and Split Protocol

The experiments use five Bellevue intersections from the Traffic Node Video
Dataset: `bellevue_116th_ne12th`, `bellevue_150th_newport`,
`bellevue_150th_eastgate`, `bellevue_150th_se38th` and `bellevue_ne8th`. The
study does not claim full eight-scene dataset validation. The canonical
trajectory cohort is divided into target-estimation, model-selection and locked
independent-test splits. HG targets and HG-SMG priors are estimated on the
target-estimation split. Clustering configurations are selected on the
model-selection split. The independent-test split is used only after the frozen
assignments are persisted.

### 3.2 Human-Defined Polygon-Rule Reference

For each scene, four entry polygons, four exit polygons and twelve legal
entry-exit mappings were manually defined. A trajectory receives a valid
reference label only when its canonical entry endpoint falls in exactly one entry
polygon, its canonical exit endpoint falls in exactly one exit polygon, and the
entry-exit pair is included in the legal mapping. Ambiguous, unassigned,
illegal-mapping and missing-geometry cases remain in the output with explicit
status codes. These labels are independent of clustering outputs, HG targets,
automatic OD assignments and EMAS scores.

### 3.3 Homography and Endpoint Micro-Modes

For a camera-space point `p = [x, y, 1]^T` and homography matrix `H`, the top-view
point is computed as `p' ~ H p`, with normalization by the third homogeneous
coordinate. The frozen implementation uses OpenCV `findHomography` with RANSAC,
a 10 px threshold, 2,000 maximum iterations and confidence 0.995. Homography is
used as a support mechanism for endpoint-structure estimation, not as the final
clustering feature space.

The original homography-guided estimator transforms trajectory endpoints into
top-view space, computes endpoint-region structure, builds supported automatic
OD pairs, and counts the supported pairs as `K_HG`. The target is therefore an
automatically estimated observed geometric-maneuver target. It is not guaranteed
to equal a semantic or legal maneuver count.

### 3.4 HG-SMG-TC

HG-SMG-TC uses the frozen endpoint micro-regions as inputs. Same-role regions are
consolidated by semantic approach consolidation (SAC). For two micro-regions
`r` and `s`, the primary preregistered distance is

`D(r,s) = max(d_bearing / R_bearing, d_heading / R_heading)`.

Regions are compatible when `D <= 1`. Complete-link consolidation forms entry
and exit supernodes without using reference labels.

For entry supernode `u` and exit supernode `v`, micro-OD support is aggregated as

`n_tilde_uv = sum_{i in u} sum_{j in v} n_ij`.

The normalized support is `q_tilde_uv = n_tilde_uv / N`, where `N` is the frozen
valid-assignment denominator. Supported semantic superedges form the semantic
maneuver graph and define the full-split `K_SMG`.

The uncertainty-aware target prior reruns the frozen `EMD -> SAC -> SMG`
sequence under 500 recording-aware hierarchical bootstrap replicates on the
target-estimation split. The primary prior is the 90% percentile interval
`[Q0.05, Q0.95]`, integerized by flooring the lower endpoint and ceiling the
upper endpoint.

Prior-constrained model selection uses existing model-selection candidate grids
only. For a candidate with cluster count `K` and prior interval `[L,U]`,
interval distance is zero if `K in [L,U]`, `L-K` if `K<L`, and `K-U` if `K>U`.
KMeans, HDBSCAN and OPTICS use the preregistered tie-break order.

### 3.5 EMAS_HG-v1

Let `K` be the number of non-noise clusters, `K_HG` the frozen
homography-derived target, `e_K = |K-K_HG|`, `N` the number of trajectories,
`N_-1` the number of noise-labelled trajectories, and `n_j` the size of
non-noise cluster `j`. The diagnostic components are:

`T = clip(1 - e_K / max(K_HG,1))`

`O = clip(1 - N_-1/N)`

`B = clip(1 - max_j(n_j) / sum_j(n_j))`

`S = clip((silhouette + 1)/2)`

`D = 1/(1 + Davies-Bouldin)`.

The diagnostic score is:

`EMAS_HG-v1 = 0.50T + 0.20O + 0.10B + 0.10S + 0.10D`.

EMAS_HG is task-oriented and heuristic. Independent validation is based on ARI,
NMI, purity and mapped F1 against the polygon-rule reference.

## 4. Experiments

The final method comparison includes original untargeted selection (A0),
original HG-aware point-target selection (A1), HG-SMG-TC with prior-constrained
model selection (A5), endpoint camera raw baseline, endpoint camera isotropic
baseline, and resampled trajectory Euclidean baseline. Each method family is
evaluated with KMeans, HDBSCAN and OPTICS where applicable. DTW and Frechet
baselines were not included because full pairwise computation was not feasible
within the revision budget.

The independent reference metrics are ARI, NMI, purity, homogeneity,
completeness, V-measure, mapped accuracy, macro precision, macro recall, macro
F1, weighted precision, weighted recall and weighted F1. Cluster counts are
compared against observed independent-test movement counts and legal movement
counts separately. The scene is the primary experimental unit; strong p-value
claims are avoided because there are five scenes.

## 5. Results

The independent-test reference contains 27,393 trajectories across the five
Bellevue scenes. Valid reference-label coverage ranges from 89.627% to 98.429%.

| Scene | Test trajectories | Valid labels | Coverage |
| --- | ---: | ---: | ---: |
| bellevue_116th_ne12th | 964 | 864 | 89.627% |
| bellevue_150th_newport | 3,924 | 3,852 | 98.165% |
| bellevue_150th_eastgate | 10,755 | 10,586 | 98.429% |
| bellevue_150th_se38th | 3,713 | 3,422 | 92.163% |
| bellevue_ne8th | 8,037 | 7,510 | 93.443% |

The frozen HG target exactly matches the independently observed movement count in
two scenes, differs by one at NE8th, overestimates Newport by three, and
overestimates SE38th by nine.

| Method family | Target error | Outlier % | ARI | NMI | Purity | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original untargeted A0 | 3.2000 | 15.5572 | 0.7218 | 0.8155 | 0.9102 | 0.6328 | 0.7896 |
| Original HG-aware A1 | 2.5333 | 9.9444 | 0.7175 | 0.8376 | 0.9339 | 0.6838 | 0.8169 |
| HG-SMG-TC A5 | 2.1333 | 10.3401 | 0.7340 | 0.8386 | 0.9301 | 0.6723 | 0.8210 |
| Endpoint raw baseline | 3.3333 | 24.5322 | 0.5591 | 0.7092 | 0.8321 | 0.5703 | 0.6882 |
| Endpoint isotropic baseline | 3.4667 | 24.4266 | 0.5594 | 0.7099 | 0.8334 | 0.5710 | 0.6884 |
| Resampled trajectory baseline | 3.0000 | 25.0206 | 0.5683 | 0.6827 | 0.7968 | 0.4752 | 0.6828 |

Relative to original untargeted selection, original HG-aware selection improves
mean observed target-count error from 3.2000 to 2.5333. Relative to original
HG-aware selection, HG-SMG-TC A5 further reduces mean target-count error to
2.1333 and slightly increases mean NMI from 0.8376 to 0.8386. It does not improve
every metric: mean macro F1 changes from 0.6838 to 0.6723 and mean outlier
percentage changes from 9.94% to 10.34%.

SE38th demonstrates the central failure mode of the original geometric target:
`K_HG=18` while the independent observed semantic movement count is 9. HG-SMG-TC
is motivated by this failure and uses semantic consolidation to reduce the
effect of micro-mode over-segmentation.

All five frozen homographies reproduce exactly and pass the Task-08 diagnostic
quality gate. Mean all-point reprojection error ranges from 8.3170 px to 13.9897
px. Endpoint extrapolation beyond the calibration hull is high in all scenes,
from 90.43% to 96.95%. Perturbation analysis indicates that SE38th remains
over-segmented under plausible small calibration perturbations.

EMAS_HG-v1 is locally stable in most development scene-method groups: local-grid
top-rank preservation is 93.88%, and the reviewer-example weights preserve the
original top candidate in 14 of 15 groups. The score remains heuristic and
diagnostic.

## 6. Discussion

The final evidence supports HG-SMG-TC as a structured, leakage-controlled
extension rather than a universally dominant clustering method. The strongest
result is improved maneuver-count alignment. The independent reference metrics
show a more nuanced picture: A5 slightly improves mean NMI relative to A1, but
mean macro F1 is lower and mean outlier percentage is slightly higher. These
differences are small in aggregate and vary by scene and method.

Endpoint baselines are credible. Endpoint KMeans is strong in several settings,
confirming that much of the maneuver signal is already present in entry and exit
endpoints. HG-SMG-TC should therefore be positioned as a selection and
scene-structure framework, not as proof that complex trajectory features always
outperform endpoint features.

## 7. Limitations

The study uses five Bellevue scenes only. The polygon-rule reference is not fully
manual per-trajectory ground truth. The frozen HG target estimator can count
geometric endpoint submodes rather than semantic maneuvers. Homography
calibration depends on manually selected point correspondences and shows high
endpoint extrapolation beyond the calibration hull. EMAS_HG is heuristic and
task-specific. The HG-SMG-TC extension is a locked post-review extension
evaluation, not a new pristine blind holdout. The number of scenes is small, so
scene-level paired analysis is descriptive.

## 8. Conclusions

This revised study introduces HG-SMG-TC, a homography-guided semantic maneuver
graph framework for unsupervised vehicle trajectory clustering at complex urban
intersections. The evidence supports cautious conclusions: original HG-aware
selection improves target-count alignment over untargeted selection, and
HG-SMG-TC further improves mean target-count alignment and slightly improves NMI
relative to the original HG-aware method while showing trade-offs in macro F1 and
outlier percentage. The revised manuscript should conclude that
homography-guided scene structure is useful for unsupervised
trajectory-clustering model selection when handled with strict split control and
independent reference evaluation. It should not claim universal superiority, full
manual ground truth, or generalization beyond the five Bellevue scenes evaluated
here.

## Data and Code Availability

The revised study is designed for reproducible release of code, configuration
files, frozen split manifests, compact result tables, reference-label protocols,
homography correspondence tables, calibration diagnostics, model-selection
manifests, final metrics and generated figures. Raw video redistribution is not
assumed. Google-derived or other third-party map imagery should not be
redistributed unless the author confirms license, attribution and metadata
requirements.

## References

[TO BE COMPLETED: Insert final formatted references from the verified literature
register and original manuscript bibliography.]
