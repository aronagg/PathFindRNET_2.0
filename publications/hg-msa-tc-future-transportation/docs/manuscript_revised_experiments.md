# Revised Experiments

## Evaluation Design

The final evaluation is split-aware. The target-estimation split is used to build
homography-guided priors. The model-selection split is used to select frozen
configurations. The independent-test split is evaluated only after cluster
assignments are persisted. This sequencing is central to the leakage-control
argument.

## Compared Method Families

The final method-comparison table includes:

1. Original untargeted selection (`A0`).
2. Original HG-aware point-target selection (`A1`).
3. HG-SMG-TC with prior-constrained model selection (`A5`).
4. Endpoint camera raw baseline.
5. Endpoint camera isotropic baseline.
6. Resampled trajectory Euclidean baseline.

Each method family is evaluated with KMeans, HDBSCAN and OPTICS where applicable.
DTW and Frechet baselines were not included because full pairwise computation was
not feasible within the revision budget; this is documented as a skipped
baseline rather than a negative result.

## Baselines

Endpoint raw uses the camera-space start and end coordinates. Endpoint isotropic
uses shared-scale isotropic normalization of the same coordinates. The resampled
trajectory Euclidean baseline uses canonical resampled trajectory coordinates
with Euclidean distance. These baselines test whether the proposed scene-structure
prior adds evidence beyond simple endpoint and shape representations.

## Independent Reference Metrics

For valid polygon-rule reference labels, the evaluation reports ARI, NMI, purity,
homogeneity, completeness, V-measure, mapped accuracy, macro precision, macro
recall, macro F1, weighted precision, weighted recall and weighted F1. Cluster
counts are compared against the observed independent-test movement count and the
legal movement count separately.

## Statistical Analysis

The scene is the primary experimental unit. Scene-method rows are useful
descriptively, but the manuscript does not treat trajectories or bootstrap rows
as independent study replicates. Paired scene-level differences are summarized
with mean, median, minimum, maximum and bootstrap confidence intervals over five
scenes. Strong p-value claims are avoided because the number of scenes is small.
