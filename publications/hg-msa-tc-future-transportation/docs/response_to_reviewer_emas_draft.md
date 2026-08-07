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
absolute difference `5.493e-13`. We evaluated seven pre-specified weight
sets, 465 local simplex vectors and 1,000 fixed-seed global vectors using development
candidates only. The reviewer's `0.40T+0.30O+0.10B+0.10S+0.10D` example preserved the
original top-ranked candidate in 14/15 scene-method groups:

| scene                 | method   | candidate_id                            |   trial_index | params_json                                 |
|:----------------------|:---------|:----------------------------------------|--------------:|:--------------------------------------------|
| bellevue_150th_se38th | hdbscan  | bellevue_150th_se38th|hdbscan|trial_001 |             1 | {"min_cluster_size": 80, "min_samples": 10} |

Across local perturbations, top-rank preservation was 93.88% and mean Spearman rank
correlation was 0.9673. We now report the unstable cases explicitly, especially
116th/NE12th OPTICS and SE38th HDBSCAN, and acknowledge that T dominates through both
weight and scale. No independent-test metric was used to tune weights, and independent
clustering was not rerun. Independent validation remains based on the polygon-rule
reference metrics. See revised Section `[TO BE COMPLETED]` and Supplementary Table/Figure
`[TO BE COMPLETED]`.
