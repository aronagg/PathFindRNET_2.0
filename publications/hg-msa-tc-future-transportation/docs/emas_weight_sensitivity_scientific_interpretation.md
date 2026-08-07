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

| scene                   | method   |   original_top_stability_pct |
|:------------------------|:---------|-----------------------------:|
| bellevue_116th_ne12th   | hdbscan  |                       100.00 |
| bellevue_116th_ne12th   | kmeans   |                       100.00 |
| bellevue_150th_eastgate | hdbscan  |                       100.00 |
| bellevue_150th_eastgate | kmeans   |                       100.00 |
| bellevue_150th_eastgate | optics   |                       100.00 |
| bellevue_150th_newport  | optics   |                       100.00 |
| bellevue_150th_se38th   | kmeans   |                       100.00 |
| bellevue_150th_se38th   | optics   |                       100.00 |
| bellevue_ne8th          | hdbscan  |                       100.00 |
| bellevue_ne8th          | kmeans   |                       100.00 |

7. **Fragile cases.** The lowest local-grid stability cases are:

| scene                  | method   |   original_top_stability_pct |   n_distinct_top_candidates |
|:-----------------------|:---------|-----------------------------:|----------------------------:|
| bellevue_116th_ne12th  | optics   |                        55.91 |                           2 |
| bellevue_150th_se38th  | hdbscan  |                        74.19 |                           2 |
| bellevue_150th_newport | hdbscan  |                        87.31 |                           2 |
| bellevue_ne8th         | optics   |                        90.97 |                           2 |
| bellevue_150th_newport | kmeans   |                        99.78 |                           2 |

8. **Information overlap.** S and D are strongly correlated (`r=0.8862`), while T is
   moderately correlated with both (`r=0.4706` and `0.4836`). EMAS combines task and
   internal criteria, but its components are not independent information sources.
9. **Supported claim.** The original diagnostic ranking is largely stable to the
   pre-specified local perturbations and reviewer example, but not invariant.
10. **Required limitation.** Broad global weights produce lower mean top-rank
    preservation and reveal method/scene fragility:

| method   |   original_top_stability_pct |
|:---------|-----------------------------:|
| hdbscan  |                        73.68 |
| kmeans   |                        73.60 |
| optics   |                        59.94 |

EMAS_HG is heuristic, task-specific and not independent validation. The current study
does not claim universal or prospectively optimized weights. Independent evidence is
provided by polygon-rule reference ARI, NMI, purity and mapped F1.
