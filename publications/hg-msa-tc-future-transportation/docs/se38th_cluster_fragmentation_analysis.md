# SE38th Cluster Fragmentation Analysis

## Diagnostic definitions

This analysis reads already persisted independent-test assignments; it does not rerun
clustering. For each manual movement, the non-noise cluster proportions are `p_j`.
The effective cluster count is `exp(-sum_j p_j log p_j)`. The normalized fragmentation
entropy divides that entropy by `log(J)` when more than one cluster is present. The
reported movement-completeness proxy is `1 - normalized entropy`. These are diagnostic
quantities, not new primary manuscript metrics.

## Strategy-level summary

| method   | selection_strategy          |   support_weighted_effective_clusters |   maximum_effective_clusters_for_one_movement |   support_weighted_noise_ratio |
|:---------|:----------------------------|--------------------------------------:|----------------------------------------------:|-------------------------------:|
| hdbscan  | hg_expected_aware_selection |                                2.0774 |                                        2.5507 |                         0.1017 |
| hdbscan  | untargeted_selection        |                                1.6679 |                                        1.9630 |                         0.2507 |
| kmeans   | hg_expected_aware_selection |                                1.8107 |                                        3.1234 |                         0.0000 |
| kmeans   | untargeted_selection        |                                1.4082 |                                        1.9009 |                         0.0000 |
| optics   | hg_expected_aware_selection |                                1.3876 |                                        2.6829 |                         0.1297 |
| optics   | untargeted_selection        |                                1.5530 |                                        1.8735 |                         0.5161 |

## HG-aware KMeans (`k=18`)

| reference_movement_id     |   movement_support |   unique_non_noise_clusters |   clusters_with_at_least_1pct_of_movement |   effective_number_of_clusters |   dominant_cluster_share_non_noise |   movement_completeness_proxy |   weighted_cluster_purity |
|:--------------------------|-------------------:|----------------------------:|------------------------------------------:|-------------------------------:|-----------------------------------:|------------------------------:|--------------------------:|
| bellevue_150th_se38th:B>E |                204 |                           4 |                                         4 |                         3.1234 |                             0.5637 |                        0.1784 |                    1.0000 |
| bellevue_150th_se38th:C>E |               1117 |                           2 |                                         2 |                         1.9009 |                             0.6580 |                        0.0733 |                    1.0000 |
| bellevue_150th_se38th:A>G |               1630 |                           4 |                                         3 |                         1.7862 |                             0.8221 |                        0.5816 |                    0.9989 |
| bellevue_150th_se38th:A>H |                182 |                           3 |                                         3 |                         1.2928 |                             0.9396 |                        0.7662 |                    0.9985 |
| bellevue_150th_se38th:B>H |                 70 |                           1 |                                         1 |                         1.0000 |                             1.0000 |                        1.0000 |                    1.0000 |
| bellevue_150th_se38th:C>F |                 12 |                           1 |                                         1 |                         1.0000 |                             1.0000 |                        1.0000 |                    1.0000 |
| bellevue_150th_se38th:C>H |                 10 |                           1 |                                         1 |                         1.0000 |                             1.0000 |                        1.0000 |                    1.0000 |
| bellevue_150th_se38th:D>F |                 56 |                           1 |                                         1 |                         1.0000 |                             1.0000 |                        1.0000 |                    1.0000 |
| bellevue_150th_se38th:D>G |                141 |                           1 |                                         1 |                         1.0000 |                             1.0000 |                        1.0000 |                    1.0000 |

The strongest KMeans fragmentation is visible for `B>E` (effective cluster count
about 3.12), followed by `C>E` and `A>G`. Support-weighted effective cluster count rises
from about 1.41 under untargeted KMeans to 1.81 under HG-aware KMeans. The corresponding
figure overlays one manual movement colored by its frozen HG-aware clusters.

The splits are spatially structured enough to be consistent with distinct geometric
path or lane-position submodes, but the current data do not prove lane semantics.
Fragmentation is therefore evidence that KMeans is especially sensitive to an
overestimated target: setting `k=18` forces every test trajectory into one of 18 groups.
Density methods are less directly controlled by the target because the target affects
configuration selection rather than fixing the fitted number of clusters.
