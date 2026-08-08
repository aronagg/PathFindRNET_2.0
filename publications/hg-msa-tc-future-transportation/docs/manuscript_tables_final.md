# Manuscript Tables Final

## Dataset and Split Table

| scene_id                |   total_trajectories |   valid_reference_labels |   coverage_pct |
|:------------------------|---------------------:|-------------------------:|---------------:|
| bellevue_116th_ne12th   |                  964 |                      864 |        89.6270 |
| bellevue_150th_eastgate |                10755 |                    10586 |        98.4290 |
| bellevue_150th_newport  |                 3924 |                     3852 |        98.1650 |
| bellevue_150th_se38th   |                 3713 |                     3422 |        92.1630 |
| bellevue_ne8th          |                 8037 |                     7510 |        93.4430 |

## Reference Coverage Table

| scene_id                |   total_trajectories |   valid_reference_labels |   coverage_pct |
|:------------------------|---------------------:|-------------------------:|---------------:|
| bellevue_116th_ne12th   |                  964 |                      864 |        89.6270 |
| bellevue_150th_eastgate |                10755 |                    10586 |        98.4290 |
| bellevue_150th_newport  |                 3924 |                     3852 |        98.1650 |
| bellevue_150th_se38th   |                 3713 |                     3422 |        92.1630 |
| bellevue_ne8th          |                 8037 |                     7510 |        93.4430 |

## Target-Estimation Validation Table

| scene_id                |   frozen_hg_target |   legal_movement_count |   observed_independent_test_movement_count |   hg_target_signed_error_vs_observed |   hg_target_abs_error_vs_observed |   hg_target_signed_error_vs_legal |   hg_target_abs_error_vs_legal |
|:------------------------|-------------------:|-----------------------:|-------------------------------------------:|-------------------------------------:|----------------------------------:|----------------------------------:|-------------------------------:|
| bellevue_116th_ne12th   |                 10 |                     12 |                                         10 |                                    0 |                                 0 |                                -2 |                              2 |
| bellevue_150th_newport  |                 12 |                     12 |                                          9 |                                    3 |                                 3 |                                 0 |                              0 |
| bellevue_150th_eastgate |                  9 |                     12 |                                          9 |                                    0 |                                 0 |                                -3 |                              3 |
| bellevue_150th_se38th   |                 18 |                     12 |                                          9 |                                    9 |                                 9 |                                 6 |                              6 |
| bellevue_ne8th          |                  9 |                     12 |                                         10 |                                   -1 |                                 1 |                                -3 |                              3 |

## Main Method Comparison Table

| method_family                  |   observed_target_abs_error |   noise_pct_all_test |    ari |    nmi |   purity |   macro_f1 |   weighted_f1 |
|:-------------------------------|----------------------------:|---------------------:|-------:|-------:|---------:|-----------:|--------------:|
| endpoint_camera_isotropic      |                      3.4667 |              24.4266 | 0.5594 | 0.7099 |   0.8334 |     0.5710 |        0.6884 |
| endpoint_camera_raw            |                      3.3333 |              24.5322 | 0.5591 | 0.7092 |   0.8321 |     0.5703 |        0.6882 |
| hg_smg_tc_A5                   |                      2.1333 |              10.3401 | 0.7340 | 0.8386 |   0.9301 |     0.6723 |        0.8210 |
| original_hg_aware_A1           |                      2.5333 |               9.9444 | 0.7175 | 0.8376 |   0.9339 |     0.6838 |        0.8169 |
| original_untargeted            |                      3.2000 |              15.5572 | 0.7218 | 0.8155 |   0.9102 |     0.6328 |        0.7896 |
| resampled_trajectory_euclidean |                      3.0000 |              25.0206 | 0.5683 | 0.6827 |   0.7968 |     0.4752 |        0.6828 |

## Baseline Comparison Table

| method_family                  |   observed_target_abs_error |    nmi |   macro_f1 |   noise_pct_all_test |
|:-------------------------------|----------------------------:|-------:|-----------:|---------------------:|
| endpoint_camera_isotropic      |                      3.4667 | 0.7099 |     0.5710 |              24.4266 |
| endpoint_camera_raw            |                      3.3333 | 0.7092 |     0.5703 |              24.5322 |
| hg_smg_tc_A5                   |                      2.1333 | 0.8386 |     0.6723 |              10.3401 |
| resampled_trajectory_euclidean |                      3.0000 | 0.6827 |     0.4752 |              25.0206 |

## Homography Quality Table

| scene_id                |   point_count |   forward_mean_error_px |   forward_rmse_error_px |   forward_p95_error_px |   endpoint_extrapolation_fraction | quality_class   | homography_passes_quality_gate   |
|:------------------------|--------------:|------------------------:|------------------------:|-----------------------:|----------------------------------:|:----------------|:---------------------------------|
| bellevue_116th_ne12th   |            26 |                 13.9897 |                 23.8590 |                41.5452 |                            0.9198 | acceptable      | True                             |
| bellevue_150th_newport  |            20 |                  9.7207 |                 13.0199 |                29.3060 |                            0.9695 | acceptable      | True                             |
| bellevue_150th_eastgate |            13 |                  8.3187 |                 14.6085 |                33.7726 |                            0.9298 | good            | True                             |
| bellevue_150th_se38th   |            24 |                 10.2540 |                 14.6544 |                31.4547 |                            0.9043 | good            | True                             |
| bellevue_ne8th          |            18 |                  8.3170 |                 11.5983 |                23.3716 |                            0.9693 | acceptable      | True                             |

## EMAS Component and Weight Table

| Component | Weight in EMAS_HG-v1 | Interpretation |
| --- | ---: | --- |
| T | 0.50 | Target-count agreement |
| O | 0.20 | Non-outlier component |
| B | 0.10 | Cluster-balance component |
| S | 0.10 | Silhouette-derived component |
| D | 0.10 | Davies-Bouldin-derived component |

## EMAS Weight-Sensitivity Summary

| family   |   original_top_stability_pct |
|:---------|-----------------------------:|
| global   |                      69.0730 |
| local    |                      93.8780 |
| named    |                      96.1900 |

## Ablation Summary Table

| ablation_id     |   mean_observed_target_abs_error |   mean_nmi |   mean_ari |   mean_purity |   mean_macro_f1 |   mean_noise_pct_all_test |
|:----------------|---------------------------------:|-----------:|-----------:|--------------:|----------------:|--------------------------:|
| A0              |                           3.2000 |     0.8155 |     0.7218 |        0.9102 |          0.6328 |                   15.5572 |
| A1              |                           2.5333 |     0.8376 |     0.7175 |        0.9339 |          0.6838 |                    9.9444 |
| A10             |                           1.8000 |     0.8505 |     0.7473 |        0.9322 |          0.6675 |                    9.9965 |
| A2              |                           1.8000 |     0.8505 |     0.7473 |        0.9322 |          0.6675 |                    9.9965 |
| A3              |                           1.8000 |     0.8505 |     0.7473 |        0.9322 |          0.6675 |                    9.9965 |
| A5              |                           2.1333 |     0.8386 |     0.7340 |        0.9301 |          0.6723 |                   10.3401 |
| A6              |                           2.1333 |     0.8386 |     0.7340 |        0.9301 |          0.6723 |                   10.3401 |
| A7              |                           4.4000 |     0.7535 |     0.6295 |        0.8207 |          0.3826 |                   15.2329 |
| A9              |                           2.1333 |     0.8386 |     0.7340 |        0.9301 |          0.6723 |                   10.3401 |
| A5_vs_A1_paired |                         nan      |   nan      |   nan      |      nan      |        nan      |                  nan      |

## Limitations Table

| Limitation | Manuscript handling |
| --- | --- |
| Five Bellevue scenes only | Do not claim full Traffic Node Video Dataset validation. |
| Polygon-rule reference is not per-trajectory manual labeling | Use precise reference terminology. |
| SE38th target over-segmentation | Treat as a visible failure mode and motivation for HG-SMG. |
| Homography point pairs are manual | Report calibration quality and uncertainty. |
| EMAS_HG is heuristic | Present independent reference metrics separately. |
| Endpoint baselines are strong | Avoid universal superiority claims. |
