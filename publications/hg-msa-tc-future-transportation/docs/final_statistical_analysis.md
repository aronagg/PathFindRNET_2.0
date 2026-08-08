# Final Statistical Analysis

This synthesis uses only persisted independent-test outputs. No clustering, model
selection, target estimation, homography calibration, EMAS sensitivity, baseline
assignment, or reference-label generation was rerun.

The primary unit for paired analysis is the scene. Each method family was first
averaged over KMeans, HDBSCAN and OPTICS within a scene, then scene-level paired
differences were computed. With only five Bellevue scenes, the analysis is
descriptive; bootstrap intervals are reported as uncertainty summaries, not as
strong inferential evidence.

## Aggregate Method Means

| method_family                  |   observed_target_abs_error |   legal_target_abs_error |   n_non_noise_clusters |   noise_pct_all_test |    ari |    nmi |   purity |   homogeneity |   completeness |   v_measure |   macro_precision |   macro_recall |   macro_f1 |   weighted_precision |   weighted_recall |   weighted_f1 |
|:-------------------------------|----------------------------:|-------------------------:|-----------------------:|---------------------:|-------:|-------:|---------:|--------------:|---------------:|------------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|
| endpoint_camera_isotropic      |                      3.4667 |                   3.9333 |                 9.0000 |              24.4266 | 0.5594 | 0.7099 |   0.8334 |        0.7302 |         0.7217 |      0.7099 |            0.6609 |         0.5423 |     0.5710 |               0.8777 |            0.6299 |        0.6884 |
| endpoint_camera_raw            |                      3.3333 |                   3.8000 |                 8.8667 |              24.5322 | 0.5591 | 0.7092 |   0.8321 |        0.7289 |         0.7218 |      0.7092 |            0.6609 |         0.5418 |     0.5703 |               0.8777 |            0.6299 |        0.6882 |
| hg_smg_tc_A5                   |                      2.1333 |                   2.7333 |                 9.6667 |              10.3401 | 0.7340 | 0.8386 |   0.9301 |        0.8877 |         0.8069 |      0.8386 |            0.7524 |         0.6371 |     0.6723 |               0.9561 |            0.7622 |        0.8210 |
| original_hg_aware_A1           |                      2.5333 |                   2.8667 |                10.2000 |               9.9444 | 0.7175 | 0.8376 |   0.9339 |        0.8993 |         0.7969 |      0.8376 |            0.7649 |         0.6489 |     0.6838 |               0.9580 |            0.7536 |        0.8169 |
| original_untargeted            |                      3.2000 |                   3.9333 |                 9.1333 |              15.5572 | 0.7218 | 0.8155 |   0.9102 |        0.8450 |         0.7982 |      0.8155 |            0.7174 |         0.5970 |     0.6328 |               0.9396 |            0.7314 |        0.7896 |
| resampled_trajectory_euclidean |                      3.0000 |                   4.8000 |                 7.2000 |              25.0206 | 0.5683 | 0.6827 |   0.7968 |        0.6800 |         0.7179 |      0.6827 |            0.5454 |         0.4722 |     0.4752 |               0.8315 |            0.6412 |        0.6828 |

## Paired Scene-Difference Summary

Positive deltas mean the left method has a larger metric value than the right
method. For target error and outlier percentage, negative values are preferable.
For ARI, NMI and macro F1, positive values are preferable.

| comparison_id                 | metric                    |   mean_delta |   median_delta |   min_delta |   max_delta |   bootstrap_ci95_low |   bootstrap_ci95_high |
|:------------------------------|:--------------------------|-------------:|---------------:|------------:|------------:|---------------------:|----------------------:|
| A1_minus_A0                   | observed_target_abs_error |      -0.6667 |        -0.6667 |     -3.3333 |      0.6667 |              -2.0000 |                0.4000 |
| A1_minus_A0                   | nmi                       |       0.0220 |         0.0116 |     -0.0728 |      0.0914 |              -0.0277 |                0.0665 |
| A1_minus_A0                   | ari                       |      -0.0043 |         0.0057 |     -0.1589 |      0.0866 |              -0.0864 |                0.0700 |
| A1_minus_A0                   | macro_f1                  |       0.0510 |         0.0766 |     -0.0806 |      0.2015 |              -0.0368 |                0.1388 |
| A1_minus_A0                   | noise_pct_all_test        |      -5.6128 |        -5.3007 |    -19.0771 |      3.6373 |             -12.4705 |                0.7177 |
| A5_minus_A1                   | observed_target_abs_error |      -0.4000 |        -0.3333 |     -2.0000 |      1.0000 |              -1.2683 |                0.4000 |
| A5_minus_A1                   | nmi                       |       0.0010 |         0.0000 |     -0.0463 |      0.0403 |              -0.0261 |                0.0264 |
| A5_minus_A1                   | ari                       |       0.0165 |         0.0000 |     -0.0525 |      0.1004 |              -0.0269 |                0.0672 |
| A5_minus_A1                   | macro_f1                  |      -0.0115 |         0.0000 |     -0.0577 |      0.0396 |              -0.0425 |                0.0186 |
| A5_minus_A1                   | noise_pct_all_test        |       0.3957 |         0.0000 |     -0.0248 |      2.0032 |              -0.0149 |                1.2019 |
| A5_minus_resampled_trajectory | observed_target_abs_error |      -0.8667 |        -0.6667 |     -2.6667 |      1.0000 |              -2.0000 |                0.2667 |
| A5_minus_resampled_trajectory | nmi                       |       0.1559 |         0.1783 |      0.0102 |      0.2286 |               0.0786 |                0.2172 |
| A5_minus_resampled_trajectory | ari                       |       0.1657 |         0.1695 |     -0.0618 |      0.3019 |               0.0467 |                0.2623 |
| A5_minus_resampled_trajectory | macro_f1                  |       0.1971 |         0.2147 |      0.1198 |      0.2397 |               0.1578 |                0.2261 |
| A5_minus_resampled_trajectory | noise_pct_all_test        |     -14.6805 |       -16.7789 |    -23.1227 |      0.1825 |             -20.6408 |               -6.6272 |
| A5_minus_strongest_endpoint   | observed_target_abs_error |      -1.3333 |        -1.3333 |     -2.6667 |      0.3333 |              -2.2000 |               -0.4000 |
| A5_minus_strongest_endpoint   | nmi                       |       0.1287 |         0.1753 |      0.0068 |      0.1998 |               0.0564 |                0.1874 |
| A5_minus_strongest_endpoint   | ari                       |       0.1746 |         0.1978 |     -0.0083 |      0.3004 |               0.0742 |                0.2578 |
| A5_minus_strongest_endpoint   | macro_f1                  |       0.1014 |         0.1221 |     -0.0052 |      0.1550 |               0.0456 |                0.1450 |
| A5_minus_strongest_endpoint   | noise_pct_all_test        |     -14.0865 |       -19.0180 |    -20.4808 |     -1.7461 |             -19.7955 |               -6.8852 |

## Main A5 versus A1 Result

HG-SMG-TC A5 reduced mean observed target-count error from
2.5333 to 2.1333
and increased mean NMI from 0.8376 to 0.8386. Mean macro F1
changed from 0.6838 to 0.6723. Mean outlier
percentage changed from 9.94% to
10.34%.

## Baseline Context

The strongest endpoint baseline by aggregate macro F1 is `endpoint_camera_isotropic`.
Compared with this endpoint baseline, HG-SMG-TC A5 has mean target error
2.1333 versus 3.4667,
mean NMI 0.8386 versus 0.7099, and mean macro F1
0.6723 versus 0.5710. The resampled trajectory
baseline has mean target error 3.0000,
mean NMI 0.6827, and mean macro F1 0.4752.
