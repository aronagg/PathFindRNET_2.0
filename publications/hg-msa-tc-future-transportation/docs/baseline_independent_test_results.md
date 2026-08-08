# Baseline Independent-Test Results

## Aggregate Baseline Summary

| baseline_id                    |   ('observed_target_abs_error', 'mean') |   ('observed_target_abs_error', 'median') |   ('ari', 'mean') |   ('ari', 'median') |   ('nmi', 'mean') |   ('nmi', 'median') |   ('purity', 'mean') |   ('purity', 'median') |   ('macro_f1', 'mean') |   ('macro_f1', 'median') |   ('noise_pct_all_test', 'mean') |   ('noise_pct_all_test', 'median') |
|:-------------------------------|----------------------------------------:|------------------------------------------:|------------------:|--------------------:|------------------:|--------------------:|---------------------:|-----------------------:|-----------------------:|-------------------------:|---------------------------------:|-----------------------------------:|
| endpoint_camera_isotropic      |                                 3.46667 |                                         3 |          0.559365 |            0.587701 |          0.709938 |            0.832407 |             0.833379 |               0.983374 |               0.570953 |                 0.668099 |                          24.4266 |                            6.78754 |
| endpoint_camera_raw            |                                 3.33333 |                                         3 |          0.559141 |            0.587701 |          0.709231 |            0.832407 |             0.832107 |               0.983374 |               0.570343 |                 0.668099 |                          24.5322 |                            6.78754 |
| resampled_trajectory_euclidean |                                 3       |                                         2 |          0.568275 |            0.758918 |          0.682667 |            0.832012 |             0.796794 |               0.913793 |               0.475222 |                 0.608525 |                          25.0206 |                            4.63237 |

## Runtime Summary

| baseline_id                    |   feature_build_time_s |   selected_trial_fit_time_s |   feature_memory_bytes |
|:-------------------------------|-----------------------:|----------------------------:|-----------------------:|
| endpoint_camera_isotropic      |             0.00035264 |                    0.261261 |       175315           |
| endpoint_camera_raw            |             0.0002992  |                    0.256406 |       175315           |
| resampled_trajectory_euclidean |             9.25851    |                    0.34105  |            1.75315e+06 |
