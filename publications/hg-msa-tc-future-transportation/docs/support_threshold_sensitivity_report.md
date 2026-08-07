# Support-Threshold and Region-Count Sensitivity Report

## Support threshold (diagnostic only)

| scene                   |   frozen_threshold_percent |   frozen_target |   minimum_target_on_grid |   maximum_target_on_grid |   thresholds_preserving_frozen_target |   minimum_pair_jaccard_vs_frozen |
|:------------------------|---------------------------:|----------------:|-------------------------:|-------------------------:|--------------------------------------:|---------------------------------:|
| bellevue_116th_ne12th   |                     0.2500 |              10 |                        8 |                       10 |                                     8 |                           0.8000 |
| bellevue_150th_newport  |                     0.1000 |              12 |                        7 |                       12 |                                     7 |                           0.5833 |
| bellevue_150th_eastgate |                     0.5000 |               9 |                        9 |                       12 |                                     4 |                           0.7500 |
| bellevue_150th_se38th   |                     0.2500 |              18 |                       14 |                       19 |                                     3 |                           0.7778 |
| bellevue_ne8th          |                     0.5000 |               9 |                        9 |                       10 |                                     6 |                           0.9000 |

The threshold curve was computed from target-estimation OD support before consulting
independent semantic counts. SE38th stays between 14 and 19 supported pairs over
0.05%-1.0%; threshold adjustment alone does not recover the independently observed 9.
Eastgate and NE8th are comparatively stable around their frozen choices, while Newport
falls from 12 to 7 as rare pairs are removed.

## Endpoint-region granularity (diagnostic only)

| scene                   |   frozen_entry_K |   frozen_exit_K |   frozen_target |   minimum_target_over_K_grid |   maximum_target_over_K_grid |   distinct_target_counts |
|:------------------------|-----------------:|----------------:|----------------:|-----------------------------:|-----------------------------:|-------------------------:|
| bellevue_116th_ne12th   |                4 |               4 |              10 |                            7 |                           22 |                       16 |
| bellevue_150th_newport  |                4 |               4 |              12 |                            8 |                           34 |                       20 |
| bellevue_150th_eastgate |                4 |               4 |               9 |                            6 |                           22 |                       16 |
| bellevue_150th_se38th   |                7 |               3 |              18 |                            8 |                           36 |                       22 |
| bellevue_ne8th          |                4 |               4 |               9 |                            6 |                           19 |                       14 |

Target count is substantially more sensitive to entry/exit K than to small local
threshold changes. SE38th ranges from 8 to 36 over the pre-specified K=3..8 grid; all
scenes show a broad region-count range. This sensitivity is not used to choose new K
values. It demonstrates that the estimator's semantic interpretation depends on
endpoint-region granularity.

No frozen threshold, K value, or target was overwritten by this analysis.
