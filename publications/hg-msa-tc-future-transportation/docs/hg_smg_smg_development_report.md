# HG-SMG SMG Development Report

| Scene | Entry nodes | Exit nodes | Supported edges / K_SMG | Unsupported edges | Threshold | Valid denominator | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bellevue_116th_ne12th | 4 | 4 | 10 | 0 | 0.0025 | 723 | 100.00% |
| bellevue_150th_newport | 3 | 4 | 8 | 0 | 0.0010 | 2,423 | 100.00% |
| bellevue_150th_eastgate | 4 | 4 | 9 | 3 | 0.0050 | 8,720 | 99.43% |
| bellevue_150th_se38th | 5 | 2 | 10 | 0 | 0.0010 | 2,482 | 100.00% |
| bellevue_ne8th | 4 | 4 | 9 | 2 | 0.0050 | 5,576 | 99.75% |

All five primary scenes had valid geometry for the complete target-estimation cohort and at least two supported edges. `K_SMG` counts supported supernode OD edges on `target_estimation`; it is not a legal or manually verified maneuver count. Thresholds were selected by the unchanged frozen heuristic, without reference labels.
