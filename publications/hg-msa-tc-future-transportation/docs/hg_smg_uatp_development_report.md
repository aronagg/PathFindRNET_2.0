# HG-SMG UATP Development Report

| scene | mode | median | interval_90_lower | interval_90_upper | entropy_bits | probability_full_split_target | failed_replicates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bellevue_116th_ne12th | 10 | 10 | 8 | 13 | 1.8282 | 0.6860 | 0 |
| bellevue_150th_newport | 8 | 9 | 7 | 13 | 2.7617 | 0.3220 | 0 |
| bellevue_150th_eastgate | 9 | 9 | 7 | 11 | 2.1688 | 0.4180 | 0 |
| bellevue_150th_se38th | 10 | 10 | 8 | 20 | 3.4542 | 0.2620 | 0 |
| bellevue_ne8th | 9 | 9 | 9 | 10 | 1.2207 | 0.5160 | 0 |

Every scene used 500 deterministic recording-aware hierarchical bootstrap replicates and reran EMD -> SAC -> SMG. The primary interval is the floor/ceil integerized 90% percentile interval.

The full run contains 10,000 rows across A5/A6/A7/A9 and zero failed replicates. For A5, the number of distinct sampled recordings per replicate ranged from 2-5, 6-12, 7-13, 4-11, and 5-12 in fixed scene order. Variation is expected because source recordings are sampled with replacement before within-recording trajectory resampling. The sampled trajectory counts and recording counts for every replicate are retained in `uatp_bootstrap_targets.csv`.

SE38th has the widest primary interval `[8,20]` and the highest entropy, while NE8th has the narrowest interval `[9,10]`. These are uncertainty diagnostics on development data and do not establish semantic target accuracy.
