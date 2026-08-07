# HG Target Reproduction Report

## Result

The canonical implementation reproduced all five frozen targets using only the
`target_estimation` split. Human polygon-reference labels were not loaded until after
this result and its lock file had been written.

| scene                   |   n_trajectories |   frozen_target |   reproduced_target |   frozen_entry_regions |   frozen_exit_regions |   frozen_support_threshold | target_exact_match   |
|:------------------------|-----------------:|----------------:|--------------------:|-----------------------:|----------------------:|---------------------------:|:---------------------|
| bellevue_116th_ne12th   |              723 |              10 |                  10 |                      4 |                     4 |                     0.0025 | True                 |
| bellevue_150th_newport  |             2423 |              12 |                  12 |                      4 |                     4 |                     0.0010 | True                 |
| bellevue_150th_eastgate |             8720 |               9 |                   9 |                      4 |                     4 |                     0.0050 | True                 |
| bellevue_150th_se38th   |             2482 |              18 |                  18 |                      7 |                     3 |                     0.0025 | True                 |
| bellevue_ne8th          |             5576 |               9 |                   9 |                      4 |                     4 |                     0.0050 | True                 |

The maximum absolute numeric difference across the complete frozen threshold-candidate,
endpoint-region-candidate, and OD-support tables was `1.110e-16`. Row schemas,
non-numeric fields, selected flags, targets, seeds, and scene ordering also matched.

## Isolation checks

- Target recomputation split: `target_estimation` only.
- Human-reference input during reproduction: none.
- Independent-test clustering executed: no.
- Frozen targets modified: no.
- Selected clustering configurations modified: no.
- Original development outputs overwritten: no.

The new diagnostic assignments are stored separately under `results/target_estimation/`.
