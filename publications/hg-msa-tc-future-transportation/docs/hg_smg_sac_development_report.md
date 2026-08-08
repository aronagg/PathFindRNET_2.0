# HG-SMG SAC Development Report

Primary SAC used 500 deterministic within-region bootstraps, q=0.95, complete-link consolidation, top-view bearing, and five-point camera-isotropic directed heading. Entry and exit roles were processed separately.

| Scene | Role | Micro-regions | Supernodes | Merges | Invalid | Compatible pairs | D min / median / max | Min abs(D-1) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| bellevue_116th_ne12th | entry | 4 | 4 | 0 | 0 | 0 | 2.412 / 4.626 / 8.021 | 1.412 |
| bellevue_116th_ne12th | exit | 4 | 4 | 0 | 0 | 0 | 2.620 / 4.236 / 8.455 | 1.620 |
| bellevue_150th_newport | entry | 4 | 3 | 1 | 0 | 1 | 0.587 / 1.886 / 3.497 | 0.375 |
| bellevue_150th_newport | exit | 4 | 4 | 0 | 0 | 0 | 5.048 / 8.649 / 14.888 | 4.048 |
| bellevue_150th_eastgate | entry | 4 | 4 | 0 | 0 | 0 | 1.345 / 2.897 / 4.419 | 0.345 |
| bellevue_150th_eastgate | exit | 4 | 4 | 0 | 0 | 0 | 1.958 / 2.982 / 8.783 | 0.958 |
| bellevue_150th_se38th | entry | 7 | 5 | 2 | 0 | 2 | 0.697 / 2.122 / 4.555 | 0.044 |
| bellevue_150th_se38th | exit | 3 | 2 | 1 | 0 | 1 | 0.512 / 1.026 / 1.980 | 0.026 |
| bellevue_ne8th | entry | 4 | 4 | 0 | 0 | 0 | 2.481 / 4.178 / 7.999 | 1.481 |
| bellevue_ne8th | exit | 4 | 4 | 0 | 0 | 0 | 4.224 / 6.884 / 10.839 | 3.224 |

All primary descriptors were valid. The smallest margins around the compatibility boundary `D=1` occur at SE38th, especially for exit regions, so its consolidation is structurally less separated than the other primary scene-role cases.

Across scene-role groups, bootstrap bearing radii ranged from 0.0581 to 1.3968 radians and heading radii from 0.2139 to 3.0674 radians. Exact per-region distributions, seeds, circular MAD values, radii, full pairwise distances, and merge traces are retained in the SAC CSV artifacts. `sac_structural_diagnostics.csv` provides the role-level summary.

No semantic/reference label was accessed, so a merge is reported only as protocol-compatible, not as semantically correct. A8 remains unexecutable because protocol v1 did not freeze a JSD compatibility threshold; no threshold was invented post hoc.
