# Homography Reproduction Report

Every frozen matrix was re-estimated from the stored correspondences with the exact
historical OpenCV call. Matrices were compared after `H[2,2]=1` normalization.

| scene_id | point_count | frozen_inlier_count | reproduced_inlier_count | inlier_mask_exact_match | matrix_max_abs_difference | matrix_exact_match |
| --- | --- | --- | --- | --- | --- | --- |
| bellevue_116th_ne12th | 26 | 17 | 17 | True | 0.000 | True |
| bellevue_150th_newport | 20 | 13 | 13 | True | 0.000 | True |
| bellevue_150th_eastgate | 13 | 11 | 11 | True | 0.000 | True |
| bellevue_150th_se38th | 24 | 16 | 16 | True | 0.000 | True |
| bellevue_ne8th | 18 | 13 | 13 | True | 0.000 | True |

Maximum matrix-element error across all scenes: `0.000e+00`.
All inlier masks are exact. The frozen JSON files were not overwritten.
