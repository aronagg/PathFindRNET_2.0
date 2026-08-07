# Manuscript-Ready HG Target-Estimation Section

## Methods: homography-guided target estimation

For each trajectory in the development-only target-estimation split, the first and
last frozen feature endpoints were transformed to top-view coordinates by the scene's
camera-to-map homography. Entry and exit endpoints were processed separately. For each
role, the coordinate-wise median endpoint `c` defined the polar reference center. An
endpoint `z` was represented as

`u(z) = [cos(phi), sin(phi), 0.25 clip(r/median(r),0,3)/3]`,

where `phi = atan2(z_y-c_y,z_x-c_x)` and `r = ||z-c||_2`. KMeans endpoint partitions
with K=3,...,8 were evaluated with ten initializations and frozen random seeds. The
region count maximized silhouette, then minimized Davies-Bouldin, with lower K as the
final tie-breaker.

For automatic entry region `a` and exit region `b`, support was
`q_ab = n_ab/N`. At threshold `theta`, the estimated target was

`K_HG(theta) = sum_ab 1[q_ab >= theta]`.

Thresholds 0.1%, 0.25%, 0.5%, 1%, and 2% were evaluated. The frozen heuristic preferred
at least 90% retained OD coverage and at least two pairs, minimized adjacent-grid target
instability, then preferred proximity to 0.5%, higher coverage, and lower threshold.
No manual maneuver labels or independent-test rows entered this computation.

## Frozen scene parameters

| scene                   |   target_estimation_trajectory_count |   chosen_entry_regions |   chosen_exit_regions |   support_threshold_percent |   minimum_required_support_count |   frozen_target |
|:------------------------|-------------------------------------:|-----------------------:|----------------------:|----------------------------:|---------------------------------:|----------------:|
| bellevue_116th_ne12th   |                                  723 |                      4 |                     4 |                      0.2500 |                                2 |              10 |
| bellevue_150th_newport  |                                 2423 |                      4 |                     4 |                      0.1000 |                                3 |              12 |
| bellevue_150th_eastgate |                                 8720 |                      4 |                     4 |                      0.5000 |                               44 |               9 |
| bellevue_150th_se38th   |                                 2482 |                      7 |                     3 |                      0.2500 |                                7 |              18 |
| bellevue_ne8th          |                                 5576 |                      4 |                     4 |                      0.5000 |                               28 |               9 |

## Results

The canonical reconstruction reproduced targets 10, 12, 9, 18, and 9 exactly. Compared
with independent polygon-rule observed counts 10, 9, 9, 9, and 10, the estimator matched
two scenes exactly, differed by one at NE8th, and overestimated Newport and SE38th.
The severe SE38th overestimate arose because the internally preferred endpoint partition
contained seven entry regions and three exit regions. Four entry regions mapped mainly
to one manual physical approach, whereas exit regions merged several manual exits.

## Discussion and limitation

These findings clarify that `K_HG` is a supported geometric endpoint-pair count rather
than a guaranteed semantic maneuver count. The estimator can resolve path or lane-position
submodes inside one physical approach. Its support threshold is moderately stable, but
its target is sensitive to endpoint-region granularity. The frozen SE38th target was not
corrected after independent evaluation.

## Future work

Future work should introduce a prospectively defined road-branch consolidation stage
using map topology, lane markings, or mandatory-turn arrows, followed by evaluation on
new scenes. Such consolidation is not part of the present results.
