# Revised Results

## Reference Coverage

The independent-test reference contains 27,393 trajectories across the five
Bellevue scenes. Valid reference-label coverage ranges from 89.627% to 98.429%.
The valid labels are used for independent reference metrics; excluded trajectories
are reported rather than silently dropped.

| Scene | Test trajectories | Valid labels | Coverage |
| --- | ---: | ---: | ---: |
| bellevue_116th_ne12th | 964 | 864 | 89.627% |
| bellevue_150th_newport | 3,924 | 3,852 | 98.165% |
| bellevue_150th_eastgate | 10,755 | 10,586 | 98.429% |
| bellevue_150th_se38th | 3,713 | 3,422 | 92.163% |
| bellevue_ne8th | 8,037 | 7,510 | 93.443% |

## Target-Estimation Validation

The frozen HG target exactly matches the independently observed movement count in
two scenes, differs by one at NE8th, overestimates Newport by three, and
overestimates SE38th by nine. SE38th is therefore a major semantic
over-segmentation case rather than an outlier to hide.

| Scene | Frozen HG target | Observed movements | Legal movements | Absolute error vs observed |
| --- | ---: | ---: | ---: | ---: |
| bellevue_116th_ne12th | 10 | 10 | 12 | 0 |
| bellevue_150th_newport | 12 | 9 | 12 | 3 |
| bellevue_150th_eastgate | 9 | 9 | 12 | 0 |
| bellevue_150th_se38th | 18 | 9 | 12 | 9 |
| bellevue_ne8th | 9 | 10 | 12 | 1 |

## Main Method Comparison

| Method family | Target error | Outlier % | ARI | NMI | Purity | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original untargeted A0 | 3.2000 | 15.5572 | 0.7218 | 0.8155 | 0.9102 | 0.6328 | 0.7896 |
| Original HG-aware A1 | 2.5333 | 9.9444 | 0.7175 | 0.8376 | 0.9339 | 0.6838 | 0.8169 |
| HG-SMG-TC A5 | 2.1333 | 10.3401 | 0.7340 | 0.8386 | 0.9301 | 0.6723 | 0.8210 |
| Endpoint raw baseline | 3.3333 | 24.5322 | 0.5591 | 0.7092 | 0.8321 | 0.5703 | 0.6882 |
| Endpoint isotropic baseline | 3.4667 | 24.4266 | 0.5594 | 0.7099 | 0.8334 | 0.5710 | 0.6884 |
| Resampled trajectory baseline | 3.0000 | 25.0206 | 0.5683 | 0.6827 | 0.7968 | 0.4752 | 0.6828 |

Relative to original untargeted selection, original HG-aware selection improves
mean observed target-count error from 3.2000 to 2.5333 and reduces mean outlier
percentage from 15.56% to 9.94%. Relative to original HG-aware selection,
HG-SMG-TC A5 further reduces mean target-count error to 2.1333 and slightly
increases mean NMI from 0.8376 to 0.8386. It does not improve every metric: mean
macro F1 changes from 0.6838 to 0.6723 and mean outlier percentage changes from
9.94% to 10.34%.

## SE38th Failure and HG-SMG Interpretation

SE38th demonstrates the central failure mode of the original geometric target:
`K_HG=18` while the independent observed semantic movement count is 9. Task-07
diagnostics show that endpoint-region granularity splits one physical approach
into multiple automatic entry regions and that some automatic OD pairs collapse
to the same manual semantic movement. HG-SMG-TC is motivated by this failure: it
uses semantic approach consolidation and interval priors to reduce the effect of
micro-mode over-segmentation. The result is a mitigation, not a post hoc
correction of the frozen target.

## Homography Quality and Sensitivity

All five frozen homographies reproduce exactly and pass the Task-08 diagnostic
quality gate. Mean all-point reprojection error ranges from 8.3170 px to
13.9897 px. However, endpoint extrapolation beyond the calibration hull is high
in all scenes, ranging from 90.43% to 96.95%. Perturbation analysis indicates
that SE38th remains over-segmented under plausible small calibration
perturbations, so calibration uncertainty alone does not explain the 18-versus-9
target error.

## EMAS Sensitivity

EMAS_HG-v1 is mathematically defined as a convex diagnostic score. The original
weighting is locally stable in most development scene-method groups: local-grid
top-rank preservation is 93.88%, and the reviewer-example weights preserve the
original top candidate in 14 of 15 groups. The score is still heuristic and
task-specific. It is not independent validation.

## Baseline Results

Endpoint baselines are credible and should not be dismissed. The strongest
endpoint baseline by aggregate macro F1 is `endpoint_camera_isotropic`, with NMI
0.7099 and macro F1 0.5710. HG-SMG-TC A5 has higher aggregate NMI and macro F1
than this baseline, but endpoint KMeans is strong in several scene-method cases.
The revised manuscript should therefore state that HG-SMG-TC improves the
structured selection evidence, not that it universally dominates all simple
representations.
