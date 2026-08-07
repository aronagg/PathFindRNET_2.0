# Homography Scientific Interpretation

1. **Calibration accuracy.** All frozen matrices reproduce exactly. Mean all-point
   errors are 8.32-13.99
   px; normalized RMSE is 0.43%-1.16%.
2. **Quality gate.** All five pass the a priori diagnostic gate: two `good`, three
   `acceptable`. High endpoint extrapolation remains a shared limitation.
3. **Method role.** Homography supports geometric endpoint-region target estimation;
   camera isotropic shared-scale remains the frozen clustering representation.
4. **Calibration sensitivity.** Small perturbations generally preserve targets, but
   Newport has a threshold discontinuity and Eastgate/NE8th show localized fragility.
5. **SE38th.** Its 18 target persists under plausible small perturbations; calibration
   uncertainty alone does not explain the semantic overestimate.

| scene_id | forward_mean_error_px | normalized_rmse_fraction_destination_diagonal | endpoint_extrapolation_fraction | automatic_od_target | observed_independent_semantic_movement_count | frozen_target_absolute_error |
| --- | --- | --- | --- | --- | --- | --- |
| 116th / NE12th | 13.9897 | 0.0116 | 0.9198 | 10 | 10 | 0 |
| 150th / Newport | 9.7207 | 0.0060 | 0.9695 | 12 | 9 | 3 |
| 150th / Eastgate | 8.3187 | 0.0068 | 0.9298 | 9 | 9 | 0 |
| 150th / SE38th | 10.2540 | 0.0068 | 0.9043 | 18 | 9 | 9 |
| NE8th | 8.3170 | 0.0043 | 0.9693 | 9 | 10 | 1 |

With five scenes, this table is descriptive only and cannot support a reliable
correlation claim between reprojection error and target error. The manuscript should
narrow any claim that homography recovers semantic maneuver counts: it provides a
reproducible geometric structure signal whose granularity can differ from semantic
maneuvers. Publish point coordinates, image dimensions, RANSAC settings, masks,
residual distributions, coverage, extrapolation and perturbation results. Do not
remove or recalibrate SE38th post hoc.
