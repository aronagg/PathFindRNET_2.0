# Objective Homography Quality Gate

The gate was specified from pixel-scale, normalized-error, correspondence coverage,
and extrapolation considerations before the independent target-error column was read.
It is a diagnostic engineering gate and is not tuned to clustering or reference-label
performance.

| Class | normalized RMSE | normalized P95 | inlier fraction | source hull area | endpoint extrapolation | Boolean pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| good | <=1% | <=2% | >=60% | >=20% | <=95% | yes |
| acceptable | <=2% | <=4% | >=40% | >=10% | <=98% | yes |
| acceptable with caution | <=3% | <=6% | >=30% | >=5% | <=99.5% | yes |
| poor | otherwise | otherwise | otherwise | otherwise | otherwise | no |

All criteria in a row must hold. The first matching row determines the class.

| scene_id | point_count | inlier_count | forward_mean_error_px | forward_median_error_px | forward_rmse_error_px | forward_p95_error_px | quality_class | homography_passes_quality_gate | normalized_rmse_percent | hull_coverage_percent | endpoint_extrapolation_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 116th / NE12th | 26 | 17 | 13.990 | 5.226 | 23.859 | 41.545 | acceptable | True | 1.157 | 43.710 | 91.978 |
| 150th / Newport | 20 | 13 | 9.721 | 6.001 | 13.020 | 29.306 | acceptable | True | 0.605 | 27.543 | 96.946 |
| 150th / Eastgate | 13 | 11 | 8.319 | 4.352 | 14.608 | 33.773 | good | True | 0.678 | 32.059 | 92.982 |
| 150th / SE38th | 24 | 16 | 10.254 | 5.208 | 14.654 | 31.455 | good | True | 0.680 | 41.390 | 90.431 |
| NE8th | 18 | 13 | 8.317 | 5.104 | 11.598 | 23.372 | acceptable | True | 0.425 | 32.167 | 96.933 |

All five scenes pass, but this does not imply uniform spatial accuracy. Endpoint
extrapolation is 90.4% to
96.9%, because calibration
features concentrate around the intersection while trajectory endpoints often lie
near image boundaries. Newport and NE8th are therefore `acceptable`, not `good`.
No scene is removed or recalibrated post hoc.
