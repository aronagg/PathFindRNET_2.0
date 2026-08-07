# Manuscript-Ready Homography Section

## Methods: homography calibration and quality control

For each scene, manually selected camera-to-top-view correspondences were used to
estimate a projective transform with OpenCV RANSAC (destination reprojection threshold
10 px, maximum 2000 iterations,
confidence 0.995). The resulting matrix was normalized by its
bottom-right element. NE8th used 18 correspondences after five previously identified
high-error points were excluded; the original point table was retained for audit.
No lens-distortion correction was applied.

Forward error was defined as the Euclidean distance between each manual top-view point
and the projection of its camera counterpart. We report all-point mean, median, RMSE,
maximum, P90 and P95 errors, together with normalized RMSE, RANSAC inlier rate,
calibration-hull coverage and endpoint extrapolation. An a priori engineering gate
combined normalized residual, coverage and extrapolation criteria. The gate was not
tuned against independent clustering/reference results.

## Results

| scene_id | point_count | inlier_count | forward_mean_error_px | forward_median_error_px | forward_rmse_error_px | forward_p95_error_px | quality_class | homography_passes_quality_gate | normalized_rmse_percent | hull_coverage_percent | endpoint_extrapolation_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 116th / NE12th | 26 | 17 | 13.9897 | 5.2264 | 23.8590 | 41.5452 | acceptable | True | 1.1568 | 43.7097 | 91.9779 |
| 150th / Newport | 20 | 13 | 9.7207 | 6.0012 | 13.0199 | 29.3060 | acceptable | True | 0.6046 | 27.5428 | 96.9459 |
| 150th / Eastgate | 13 | 11 | 8.3187 | 4.3523 | 14.6085 | 33.7726 | good | True | 0.6784 | 32.0595 | 92.9817 |
| 150th / SE38th | 24 | 16 | 10.2540 | 5.2082 | 14.6544 | 31.4547 | good | True | 0.6805 | 41.3896 | 90.4311 |
| NE8th | 18 | 13 | 8.3170 | 5.1041 | 11.5983 | 23.3716 | acceptable | True | 0.4252 | 32.1668 | 96.9333 |

All five matrices were reproduced exactly from stored correspondences and all scenes
passed the diagnostic gate. However, 90.4-96.9% of target-estimation endpoints lay
outside the source calibration hull. This high extrapolation rate limits claims of
uniform spatial accuracy.

Calibration sensitivity was evaluated by leaving out each correspondence and by
adding fixed-seed bounded source-point perturbations (`+/-1, +/-2, +/-3, +/-5, +/-10 px`; 20
replicates per scene-scale). Each replicate reran the complete frozen target estimator
on `target_estimation`. SE38th retained `K_HG=18` in every replicate through `+/-5 px`;
at `+/-10 px`, 14/20 retained 18 and no replicate approached the independent semantic
count of 9. The SE38th discrepancy is therefore better explained by endpoint-region
granularity and semantic duplication than by calibration noise alone, although point
omission reveals local fragility.

Homography is a support layer for target estimation, not the final clustering feature
space. The method does not claim metric rectification, dense geometric ground truth,
or universally correct semantic maneuver counts. Manual correspondence uncertainty,
planar-scene assumptions, lens distortion, sparse calibration coverage, Google-image
alignment and imagery provenance remain limitations.
