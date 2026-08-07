# SE38th Homography Diagnosis

The frozen SE38th calibration has 24 correspondences and 16 RANSAC inliers. Its
all-point forward mean is 10.254 px, median
5.208 px, RMSE
14.654 px and P95
31.455 px. Normalized RMSE is
0.680% of the
top-view diagonal. The calibration hull covers
41.4% of the camera image,
while 90.4% of target-estimation
endpoints lie outside that hull.

The target remains 18 in all 80
replicates through `+/-5 px`. At `+/-10 px`, preservation is
70%
and the target range is
16-37.
None of the 100 perturbation runs produces 9. Leave-one-point-out preserves
18 in 91.7% of cases; omitting
inlier point 5 or 6 changes the endpoint partition to 7 entry and 8 exit regions and
raises the target to 43, indicating a localized calibration dependency rather than a
plausible correction toward the semantic count.

Task 07 showed four automatic entry regions dominated by one physical approach and
semantic duplication among supported OD pairs. The new evidence therefore supports
the diagnosis that SE38th is primarily an endpoint-region granularity/semantic
over-segmentation problem. Calibration extrapolation and two influential points are
secondary fragility factors, but realistic small perturbations do not explain the
18-versus-9 error.
