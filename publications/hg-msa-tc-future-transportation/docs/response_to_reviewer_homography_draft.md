# Response to Reviewer: Homography Calibration

**Comment: The homography implementation and RANSAC settings are insufficiently specified.**

Response: We now state the exact OpenCV estimator, 10 px destination-space RANSAC
threshold, 2000-iteration limit, 0.995 confidence, point ordering, matrix
normalization and fallback behavior. [TO BE COMPLETED: section/page/line]

**Comment: Calibration points, inliers and image dimensions are not reproducible.**

Response: We provide all 101 source/destination pairs, normalized coordinates, image
dimensions, point IDs, inlier masks, source/config hashes and provenance status in a
machine-readable table. Point-level landmark type was not historically recorded and
is disclosed as unknown. [TO BE COMPLETED: supplement/table citation]

**Comment: Qualitative quality labels need an objective gate.**

Response: We replaced the earlier mean-error-only labels with a deterministic gate
combining normalized RMSE, normalized P95, inlier fraction, source-hull coverage and
endpoint extrapolation. The thresholds were specified independently of target/reference
performance, and no scene was removed post hoc. [TO BE COMPLETED: table/line]

**Comment: Calibration sensitivity is not quantified.**

Response: We added leave-one-correspondence-out analysis and 500 fixed-seed bounded
source-point perturbation runs. Each run propagates uncertainty through the full frozen
target estimator on the development target split. [TO BE COMPLETED: figure/table]

**Comment: Could SE38th's target error be a calibration artifact?**

Response: SE38th retained 18 in all perturbations through +/-5 px and never approached
9 at any tested scale. Two point omissions produce strong upward instability, but no
evidence supports calibration uncertainty as a correction toward the semantic count.
We now attribute the primary failure to endpoint-region granularity, with calibration
extrapolation as a secondary limitation. [TO BE COMPLETED: discussion lines]

**Comment: Map imagery source/licensing is unclear.**

Response: The top-view rasters are repository-identified Google Maps screenshots with
incomplete attribution metadata. We do not redistribute them in the review package and
recommend schematic or appropriately licensed replacements in the manuscript. This is
an evidence audit, not a legal conclusion. [TO BE COMPLETED: data/figure statement]
