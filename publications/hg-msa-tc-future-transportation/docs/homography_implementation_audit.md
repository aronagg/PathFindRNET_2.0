# Homography Implementation Audit

## Frozen implementation

The five-scene HG-MSA-TC homographies are read from
`research_experiments/fov2026_trajectory_clustering/configs/hg_msa_tc_five_scene/`.
The historical calibration runner is
`research_experiments/fov2026_trajectory_clustering/scripts/hg_msa_tc_five_scene/run_homography_calibration_five_scene.py`.
It calls `cv2.findHomography(source, destination, cv2.RANSAC, 10.0)` and normalizes
the returned matrix by `H[2,2]`. Under OpenCV 4.12.0, the omitted arguments resolve
to `maxIters=2000` and `confidence=0.995`. A direct
least-squares call (`method=0`) is used only if RANSAC fails or returns fewer than four
inliers; no frozen scene used that fallback.

NE8th uses the same estimator after excluding manually identified point IDs
`12, 13, 15, 21, 23`; the original 23-row CSV remains unchanged and the frozen
calibration uses the 18-row filtered file. No lens-distortion correction, point
normalization, nonlinear refinement, or manual matrix postprocessing is present.

## Exact data flow

1. Correspondences are ordered by their CSV row order and interpreted as camera
   `(x,y)` to top-view `(x,y)` pairs.
2. RANSAC uses a 10 px threshold in destination/top-view pixel coordinates.
3. OpenCV returns the inlier mask and a refined matrix estimated from the consensus set.
4. The matrix is divided by `H[2,2]` and stored as camera-to-top-view.
5. The target estimator transforms only canonical first/last endpoints from the
   `target_estimation` split.

## Scene inputs

| scene_id | point_count | inlier_count | source_image_width | source_image_height | destination_image_width | destination_image_height |
| --- | --- | --- | --- | --- | --- | --- |
| bellevue_116th_ne12th | 26 | 17 | 1280 | 720 | 1846 | 920 |
| bellevue_150th_newport | 20 | 13 | 1280 | 720 | 1915 | 985 |
| bellevue_150th_eastgate | 13 | 11 | 1280 | 720 | 1915 | 985 |
| bellevue_150th_se38th | 24 | 16 | 1280 | 720 | 1915 | 985 |
| bellevue_ne8th | 18 | 13 | 1280 | 720 | 2445 | 1209 |

## Duplicate and superseded implementations

- `homography_extension/scripts/run_multiscene_homography_v2.py` uses `method=0` and
  belongs to an earlier extension; it is not the frozen five-scene estimator.
- `homography_extension/scripts/run_final_homography_extension_v1.py` contains earlier
  direct-fit sensitivity code; it is not used here.
- Task 08 centralizes the frozen behavior in `code/homography/calibration.py`.

The final five-scene repository evidence supports RANSAC. Any manuscript wording that
describes all historical extension scripts as RANSAC would still be inaccurate and
must distinguish the final frozen pipeline from earlier case-study code.
