# OneDrive Upload Batches

Generated: `2026-08-08T23:23:19Z`

The priority labels are derived from the Task 15 upload manifest and the Task 16 release rules. They do not imply that any file has already been uploaded.

## Priority Definitions

- `P0`: required before manuscript submission or directly referenced by the manuscript, GitHub Pages, data availability statement, or reproducibility package.
- `P1`: required for full technical reproducibility, including large source videos, detector/tracker exports and intermediate trajectory artifacts.
- `P2`: useful supplementary material and development diagnostics.
- `P3`: do not upload or review manually before public release.

## Batch Summary

| Priority | Files | Size GB | Top target folders | Source examples | Upload order |
| --- | ---: | ---: | --- | --- | ---: |
| P0 | 2880 | 19.030 | `04_processed_trajectories_and_features/` (2346), `06_publication_results_and_docs/` (265), `07_figures_and_website_assets/` (163), `05_reference_labels_and_protocols/` (85) | `TNVD2_UPLOAD_PACKAGE/06_homography_map_correction/data/processed/bellevue_150th_eastgate/calibration/topview_homography_v1/calibration_preview.png`<br>`TNVD2_UPLOAD_PACKAGE/06_homography_map_correction/data/processed/bellevue_150th_eastgate/calibration/topview_homography_v1/camera_frame.png`<br>`TNVD2_UPLOAD_PACKAGE/06_homography_map_correction/data/processed/bellevue_150th_eastgate/calibration/topview_homography_v1/topview_homography.json`<br>`TNVD2_UPLOAD_PACKAGE/06_homography_map_correction/data/processed/bellevue_150th_eastgate/calibration/topview_homography_v1/warped_camera.png` | 1 |
| P1 | 595 | 157.450 | `03_yolo_tracking_outputs/` (239), `01_original_videos/` (230), `04_processed_trajectories_and_features/` (123), `02_yolov11x_detections/` (3) | `TNVD2_UPLOAD_PACKAGE/01_original_videos/data/raw/bellevue_116th_ne12th/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4`<br>`TNVD2_UPLOAD_PACKAGE/01_original_videos/data/raw/bellevue_116th_ne12th/Bellevue_116th_NE12th__2017-09-10_20-09-12.mp4`<br>`TNVD2_UPLOAD_PACKAGE/01_original_videos/data/raw/bellevue_116th_ne12th/Bellevue_116th_NE12th__2017-09-10_21-08-54.mp4`<br>`TNVD2_UPLOAD_PACKAGE/01_original_videos/data/raw/bellevue_116th_ne12th/Bellevue_116th_NE12th__2017-09-10_22-08-50.mp4` | 2 |
| P2 | 83 | 0.064 | `06_publication_results_and_docs/` (41), `07_figures_and_website_assets/` (27), `04_processed_trajectories_and_features/` (13), `08_reproducibility_configs_and_manifests/` (2) | `TNVD2_UPLOAD_PACKAGE/08_figures_for_paper_and_presentation/data/processed/bellevue_116th_ne12th/rebuild_v1/trajectories_before_after.png`<br>`TNVD2_UPLOAD_PACKAGE/08_figures_for_paper_and_presentation/data/processed/bellevue_116th_ne12th/trajectories_before_after.png`<br>`TNVD2_UPLOAD_PACKAGE/08_figures_for_paper_and_presentation/data/processed/bellevue_150th_eastgate/experiments/eastgate_edge_touch_trim_v1/edge_touch_bottom_center_before_after.png`<br>`TNVD2_UPLOAD_PACKAGE/08_figures_for_paper_and_presentation/data/processed/bellevue_150th_eastgate/experiments/eastgate_edge_touch_trim_v1/trajectories_before_after.png` | 3 |
| P3 | 177 | 0.072 | `99_misc_review/` (161), `07_figures_and_website_assets/` (8), `04_processed_trajectories_and_features/` (6), `08_reproducibility_configs_and_manifests/` (1) | `TNVD2_UPLOAD_PACKAGE/06_homography_map_correction/data/Google_Maps_Pics/bellevue_116th_ne12th_google_maps.png`<br>`TNVD2_UPLOAD_PACKAGE/06_homography_map_correction/data/Google_Maps_Pics/bellevue_150th_eastgate_google_maps.png`<br>`TNVD2_UPLOAD_PACKAGE/06_homography_map_correction/data/Google_Maps_Pics/bellevue_150th_newport_google_maps.png`<br>`TNVD2_UPLOAD_PACKAGE/06_homography_map_correction/data/Google_Maps_Pics/bellevue_150th_se38th_google_maps.png` | 99 |

## Recommended Upload Order

1. Stage and verify `P0`.
2. Confirm OneDrive share permissions and publication page links.
3. Stage `P1` large data after storage and licensing checks.
4. Review `P2` selectively.
5. Keep `P3` out of the public upload unless a maintainer explicitly reclassifies it.
