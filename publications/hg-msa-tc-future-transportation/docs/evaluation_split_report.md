# Leakage-Free Evaluation Split Report

Protocol: `future-transportation-evaluation-v1`.

## Protocol

The split uses complete source-recording blocks in chronological order for each scene. The earliest block is reserved for homography-guided target estimation, the middle block for candidate/model selection, and the latest block for independent testing. Frame IDs reset in each hourly recording, so frame-only ordering would not be temporally valid.

The requested 30/30/40 proportions are targets rather than row-level cut points. Boundaries are selected jointly to minimize count deviation while keeping recordings intact.

## Scene and Subset Counts

| scene | subset | n | percent | recordings | time_start | time_end | frame_min | frame_max | length_median | length_min | length_max | duplicate_rows | missing_id | missing_time_or_frame | filtering_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bellevue_116th_ne12th | target_estimation | 723 | 31.15 | 5 | 2017-09-10T19:08:25.000 | 2017-09-11T00:05:31.466 | 0 | 106404 | 273.0 | 55 | 698 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=723 |
| bellevue_116th_ne12th | model_selection | 634 | 27.32 | 8 | 2017-09-11T00:09:41.133 | 2017-09-11T07:15:04.633 | 0 | 106851 | 279.5 | 48 | 703 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=634 |
| bellevue_116th_ne12th | independent_test | 964 | 41.53 | 8 | 2017-09-11T08:08:50.000 | 2017-09-11T17:12:46.233 | 0 | 25926 | 198.0 | 44 | 701 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=964 |
| bellevue_150th_newport | target_estimation | 2423 | 25.65 | 14 | 2017-09-10T18:08:24.000 | 2017-09-11T08:08:26.266 | 0 | 107974 | 255.0 | 32 | 633 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=2423 |
| bellevue_150th_newport | model_selection | 3101 | 32.82 | 6 | 2017-09-11T08:08:31.000 | 2017-09-11T14:01:49.166 | 0 | 107945 | 244.0 | 34 | 633 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=3101 |
| bellevue_150th_newport | independent_test | 3924 | 41.53 | 4 | 2017-09-11T14:08:31.000 | 2017-09-11T18:08:29.333 | 0 | 107936 | 239.0 | 39 | 631 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=3924 |
| bellevue_150th_eastgate | target_estimation | 8720 | 33.20 | 15 | 2017-09-10T18:08:24.000 | 2017-09-11T09:08:25.066 | 0 | 107904 | 229.0 | 38 | 709 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=8720 |
| bellevue_150th_eastgate | model_selection | 6791 | 25.85 | 4 | 2017-09-11T09:08:31.000 | 2017-09-11T13:08:27.366 | 0 | 107861 | 249.0 | 48 | 709 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=6791 |
| bellevue_150th_eastgate | independent_test | 10755 | 40.95 | 5 | 2017-09-11T13:08:32.000 | 2017-09-11T18:08:27.100 | 0 | 107882 | 210.0 | 33 | 709 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=10755 |
| bellevue_150th_se38th | target_estimation | 2482 | 27.40 | 12 | 2017-09-10T18:08:24.000 | 2017-09-11T06:08:21.000 | 0 | 107999 | 279.0 | 74 | 708 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=2482 |
| bellevue_150th_se38th | model_selection | 2865 | 31.62 | 5 | 2017-09-11T06:08:36.533 | 2017-09-11T11:06:57.100 | 0 | 107412 | 234.0 | 55 | 707 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=2865 |
| bellevue_150th_se38th | independent_test | 3713 | 40.98 | 6 | 2017-09-11T11:08:34.000 | 2017-09-11T17:16:07.766 | 0 | 105643 | 238.0 | 69 | 708 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=3713 |
| bellevue_ne8th | target_estimation | 5576 | 27.97 | 13 | 2017-09-10T18:08:23.000 | 2017-09-11T07:08:28.166 | 0 | 107985 | 312.0 | 53 | 1429 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=5576 |
| bellevue_ne8th | model_selection | 6321 | 31.71 | 5 | 2017-09-11T07:08:31.000 | 2017-09-11T12:08:30.066 | 0 | 107983 | 266.0 | 43 | 1429 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=6321 |
| bellevue_ne8th | independent_test | 8037 | 40.32 | 5 | 2017-09-11T12:08:31.000 | 2017-09-11T17:08:31.300 | 0 | 107979 | 259.0 | 41 | 1429 | 0 | 0 | 0 | included_in_features_trimmed_frame_disp_norm=8037 |

## Totals

| split | n | scenes | percent |
| --- | --- | --- | --- |
| target_estimation | 19924 | 5 | 29.72 |
| model_selection | 19712 | 5 | 29.41 |
| independent_test | 27393 | 5 | 40.87 |

## Leakage Checks

- Unique trajectory IDs: `67,029` of `67,029` rows.
- Exact fingerprint groups crossing subsets: `0`.
- Rows reassigned to keep an exact duplicate group together: `0`.
- Approximate near-duplicate candidate groups crossing subsets: `0`.
- Missing trajectory IDs: `0`.
- Missing time/frame rows: `0`.

## Filtering Status

The canonical cohort consists only of rows retained in `features_trimmed_frame_disp_norm.parquet`. Consequently every split row is marked pipeline-valid with the same final filtering status. Earlier rejected trajectories are outside this evaluation cohort and are summarized in the repository audit.

## Imbalance and Boundary Effects

Count deviations from 30/30/40 reflect preservation of complete hourly recording blocks. This is preferable to cutting a recording or randomly mixing trajectories from the same time period across protocol stages. All five scenes occur in all three subsets.

## Issues Before Manual Annotation

- Exact vehicle subclass is unavailable in the final processed trajectory schema; the upstream class filter retained car, bus, and truck detections but discarded class metadata.
- Source recording provenance is reconstructed from the verified `merge_tracks.py` ID-offset rule because the merged trajectory table does not retain `video_id`.
- Recording timestamps are inferred from filenames at 30 fps and are local wall-clock values; the repository contains no explicit timezone field.
- Approximate near-duplicate candidates must be reviewed if any cross subsets; they are not automatically removed because geometric similarity can represent distinct vehicles.
- Annotation must begin only after freezing this versioned split and checksum set.

## Manual-Label Isolation

The manually annotated labels are isolated from homography-guided target estimation and clustering model selection. They are used only for independent final evaluation and explicitly identified post hoc diagnostic analyses.

The `independent_test` labels must remain inaccessible to all target estimation, hyperparameter search, model selection, threshold setting, and stopping decisions.

## Configuration Trace

- Manifest checksum: `0b26f59168f0c3996290cc1832ddd423321a5988e27fca93e78d2a02ecde7a03`.
- Random seed recorded but unused by splitting: `20260702`.
- Full recording lists and temporal boundaries are stored in `configs/evaluation_split.yaml`.
