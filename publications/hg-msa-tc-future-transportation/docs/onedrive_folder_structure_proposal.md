# Public OneDrive Folder Structure Proposal

Generated: `2026-08-08T22:47:51Z`

Public OneDrive folder:

`https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84MGZhYWQ2ZmVhMDViMDhjL0lnQlBxblBYemRWd1I1NERrbHVfUzB0bkFUMWJFUlBzSk1XTHM3TVJ3ZDNPZFRJP2U9b2NpOXBM&id=80FAAD6FEA05B08C%21sd773aa4fd5cd47709e03925bbf4b4b67&cid=80FAAD6FEA05B08C`

## Proposed Structure

```text
PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release/
  00_README_AND_MANIFESTS/
  01_original_videos/
  02_yolov11x_detections/
  03_yolo_tracking_outputs/
  04_processed_trajectories_and_features/
  05_reference_labels_and_protocols/
  06_publication_results_and_docs/
  07_figures_and_website_assets/
  08_reproducibility_configs_and_manifests/
  09_licenses_and_provenance/
  99_misc_review/
```

## Mapping From Local Inventory

| Category | Files | Size GB |
| --- | ---: | ---: |
| `configuration_or_manifest` | 22 | 0.000 |
| `figure_or_website_asset` | 198 | 0.110 |
| `other` | 161 | 0.026 |
| `processed_trajectory_or_feature` | 2488 | 37.814 |
| `publication_result_or_documentation` | 307 | 0.417 |
| `raw_video` | 230 | 121.736 |
| `reference_label_or_annotation_protocol` | 85 | 0.279 |
| `tracking_output` | 240 | 16.233 |
| `yolov11_detection_or_detection_statistics` | 4 | 0.000 |

## Required Upload Policy

- Keep GitHub focused on code, compact CSV summaries, documentation, reproducibility manifests and manuscript-support material.
- Put large raw videos, detector/tracking exports and large trajectory data in OneDrive.
- Do not upload Google-derived rasters or third-party map screenshots unless redistribution rights and attribution are confirmed.
- Keep raw video and large data out of Git history.
- Update `docs/onedrive_upload_manifest.csv` after any manual upload or rename.
