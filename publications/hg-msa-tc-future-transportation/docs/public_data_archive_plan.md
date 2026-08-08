# Public OneDrive Data Release Plan

Generated/updated for Task 15 on `2026-08-08T22:47:51Z`.

The release plan for this manuscript is GitHub plus GitHub Pages plus public OneDrive. OneDrive is used for large redistributable artifacts that are not appropriate for Git history.

Public OneDrive folder:

`https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84MGZhYWQ2ZmVhMDViMDhjL0lnQlBxblBYemRWd1I1NERrbHVfUzB0bkFUMWJFUlBzSk1XTHM3TVJ3ZDNPZFRJP2U9b2NpOXBM&id=80FAAD6FEA05B08C%21sd773aa4fd5cd47709e03925bbf4b4b67&cid=80FAAD6FEA05B08C`

## Required Manifests

- `docs/local_release_artifact_inventory.csv`
- `docs/onedrive_upload_manifest.csv`
- `docs/new_publication_artifacts_to_upload.md`
- `docs/yolov11_raw_detection_file_locations.md`

## Upload Groups

1. `01_original_videos/`
2. `02_yolov11x_detections/`
3. `03_yolo_tracking_outputs/`
4. `04_processed_trajectories_and_features/`
5. `05_reference_labels_and_protocols/`
6. `06_publication_results_and_docs/`
7. `07_figures_and_website_assets/`
8. `08_reproducibility_configs_and_manifests/`
9. `09_licenses_and_provenance/`

## Manual Checks Before Public Sharing

- Verify that each uploaded item is listed in `docs/onedrive_upload_manifest.csv`.
- Confirm share permissions from a browser session that is not signed in.
- Exclude Google-derived or third-party map rasters unless redistribution is documented.
- Keep generated ZIP review packages out of the public data folder unless explicitly intended.
