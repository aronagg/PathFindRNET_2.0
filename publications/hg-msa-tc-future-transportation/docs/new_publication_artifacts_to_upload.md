# New Publication Artifacts To Upload To OneDrive

Generated: `2026-08-08T22:47:51Z`

This list is derived from `docs/onedrive_upload_manifest.csv`. It is a release-planning list only; no upload was performed.

## High-Priority Groups

| Group | Local evidence | Suggested OneDrive folder |
| --- | ---: | --- |
| Raw videos | 230 files | `01_original_videos/` |
| YOLOv11 detection/statistics files | 4 files | `02_yolov11x_detections/` |
| YOLO tracking outputs | 240 files | `03_yolo_tracking_outputs/` |
| Processed trajectories/features | 2488 files | `04_processed_trajectories_and_features/` |
| Reference labels/protocols | 85 files | `05_reference_labels_and_protocols/` |
| Publication results/docs | 307 files | `06_publication_results_and_docs/` |
| Figures/site assets | 198 files | `07_figures_and_website_assets/` |

## Notes

- The full itemized upload list is `docs/onedrive_upload_manifest.csv`.
- Large files have size metadata but may not have local SHA-256 hashes if they exceed the configured audit limit of 268435456 bytes.
- Publication result CSVs, reference-label summaries and final figures should also remain in GitHub when compact enough.
- OneDrive should be treated as the public large-artifact mirror, not as evidence that files were generated without the repository provenance records.
