# Task 17 Execution Report

Generated: `2026-08-09T22:14:16Z`

Branch: `feature/futuretransp-onedrive-final-staging`

Base commit at generation time: `fe6ae4b5022ce06c5881dbc3c669a057921bf80a`

## data/interim Detection Audit Result

True raw YOLOv11 detections found: `false`

## Detection/Tracking/Intermediate Files Found

| Type | Files |
| --- | ---: |
| `detector summary/statistics` | 2 |
| `processed trajectory` | 47 |
| `schema/metadata only` | 19 |
| `tracker output` | 284 |
| `trajectory/intermediate output` | 340 |
| `unclear` | 236 |

## Updated Priority Summary

| Priority | Files | Size GB |
| --- | ---: | ---: |
| P0 | 2310 | 38.234 |
| P1 | 517 | 138.155 |
| P2 | 927 | 0.183 |
| P3 | 40 | 0.045 |

## OneDrive Staging

Local OneDrive root used:

`C:\Users\aggko\OneDrive\aggaron\Education\PhD\Traffic_Node_Video_Dataset_2_0`

P0 missing after local comparison: `0`

P0 mismatched after local comparison: `0`

P0 complete locally: `true`

Cloud sync status: manual verification still required in OneDrive.

## Data Availability Status

The wording distinguishes local P0 staging from cloud-sync verification and does not claim raw YOLOv11 detector exports are available when they were not found.

## Checks

- Python compile passed for `scripts/generate_task17_final_staging_docs.py`.
- Ruff passed for `scripts/generate_task17_final_staging_docs.py`.
- PowerShell parser check passed for `scripts/stage_onedrive_release.ps1`.
- Bash syntax check for `scripts/stage_onedrive_release.sh` was skipped because `bash` was not available in this Windows shell.
- Final P0 dry-run passed: 2310 files, 38.234 GB, 0 missing sources, 0 P3 selected.
- Final local P0 staging verification passed: 0 missing files and 0 mismatched hashes.
- ZIP creation was intentionally skipped for Task 17.
- No experiments were run.
