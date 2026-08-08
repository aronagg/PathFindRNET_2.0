# Task 15 Execution Report

Generated: `2026-08-08T22:47:51Z`

Branch: `feature/futuretransp-onedrive-release-and-submission`

Base commit at generation time: `bb0eb104aeca424f356b0b4f7466bb533b2e5773`

## Scope

Task 15 replaced the earlier permanent-archive-oriented release wording with a GitHub + GitHub Pages + public OneDrive release plan. No experiments were run and no frozen scientific outputs were modified intentionally.

Public OneDrive folder:

`https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84MGZhYWQ2ZmVhMDViMDhjL0lnQlBxblBYemRWd1I1NERrbHVfUzB0bkFUMWJFUlBzSk1XTHM3TVJ3ZDNPZFRJP2U9b2NpOXBM&id=80FAAD6FEA05B08C%21sd773aa4fd5cd47709e03925bbf4b4b67&cid=80FAAD6FEA05B08C`

## OneDrive Audit Mode

`mode_b_manifest_only_no_local_onedrive_sync_root`

If this is mode B, the task produced a local upload manifest but did not compare against a synced OneDrive directory because `ONEDRIVE_RELEASE_ROOT` was not set to a valid local folder.

## Inventory Summary

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

## Preview Images

| Source | Preview | Status | Width | Height | Size bytes |
| --- | --- | --- | ---: | ---: | ---: |
| `publications/hg-msa-tc-future-transportation/figures/final_synthesis/final_hg_smg_pipeline_schematic.png` | `publications/hg-msa-tc-future-transportation/site/assets/images/final_hg_smg_pipeline_schematic_preview.png` | created | 1200 | 349 | 37366 |
| `publications/hg-msa-tc-future-transportation/figures/final_synthesis/final_target_count_error.png` | `publications/hg-msa-tc-future-transportation/site/assets/images/final_target_count_error_preview.png` | created | 1200 | 663 | 40275 |
| `publications/hg-msa-tc-future-transportation/figures/final_synthesis/final_agreement_metrics.png` | `publications/hg-msa-tc-future-transportation/site/assets/images/final_agreement_metrics_preview.png` | created | 1200 | 643 | 35285 |
| `publications/hg-msa-tc-future-transportation/figures/final_synthesis/final_baseline_comparison_summary.png` | `publications/hg-msa-tc-future-transportation/site/assets/images/final_baseline_comparison_summary_preview.png` | created | 1200 | 734 | 61040 |
| `publications/hg-msa-tc-future-transportation/figures/final_synthesis/final_se38th_fragmentation_story.png` | `publications/hg-msa-tc-future-transportation/site/assets/images/final_se38th_fragmentation_story_preview.png` | created | 1200 | 520 | 69504 |

## Created Or Updated Files

- `scripts/generate_task15_onedrive_release_docs.py`
- `docs/local_release_artifact_inventory.csv`
- `docs/onedrive_upload_manifest.csv`
- `docs/onedrive_folder_structure_proposal.md`
- `docs/yolov11_raw_detection_file_locations.md`
- `docs/new_publication_artifacts_to_upload.md`
- `docs/preview_image_quality_audit.md`
- `DATA_AVAILABILITY.md`
- `docs/final_data_licensing_statement.md`
- `docs/manuscript_revised_data_code_availability_final.md`
- `docs/public_release_scope.md`
- `docs/release_tagging_plan.md`
- `docs/final_submission_checklist.md`
- `docs/manuscript_data_availability_onedrive_final.md`
- `docs/public_data_archive_plan.md`
- `README.md`
- `docs/root_readme_update_proposal.md`
- `../README_publication_section_draft.md`
- `site/publications/hg-smg-tc/index.md`
- `docs/github_pages_integration_plan.md`
- `docs/final_placeholder_and_consistency_report.md`
- `docs/task_15_execution_report.md`
- `site\assets\images\final_hg_smg_pipeline_schematic_preview.png`
- `site\assets\images\final_target_count_error_preview.png`
- `site\assets\images\final_agreement_metrics_preview.png`
- `site\assets\images\final_baseline_comparison_summary_preview.png`
- `site\assets\images\final_se38th_fragmentation_story_preview.png`

## Skipped Files

- No OneDrive existing/missing/extra comparison CSVs were created unless a valid `ONEDRIVE_RELEASE_ROOT` was present.
- No ZIP package or SHA-256 sidecar was created because Task 15 explicitly did not request a ZIP.
- Raw videos and large source datasets were inventoried but not copied or modified.
- The active repository GitHub Pages source was not modified because it could not be safely determined from this checkout.

## Failed Steps

- None at generation time. External upload and deployed Pages verification remain manual release steps.

## Commands

```powershell
git switch -c feature/futuretransp-onedrive-release-and-submission
.\.venv\Scripts\python.exe .\publications\hg-msa-tc-future-transportation\scripts\generate_task15_onedrive_release_docs.py
```
