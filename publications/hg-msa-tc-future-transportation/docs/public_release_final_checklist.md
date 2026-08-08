# Public Release Final Checklist

| Item | Status | Notes |
| --- | --- | --- |
| Code | ready for review | Publication code modules are present; run Ruff before release. |
| Configs | ready for review | Frozen configs must be archived with hashes. |
| Processed trajectories | local only / release planning needed | Do not include raw videos in lightweight review artifacts. |
| Reference labels | ready for release planning | Include CSV/Parquet and protocol hash if allowed by data policy. |
| Split manifests | ready for release planning | Required for leakage-free reproduction. |
| Homography correspondences | ready for release planning | Manual point pairs and provenance should be published if licensing permits. |
| Intermediate outputs | selective release | Prefer compact CSV summaries and checksums. |
| Figures | ready for manuscript review | Final synthesis figures generated from persisted outputs. |
| Metrics | ready | Final comparison tables are under `results/final_synthesis/`. |
| Licenses | manual review required | Google-derived imagery should not be redistributed unless permitted. |
| DOI archive plan | pending | Archive code, configs, compact metrics, and documentation. |
| GitHub release plan | pending | Tag final revision state after manuscript updates. |
| Raw video redistribution | not included | Treat raw video as non-redistributed unless dataset license permits. |
