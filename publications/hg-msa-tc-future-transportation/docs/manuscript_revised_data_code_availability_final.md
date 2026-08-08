# Data and Code Availability

The code, configuration files, compact result tables, final synthesis figures,
manuscript-support documents and reproducibility scripts for this study are
prepared for release in the public PathFindRNET 2.0 repository:

`https://github.com/aronagg/PathFindRNET_2.0`

Publication package path:

`publications/hg-msa-tc-future-transportation/`

GitHub release tag:

`[GitHub release tag to be added]`

A DOI archive containing the release snapshot and permitted compact data
artifacts will be added after deposit:

`[DOI to be added]`

Archive URL:

`[Archive URL to be added]`

Raw video files are not redistributed in this manuscript package and should be
obtained from the official dataset source under its own access terms. Large
trajectory-level intermediate outputs may be regenerated from the released code,
configuration files and locally available source data.

Google-derived or third-party map raster images are not redistributed unless
licensing, attribution and metadata requirements are confirmed. Homography
calibration is documented using numeric point correspondences, matrices,
residuals and quality metrics.

Core result synthesis can be regenerated from persisted result tables using:

```powershell
.\publications\hg-msa-tc-future-transportation\scripts\reproduce_core_results.ps1
```

The script regenerates final synthesis tables and figures only; it does not
rerun clustering, target estimation, homography calibration, baseline generation
or reference-label generation.
