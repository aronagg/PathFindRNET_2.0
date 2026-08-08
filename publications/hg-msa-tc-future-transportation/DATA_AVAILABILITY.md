# Data Availability

The revised manuscript uses five Bellevue intersections from the Traffic Node
Video Dataset. This publication package contains compact derived artifacts,
protocols, metrics, figures and manuscript-support files. It does not redistribute
raw videos or third-party map imagery.

## GitHub-Suitable Artifacts

- source code and tests;
- frozen configuration files;
- compact metric CSV files;
- final synthesis tables and figures;
- protocol reports and reproducibility documentation;
- homography correspondence tables when redistribution is permitted;
- manuscript-support Markdown files.

## DOI Archive Candidates

The following should be deposited in Zenodo, Figshare, OSF or an equivalent DOI
archive:

- exact release snapshot of this publication package;
- compact CSV/Parquet result tables;
- split manifests and checksums where size permits;
- reference-label protocol and labels where data policy permits;
- homography correspondences and residual summaries;
- final manuscript figures and tables;
- environment and reproduction scripts.

DOI placeholder: `[DOI to be added after archive deposit]`.

## Excluded or Restricted Artifacts

- raw videos;
- large local trajectory-level intermediate outputs not suitable for GitHub;
- Google Maps or Google-derived top-view raster images unless redistribution is
  explicitly permitted;
- manual SQLite annotation databases;
- previous task ZIP review packages;
- machine-specific caches and virtual environments.

## Raw Video Handling

Raw video files should be obtained from the official dataset source under its
own license or access rules. The publication package can provide scripts and
manifests to reproduce derived results after the raw files are available locally.

## Google Imagery Handling

Google-derived imagery is treated as restricted until licensing, attribution and
redistribution permissions are confirmed. The public release should prefer
author-created diagrams, source video frames, compact numeric calibration tables,
or other permitted alternatives.
