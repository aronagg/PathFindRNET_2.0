# Public Release Plan

The release separates lightweight version-controlled materials from larger DOI-backed
research artifacts and from restricted source imagery.

## GitHub Release

- publication-specific source code and tests;
- frozen protocols, ablations, seeds, schemas, and checksums;
- mathematical, leakage-control, and execution documentation;
- experiment registry and artifact manifest;
- compact aggregate CSV/JSON results and figure-source tables;
- platform-neutral Python entry point and PowerShell wrapper;
- `CITATION.cff`, data dictionary, provenance, licensing, and reproduction guides.

## DOI-Backed Archive

Subject to source licenses and privacy review, archive:

- processed trajectory point and feature tables;
- canonical trajectory and split manifests;
- homography correspondence tables, matrices, and provenance checksums;
- scene polygons and exhaustive polygon-rule labels;
- EMD micro-region assignments and descriptors;
- SAC supernode memberships and descriptor/bootstrap tables;
- SMG edge counts and threshold tables;
- UATP bootstrap target distributions;
- candidate model-selection tables and frozen selections;
- independent cluster assignments and reference-evaluation tables;
- baselines, ablations, nuisance sensitivities, statistical tables, and figure sources.

Every archived file must have SHA-256, schema version, generation command, commit,
split, and reference-access status in the registry/manifest.

## Restricted or Conditional Assets

- Google-derived top-view rasters are not redistributed until rights and attribution
  metadata are resolved. Publish calibration coordinates, homography matrices, hashes,
  provenance, and author-created blank-coordinate or schematic alternatives instead.
- Raw videos are released only if the Traffic Node Video Dataset terms permit it.
- Local paths and hashes may document unavailable source assets without copying them.
- Annotation SQLite databases, credentials, caches, and local environments are never
  release artifacts.

## Reproducibility Boundary

The public package must distinguish regenerable outputs from source-restricted inputs.
The later HG-SMG-TC evaluation remains post-review; public preregistration timestamps
and hashes establish only that extension decisions were frozen before its first run.
