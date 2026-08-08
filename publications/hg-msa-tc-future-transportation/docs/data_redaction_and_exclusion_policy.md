# Data Redaction and Exclusion Policy

## Excluded by Default

- raw traffic videos;
- Google-derived or third-party map raster images;
- local annotation SQLite databases;
- prior task ZIP packages and sidecar hashes;
- virtual environments, caches and build outputs;
- large trajectory-level exports that are reproducible from scripts and source
  data.

## Redaction Principles

1. Prefer compact summaries over trajectory-level exports when full data are too
   large or restricted.
2. Preserve checksums and manifests so excluded local files can be verified.
3. Do not include machine-specific absolute paths in public scientific tables.
4. Keep raw data access separate from manuscript reproducibility claims.
5. Document every excluded class of artifact in `DATA_AVAILABILITY.md` and the
   DOI archive readme.

## Reference Labels

Polygon-rule reference labels are derived artifacts. They should be released only
if consistent with the underlying dataset license and institutional policy. If
full labels cannot be released, provide the protocol, checksums, compact coverage
tables and generation scripts.

## Homography Assets

Calibration point correspondences and numeric residuals are preferred for public
release. Map rasters or Google-derived top-view images should be excluded unless
redistribution is explicitly permitted.
