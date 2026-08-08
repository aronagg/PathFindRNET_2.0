# Data Availability Plan

The release package should provide the frozen polygon protocol and checksums,
generator and evaluation code, legal and observed movement inventories, coverage and
sensitivity summaries, QC figures, and the compressed all-trajectory Parquet file.
The uncompressed CSV and split CSV exports remain reproducible local artifacts when
size limits prevent direct distribution; their paths, row counts, and SHA-256 hashes
are recorded in `annotations/reference_labels/reference_output_manifest.json`.

Raw traffic videos and the repository's large processed trajectory shards are not
duplicated in the revision package. Access remains subject to the Traffic Node Video
Dataset distribution conditions. The package must not include annotation SQLite
databases or independent-test clustering outputs.

After the authorized final evaluation, the compact release should also contain the
independent-test run/evaluation manifests, checksums, aggregate and per-movement
metrics, target-validation table, sensitivity results, and figures. The full
assignment CSV is a reproducible large artifact; compressed Parquet plus its checksum
is sufficient when package limits require omitting the CSV. The assignment table
contains trajectory IDs and cluster labels, not raw images or videos.

The EMAS_HG revision package should include the canonical metric source, frozen-score
reproduction table, pre-specified weight configuration, local/global weight vectors,
development-only ranking summaries, figures and scientific documentation. It should
not duplicate independent-test assignments or reference-label exports because Task 06
only records their hashes and does not rerun or retune the test workflow.

The target-estimation formalization package should include the canonical estimator,
exact reproduction table, frozen scene-parameter table, compact endpoint/OD diagnostic
tables, threshold and region-count sensitivity results, source tests, figures, and
scientific reports. It should include checksums rather than duplicate the large source
feature files, polygon-reference exports, or independent-test assignment table. The
SE38th fragmentation analysis is reproducible from persisted assignments and source
trajectory shards but does not distribute raw videos or trajectory shards. Human
reference data are diagnostic-only inputs loaded after frozen-target reproduction.


The homography-quality revision package should include canonical calibration/quality
source, the frozen correspondence table, matrices, compact quality and sensitivity
tables, generated schematic/camera diagnostic figures, tests, manifests and reports.
It must not duplicate raw video, trajectory data, independent-test assignments, or
Google-derived top-view raster images. Source-image paths and hashes preserve local
auditability without asserting redistribution permission.

The HG-SMG-TC preregistration package adds only protocols, literature/novelty
positioning, reproducibility manifests, validation code, and tests. It contains no new
scientific result, independent-test label, assignment, or metric. A later DOI-backed
release should include processed trajectory tables, EMD micro-regions, SAC supernode
memberships, SMG edges, UATP bootstrap distributions, PCMS candidates/selections,
assignments, ablations, sensitivity tables, and figure-source data with SHA-256 and
split/access metadata. Google-derived top-view rasters remain excluded until
redistribution rights and attribution are resolved; calibration coordinates, matrices,
checksums, and author-created schematic alternatives are publishable substitutes.

The Task 09B development package adds the canonical HG-SMG source, exact EMD
reproduction, SAC descriptor/compatibility/merge tables, SMG edges, the complete
10,000-row UATP development sequence, UATP summaries, PCMS candidate and selection
tables, development-only diagnostic figures, tests, and the development-freeze
manifest. It excludes raw videos, source trajectory shards, Google-derived rasters,
manual scene guides, polygon-reference exports, independent-test assignments, and
independent-test metrics. Repository-relative identifiers and SHA-256 manifests are
used in place of machine-specific scientific paths.
