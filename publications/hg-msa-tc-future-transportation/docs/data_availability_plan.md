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
