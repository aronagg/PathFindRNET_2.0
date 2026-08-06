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
