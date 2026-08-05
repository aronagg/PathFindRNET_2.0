# Frozen Canonical Cohort Limitation

The split-aware runner uses the existing canonical rows in:

`data/processed/<scene>/feature_analysis/features_trimmed_frame_disp_norm.parquet`.

Before the revision split existed, this cohort was constructed using scene-level
frame-span quantiles from the complete scene and a normalized displacement filter. The
quantiles therefore include unlabeled distributions from time periods that now belong
to target estimation, model selection, and independent test.

This preprocessing did not use manual maneuver labels, annotation-derived targets, or
test evaluation metrics. It nevertheless means that the entire preprocessing chain is
not strictly inductive or fully leakage-free. It is a transductive preprocessing
limitation and must remain visible in the revision.

The task does not redesign preprocessing because doing so would change the frozen
canonical cohort and exceed the agreed scope. A later sensitivity analysis should fit
frame-span and displacement thresholds on development recordings only, apply those
fixed thresholds to the locked test recordings, and compare retention and final
external metrics with the current cohort.

The feature-layer manifests name
`data/processed/<scene>/trajectories_filtered_filled.parquet` as the cleaned parent for
each scene. Those five active parent files are missing. Legacy copies cannot be assumed
byte-identical. The present final feature files are available and checksummed, but
byte-level full lineage back to the missing point-level parents cannot be claimed.
