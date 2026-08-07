# Independent-Test Result Manifest Contract

`clustering_run_manifest.json` is written only after the CSV and Parquet assignments
and clustering data-access log exist. It records frozen configuration and unlock
hashes, run identity, persistence time, 30 run configurations, feature checksums,
cluster/noise counts, and explicit declarations that references were not read and no
target, hyperparameter, or normalization was recomputed.

`clustering_run_checksums.sha256` covers the CSV, Parquet, clustering manifest, and
data-access log. The clustering manifest is immutable during later evaluation.

`evaluation_run_manifest.json` separately records assignment/reference checksums,
the order of assignment and reference access, the valid-reference filter, noise and
Hungarian mapping policies, EMAS_HG's non-independent diagnostic role, and hashes of
all primary metric tables. EMAS_HG was not a model-selection key. Reference
sensitivity is a separate output and cannot overwrite the primary metric checksums.

Required assignment identity is `(scene_id, trajectory_id, method,
selection_strategy)`. Exactly 164,358 rows are expected: 27,393 test trajectories
times three methods times two frozen strategies.
