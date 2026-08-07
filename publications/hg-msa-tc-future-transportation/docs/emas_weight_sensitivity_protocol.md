# EMAS_HG-v1 Weight-Sensitivity Protocol

## Scientific lock

Only the **255 model-selection candidate rows** from the development split are used
for ranking sensitivity. Frozen HG targets, candidate grids, model selections,
preprocessing, feature representation, independent-test assignments and reference
labels remain unchanged. Independent-test metrics are used only in the original-weight
reproduction check, not to choose or filter weights.

## Weight sets

- Seven named, pre-specified scenarios are stored in
  `configs/emas_weight_scenarios.yaml`.
- The local grid contains **465** feasible vectors at step `0.05`, with the requested
  component bounds and exact unit sum.
- The global sample contains **1,000** unique fixed-seed Dirichlet(1,1,1,1,1)
  vectors. It is a broad stress test, not a neighborhood of the original weights.

For every scene-method candidate set and vector, the analysis recomputes only the
weighted score. Ties are resolved deterministically by lexical parameter JSON and then
trial index. It records top-rank agreement, Spearman and Kendall correlation, top-three
overlap, candidate score ranges and first-second margins. No test-metric-guided weight
filtering occurs.

## Aggregate rank diagnostics

| family   |   weight_vectors |   mean_spearman |   median_spearman |   mean_kendall |   mean_top3_overlap |   top_rank_preservation_pct |
|:---------|-----------------:|----------------:|------------------:|---------------:|--------------------:|----------------------------:|
| global   |             1000 |          0.7956 |            0.9236 |         0.7156 |              0.7156 |                     69.0733 |
| local    |              465 |          0.9673 |            0.9882 |         0.9198 |              0.9130 |                     93.8781 |
