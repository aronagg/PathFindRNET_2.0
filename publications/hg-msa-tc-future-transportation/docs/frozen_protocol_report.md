# Frozen Split-Aware HG-MSA-TC Protocol

- Protocol version: `future-transportation-split-aware-v1`
- Git implementation commit: `0574efa38e4fcbe8e77fb054ec63c962170ebf23`
- Frozen configuration hash: `829c95c4f0a012d08433536b22379afad4d5a79122e52f8fd6ac12817882167c`
- Independent test locked: **yes**
- Target rows were read only from `target_estimation`.
- Candidate selection rows were read only from `model_selection`.
- No manual labels were read by either phase.
- The real `independent_test` feature cohort was not loaded.

## Final Test Design

The later final evaluation is transductive. Each frozen method and every selected
hyperparameter will be kept fixed, then the clustering method will be fitted on the
independent-test feature vectors without labels. Cluster assignments must be written
and checksummed before a separate evaluation step may read manual labels. KMeans,
HDBSCAN, and OPTICS therefore follow one consistent test protocol.

## Frozen Artifacts

- `publications/hg-msa-tc-future-transportation/configs/frozen_evaluation_protocol.yaml`
- `publications/hg-msa-tc-future-transportation/results/development/frozen_selection_manifest.json`
- `publications/hg-msa-tc-future-transportation/results/development/target_estimates.csv`
- `publications/hg-msa-tc-future-transportation/results/development/selected_configurations.csv`
