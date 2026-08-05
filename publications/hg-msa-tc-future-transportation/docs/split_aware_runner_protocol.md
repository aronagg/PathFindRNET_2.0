# Split-Aware HG-MSA-TC Protocol

## Scientific Isolation Rule

The homography-derived observed maneuver target is estimated exclusively from the
target-estimation subset. Clustering configurations are selected exclusively from the
model-selection subset. The independent-test subset is not accessed until all targets,
hyperparameters, selection rules and software configurations have been frozen. Manual
labels are used only after test clustering assignments have been generated and
persisted.

Historical declared target values are not method inputs. Prior full-data selected
configurations are not method inputs. The only target passed to model selection is the
output produced by the current protocol's target phase.

## Phase A: Target Estimation

The target phase admits only rows marked `target_estimation`. Camera start and end
points are transformed with the scene's existing calibrated homography. Entry and exit
regions are selected independently by polar endpoint KMeans over candidate region
counts 3 through 8. Every region-count candidate is retained with silhouette and
Davies-Bouldin metrics.

Observed OD combinations are evaluated at support thresholds 0.1%, 0.25%, 0.5%, 1%,
and 2%. The selected threshold prefers at least 90% OD coverage and a target of at
least two, then minimizes adjacent-threshold target instability. Remaining ties use
distance to 0.5% support, higher coverage, and lower threshold. All target-estimation
trajectories are fitted; no 6000-row cap is used.

## Phase B: Model Selection

The select phase admits only rows marked `model_selection` and reads the frozen target
table from Phase A. Camera endpoint features are ordered as `start_x`, `start_y`,
`end_x`, `end_y`. Isotropic shared-scale normalization parameters are fitted only on
the scene's model-selection rows and recorded for later test application.

The submitted candidate grids are preserved:

- KMeans: cluster counts 2 through 30, `n_init=10`, `max_iter=300`;
- HDBSCAN: minimum cluster sizes 80, 160, 320 crossed with minimum samples 10, 20;
- OPTICS: minimum samples 40, 80; `xi` 0.05, 0.07; and four `max_eps` values derived
  from 50%, 70%, 85%, and 95% nearest-neighbor distance quantiles.

OPTICS quantiles are calculated only from the model-selection feature matrix. The
untargeted rule uses internal metrics only. The HG-aware rule first minimizes error
against the fixed HG target and, for density methods, then minimizes outliers before
using internal metrics. Lexical parameter order is the final deterministic tie-breaker;
runtime is never used for model selection.

## Phase C: Freeze

The freeze phase writes a frozen YAML protocol and JSON manifest containing the Git
implementation commit, input and output checksums, all targets, all selected
configurations, candidate grids, rules, seeds, software versions, feature order,
normalization parameters, sampling policy, and a complete canonical configuration
hash. An existing frozen protocol is validated and not overwritten.

Target or selection execution after freezing is rejected. A later protocol requires a
different protocol version and a different output directory.

## Phase D: Transductive Test

The real test path is implemented but locked. It requires a valid frozen protocol,
matching hashes, an externally created `configs/INDEPENDENT_TEST_UNLOCK.json`, and the
explicit confirmation `I_CONFIRM_FROZEN_TRANSDUCTIVE_TEST`.

After authorization, each frozen KMeans, HDBSCAN, and OPTICS configuration is fitted
transductively on the independent-test feature vectors without labels. The model type,
hyperparameters, feature order, coordinate representation, model-selection-fitted
normalization parameters, and random seeds remain fixed. Assignments are written and
checksummed before any separate evaluation code may read manual labels. This avoids an
inconsistent protocol in which KMeans predicts from development centroids while OPTICS
is refitted on test data.

## Access Enforcement

Each phase verifies the unique observed split value before features enter scientific
code. Every input is recorded in `results/development/data_access_log.jsonl` with
phase, scene, path, allowed split, observed split values, row count, checksum, and
timestamp. Target and select reject annotation-like input paths explicitly.

The split and trajectory manifests are control metadata used to identify authorized
rows. Feature tables are queried by authorized trajectory IDs and the returned cohort
is joined back one-to-one to the phase index before use.
