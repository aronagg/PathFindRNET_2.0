# HG-MSA-TC Future Transportation Evaluation Protocol

This publication-specific directory contains the data audit, canonical trajectory
manifest, leakage-controlled evaluation split, split-aware HG-MSA-TC runner,
blind manual-annotation workflow, and validation tests for the Future Transportation major revision.
It does not change the repository's trajectory preprocessing, homography calibration,
prior full-data results, figures, or manuscript.

The protocol covers exactly these five scenes:

- `bellevue_116th_ne12th`
- `bellevue_150th_newport`
- `bellevue_150th_eastgate`
- `bellevue_150th_se38th`
- `bellevue_ne8th`

## Leakage Prevention

**The manually annotated labels are isolated from homography-guided target estimation
and clustering model selection. They are used only for independent final evaluation
and explicitly identified post hoc diagnostic analyses.**

The three subsets have fixed roles:

| Subset | Permitted role | Manual labels available to method code |
| --- | --- | --- |
| `target_estimation` | Estimate the homography-guided observed maneuver target | No |
| `model_selection` | Select candidate parameters using the frozen target | No |
| `independent_test` | Final evaluation and explicitly marked diagnostics | Only after all decisions are frozen |

No independent-test label may affect preprocessing choices, coordinate normalization
choices, target estimation, candidate grids, hyperparameter selection, stopping,
claim selection, or figure selection.

## Canonical Cohort

The manifest uses the exact one-row-per-trajectory inputs consumed by the completed
five-scene HG-MSA-TC experiments:

```text
data/processed/<scene>/feature_analysis/features_trimmed_frame_disp_norm.parquet
```

These files contain 67,029 pipeline-valid trajectories. The manifest does not copy
trajectory arrays or modify source data. Original merged `track_id` values are
preserved and namespaced by scene in `trajectory_id`.

The merged feature files do not retain `video_id`. The generator reconstructs source
recording provenance from the verified deterministic offset rule in
`scripts/merge_tracks.py` and checks that every final trajectory maps to exactly one
per-recording shard. Recording timestamps come from source filenames; frame time uses
the configured 30 fps.

## Generate the Manifest

From the repository root:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\data\build_trajectory_manifest.py
```

Output:

```text
publications/hg-msa-tc-future-transportation/data/manifests/trajectory_manifest.csv
```

The script is read-only with respect to `data/`. It computes source checksums,
deterministic IDs, trajectory fingerprints, exact duplicate groups, and conservative
near-duplicate candidates.

The fingerprint hashes scene, frame range, point count, camera endpoints, path length,
displacement, and straightness. The near-duplicate flag uses narrow quantized bins over
the same structural fields and is a review signal, not a deletion rule.

## Generate the Evaluation Split

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\data\build_evaluation_split.py
```

Outputs:

- `data/splits/evaluation_split.csv`
- `configs/evaluation_split.yaml`
- `docs/evaluation_split_report.md`
- `annotations/annotation_template.csv`

The split is chronological within scene and uses complete hourly recordings as its
atomic units. The requested 30/30/40 ratios are optimized at recording boundaries;
ordinary random row splitting is not used. Exact duplicate groups are forced into one
subset. Approximate near-duplicates are reported and never removed automatically.

## Run Validation

```powershell
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
```

The tests cover deterministic IDs and split assignment, required schemas and scene
membership, split proportions, ordering, duplicate isolation, annotation-label
emptiness, and source-file checksum preservation.

The latest executed result is recorded in `docs/validation_results.md`.

## Run the Split-Aware Development Protocol

The development phases are technically separate:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py benchmark
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py target
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py select
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py freeze
```

The target phase uses only `target_estimation`. The select phase uses only
`model_selection` and the frozen target table. The freeze phase records the selected
configurations, data and configuration checksums, software versions, random seeds,
candidate grids, selection rules, and the locked status of `independent_test`.

The final test protocol is transductive for all three methods. The algorithm and all
hyperparameters remain frozen, but KMeans, HDBSCAN, and OPTICS are each fitted on the
independent-test feature vectors without labels. Cluster assignments are persisted
and checksummed before a separate evaluation step may read manual labels. The real
test command refuses to run without an external unlock file and explicit confirmation.

The generated trajectory-level CSV files are intentionally ignored by Git because the
manifest and split are about 50 MB and 27 MB respectively. Recreate them from the
versioned scripts/configuration instead of committing them as dataset content.

## Directory Contents

```text
annotations/   annotation schema and empty label template
code/data/     read-only manifest and split generators
code/pipeline/ split-aware HG-MSA-TC implementation and data guards
configs/       versioned split strategy, boundaries, and checksums
data/          generated lightweight CSV manifests only
docs/          repository/data audit and split audit
results/       development-only target, candidate, and frozen protocol artifacts
tests/         protocol validation
```

## Blind Manual Annotation

The publication-specific `code/annotation_app/` package provides a Streamlit UI,
deterministic recording-stratified pilot and primary queues, append-only SQLite
storage, agreement analysis, separate adjudication, and post-consensus inventory
generation. Install the optional dependencies and inspect exact commands in
`docs/annotation_user_guide.md`.

The independent-test queues contain all 27,393 trajectories twice, once for each
independent annotator in a different deterministic order. They remain locked until an
authorized protocol designer manually configures all five scene guides and freezes
the annotation protocol. No real human labels are included.

**The annotation application renders only source trajectory geometry and source
imagery. It does not access homography-derived targets, automatic OD assignments,
clustering outputs, pseudo-reference labels, or model-selection metrics. The
scientific model-selection protocol was frozen before manual labels are collected.**

## Intentionally Not Implemented

- manual maneuver labels;
- manual maneuver annotation;
- real independent-test clustering or evaluation;
- manuscript or final-claim regeneration;
- split-fitted preprocessing sensitivity analysis.

## Exhaustive Polygon-Rule Reference Labels

The five completed scene guides are frozen separately as
`annotations/protocol/polygon_reference_protocol_v1.yaml`. The reference-label
generator applies their manually specified entry/exit polygons and legal movement
mappings to every canonical trajectory. It uses the first and last finite point in
the canonical interval and Shapely `covers`; no nearest-polygon fallback is used.

The result is an **exhaustive human-defined polygon-rule-based reference**, not a
claim of fully independent per-trajectory manual ground truth. All 67,029 rows are
retained, including unassigned and ambiguous cases. The future scientific evaluator
may use only valid rows from `independent_test_reference_labels.csv`; generation of
labels for the two development splits is diagnostic and does not reopen frozen model
selection.

Run the stages from the repository root:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py freeze
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py generate
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py inventories
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py sensitivity
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py figures
```

`cli.py all` runs generation and all downstream summaries after the protocol has
already been frozen. See `docs/polygon_reference_generation_protocol.md` for the
schema, status rules, scientific isolation constraints, and reproducibility details.

## Locked Independent-Test Evaluation

The independent-test runner is a separate, single-use workflow. It validates all
frozen development hashes before test-feature access, requires the versioned unlock
file and an explicit confirmation flag, runs both frozen selection strategies for
KMeans, HDBSCAN, and OPTICS, and persists assignments before any reference row is
loaded. Clustering and reference evaluation are separate CLI commands.

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py preflight
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py unlock
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py cluster --confirm-independent-test-evaluation
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py evaluate
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py sensitivity
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py figures
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py reports
```

The frozen HG targets and selected configurations are never recomputed. Independent
metrics use only valid polygon-reference rows from `independent_test`. EMAS_HG is
reported as the frozen task-specific diagnostic composite, not as a selection key or
independent validation metric.

## EMAS_HG-v1 Formalization and Sensitivity

The canonical score implementation is `code/metrics/emas_hg.py`. It exactly reproduces
the stored frozen scores and explicitly defines component transformations, clipping,
noise handling, missing-metric fallbacks and invalid inputs. The audit confirms that
EMAS_HG was reported after candidate evaluation but did not participate in either
frozen model-selection key.

Weight sensitivity therefore concerns development-candidate ranking, not re-selection.
It uses seven pre-specified scenarios, a deterministic 465-vector local grid and 1,000
fixed-seed global simplex vectors. It does not read independent reference labels for
weight analysis and does not rerun independent-test clustering.

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_emas_sensitivity.py analyze
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_emas_reporting.py figures
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_emas_reporting.py documents
```

See `docs/emas_hg_mathematical_definition.md`,
`docs/emas_weight_sensitivity_scientific_interpretation.md` and
`docs/emas_result_manifest.md`.

## Frozen HG Target Formalization and Failure Analysis

The canonical target estimator is
`code/target_estimation/hg_target_estimator.py`. It reproduces the frozen targets
`10, 12, 9, 18, 9` and the complete threshold, endpoint-region, and OD-support tables
from `target_estimation` only. The human polygon reference is loaded only after this
reproduction is persisted and locked, and then only for diagnostic mapping.

The analysis formalizes the separate entry/exit median centers, circular endpoint
features, KMeans region-count rule, OD support, and data-dependent threshold heuristic.
It also documents the SE38th failure: the frozen estimator selected seven entry and
three exit regions, yielding 18 supported geometric OD pairs versus nine independently
observed semantic movements. This result is retained without post hoc correction.

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_target_estimation_analysis.py analyze
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_target_estimation_reporting.py figures
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_target_estimation_reporting.py documents
```

See `docs/hg_target_estimator_mathematical_definition.md`,
`docs/se38th_target_failure_analysis.md`, and
`docs/target_estimation_result_manifest.md`. The sensitivity analyses are diagnostic;
they do not replace frozen thresholds, endpoint K values, targets, or cluster selections.
