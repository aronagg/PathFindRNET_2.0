# Locked Independent-Test Execution Protocol

## Scientific sequence

1. Validate every frozen development hash without loading independent-test features
   or reference rows.
2. Create the one-time versioned authorization file.
3. Require the explicit `--confirm-independent-test-evaluation` flag.
4. Load only `independent_test` camera endpoint features.
5. Apply the model-selection-fitted isotropic normalization parameters.
6. Fit every frozen KMeans, HDBSCAN, and OPTICS configuration for both selection
   strategies without recomputing any target, grid, threshold, or hyperparameter.
7. Persist and checksum all assignments.
8. In a separate command, load the persisted assignments before reading the
   independent polygon-rule reference.
9. Evaluate valid reference rows and report excluded-reference coverage.

The clustering module has no reference-label dependency. The output directory is
single-use and must not exist before the authorized run.

## Commands

Run from the repository root:

```powershell
# Hash-only preflight; no real test feature or label row is loaded
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py preflight

# One-time authorization
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py unlock

# First real transductive independent-test clustering
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py cluster --confirm-independent-test-evaluation

# Separate reference evaluation
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py evaluate

# Frozen-reference sensitivity, figures, and reports
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py sensitivity
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py figures
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py reports

# Validation
.\.venv\Scripts\python.exe -m ruff check publications\hg-msa-tc-future-transportation\code\independent_test publications\hg-msa-tc-future-transportation\tests\test_independent_test_evaluation.py
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
```

## Evaluation semantics

ARI, NMI, purity, homogeneity, completeness, and V-measure retain density-method
noise label `-1` as part of the predicted partition. Hungarian mapping is computed
separately per scene, method, and strategy using only non-noise clusters. It maximizes
one-to-one overlap with observed movement classes. Unmatched clusters predict an
explicit unmatched token; unmatched movements receive zero recall. This prevents a
large noise set from being rewarded as a maneuver class.

Observed independent-test movement count is the primary count comparator. The 12
human-defined legal movements are secondary context. The frozen HG target is reported
unchanged even where it differs strongly, especially SE38th.
