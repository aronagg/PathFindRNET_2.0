# Task 07 Execution Report

## Execution summary

- Branch: `feature/futuretransp-target-estimation-formalization`
- Canonical module: `code/target_estimation/hg_target_estimator.py`
- Frozen targets reproduced: 10, 12, 9, 18, 9
- Reference labels read during reproduction: no
- Independent-test clustering rerun: no
- Frozen targets, thresholds, K values, and selected configurations changed: no
- Pytest: `108 passed in 79.02s`
- Ruff: `All checks passed`

## Exact commands

```powershell
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_target_estimation_analysis.py analyze
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_target_estimation_reporting.py figures
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_target_estimation_reporting.py documents
./.venv/Scripts/python.exe -m pytest publications/hg-msa-tc-future-transportation/tests -q
./.venv/Scripts/python.exe -m ruff check publications/hg-msa-tc-future-transportation/code publications/hg-msa-tc-future-transportation/tests
```

## Principal findings

- SE38th target error is primarily endpoint-region granularity failure, not a threshold-only problem.
- The 18 supported SE38th geometric OD pairs contain repeated and mixed manual semantics.
- Threshold sensitivity is moderate; region-count sensitivity is broad.
- Persisted SE38th KMeans assignments show manual movement fragmentation under `k=18`.
- The estimator should be described as geometric and heuristic, not as a guaranteed semantic counter.

## Limitations

- Five fixed scenes provide limited scene-level replication.
- Manual point-pair homographies retain calibration uncertainty.
- The human-defined polygon reference shares endpoint information with the maneuver problem.
- Region-to-approach mapping is diagnostic and was performed after target reproduction.
- Lane-level interpretation is plausible but unverified without lane semantics.
