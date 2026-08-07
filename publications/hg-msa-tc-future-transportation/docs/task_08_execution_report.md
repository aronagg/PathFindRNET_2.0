# Task 08 Execution Report

## Execution summary

- Branch: `feature/futuretransp-homography-quality-sensitivity`
- Frozen matrix reproduction: exact for all five scenes
- Calibration correspondences: 101
- Jackknife runs: 101
- Perturbation runs: 500
- Quality-gate pass: 5/5
- Target propagation split: `target_estimation` only
- Frozen matrices/targets/configurations modified: no
- Independent-test clustering rerun: no

## Commands

```powershell
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_homography_analysis.py analyze --replicates 20
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_homography_reporting.py all
./.venv/Scripts/python.exe -m pytest publications/hg-msa-tc-future-transportation/tests -q
./.venv/Scripts/python.exe -m ruff check publications/hg-msa-tc-future-transportation/code publications/hg-msa-tc-future-transportation/tests
```

## Principal findings

- Exact reproduction confirms the final five-scene implementation is RANSAC with a
  10 px destination-space threshold.
- All scenes pass the diagnostic gate, but endpoint extrapolation is high.
- SE38th's 18 target persists under plausible small calibration perturbations.
- Newport exhibits a target-threshold discontinuity; Eastgate and NE8th have localized
  sensitivity that must remain visible in the paper.
- Google-derived top-view rasters are not included in the review package.

## Verification status

- Pytest: `117 passed in 86.68s`
- Ruff: `All checks passed!`
