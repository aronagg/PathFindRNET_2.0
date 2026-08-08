# Reproducibility

This file summarizes what can be reproduced from the publication package without
rerunning new scientific experiments.

## Recommended Environment

Use the repository root virtual environment if available, or create a fresh
environment from:

`publications/hg-msa-tc-future-transportation/environment/reproducibility_environment.yml`

## Lightweight Verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
.\.venv\Scripts\python.exe -m ruff check publications\hg-msa-tc-future-transportation\code publications\hg-msa-tc-future-transportation\tests
```

These checks validate code and table consistency. They do not run new clustering
experiments.

## Core Result Regeneration

The final synthesis tables and figures can be regenerated from persisted result
tables:

```powershell
.\publications\hg-msa-tc-future-transportation\scripts\reproduce_core_results.ps1
```

or, on POSIX-like shells:

```bash
bash publications/hg-msa-tc-future-transportation/scripts/reproduce_core_results.sh
```

The scripts rerun only the final synthesis layer. They do not recompute frozen
homographies, targets, clustering assignments, EMAS analyses, baselines, or
reference labels.

## Frozen Scientific Artifacts

The following artifacts are treated as frozen inputs:

- polygon-rule reference protocol and generated labels;
- homography matrices and calibration diagnostics;
- frozen target-estimation outputs;
- independent-test cluster assignments;
- HG-SMG-TC independent-test metrics;
- baseline assignment and metric tables;
- EMAS_HG formalization and sensitivity outputs.

## Non-Reproducible Without External Data

Raw video processing and full trajectory reconstruction require access to the
original Traffic Node Video Dataset files. Raw videos are not redistributed in
this publication package.
