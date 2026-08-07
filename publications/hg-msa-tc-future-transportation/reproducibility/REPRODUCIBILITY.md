# Reproducibility Guide

Task 09A freezes a protocol only. The supplied CLI validates hashes and lists planned
experiments; it deliberately cannot run HG-SMG-TC.

## Validate the Preregistration

From the repository root:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reproduction\cli.py validate-preregistration
```

Platform-neutral equivalent:

```bash
python publications/hg-msa-tc-future-transportation/code/reproduction/cli.py validate-preregistration
```

List the frozen registry or inspect one planned command:

```bash
python publications/hg-msa-tc-future-transportation/code/reproduction/cli.py list
python publications/hg-msa-tc-future-transportation/code/reproduction/cli.py plan --experiment hg_smg_A5
```

PowerShell wrapper:

```powershell
publications\hg-msa-tc-future-transportation\reproduce_publication.ps1 validate-preregistration
```

## Run Verification

```powershell
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
.\.venv\Scripts\python.exe -m ruff check publications\hg-msa-tc-future-transportation\code publications\hg-msa-tc-future-transportation\tests
```

## Environment

Use the repository virtual environment and retain exact package versions in the later
release lock. Relevant frozen historical facts include OpenCV 4.12 and scikit-learn
behavior captured by the prior result manifests. Task 09A adds no dependency.

## Scientific Guard

The extension implementation is intentionally absent. Future implementation must be a
new versioned task, verify this protocol hash, keep reference labels unavailable during
target estimation/model selection, and persist test assignments before separate
evaluation. Any amendment must create a new protocol file and hash; v1 is never
overwritten.
