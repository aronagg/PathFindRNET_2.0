# Reproducibility Guide

Task 09A freezes the protocol. Task 09B implements and freezes the extension on
`target_estimation` and `model_selection` only. The independent-test guard remains
active and the future test command always refuses execution in Task 09B.

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

## Reproduce the Development Freeze

From the repository root:

```powershell
$env:PYTHONPATH = 'publications/hg-msa-tc-future-transportation/code'
./.venv/Scripts/python.exe -m hg_smg.cli preflight
./.venv/Scripts/python.exe -m hg_smg.cli full-split --jobs 3
./.venv/Scripts/python.exe -m hg_smg.cli uatp --replicates 500 --sac-replicates 500 --replicate-jobs 10 --variants A5,A6,A7,A9
./.venv/Scripts/python.exe -m hg_smg.cli pcms
./.venv/Scripts/python.exe -m hg_smg.cli determinism --replicates 500 --sac-replicates 500 --replicate-jobs 10
./.venv/Scripts/python.exe -m hg_smg.cli report
```

The PowerShell/platform-neutral reproduction entrypoint also exposes explicitly
confirmed development stages:

```powershell
publications/hg-msa-tc-future-transportation/reproduce_publication.ps1 hg-smg-development -Stage preflight -ConfirmDevelopmentOnly I_CONFIRM_DEVELOPMENT_SPLITS_ONLY
python publications/hg-msa-tc-future-transportation/code/reproduction/cli.py hg-smg-development --stage full-split --confirm-development-only I_CONFIRM_DEVELOPMENT_SPLITS_ONLY
```

The frozen protocol hash is
`2e892f25b1c1770a55fa29e36206b634f424f9ff2508bf045c92d388e69b4be6`;
the Task 09B development-freeze hash is
`c86830ea99b331fa8322915c33425398ac3edee76d242e0915f05c7c5ff0a14a`.

## Environment

Use the repository virtual environment and retain exact package versions in the later
release lock. Relevant frozen historical facts include OpenCV 4.12 and scikit-learn
behavior captured by the prior result manifests. Task 09A adds no dependency.

## Scientific Guard

Task 09B reads only development cohorts and frozen method inputs. It does not import
manual scene-guide or polygon-reference modules. A8 remains non-identifiable because
protocol v1 omits a JSD compatibility threshold. Any correction must create a new
versioned protocol and hash; v1 is never overwritten. The future independent-test
runner requires a separate task and gate and continues to refuse even when given the
reserved confirmation phrase.
