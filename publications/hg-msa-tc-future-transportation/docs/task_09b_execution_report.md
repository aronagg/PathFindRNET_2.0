# Task 09B Execution Report

## Scope and freeze

- Branch: `feature/futuretransp-hg-smg-development`.
- Required base: `7ffcbca365122a7acbf5232d515d829f1e27b8bd`.
- Scientific code commit: `55bab42853ce341c75b31404ed39eb34e9841c6b`.
- Task-09A protocol SHA-256: `2e892f25b1c1770a55fa29e36206b634f424f9ff2508bf045c92d388e69b4be6`.
- Development-freeze SHA-256: `c86830ea99b331fa8322915c33425398ac3edee76d242e0915f05c7c5ff0a14a`.
- Implementation: `hg-smg-tc-development-v1`; master seed: `20260901`.
- Permitted splits used: `target_estimation`, `model_selection`.
- Independent-test/reference access: none.

## EMD reproduction and primary SAC/SMG

Frozen EMD targets `10, 12, 9, 18, 9`, entry/exit region counts, thresholds, and all trajectory-level region assignments reproduced exactly before SAC execution.

| Scene | Entry micro-regions | Exit micro-regions | Entry supernodes | Exit supernodes | A5 K_SMG | Threshold | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bellevue_116th_ne12th | 4 | 4 | 4 | 4 | 10 | 0.0025 | 100.00% |
| bellevue_150th_newport | 4 | 4 | 3 | 4 | 8 | 0.0010 | 100.00% |
| bellevue_150th_eastgate | 4 | 4 | 4 | 4 | 9 | 0.0050 | 99.43% |
| bellevue_150th_se38th | 7 | 3 | 5 | 2 | 10 | 0.0010 | 100.00% |
| bellevue_ne8th | 4 | 4 | 4 | 4 | 9 | 0.0050 | 99.75% |

Primary A5 merges occurred only where the complete-link compatibility rule permitted them. No primary scene collapsed to one entry/exit node, no primary scene had zero supported SMG edges, and no material undefined-heading or numerical-floor condition was detected.

## UATP

The official table has 10,000 rows: five scenes x four executable UATP variants x 500 hierarchical recording-aware replicates. Every replicate reran EMD -> SAC -> SMG with 500 within-region SAC bootstraps. There were zero failed replicates.

| Scene | A5 full K | Mode | Median | 90% interval | Entropy (bits) | P(full K) |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| bellevue_116th_ne12th | 10 | 10 | 10 | [8, 13] | 1.8282 | 68.6% |
| bellevue_150th_newport | 8 | 8 | 9 | [7, 13] | 2.7617 | 32.2% |
| bellevue_150th_eastgate | 9 | 9 | 9 | [7, 11] | 2.1688 | 41.8% |
| bellevue_150th_se38th | 10 | 10 | 10 | [8, 20] | 3.4542 | 26.2% |
| bellevue_ne8th | 9 | 9 | 9 | [9, 10] | 1.2207 | 51.6% |

SE38th has the broadest interval and highest entropy. This is a development-time uncertainty diagnosis, not evidence of semantic correctness.

## PCMS

PCMS evaluated only the existing frozen model-selection grids: 29 KMeans, 6 HDBSCAN, and 16 OPTICS candidates per scene. It selected 15 scene-method configurations using interval distance and the preregistered method-specific internal-metric tie breaks. EMAS_HG was diagnostic only and did not enter the key. Exact parameters are in `pcms_selected_configurations.csv` and `docs/hg_smg_pcms_development_report.md`.

## Ablations and sensitivities

- A0/A1: frozen original outputs reused; no scientific recomputation.
- A2/A3/A4/A5/A6/A7/A9/A10: completed on development splits.
- A7 heading-only: strong structural collapse (`K_SMG` 1-4), demonstrating that heading alone does not preserve the preregistered approach-side structure.
- Heading windows 3/5/7: full-split A5 target unchanged in every scene.
- Self-consistency q=0.90/0.95/0.975: Newport and SE38th changed; the other scenes were unchanged.
- Camera-isotropic versus top-view heading: full-split target unchanged in every scene.
- UATP 80/90/95% intervals: reported without replacing the primary 90% prior.
- A8: not identifiable because protocol v1 defines JSD but no executable compatibility threshold. No value was invented.

## Determinism and validity

The complete primary development pipeline was run twice from clean output directories. SHA-256 hashes matched for all six designated artifacts: SAC supernode assignments, SMG edges, the complete UATP sequence, UATP summary, PCMS selections, and PCMS provenance. Primary A5 passed all implemented degeneracy checks. A8 requires a versioned amendment before affected test execution.

## Exact commands

Run from the publication workspace:

```powershell
$env:PYTHONPATH = 'code'
../../.venv/Scripts/python.exe -m hg_smg.cli preflight
../../.venv/Scripts/python.exe -m hg_smg.cli full-split --jobs 3
../../.venv/Scripts/python.exe -m hg_smg.cli uatp --replicates 500 --sac-replicates 500 --replicate-jobs 10 --variants A5,A6,A7,A9
../../.venv/Scripts/python.exe -m hg_smg.cli pcms
../../.venv/Scripts/python.exe -m hg_smg.cli determinism --replicates 500 --sac-replicates 500 --replicate-jobs 10
../../.venv/Scripts/python.exe -m hg_smg.cli report
../../.venv/Scripts/python.exe -m pytest -q tests
../../.venv/Scripts/python.exe -m ruff check code tests
```

## Scientific guard

No independent-test feature, assignment, ARI/NMI/purity/F1 result, movement inventory, polygon-reference label, manual scene guide, or semantic tuning signal was read. No HG-SMG independent-test execution occurred, and no unlock file was created.
