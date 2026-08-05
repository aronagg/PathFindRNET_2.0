# Task 02 Execution Report: Split-Aware HG-MSA-TC Runner

## Scope and Branch

- Branch: `feature/futuretransp-split-aware-runner`
- Base commit: `1c82be2863679b021d35abfc932d4a7247554959`
- Implementation commit used by the frozen protocol: `0574efa`
- Protocol version: `future-transportation-split-aware-v1`
- Frozen configuration SHA-256:
  `829c95c4f0a012d08433536b22379afad4d5a79122e52f8fd6ac12817882167c`

The task implemented and executed only development target estimation and model
selection. It did not run real independent-test clustering, annotation, or manual-label
evaluation.

## Development Target Estimates

| Scene | Entry regions | Exit regions | Support | Absolute support | HG target | OD coverage | Trajectories |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bellevue_116th_ne12th | 4 | 4 | 0.25% | 2 | 10 | 100.00% | 723 |
| bellevue_150th_newport | 4 | 4 | 0.10% | 3 | 12 | 100.00% | 2,423 |
| bellevue_150th_eastgate | 4 | 4 | 0.50% | 44 | 9 | 99.43% | 8,720 |
| bellevue_150th_se38th | 7 | 3 | 0.25% | 7 | 18 | 99.76% | 2,482 |
| bellevue_ne8th | 4 | 4 | 0.50% | 28 | 9 | 99.75% | 5,576 |

The SE38th and NE8th development targets differ from former complete-cohort values.
They were not manually adjusted because doing so would reintroduce information from
outside the authorized target-estimation subset.

## Frozen Development Configurations

Exact metrics, normalization parameters, seeds, selection keys, and parameter JSON
are stored in `results/development/selected_configurations.csv`. The compact frozen
configuration comparison is:

| Scene | Method | Untargeted: clusters / parameters | HG-aware: clusters / parameters |
| --- | --- | --- | --- |
| 116th NE12th | KMeans | 10 / `k=10` | 10 / `k=10` |
| 116th NE12th | HDBSCAN | 4 / `mcs=80, ms=10` | 4 / `mcs=80, ms=10` |
| 116th NE12th | OPTICS | 3 / `eps=0.028529, ms=40, xi=0.05` | 5 / `eps=0.318739, ms=40, xi=0.05` |
| 150th Newport | KMeans | 7 / `k=7` | 12 / `k=12` |
| 150th Newport | HDBSCAN | 6 / `mcs=80, ms=20` | 6 / `mcs=80, ms=10` |
| 150th Newport | OPTICS | 4 / `eps=0.012798, ms=80, xi=0.05` | 7 / `eps=0.036344, ms=40, xi=0.07` |
| 150th Eastgate | KMeans | 11 / `k=11` | 9 / `k=9` |
| 150th Eastgate | HDBSCAN | 12 / `mcs=160, ms=10` | 10 / `mcs=320, ms=20` |
| 150th Eastgate | OPTICS | 11 / `eps=0.039122, ms=40, xi=0.07` | 10 / `eps=0.022171, ms=80, xi=0.07` |
| 150th SE38th | KMeans | 12 / `k=12` | 18 / `k=18` |
| 150th SE38th | HDBSCAN | 3 / `mcs=160, ms=20` | 7 / `mcs=80, ms=10` |
| 150th SE38th | OPTICS | 3 / `eps=0.020337, ms=80, xi=0.05` | 12 / `eps=0.096239, ms=40, xi=0.05` |
| NE8th | KMeans | 10 / `k=10` | 9 / `k=9` |
| NE8th | HDBSCAN | 14 / `mcs=80, ms=20` | 9 / `mcs=160, ms=10` |
| NE8th | OPTICS | 10 / `eps=0.044077, ms=40, xi=0.07` | 9 / `eps=0.044077, ms=80, xi=0.05` |

`mcs` means HDBSCAN `min_cluster_size`; `ms` means `min_samples`; `eps` means
OPTICS `max_eps`.

## Development-Only Comparison

| Strategy | Mean target error | Mean outlier ratio | Mean EMAS_HG |
| --- | ---: | ---: | ---: |
| Untargeted | 5.4667 | 18.7891% | 0.6908 |
| HG expected-aware | 2.7333 | 10.0279% | 0.8114 |

Target error improved in 12 of 15 scene-method comparisons, was unchanged in 3, and
worsened in 0. These are development model-selection diagnostics, not independent
test evidence. In particular, HDBSCAN and OPTICS retain large target errors for
SE38th, and high outlier ratios remain for some OPTICS configurations.

## Sampling Decision

No trajectory-level fit sampling was used. Target estimation fitted all 19,924
authorized target-estimation trajectories. Candidate fitting used all 19,712
authorized model-selection trajectories. The 3000-row cluster metric sample and
2500-row endpoint-region metric sample affect metric calculation only.

The largest-cohort benchmark measured 1.3448 s for KMeans, 0.1381 s for HDBSCAN, and
3.3443 s for OPTICS on 6791 Eastgate model-selection trajectories. The former arbitrary
6000-row cap was therefore removed. No `sampled_trajectory_ids.csv` was created.

## Access and Lock Verification

- Access log: 7 target records and 7 select records.
- Observed target split values: only `target_estimation`.
- Observed select split values: only `model_selection`.
- Test access records: none.
- Real test output directory: absent.
- Independent-test unlock file: absent.
- Manual annotations read by target/select: no.
- Candidate fits: 255 total, 0 failed.
- Synthetic transductive assignment smoke test: passed.
- Real test command with explicit CLI confirmation: refused because the unlock file is
  absent, before test feature loading.
- Target rerun after freeze: refused before feature loading.

## Commands Executed

```powershell
.\.venv\Scripts\python.exe -m py_compile publications\hg-msa-tc-future-transportation\code\pipeline\hg_msa_tc_core.py publications\hg-msa-tc-future-transportation\code\pipeline\split_aware_io.py publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests\test_split_aware_runner.py -q
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py benchmark
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py target
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py select
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py test --synthetic-fixture
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py freeze
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\pipeline\run_split_aware_hg_msa_tc.py test --confirm-independent-test I_CONFIRM_FROZEN_TRANSDUCTIVE_TEST
```

Short-timeout benchmark, target, and select startup attempts were terminated before
their consolidated scientific outputs were written. Each was rerun successfully with
an adequate command timeout. The completed outputs and provenance come only from the
successful runs.

## Remaining Limitations and Blockers

1. The frozen cohort uses unlabeled complete-scene frame-span distributions. This is a
   transductive preprocessing limitation, not a manual-label leak.
2. The five active `trajectories_filtered_filled.parquet` parents are missing, so
   byte-level full lineage to those cleaned point-level inputs remains blocked.
3. The Newport dataset config has a cross-scene single-video path. The publication
   runner does not use it and a validator rejects it.
4. Manual maneuver annotations are not populated; independent external validity cannot
   yet be evaluated.
5. SE38th's development-only target of 18 and residual density-method target errors
   require careful interpretation in the later independent evaluation.
6. No claim about independent-test performance is supported by this task.
