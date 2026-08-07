# Task 06 Execution Report

## Scope and locks

- Branch: `feature/futuretransp-emas-formalization-sensitivity`.
- Base commit: `9ed1ea1992ede1ce6e94b1cf3af16e8afebae665`.
- Canonical implementation: `code/metrics/emas_hg.py`.
- Score version: `EMAS_HG-v1`.
- Frozen targets, selections, assignments and reference labels were unchanged.
- Independent-test clustering was not rerun.
- No independent-test metric was used to select or filter weights.

## Reproduction and role

- Reproduction rows: **345**.
- Maximum absolute error: **5.4933835258452756e-13** (`PASS`, tolerance `1e-12`).
- Verified role: post hoc diagnostic composite; not a selection key or tie-breaker.

## Sensitivity summary

| scenario               |   preserved_cases |   total_cases |   preservation_pct |
|:-----------------------|------------------:|--------------:|-------------------:|
| balanced_task_internal |                14 |            15 |              93.33 |
| equal_weights          |                14 |            15 |              93.33 |
| moderate_target        |                15 |            15 |             100.00 |
| original               |                15 |            15 |             100.00 |
| outlier_heavy          |                14 |            15 |              93.33 |
| reviewer_example       |                14 |            15 |              93.33 |
| target_heavy           |                15 |            15 |             100.00 |

Local-grid mean stability by method:

| method   |   original_top_stability_pct |
|:---------|-----------------------------:|
| hdbscan  |                        92.30 |
| kmeans   |                        99.96 |
| optics   |                        89.38 |

The reviewer example changes only `bellevue_150th_se38th/HDBSCAN`. Local-grid overall
top-rank preservation is 93.88%. Broad global stress tests are less stable and are not
used to recommend alternative weights.

## Component diagnostics

| statistic                  | variable_x   |   value |
|:---------------------------|:-------------|--------:|
| variance                   | T            |  0.0941 |
| pearson_with_original_EMAS | T            |  0.9438 |
| fraction_at_zero           | T            |  0.1922 |
| fraction_at_one            | T            |  0.0314 |
| variance                   | O            |  0.0548 |
| pearson_with_original_EMAS | O            |  0.3165 |
| fraction_at_zero           | O            |  0.0157 |
| fraction_at_one            | O            |  0.5725 |
| variance                   | B            |  0.0217 |
| pearson_with_original_EMAS | B            |  0.2192 |
| fraction_at_zero           | B            |  0.0157 |
| fraction_at_one            | B            |  0.0000 |
| variance                   | S            |  0.0093 |
| pearson_with_original_EMAS | S            |  0.5654 |
| fraction_at_zero           | S            |  0.0000 |
| fraction_at_one            | S            |  0.0000 |
| variance                   | D            |  0.0123 |
| pearson_with_original_EMAS | D            |  0.4730 |
| fraction_at_zero           | D            |  0.0000 |
| fraction_at_one            | D            |  0.0000 |

## Commands

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_emas_sensitivity.py analyze
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_emas_reporting.py figures
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_emas_reporting.py documents
.\.venv\Scripts\python.exe -m ruff check publications\hg-msa-tc-future-transportation\code publications\hg-msa-tc-future-transportation\tests
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
```

## Validation status

- Ruff over publication code and tests: **PASS**.
- Pytest over the complete publication suite: **94 passed in 80.09 seconds**.
- Frozen-input post-analysis checksum verification: **PASS**.
- Generated figure inspection: **PASS**; five PNG/PDF figure pairs are readable.
- One initial pytest invocation was terminated by an undersized shell timeout before
  producing a result; the unchanged command then completed under the correct timeout.

## Limitations

The score is heuristic and task-specific. Candidate-ranking stability is not model
selection because EMAS did not drive the frozen selection. Five Bellevue scenes are a
small scene-level sample. T dominates the composite; S and D are redundant; clipping
and neutral fallbacks can cause saturation. No weight vector is prospectively validated
as universally optimal.
