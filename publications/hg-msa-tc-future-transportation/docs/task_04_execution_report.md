# Task 04 Execution Report

## Outcome

The five scene guides were validated and frozen, and all **67,029** canonical
trajectories were processed exactly once. The output contains **63,188 valid
human-defined polygon-rule-based reference labels (94.269%)** and **3,841 retained
unassigned rows (5.731%)**. No trajectory was discarded.

- Branch: `feature/futuretransp-polygon-reference-labels`
- Guide commit: `a86660c0b3d31ebcf356d556e4b5b2e333bfedb1`
- Protocol hash: `b3ed9c7a211d461e21d7b3f39a1c9b34793ddcea5fea8036ebf4f02128377d6a`
- Frozen protocol file SHA-256: `6609f29e6251d477089ad210d00ac2160ed0daea73d14c1a26e5fdfa401ede96`
- All-label CSV SHA-256: `6630bc1f32f9a3ac5bfe78b6312432aafa7ac0327d43251ff01b05e2a9bfbd60`
- All-label Parquet SHA-256: `63c88c91925560fc1a636b8cb165abd5d90d8ed2b63adfad293727edec32648c`

## Coverage by scene and split

| Scene | Target estimation | Model selection | Independent test | Total |
| --- | ---: | ---: | ---: | ---: |
| bellevue_116th_ne12th | 657/723 (90.871%) | 577/634 (91.009%) | 864/964 (89.627%) | 2,098/2,321 (90.392%) |
| bellevue_150th_newport | 2,327/2,423 (96.038%) | 2,967/3,101 (95.679%) | 3,852/3,924 (98.165%) | 9,146/9,448 (96.804%) |
| bellevue_150th_eastgate | 8,474/8,720 (97.179%) | 6,615/6,791 (97.408%) | 10,586/10,755 (98.429%) | 25,675/26,266 (97.750%) |
| bellevue_150th_se38th | 2,282/2,482 (91.942%) | 2,648/2,865 (92.426%) | 3,422/3,713 (92.163%) | 8,352/9,060 (92.185%) |
| bellevue_ne8th | 4,553/5,576 (81.654%) | 5,854/6,321 (92.612%) | 7,510/8,037 (93.443%) | 17,917/19,934 (89.882%) |

## Movement and exclusion inventory

Every scene has twelve frozen legal movements. Observed movements over all splits are
10, 9, 9, 10, and 10 in the table's scene order; independent-test observed counts are
10, 9, 9, 9, and 10. There are no accepted multiple-polygon cases, no non-legal pair
assigned as valid, and no missing source geometry.

| Primary row outcome | Count |
| --- | ---: |
| valid | 63,188 |
| entry_no_polygon | 1,294 |
| exit_no_polygon | 2,547 |

## Sensitivity

Relative to the primary first/last-point rule, movement/status changed for 309 rows
(0.461%) with median first/last 3 points and 584 rows (0.871%) with median first/last
5 points. Shrinking polygons by 3 pixels changed 4,449 rows (6.637%); expanding them
by 3 pixels changed 1,097 rows (1.636%). NE8th was most sensitive to inward buffering
(16.118%), consistent with its lower target-estimation coverage and many endpoints
near polygon boundaries. The primary protocol remains unchanged.

## Commands executed

```powershell
.\.venv\Scripts\python.exe -m pip install "shapely>=2.0,<3" "tabulate>=0.9,<1"
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py freeze
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py all
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py inventories
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py sensitivity
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py figures
.\.venv\Scripts\python.exe -m ruff check publications\hg-msa-tc-future-transportation\code\reference_labels publications\hg-msa-tc-future-transportation\code\evaluation publications\hg-msa-tc-future-transportation\tests\test_polygon_reference_labels.py publications\hg-msa-tc-future-transportation\tests\test_polygon_reference_evaluation.py
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
```

The initial combined `all` command completed primary CSV/Parquet generation but
stopped at Markdown rendering because `tabulate` was absent. After installing the
declared dependency, downstream stages completed without regenerating source
endpoints. Rewriting the primary outputs from their Parquet representation produced
identical CSV and Parquet hashes.

## Validation and isolation

- Ruff: all targeted files passed.
- Pytest: **68 passed**.
- Synthetic ARI/NMI/purity and deterministic cluster-to-movement mapping tests passed.
- The frozen model-selection manifest remains checksum-identical and reports
  `independent_test_locked: true`.
- No manual cherry-picked label set was created or used.
- No cluster assignment, HG target, EMAS score, pseudo-reference label, or automatic
  OD assignment was read by the generator.
- Independent-test clustering was neither unlocked nor executed.
- Source video, processed trajectory data, and manuscript text were not modified.

## Limitations

The reference shares endpoint information with maneuver identification and is
sensitive to camera-space polygon definitions, incomplete tracks, and boundary
placement. It is not lane-level legal ground truth and does not replace manual
per-trajectory inspection. Manual review of the 980-row deterministic QA queue remains
a recommended quality-control step, not a source of labels.
