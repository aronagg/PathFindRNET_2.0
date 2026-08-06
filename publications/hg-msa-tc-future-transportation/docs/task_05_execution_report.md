# Task 05 Execution Report

## Execution identity and safeguards

- Branch: `feature/futuretransp-independent-test-evaluation`
- Base polygon-reference commit: `50abade318effcd657f797b026e9bfee854caa39`
- Pre-execution implementation commit: `96ee0c4525d2cca9f01c86a15395fa4491774214`
- Frozen configuration hash: `829c95c4f0a012d08433536b22379afad4d5a79122e52f8fd6ac12817882167c`
- Polygon-reference protocol hash: `b3ed9c7a211d461e21d7b3f39a1c9b34793ddcea5fea8036ebf4f02128377d6a`
- Reference export SHA-256: `127a52ff80af8906fdc3bb6a7436a222baedd42d8c542cfb2e234e67e43b4f5f`

All 25 preflight hash checks passed before any real independent-test feature vector
or reference row was loaded. The one-time unlock records author authorization and a
commitment not to retune after seeing results. Clustering required the additional
`--confirm-independent-test-evaluation` flag.

## Persisted test assignments

The first real transductive test run produced **164,358 rows**: 27,393 trajectories,
three methods, and two frozen strategies. Every `(scene_id, trajectory_id, method,
selection_strategy)` key is unique. All feature-access records contain only
`independent_test`.

- Assignment CSV SHA-256: `81185f1f92f4411a9b8e575769e58d45bc386adb4b4e753f2c6824e86998f47d`
- Assignment Parquet SHA-256: `137cd7e39f8886abbfbb49f4599d4f000626614f78342cf08f74f389b4963169`
- The clustering manifest records `reference_labels_read: false` and
  `evaluation_started: false`.
- Reference access occurred in a later command and a separate access log.

## Reference coverage

| Scene | Test trajectories | Valid reference | Excluded | Coverage |
| --- | ---: | ---: | ---: | ---: |
| bellevue_116th_ne12th | 964 | 864 | 100 | 89.627% |
| bellevue_150th_newport | 3,924 | 3,852 | 72 | 98.165% |
| bellevue_150th_eastgate | 10,755 | 10,586 | 169 | 98.429% |
| bellevue_150th_se38th | 3,713 | 3,422 | 291 | 92.163% |
| bellevue_ne8th | 8,037 | 7,510 | 527 | 93.443% |

## Frozen target validation

| Scene | Frozen HG target | Observed test movements | Legal movements | HG absolute error vs observed |
| --- | ---: | ---: | ---: | ---: |
| bellevue_116th_ne12th | 10 | 10 | 12 | 0 |
| bellevue_150th_newport | 12 | 9 | 12 | 3 |
| bellevue_150th_eastgate | 9 | 9 | 12 | 0 |
| bellevue_150th_se38th | 18 | 9 | 12 | 9 |
| bellevue_ne8th | 9 | 10 | 12 | 1 |

SE38th is a substantial frozen target-estimation failure and was not corrected.

## Aggregate descriptive findings

Across the 15 method-scene comparisons, HG-aware versus untargeted means were:

| Metric | Untargeted | HG-aware | Paired outcome (better/equal/worse) |
| --- | ---: | ---: | ---: |
| observed-count absolute error | 3.2000 | 2.5333 | 7 / 4 / 4 |
| legal-count absolute error | 3.9333 | 2.8667 | 7 / 4 / 4 |
| ARI | 0.7218 | 0.7175 | 7 / 2 / 6 |
| NMI | 0.8155 | 0.8376 | 7 / 2 / 6 |
| purity | 0.9102 | 0.9339 | 7 / 3 / 5 |
| macro F1 | 0.6328 | 0.6838 | 7 / 2 / 6 |
| weighted F1 | 0.7896 | 0.8169 | 8 / 2 / 5 |
| all-test outlier percentage | 15.5572% | 9.9444% | 5 / 7 / 3 |
| silhouette | 0.7954 | 0.7603 | 5 / 2 / 8 |
| Davies-Bouldin | 0.3076 | 0.3544 | 5 / 2 / 8 |
| EMAS_HG | 0.6791 | 0.7845 | 11 / 2 / 2 |

These are descriptive method-scene summaries, not 15 independent-intersection
replicates. The scene-level bootstrap uses five scenes separately by method.

KMeans HG-aware is harmed by target mismatch on Newport, SE38th, and NE8th; its mean
observed-count error increases from 1.4 to 2.6 and mean ARI decreases from 0.9273 to
0.8430. OPTICS shows the clearest alignment gain: mean observed-count error decreases
from 4.2 to 1.8, macro F1 increases from 0.4565 to 0.5903, and outliers decrease from
37.54% to 24.09%, although scene-level intervals remain broad. HDBSCAN changes are
smaller and mixed.

## Reference sensitivity

The HG-aware versus untargeted macro-F1 ranking remains 7 better, 2 equal, and 6
worse for the primary reference and all four sensitivity variants. Mean absolute
metric changes from the primary reference are very small for 3/5-point endpoint
medians. A 3-pixel inward polygon buffer has the largest effect: minimum scene
coverage falls to 75.849% at NE8th and mean absolute ARI change reaches 0.0122.

## Commands

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py preflight
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py unlock
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py cluster --confirm-independent-test-evaluation
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py evaluate
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py sensitivity
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py figures
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py reports
.\.venv\Scripts\python.exe -m ruff check publications\hg-msa-tc-future-transportation\code\independent_test publications\hg-msa-tc-future-transportation\code\run_independent_test_evaluation.py publications\hg-msa-tc-future-transportation\tests\test_independent_test_evaluation.py
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
```

## Validation and limitations

- Ruff passed.
- Pytest: **80 passed in 105.21 seconds**.
- No target, candidate grid, max-eps value, normalization parameter, clustering
  hyperparameter, EMAS weight, or selection rule was recomputed.
- Primary metrics use only valid independent-test reference rows.
- Excluded-reference rows are counted and reported.
- EMAS_HG is not treated as independent validation.
- The main result is a scene- and method-dependent trade-off, not general superiority.
- The five scenes are one Bellevue subset, and polygon-rule labels are not independent
  per-trajectory manual ground truth.

During implementation, the first evaluation access record was appended to the
clustering log, changing its checksum. No clustering was rerun. The seven original
clustering records were restored byte-for-byte, their checksum was verified, and the
evaluation now uses a separate immutable access log. Two older tests that required
test outputs to remain absent were updated to require the authorized output contract.
