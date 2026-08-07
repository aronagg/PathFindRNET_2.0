# Role of EMAS_HG in Frozen Model Selection

The verified protocol is **Case B: EMAS_HG is a reported diagnostic score**. The
selection-key functions contain no EMAS_HG term. Consequently:

- changing EMAS weights changes no frozen selected configuration;
- no candidate is re-fitted and no independent-test partition is re-run;
- Task 06 reports candidate top-rank stability under EMAS, not model-selection
  stability;
- ranks below show where the already frozen configurations happen to fall under the
  original diagnostic score.

| scene                   | method   |   frozen_untargeted_original_emas_rank |   frozen_hg_aware_original_emas_rank |
|:------------------------|:---------|---------------------------------------:|-------------------------------------:|
| bellevue_116th_ne12th   | hdbscan  |                                      1 |                                    1 |
| bellevue_116th_ne12th   | kmeans   |                                      1 |                                    1 |
| bellevue_116th_ne12th   | optics   |                                     11 |                                    1 |
| bellevue_150th_eastgate | hdbscan  |                                      3 |                                    2 |
| bellevue_150th_eastgate | kmeans   |                                      4 |                                    1 |
| bellevue_150th_eastgate | optics   |                                      7 |                                    1 |
| bellevue_150th_newport  | hdbscan  |                                      2 |                                    1 |
| bellevue_150th_newport  | kmeans   |                                     10 |                                    1 |
| bellevue_150th_newport  | optics   |                                     13 |                                    1 |
| bellevue_150th_se38th   | hdbscan  |                                      5 |                                    2 |
| bellevue_150th_se38th   | kmeans   |                                     12 |                                    1 |
| bellevue_150th_se38th   | optics   |                                     11 |                                    1 |
| bellevue_ne8th          | hdbscan  |                                      6 |                                    1 |
| bellevue_ne8th          | kmeans   |                                      2 |                                    1 |
| bellevue_ne8th          | optics   |                                      7 |                                    2 |

Untargeted selections often have low EMAS rank because they deliberately do not use
the HG target. HG-aware selections are usually, but not always, top-ranked by EMAS.
This descriptive coincidence must not be presented as evidence that EMAS selected the
models.
