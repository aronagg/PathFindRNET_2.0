# EMAS_HG-v1 Reproduction Report

All available publication development and independent-test score rows were recomputed
from stored component inputs with `code/metrics/emas_hg.py`.

| source                   |   rows |   maximum_absolute_difference |
|:-------------------------|-------:|------------------------------:|
| development_candidates   |    255 |                     2.220e-16 |
| development_selected     |     30 |                     1.110e-16 |
| frozen_protocol_selected |     30 |                     1.110e-16 |
| independent_test_metrics |     30 |                     5.493e-13 |

- Total checked rows: **345**.
- Maximum absolute difference: **5.4933835258452756e-13**.
- Acceptance threshold: **1e-12**.
- Largest difference source: `independent_test_metrics`, `bellevue_150th_eastgate`,
  `optics`, `untargeted_selection`.
- Result: **PASS**.

The largest difference is consistent with decimal CSV serialization. Frozen files
were read-only and were not overwritten. Sensitivity analysis proceeded only after
this check passed.
