# HG-SMG Development Ablation Report

| ablation_id | status | split | selected_configuration_rows | reference_label_access | note |
| --- | --- | --- | --- | --- | --- |
| A0 | reused_frozen_original | target_estimation_then_model_selection | 0 | False | Frozen untargeted selection reused; no scientific recomputation. |
| A1 | reused_frozen_original | target_estimation_then_model_selection | 0 | False | Frozen original HG-aware point-target selection reused. |
| A2 | completed | target_estimation_then_model_selection | 15 | False |  |
| A3 | completed | target_estimation_then_model_selection | 15 | False |  |
| A4 | completed | target_estimation | 0 | False | UATP target distribution only; no model selection by definition. |
| A5 | completed | target_estimation_then_model_selection | 15 | False |  |
| A6 | completed | target_estimation_then_model_selection | 15 | False |  |
| A7 | completed | target_estimation_then_model_selection | 15 | False |  |
| A8 | not_identifiable | target_estimation_then_model_selection | 0 | False | Protocol v1 defines JSD but no executable JSD compatibility threshold. |
| A9 | completed | target_estimation_then_model_selection | 15 | False |  |
| A10 | completed | target_estimation_then_model_selection | 15 | False |  |

The heading-only A7 variant shows substantial structural collapse in several scenes. Sensitivities remain diagnostics and cannot replace A5. A8 remains unexecuted because its compatibility threshold is absent from protocol v1.

The A5, A6 bearing-only, and A9 top-view-heading variants produced the same deterministic full-split targets in all five scenes (`10, 8, 9, 10, 9`), although Newport UATP distributions differ slightly across variants. A7 heading-only produced full-split targets `1, 2, 4, 1, 3`, showing that directed heading alone does not retain the preregistered side/bearing structure.

Heading-window sensitivities at 3, 5 and 7 points did not change the five deterministic full-split targets. Changing q from 0.95 to 0.90/0.975 changed Newport from 8 to 12/8 and SE38th from 10 to 12/8; other scenes were unchanged. Primary 80/90/95% interval choices are retained separately and no sensitivity replaces the preregistered 90% PCMS prior.
