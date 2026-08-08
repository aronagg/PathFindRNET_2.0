# Final Claim Audit

| claim                                                     | status                                | evidence_or_required_change                                                                                 |
|:----------------------------------------------------------|:--------------------------------------|:------------------------------------------------------------------------------------------------------------|
| Leakage-free validation                                   | supported                             | Frozen split protocol, persisted assignments before reference evaluation, independent-test reports.         |
| Independent reference labels exist                        | supported with terminology constraint | Exhaustive human-defined polygon-rule labels; not per-trajectory manual ground truth.                       |
| Original HG-aware improves target alignment vs untargeted | supported                             | A1 mean observed target error lower than A0 in final comparison.                                            |
| HG-SMG improves over original HG-aware                    | partially supported                   | A5 improves target alignment and NMI on average; macro-F1/outlier trade-offs must be reported.              |
| HG-SMG is universally superior to all baselines           | unsupported                           | Endpoint KMeans and resampled baselines are strong in some scene-method cases, especially SE38th.           |
| EMAS_HG is an independent validation metric               | must be removed                       | EMAS_HG is task-specific development/ranking score, not independent reference validation.                   |
| Target estimator recovers semantic maneuver counts        | must be narrowed                      | SE38th frozen HG target 18 versus observed 9 shows semantic over-segmentation.                              |
| Homography is reliable enough for all claims              | partially supported                   | Quality gate passes but extrapolation and point-pair uncertainty remain material limitations.               |
| Generalization across all TNVD scenes                     | unsupported                           | Study uses five Bellevue scenes only.                                                                       |
| Novelty of HG-SMG                                         | supported but narrow                  | Novelty is integration of homography-guided target estimation, SAC/SMG prior and leakage-locked evaluation. |
