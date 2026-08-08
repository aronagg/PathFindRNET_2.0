# Data Dictionary

## HG-SMG-TC Development Tables

Task 09B generates the following development-only tables under
`results/hg_smg/development/`.

| Table | Unit | Required core fields |
| --- | --- | --- |
| `emd_reproduction.csv` | scene | frozen/reproduced target, entry/exit region counts, threshold, OD coverage, exact-assignment flags, seed |
| `sac_microregion_descriptors.csv` | scene, variant, role, micro-region | support, centroid, bearing/heading locations, bootstrap radii, descriptor-validity flags |
| `sac_pairwise_compatibility.csv` | scene, variant, role, micro-region pair | circular separations, pair tolerances, normalized distances, primary D, compatibility |
| `sac_merge_trace.csv` | deterministic merge step | scene, variant, role, member sets, complete-link distance |
| `sac_supernode_assignments.csv` | micro-region | scene, variant, role, micro-region, supernode ID |
| `smg_edges` | semantic entry/exit pair | scene_id, entry_supernode, exit_supernode, count, share, threshold, supported |
| `uatp_bootstrap_targets.csv` | scene, variant, bootstrap replicate | bootstrap/EMD seeds, sample counts, region/supernode counts, threshold, K_SMG, coverage, status |
| `uatp_summary.csv` | scene, variant | mode, median, entropy, 80/90/95% intervals, full-split-target probability, failure count |
| `pcms_candidates` | scene, method, candidate | parameters, nonnoise_K, interval_distance, outliers, internal metrics, selection key |
| `pcms_selected_configurations.csv` | scene, method | frozen candidate fields, UATP interval, interval distance, selected parameters |
| `ablation_development_summary.csv` | ablation | status, split, selected-row count, reference-access flag, note |
| `preregistered_sensitivity_summary.csv` | scene, sensitivity | deterministic K_SMG or UATP interval sensitivity summary |

## Shared Identifiers

- `scene_id`: one of the five frozen Bellevue scene slugs.
- `trajectory_id`: scene-namespaced canonical trajectory identifier.
- `recording_id`: deterministic source recording identifier.
- `split`: exactly `target_estimation`, `model_selection`, or `independent_test`.
- `protocol_hash`: SHA-256 of `configs/hg_smg_protocol_v1.yaml`.
- `reference_label_access`: boolean recorded for every experiment stage.

Task 09B creates no `hg_smg_assignments` or `hg_smg_reference_metrics` table because
independent-test execution and reference evaluation remain locked.

All percentages must state their denominator. Noise is label `-1`; non-noise cluster
count excludes it.
