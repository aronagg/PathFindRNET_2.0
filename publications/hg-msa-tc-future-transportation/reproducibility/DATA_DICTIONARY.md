# Data Dictionary

## Future HG-SMG-TC Tables

No table below is generated in Task 09A. Names and core fields are preregistered for
the later implementation.

| Table | Unit | Required core fields |
| --- | --- | --- |
| `emd_micro_regions` | scene, role, micro-region | scene_id, split, role, micro_region_id, member_count, centroid, bearing, frozen EMD config |
| `sac_region_descriptors` | scene, role, micro-region | bearing, heading, robust dispersions, valid sample counts, bootstrap radii, status |
| `sac_supernode_membership` | micro-region | scene_id, role, micro_region_id, supernode_id, merge order, protocol hash |
| `smg_edges` | semantic entry/exit pair | scene_id, entry_supernode, exit_supernode, count, share, threshold, supported |
| `uatp_bootstrap_targets` | bootstrap replicate | scene_id, replicate, seed, K_SMG, threshold, status |
| `uatp_summary` | scene | mode, median, entropy_bits, interval_level, lower_K, upper_K |
| `pcms_candidates` | scene, method, candidate | parameters, nonnoise_K, interval_distance, outliers, internal metrics, selection key |
| `hg_smg_assignments` | trajectory, method, ablation | scene_id, trajectory_id, split, method, ablation_id, cluster_label, is_noise |
| `hg_smg_reference_metrics` | scene, method, ablation | coverage, ARI, NMI, purity, completeness, macro_F1, noise percentage |

## Shared Identifiers

- `scene_id`: one of the five frozen Bellevue scene slugs.
- `trajectory_id`: scene-namespaced canonical trajectory identifier.
- `recording_id`: deterministic source recording identifier.
- `split`: exactly `target_estimation`, `model_selection`, or `independent_test`.
- `protocol_hash`: SHA-256 of `configs/hg_smg_protocol_v1.yaml`.
- `reference_label_access`: boolean recorded for every experiment stage.

All percentages must state their denominator. Noise is label `-1`; non-noise cluster
count excludes it.
