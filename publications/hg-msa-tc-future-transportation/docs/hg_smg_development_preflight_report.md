# HG-SMG Development Preflight Report

- Timestamp: `2026-08-07T23:19:52+00:00`
- Git HEAD at preflight: `7ffcbca365122a7acbf5232d515d829f1e27b8bd`
- Required base commit: `7ffcbca365122a7acbf5232d515d829f1e27b8bd` (verified ancestor)
- Protocol SHA-256: `2e892f25b1c1770a55fa29e36206b634f424f9ff2508bf045c92d388e69b4be6` (match)
- Ablation SHA-256: `e02cb6a22355fa66cd66c392e3199ccbb7508413e2db3b0f3977fc1796f0ccd6` (match)
- Independent-test access guard: **active**
- Existing HG-SMG scientific outputs before implementation: **none**
- Semantic/reference labels accessed: **no**

## Frozen Inputs

| Input | SHA-256 match | Repository-relative path |
| --- | --- | --- |
| `trajectory_manifest` | yes | `publications/hg-msa-tc-future-transportation/data/manifests/trajectory_manifest.csv` |
| `evaluation_split` | yes | `publications/hg-msa-tc-future-transportation/data/splits/evaluation_split.csv` |
| `frozen_evaluation_protocol` | yes | `publications/hg-msa-tc-future-transportation/configs/frozen_evaluation_protocol.yaml` |
| `frozen_selection_manifest` | yes | `publications/hg-msa-tc-future-transportation/results/development/frozen_selection_manifest.json` |
| `task_08_result_manifest` | yes | `publications/hg-msa-tc-future-transportation/results/homography/task_08_result_manifest.json` |
| `task_08_quality_metrics` | yes | `publications/hg-msa-tc-future-transportation/results/homography/homography_quality_metrics.csv` |
| `frozen_scene_parameters` | yes | `publications/hg-msa-tc-future-transportation/results/target_estimation/frozen_scene_parameters.csv` |

The preflight read only preregistered control files and hashes. It did not read independent-test assignments, metrics, polygon labels, scene guides, or movement inventories.
