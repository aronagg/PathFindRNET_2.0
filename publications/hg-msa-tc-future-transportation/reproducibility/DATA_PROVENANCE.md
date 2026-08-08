# Data Provenance

## Primary Sources

- Raw recordings: five Bellevue intersections from the Traffic Node Video Dataset,
  retained under repository `data/raw/<scene>/` and governed by source distribution
  terms.
- Canonical processed features:
  `data/processed/<scene>/feature_analysis/features_trimmed_frame_disp_norm.parquet`.
- Canonical publication manifest: 67,029 trajectories across 115 source recordings.
- Split manifest: recording-aware `target_estimation`, `model_selection`, and
  `independent_test` roles.
- Homographies: manually selected camera/top-view correspondences and frozen OpenCV
  RANSAC matrices, with checksums and Task-08 quality diagnostics.
- Human-defined polygon-rule reference: frozen independently from clustering outputs;
  evaluation-only and never an HG-SMG-TC method input.

## Transform Chain

```text
video -> tracking -> canonical trajectory interval -> processed features
      -> recording-aware split
target_estimation -> frozen EMD -> SAC -> SMG -> UATP
model_selection   -> frozen candidate grid -> PCMS
persisted test assignments -> separate polygon-reference evaluation
```

The extension is post-review. Its equations and analysis plan are frozen before the
first extension test run, but the motivating original failure was already known.

Task 09B executes the upper two branches only. EMD/SAC/SMG/UATP read
`target_estimation`; PCMS reads the frozen `model_selection` candidate table. The
implementation records `reference_label_access=false` and
`independent_test_access=false`, and rejects paths containing reference/test markers.
No test assignment or semantic metric is generated.

## Immutable Inputs

The protocol records SHA-256 values for the trajectory manifest, split manifest,
frozen evaluation protocol, frozen model-selection manifest, Task-08 result manifest,
Task-08 quality table, and frozen scene parameters. Reproduction must fail on mismatch.
Task 09B additionally freezes its exact code commit, master seed, input hashes,
development result hashes, deterministic rerun comparison, and package versions in
`configs/hg_smg_development_freeze_v1.yaml`.
