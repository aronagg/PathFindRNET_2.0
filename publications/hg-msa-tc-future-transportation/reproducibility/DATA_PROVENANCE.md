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

## Immutable Inputs

The protocol records SHA-256 values for the trajectory manifest, split manifest,
frozen evaluation protocol, frozen model-selection manifest, Task-08 result manifest,
Task-08 quality table, and frozen scene parameters. Reproduction must fail on mismatch.
