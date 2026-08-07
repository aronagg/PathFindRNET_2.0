# Homography-to-Target Sensitivity Report

## Protocol

The analysis perturbs camera/source calibration points independently with bounded
uniform noise at `+/-1, +/-2, +/-3, +/-5, +/-10 px`. Destination points remain fixed. Each
scene-scale cell has 20
fixed-seed replicates. Every replicate uses the exact historical homography estimator
and the full frozen target-estimation algorithm on `target_estimation`; no K, threshold,
or target is held artificially fixed. This is a finite deterministic Monte Carlo
diagnostic, not a high-precision probability estimate.

| scene_id | noise_scale_px | target_preservation_probability | target_min | target_max | entry_k_preservation_probability | exit_k_preservation_probability | threshold_preservation_probability | endpoint_displacement_p95_px | od_pair_change_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 116th / NE12th | 1.0000 | 1.0000 | 10 | 10 | 1.0000 | 1.0000 | 1.0000 | 2.7083 | 0.0000 |
| 116th / NE12th | 2.0000 | 1.0000 | 10 | 10 | 1.0000 | 1.0000 | 1.0000 | 5.1296 | 0.0000 |
| 116th / NE12th | 3.0000 | 1.0000 | 10 | 10 | 1.0000 | 1.0000 | 1.0000 | 8.8132 | 0.0000 |
| 116th / NE12th | 5.0000 | 1.0000 | 10 | 10 | 1.0000 | 1.0000 | 1.0000 | 15.3338 | 0.0000 |
| 116th / NE12th | 10.0000 | 1.0000 | 10 | 10 | 1.0000 | 1.0000 | 1.0000 | 30.0581 | 0.0000 |
| 150th / Newport | 1.0000 | 0.8500 | 7 | 12 | 1.0000 | 1.0000 | 0.8500 | 4.0555 | 0.0015 |
| 150th / Newport | 2.0000 | 0.8500 | 7 | 12 | 1.0000 | 1.0000 | 0.8500 | 13.0959 | 0.0024 |
| 150th / Newport | 3.0000 | 0.9500 | 7 | 12 | 1.0000 | 1.0000 | 0.9500 | 22.8783 | 0.0029 |
| 150th / Newport | 5.0000 | 0.8500 | 7 | 12 | 1.0000 | 1.0000 | 0.8500 | 26.3334 | 0.0028 |
| 150th / Newport | 10.0000 | 0.7000 | 7 | 13 | 0.9000 | 1.0000 | 0.7500 | 44.5083 | 0.0095 |
| 150th / Eastgate | 1.0000 | 1.0000 | 9 | 9 | 1.0000 | 1.0000 | 1.0000 | 1.8426 | 0.0006 |
| 150th / Eastgate | 2.0000 | 1.0000 | 9 | 9 | 1.0000 | 1.0000 | 1.0000 | 8.8219 | 0.0013 |
| 150th / Eastgate | 3.0000 | 0.9500 | 8 | 9 | 0.9500 | 1.0000 | 1.0000 | 8.0420 | 0.0062 |
| 150th / Eastgate | 5.0000 | 0.9000 | 8 | 10 | 0.9500 | 1.0000 | 1.0000 | 12.7752 | 0.0082 |
| 150th / Eastgate | 10.0000 | 0.8500 | 7 | 10 | 0.9000 | 1.0000 | 0.9500 | 21.0330 | 0.0123 |
| 150th / SE38th | 1.0000 | 1.0000 | 18 | 18 | 1.0000 | 1.0000 | 1.0000 | 2.1293 | 0.0020 |
| 150th / SE38th | 2.0000 | 1.0000 | 18 | 18 | 1.0000 | 1.0000 | 1.0000 | 6.4439 | 0.0048 |
| 150th / SE38th | 3.0000 | 1.0000 | 18 | 18 | 1.0000 | 1.0000 | 1.0000 | 32.5311 | 0.0067 |
| 150th / SE38th | 5.0000 | 1.0000 | 18 | 18 | 1.0000 | 1.0000 | 1.0000 | 41.8040 | 0.0105 |
| 150th / SE38th | 10.0000 | 0.7000 | 16 | 37 | 0.9000 | 0.8000 | 0.7000 | 37.8592 | 0.0971 |
| NE8th | 1.0000 | 1.0000 | 9 | 9 | 1.0000 | 1.0000 | 1.0000 | 21.8036 | 0.0001 |
| NE8th | 2.0000 | 1.0000 | 9 | 9 | 1.0000 | 1.0000 | 1.0000 | 44.4124 | 0.0001 |
| NE8th | 3.0000 | 0.9500 | 9 | 10 | 1.0000 | 1.0000 | 0.9500 | 56.4108 | 0.0002 |
| NE8th | 5.0000 | 0.9000 | 9 | 10 | 1.0000 | 1.0000 | 0.9000 | 114.8448 | 0.0003 |
| NE8th | 10.0000 | 0.8000 | 9 | 10 | 1.0000 | 1.0000 | 0.8000 | 218.3504 | 0.0005 |

## Jackknife

| scene_id | omissions | target_preservation | target_min | target_max | median_p95_endpoint_displacement_px |
| --- | --- | --- | --- | --- | --- |
| 116th / NE12th | 26 | 1.0000 | 10 | 10 | 29.4639 |
| 150th / Eastgate | 13 | 0.6154 | 9 | 10 | 7.4710 |
| 150th / Newport | 20 | 0.9500 | 7 | 12 | 41.8637 |
| 150th / SE38th | 24 | 0.9167 | 18 | 43 | 15.1671 |
| NE8th | 18 | 0.8889 | 9 | 10 | 68.4757 |

## Interpretation

- 116th/NE12th preserves the target in every jackknife and perturbation run.
- Newport shows a threshold-linked discontinuity: small perturbations occasionally
  move the result from 12 to 7 even though mean OD assignment changes remain below 1%.
- Eastgate preserves 9 in 61.5% of leave-one-point-out runs and at least 85% of
  perturbations at every tested scale.
- NE8th has large top-view displacement under perturbation, consistent with strong
  projective amplification outside the calibration hull, while OD changes remain low.
- SE38th preserves 18 in every `+/-1-5 px` replicate. At `+/-10 px`, 14/20 runs preserve
  18 and the observed range is 16-37. Calibration uncertainty changes the geometric
  target only under large perturbation and does not move it toward 9.
