# HG Target Estimator Scientific Interpretation

## Five-scene accuracy

| scene                   |   automatic_od_target |   observed_independent_semantic_movement_count |   absolute_error_vs_observed |   automatic_entry_region_count |   automatic_exit_region_count |   semantic_duplication_ratio |
|:------------------------|----------------------:|-----------------------------------------------:|-----------------------------:|-------------------------------:|------------------------------:|-----------------------------:|
| bellevue_116th_ne12th   |                    10 |                                             10 |                            0 |                              4 |                             4 |                       0.0000 |
| bellevue_150th_newport  |                    12 |                                              9 |                            3 |                              4 |                             4 |                       0.0000 |
| bellevue_150th_eastgate |                     9 |                                              9 |                            0 |                              4 |                             4 |                       0.0000 |
| bellevue_150th_se38th   |                    18 |                                              9 |                            9 |                              7 |                             3 |                       0.5000 |
| bellevue_ne8th          |                     9 |                                             10 |                            1 |                              4 |                             4 |                       0.0000 |

The frozen target exactly matches the independent observed count in two of five scenes
(116th/NE12th and Eastgate), differs by one at NE8th, overestimates Newport by three,
and overestimates SE38th by nine. Mean absolute scene-level error is `2.60`.
With only five scenes, this is descriptive evidence, not a population-level accuracy claim.

## What the estimator counts

The estimator counts supported geometric endpoint-region pairs. In favorable scenes
those pairs align with semantic maneuvers. SE38th proves that the equivalence is not
guaranteed: one physical approach is split into several automatic entry modes, while
the exit partition merges manual exits. The method can therefore resolve lane-position
or geometric trajectory submodes in addition to semantic maneuver classes.

## Why KMeans is most sensitive

For HG-aware KMeans the frozen target directly sets `k`. An overestimate therefore
forces additional non-noise partitions. HDBSCAN and OPTICS do not take target count as
a fit parameter; the target only changes which already evaluated configuration is
selected. They can still fragment or merge movements, but the propagation is indirect.

## Robustness

The support threshold is locally stable for several scenes, and SE38th remains far
above nine throughout the diagnostic threshold grid. Endpoint-region count is much
more influential: plausible K combinations create broad target ranges in every scene.
The estimator is therefore not robust to endpoint-region granularity in a semantic sense.

## Claims to retain, narrow, or remove

- Retain: homography enables a deterministic, label-free geometric target-estimation layer.
- Retain: the frozen implementation is exactly reproducible from `target_estimation` only.
- Narrow: target alignment improves in several scenes, not uniformly across all scenes.
- Narrow: `K_HG` is an automatically estimated observed geometric-maneuver target, not
  a guaranteed semantic or legal maneuver count.
- Remove: any implication that all physical road branches are recovered one-to-one.
- Remove: any universal-superiority claim based only on EMAS_HG or cluster-count agreement.

## Defensible limitation and future work

The endpoint-region K and OD threshold are heuristic and can resolve multiple modes
within one physical approach or merge distinct exits. Calibration uncertainty,
incomplete trajectories, and camera-space reference boundaries add further uncertainty.
A future road-branch consolidation stage could merge geometric endpoint modes using
road topology, lane markings, or mandatory-turn arrows. It must be prospectively
specified and evaluated; it is not applied to current results.
