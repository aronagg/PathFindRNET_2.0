# HG-SMG-TC Protocol Freeze Report

## Freeze Identity

- Protocol: `hg-smg-tc-preregistration-v1`
- Method: **HG-SMG-TC, Homography-Guided Semantic Maneuver Graph Trajectory Clustering**
- Base commit: `8bfb4826725ea5c4c04042c037b521edcf216ec4`
- Branch: `feature/futuretransp-hg-smg-preregistration`
- Protocol SHA-256: `2e892f25b1c1770a55fa29e36206b634f424f9ff2508bf045c92d388e69b4be6`
- Freeze timestamp: `2026-08-07T22:30:12Z`
- Scientific status: protocol-only; HG-SMG-TC is not implemented or run.

## Frozen Primary Method

1. Reuse frozen EMD on `target_estimation` only.
2. Consolidate same-role micro-regions by complete-link SAC using top-view side/bearing,
   five-point directed camera-isotropic heading, and bootstrap self-consistency radii.
3. Do not require OD-profile similarity in primary SAC.
4. Aggregate micro-OD support into SMG superedges and reuse the frozen support-grid
   heuristic.
5. Run 500 recording-aware bootstrap replicates and use the primary 90% percentile
   interval for `K_SMG`.
6. Replace point-target error with interval distance in model selection, then apply the
   exact frozen method-specific tie-break order. Do not force KMeans to one K.

The master seed, derivation rule, mathematical edge cases, deterministic ordering,
A0-A10 ablations, and H1-H7 hypotheses are all frozen in versioned files.

## Sensitivity-Only Variants

- heading windows 3, 5, and 7;
- self-consistency quantiles 0.90, 0.95, and 0.975;
- camera-isotropic and homography-top-view heading;
- UATP intervals 80%, 90%, and 95%;
- A8 OD-profile compatibility diagnostic.

None may replace the primary protocol after test outcomes are seen.

## Leakage and Post-Review Status

The original independent-test failure is already known, so this is not described as
blind prospective method development. The prospective safeguard is narrower and
explicit: all extension choices are frozen before its first test execution. The
preregistration code path cannot read independent polygon labels, assignments,
agreement metrics, mapping tables, or accidental extension outputs.

No unused same-scene recording was found: all 115 raw recordings are represented in
the 67,029-trajectory canonical manifest. Future testing must be called a **locked
post-review extension evaluation**.

## Inherited Task-08 Facts

- frozen homographies reproduce exactly;
- OpenCV 4.12 `findHomography`, RANSAC 10 px, 2,000 iterations, confidence 0.995;
- all five scenes pass the diagnostic engineering screening classification;
- 90.4-96.9% endpoint extrapolation beyond the source calibration hull;
- the SE38th target 18 persists for every +/-1 to +/-5 px Task-08 perturbation run;
- semantic endpoint-region granularity is the primary failure mechanism;
- Google-derived top-view raster redistribution metadata remain incomplete.

No scene is removed and no frozen Task-08 artifact is changed.

## Amendment Rule

Version 1 is immutable. Before the first extension test run, any necessary correction
requires a separately named protocol, rationale, hash, and leakage audit. After the
first run, primary equations, parameters, ablations, seeds, selection logic, and
hypotheses cannot be replaced. Deviations must be reported rather than overwritten.
