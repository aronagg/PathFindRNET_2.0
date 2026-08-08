# Baseline Method Definitions

All baselines are deterministic, reviewer-focused comparisons on the frozen independent-test split.

## Endpoint-only baseline
Uses raw camera-space `start_x,start_y,end_x,end_y` endpoint coordinates.

## Isotropic endpoint baseline
Uses the same endpoint coordinates after camera-space shared-scale isotropic normalization fitted without labels.

## Resampled-trajectory Euclidean baseline
Resamples each canonical trajectory interval to `20` points, flattens x/y coordinates, and applies a shared camera-space x/y scale.

## Selection
Candidate grids are small and deterministic. Selection uses only internal clustering metrics before any reference labels are loaded.
