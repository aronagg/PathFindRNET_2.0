# Split-Aware Runner Sampling Policy

## Decision

All available authorized trajectories are used for target fitting and model-selection
candidate fitting. The former full-data `MAX_EVAL_TRACKS=6000` cap is not retained.

The largest model-selection cohort is Eastgate with 6791 trajectories. A small
full-cohort benchmark produced:

| Method | Trajectories | Representative fit time (s) |
| --- | ---: | ---: |
| KMeans | 6791 | 1.3448 |
| HDBSCAN | 6791 | 0.1381 |
| OPTICS | 6791 | 3.3443 |

These measurements show that all-trajectory fitting is computationally feasible for
the submitted candidate grids on the current machine. The benchmark does not choose a
model and does not change scientific outputs.

## Metric Evaluation Sampling

Cluster methods are fitted on all authorized trajectories. Clustered-only internal
metrics use a deterministic sample of at most 3000 clustered trajectories to control
the quadratic cost of silhouette evaluation. Endpoint-region candidate metrics use at
most 2500 trajectories. These samples affect metric calculation only, not clustering
fit or cohort membership, and their seeds are frozen.

## Contingency Policy

If a later protocol version requires a fit cap, it must be explicit in YAML and use
deterministic recording-stratified sampling. Every recording block must retain rows,
sampled trajectory IDs must be saved, and per-scene coverage must be reported. No row
may be borrowed from another split.

No fit sampling is used in the current protocol, so
`results/development/sampled_trajectory_ids.csv` is intentionally absent.
