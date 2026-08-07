# HG-SMG-TC Novelty and Overlap Matrix

Legend: `yes` means an explicit central component, `partial` means related but not
equivalent, `no` means absent from the described method, and `unknown` means the
available primary metadata was insufficient for a defensible classification.

| Work | Endpoint discovery | Lane/geometric modes | OD extraction | Graph modelling | Semantic approach consolidation | Semantic maneuver graph | Uncertainty-aware K prior | Prior-constrained clustering | Exhaustive reference | Reproducibility/public release |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rathore et al. 2026 | yes | partial | yes | no | no | no | no | no | partial | partial |
| Wan et al. 2025 | partial | yes | partial | yes | no | no | no | no | no | yes |
| Yuan et al. 2024 | partial | yes | partial | yes | partial | partial | no | no | no | yes |
| Wang et al. 2017 | partial | partial | partial | partial | partial | partial | no | no | partial | unknown |
| Uduwaragoda et al. 2013 | no | yes | no | no | no | no | no | no | no | unknown |
| Rezaie and Saunier 2021 | yes | partial | yes | no | no | no | no | no | partial | partial |
| Sekh et al. 2020 | partial | yes | partial | no | no | no | no | partial | no | unknown |
| Betru et al. 2025 | yes | partial | yes | no | no | no | no | no | partial | partial |
| Du et al. 2023 | partial | yes | partial | partial | no | no | no | no | partial | yes |
| Original HG-MSA-TC | yes | yes | yes | partial | no | partial | no | yes | yes | yes |
| Proposed HG-SMG-TC | inherited | inherited | yes | yes | yes | yes | yes | yes | inherited | yes |

## Interpretation

- Endpoint regions, OD pairs, graphs, and lane/geometric modes cannot be claimed as
  isolated novelties.
- The original HG-MSA-TC already links homography-derived endpoint modes to a point
  target and expected-aware selection; HG-SMG-TC must not relabel that contribution.
- The post-review extension adds a frozen, label-free consolidation level between
  geometric micro-modes and maneuver structure, then treats target count as a
  bootstrap interval rather than a single forced value.
- The exhaustive polygon-rule reference is an inherited evaluation asset. It is
  independent of cluster outputs but is not fully manual per-trajectory ground truth.

## Conservative Novelty Claim

The defensible claim is a new coupling and evaluation protocol, not a new primitive:
HG-SMG-TC combines directional-geometric semantic approach consolidation, a semantic
maneuver graph, an uncertainty-aware target interval, and interval-constrained model
selection while preserving strict split isolation and exhaustive reference coverage.
