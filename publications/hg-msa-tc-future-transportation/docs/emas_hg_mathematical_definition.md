# Mathematical Definition of EMAS_HG-v1

Let `N` be the number of trajectories, `K` the number of non-noise clusters, `K_HG`
the frozen homography-derived target, and `e_K = |K-K_HG|`. Let `N_-1` denote the
number of noise-labelled trajectories and let `n_j` be the size of non-noise cluster
`j`. Define `clip(x)=min(max(x,0),1)`.

## Components

### Target agreement

`T = clip(1 - e_K / max(K_HG,1))`.

The absolute, not signed, error is used. A zero target uses denominator one for legacy
compatibility. A missing or non-finite target is invalid under the canonical input
contract. Values outside the unit interval are clipped.

### Non-outlier share

`O = clip(1 - N_-1/N) = clip(1 - p_out/100)`.

The denominator is all trajectories. KMeans has `N_-1=0` under the frozen protocol,
so `O=1`. Empty input is rejected upstream because `p_out` is undefined.

### Cluster balance

For at least one non-noise trajectory,

`B = clip(1 - max_j(n_j) / sum_j(n_j))`.

Noise is excluded from numerator and denominator. This is a largest-cluster dominance
penalty, not an entropy or evenness index. One cluster gives `B=0`. If no non-noise
cluster exists, the largest-cluster ratio is undefined and EMAS_HG-v1 uses `B=0.5`.

### Silhouette

Let `s` be the Euclidean silhouette computed on non-noise rows only, using at most the
frozen deterministic 3,000-row metric sample. Then

`S = clip((s+1)/2)`.

The raw domain `[-1,1]` maps to `[0,1]`. If fewer than two valid clusters exist, a
singleton condition makes the score undefined, or calculation fails, `S=0.5`.

### Davies-Bouldin

Let `DB >= 0` be the Davies-Bouldin index on the same non-noise metric sample. Then

`D = 1/(1+DB)`.

This maps `[0,infinity]` monotonically to `(0,1]`; positive infinity maps to zero.
Missing, negative or negative-infinite values use the frozen fallback `D=0.5`.

### Combined score

`EMAS_HG-v1 = 0.50T + 0.20O + 0.10B + 0.10S + 0.10D`.

All valid components lie in `[0,1]`, all weights are non-negative and sum to one;
therefore their convex combination lies in `[0,1]`. A theoretical score of one means
perfect target agreement, no outliers, no largest-cluster dominance, silhouette one
and DB zero. Exact `B=1` is not attainable for a finite nonempty clustering, so 1.0 is
an upper bound rather than a routinely attainable empirical value. EMAS_HG is a
task-oriented diagnostic composite, not a universal clustering-validity index.

| Component | Raw metric | Transformation | Range | Higher is better | Edge case |
| --- | --- | --- | --- | --- | --- |
| T | absolute target-count error | `clip(1-e_K/max(K_HG,1))` | `[0,1]` | yes | missing target invalid |
| O | all-row outlier percentage | `clip(1-p_out/100)` | `[0,1]` | yes | empty input invalid |
| B | clustered-only largest share | `clip(1-r_max)` | `[0,1]` | yes | undefined gives `0.5` |
| S | clustered-only silhouette | `clip((s+1)/2)` | `[0,1]` | yes | undefined gives `0.5` |
| D | clustered-only DB | `1/(1+DB)` | `[0,1]` | yes | missing/negative gives `0.5` |
