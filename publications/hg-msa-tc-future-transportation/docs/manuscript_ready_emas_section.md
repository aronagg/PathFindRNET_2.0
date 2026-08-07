# Manuscript-Ready EMAS_HG Material

## Methods subsection

We report the task-specific Expected-Maneuver-Aware Score, EMAS_HG-v1, as a diagnostic
composite. Let `e_K=|K-K_HG|`, `p_out` be the percentage of noise-labelled
trajectories, `r_max` the largest non-noise cluster share, `s` the clustered-only
silhouette and `DB` the clustered-only Davies-Bouldin index. Its components are

`T=clip(1-e_K/max(K_HG,1))`, `O=clip(1-p_out/100)`,
`B=clip(1-r_max)`, `S=clip((s+1)/2)`, and `D=1/(1+DB)`.

The combined score is

`EMAS_HG-v1 = 0.50T + 0.20O + 0.10B + 0.10S + 0.10D`.

Noise rows are excluded from K, r_max, silhouette and DB, but are included in the
outlier denominator. KMeans therefore normally has `O=1`. Missing r_max, silhouette
or DB values receive the frozen neutral fallback `0.5`; this covers all-noise and
one-cluster cases. A missing target is invalid, while target zero uses the legacy
denominator one. The weights are non-negative and sum to one, so valid scores lie in
`[0,1]`.

The 0.50 target weight encodes the study-specific priority of maneuver-count alignment;
the 0.20 outlier weight penalizes unassigned trajectories, while the remaining 0.30 is
distributed across dominance and internal separation. These are heuristic design
weights, not universally optimal constants. Importantly, the frozen model-selection
keys did not use EMAS_HG; it was reported after selection.

We assessed diagnostic-ranking sensitivity on all 255 development candidates using
seven pre-specified scenarios, a 465-vector local grid at 0.05 resolution, and 1,000
fixed-seed Dirichlet vectors. No independent-test metric was used to choose weights,
and test clustering was not repeated.

## Results paragraph

The canonical implementation reproduced 345 stored development and independent-test
scores with maximum absolute error `5.493e-13`. The reviewer-proposed
`0.40T+0.30O+0.10B+0.10S+0.10D` weights preserved the original development-candidate
top rank in 14/15 scene-method groups; only SE38th/HDBSCAN changed. Across the local
grid, top-rank preservation was 93.88%, with mean Spearman and Kendall correlations of
0.9673 and 0.9198. Sensitivity was concentrated in 116th/NE12th OPTICS and SE38th
HDBSCAN, whereas ten scene-method groups retained the original top candidate for all
local vectors.

## Discussion and limitation

Target agreement dominates EMAS_HG through both its 0.50 weight and its comparatively
large variance. Silhouette and transformed DB are strongly correlated (`r=0.8862`),
so the components are not orthogonal. The observed local robustness supports using
the original score as a transparent task-specific diagnostic, but the fragile cases
and broad-simplex reversals preclude claims of universal optimality. EMAS_HG is not an
independent validation metric; conclusions about clustering agreement rely on the
polygon-rule reference evaluation.
