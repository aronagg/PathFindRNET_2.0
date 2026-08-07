# HG-SMG-TC Mathematical Specification

This document freezes the post-review extension before its first execution on the
extension test split. The method is not implemented and no HG-SMG-TC result is
generated in this task.

## 1. Inputs and Split Isolation

For scene `s`, let `T_s^te` contain only `target_estimation` trajectories and
`T_s^ms` contain only `model_selection` trajectories. EMD, SAC, SMG, and UATP may
read only `T_s^te`; PCMS may read only `T_s^ms` plus the already frozen UATP prior.
No semantic reference label or independent-test result is a method input.

The five scenes are fixed. Task-08 engineering screening cannot remove a scene.

## 2. EMD: Frozen Endpoint Micro-Mode Discovery

EMD is unchanged from `split-aware-hg-msa-tc-v1`. Canonical camera endpoints are
mapped by the frozen homography and represented by polar angle plus the frozen weak
radial term. Entry and exit KMeans candidates are `k in {3,...,8}`. Candidate choice,
seeds, metric sampling, and tie-breaking remain frozen. EMD returns entry micro-
regions `E_i`, exit micro-regions `X_j`, trajectory assignments, and micro-OD counts
`n_ij`.

## 3. SAC: Semantic Approach Consolidation

SAC operates separately on entry and exit roles. An entry micro-region can never be
merged with an exit micro-region.

### 3.1 Region descriptors

Let `P_r` be the top-view endpoint set for micro-region `r`. Its robust centroid is
the componentwise median `m_r`. With the frozen role-specific EMD center `c_role`,
the region-side bearing is:

```text
beta_r = atan2(m_r,y - c_role,y, m_r,x - c_role,x).
```

For each trajectory, exactly five finite points adjacent to its canonical endpoint
are used. The entry heading is the directed vector from point 1 to point 5. The exit
heading is the directed vector from the fifth-last point to the last point. A
trajectory with fewer than five such points has undefined heading. Camera coordinates
are normalized by one shared isotropic scale before heading calculation; isotropic
translation and scale preserve direction. Top-view heading is sensitivity-only.

For angles `a,b`, define circular distance:

```text
d_c(a,b) = abs(((a - b + pi) mod (2*pi)) - pi).
```

The region heading `eta_r` is the circular L1 median of valid directed headings:
the observed angle minimizing the sum of circular distances, with ties resolved by
the lowest wrapped angle in `[-pi,pi)`. Bearing and heading dispersions are circular
median absolute deviations.

### 3.2 Self-calibrated compatibility

For each region and descriptor, perform 500 deterministic trajectory bootstraps. In
each replicate calculate the circular center and the 0.95 quantile of circular
residuals. The region self-consistency radius is the 0.95 quantile of those 500
bootstrap residual radii. Let these be `R_beta,r` and `R_eta,r`.

For same-role regions `r,s`:

```text
z_beta(r,s) = d_c(beta_r,beta_s) / max(R_beta,r + R_beta,s, eps)
z_eta(r,s)  = d_c(eta_r,eta_s)   / max(R_eta,r  + R_eta,s,  eps)
D(r,s) = max(z_beta(r,s), z_eta(r,s)).
```

`eps` is eight times float64 machine epsilon in radians. The max construction gives
equal logical necessity to bearing and heading without fitted feature weights.
Regions are compatible exactly when `D <= 1`.

Start with singleton clusters. Complete-link distance between two clusters is the
maximum `D(r,s)` across all cross-cluster region pairs. Repeatedly merge the eligible
pair with the smallest complete-link distance. Ties use the lexicographically sorted
tuples of original micro-region IDs. Stop when no distance is at most one. Supernodes
are ordered by role, wrapped bearing, heading, then minimum member ID.

This rule was selected before implementation for identifiability, deterministic
behavior, lack of manually fitted feature weights, and explicit use of within-mode
dispersion. It was not selected against semantic labels or test metrics.

### 3.3 Edge cases

- Fewer than two finite bearing samples: retain the region as an isolated supernode.
- Fewer than two valid five-point headings: retain it as isolated in primary SAC.
- One region for a role: return it unchanged.
- Zero dispersion: use the numerical floor; only numerically identical descriptors
  can pass that component.
- Non-finite descriptor or bootstrap failure: retain an invalid isolated node and
  report it; no nearest-region fallback is allowed.
- A zero-support region is retained and reported.

### 3.4 OD profile diagnostic

Normalized entry profiles use destination micro-region probabilities; exit profiles
use source micro-region probabilities. Jensen-Shannon distance uses base-2 logarithms
and its square root. A zero-support profile is undefined and isolated. OD compatibility
is used only in A8 and is not a primary SAC requirement.

## 4. SMG: Semantic Maneuver Graph

Let `u` and `v` be SAC entry and exit supernodes. Aggregate all micro-OD counts:

```text
n_tilde_uv = sum_(i in u) sum_(j in v) n_ij
q_tilde_uv = n_tilde_uv / N
K_SMG = sum_(u,v) I[q_tilde_uv >= theta].
```

`N` is the number of target-estimation trajectories with valid entry and exit
supernode assignments. Missing geometry is excluded from `N` and counted. No endpoint
or edge is imputed.

The frozen threshold grid `{0.001, 0.0025, 0.005, 0.01, 0.02}` and original selection
heuristic are reused after aggregation: require coverage at least 0.90 and at least two
supported edges, then minimize adjacent-threshold target instability, minimize
distance to 0.005, maximize coverage, and choose the lower threshold. The existing
0.80 coverage and all-candidate fallbacks remain unchanged.

## 5. UATP: Uncertainty-Aware Target Prior

UATP runs 500 deterministic hierarchical bootstraps of `T_s^te`. Each replicate
samples recordings with replacement and then samples trajectories within each selected
recording occurrence with replacement while preserving its original size. The complete
EMD to SAC to SMG pipeline is rerun, producing `K_SMG^(b)`.

Report the mode (smaller K on ties), median, Shannon entropy in bits, and the primary
90% percentile interval:

```text
[L,U] = [floor(Q_0.05(K_SMG)), ceil(Q_0.95(K_SMG))].
```

Quantiles use the linear method. The 80%, 90%, and 95% interval variants are frozen
sensitivities; 90% remains primary regardless of later outcomes.

## 6. PCMS: Prior-Constrained Model Selection

For candidate non-noise cluster count `K` and prior `[L,U]`:

```text
d_I(K) = 0       if L <= K <= U
         L - K   if K < L
         K - U   if K > U.
```

This replaces point-target absolute error at exactly the first position of the frozen
HG-aware selection key. Density methods next minimize outlier percentage; all methods
then use the existing silhouette, Davies-Bouldin, Calinski-Harabasz, largest-cluster,
and lexical parameter tie-break order. KMeans chooses from the existing candidate
grid and is never forced to one target. HDBSCAN and OPTICS use non-noise counts.
EMAS_HG remains diagnostic only.

## 7. Randomness and Sensitivities

The master seed is `20260901`. Derived seeds are the first eight bytes of SHA-256 over
`master_seed|scene|module|role|region_or_replicate`, interpreted big-endian modulo
2,147,483,647. Full non-bootstrap EMD retains its frozen seeds.

Preregistered sensitivities are heading windows 3/5/7, self-consistency quantiles
0.90/0.95/0.975, camera-isotropic versus top-view heading, and UATP interval levels
80/90/95%. None may replace the primary after test inspection.

## 8. Interpretation Boundary

`K_SMG` estimates supported observed maneuver structure under the frozen endpoint and
support rules. It is not a true legal maneuver count. SAC is label-free but remains a
geometric heuristic. The extension is post-review and its original independent-test
motivation is already known; only its subsequent protocol freeze is prospective.
