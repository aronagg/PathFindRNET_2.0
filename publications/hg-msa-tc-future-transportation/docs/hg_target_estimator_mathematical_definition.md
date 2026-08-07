# HG Target Estimator Mathematical Definition

## Trajectories and homography

For target-estimation trajectory `tau_i`, let the frozen camera-space feature endpoints be

`p_i^in = (x_i^in, y_i^in)^T` and `p_i^out = (x_i^out, y_i^out)^T`.

For `p = (x,y)^T`, the camera-to-top-view homography is

`h = H [x, y, 1]^T`, and `pi_H(p) = (h_1/h_3, h_2/h_3)^T`.

The implementation rejects `|h_3| < 1e-12`. It applies this transformation to every
authorized endpoint and does not read human reference labels.

## Separate entry and exit centers

For role `r` in `{in,out}`, define a separate coordinate-wise median center:

`c_r = (median_i z_{ir,x}, median_i z_{ir,y})^T`,

where `z_ir = pi_H(p_i^r)`. The executed code therefore has an entry center and an
exit center, not one joint intersection center.

## Polar-radius endpoint representation

For role `r`:

`d_ir = z_ir - c_r`,

`phi_ir = atan2(d_ir,y, d_ir,x)`,

`rho_ir = ||d_ir||_2`,

`rho'_ir = clip(rho_ir / max(median_j rho_jr, 1e-12), 0, 3) / 3`,

and the endpoint feature vector is

`u_ir = [cos(phi_ir), sin(phi_ir), 0.25 rho'_ir]^T`.

The circular encoding avoids an angle discontinuity at `-pi/pi`; the clipped radius
retains weak distance information without giving it the same scale as the two angular terms.

## Endpoint-region selection

For each role and each candidate `K in {3,4,5,6,7,8}`, KMeans minimizes

`sum_a sum_{i in R_a} ||u_ir - mu_ar||_2^2`

with `n_init=10`, Lloyd updates, and a frozen seed. Internal metrics are computed on
all rows when `N <= 2500`, otherwise on a deterministic 2500-row sample. The chosen K
is the lexicographic optimum:

1. maximum silhouette;
2. minimum Davies-Bouldin;
3. minimum K.

The resulting entry regions are `E_a`; exit regions are `X_b`.

## OD support and target

For every observed automatic pair `(a,b)`:

`n_ab = sum_i 1[z_i^in in E_a and z_i^out in X_b]`,

`q_ab = n_ab / N`.

For candidate support threshold `theta`, the target and retained coverage are

`K(theta) = sum_(a,b observed) 1[q_ab >= theta]`,

`C(theta) = sum_(a,b observed) n_ab 1[q_ab >= theta] / N`.

The comparison is inclusive (`>=`). Unobserved zero-support combinations do not appear
in the sum. The reported absolute support count is `ceil(theta N)`.

## Threshold heuristic

For ordered candidates `theta_j`, local target instability is

`I_j = mean(|K(theta_j)-K(theta_l)| : l is an adjacent grid index)`.

The estimator first retains candidates with `C >= 0.90` and `K >= 2`; if none exist it
uses `C >= 0.80`, then all candidates as a final fallback. It minimizes the tuple

`(I_j, |theta_j-0.005|, -C(theta_j), theta_j)`.

The final frozen output is `K_HG = K(theta*)`.

## Determinism and interpretation

KMeans labels are arbitrary integer identifiers, but frozen seeds and stable sorting
make the partition reproducible. The target is an automatically estimated count of
supported top-view geometric endpoint-pair modes. It is not guaranteed to equal a
legal or semantic maneuver count.
