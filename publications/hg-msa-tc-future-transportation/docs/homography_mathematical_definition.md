# Homography Mathematical Definition

For camera point \(p_i=[x_i,y_i,1]^T\) and projective matrix \(H\), the destination
point is \(\hat q_i \sim H p_i\). If the rows of \(H\) are \(h_1^T,h_2^T,h_3^T\),

\[
\hat x_i' = \frac{h_1^T p_i}{h_3^T p_i},\qquad
\hat y_i' = \frac{h_2^T p_i}{h_3^T p_i}.
\]

The primary all-point forward residual is
\(e_i^f=\lVert\hat q_i-q_i\rVert_2\), measured in top-view pixels. We report
mean, median, root-mean-square (RMSE), maximum, P90 and P95 residuals. The normalized
RMSE is RMSE divided by the top-view image diagonal. Inverse residuals apply
\(H^{-1}\) and are reported separately in camera pixels. A unitless symmetric
diagnostic is

\[
e_i^{sym,n}=e_i^f/d_{top}+e_i^b/d_{camera}.
\]

RANSAC classifies a correspondence as an inlier when its destination reprojection
residual is within 10 px according to OpenCV's implementation. The final matrix is
normalized by \(H_{33}\). The normalized-DLT effective condition and null-space gap
are diagnostic only; they do not replace or alter the frozen matrix.

Calibration coverage is the source-point convex-hull area divided by image area.
Endpoint extrapolation is the fraction of target-estimation entry/exit endpoints not
covered by that source convex hull. Sparse residual interpolation is shown only inside
the calibration hull and is explicitly not a measured dense error field.
