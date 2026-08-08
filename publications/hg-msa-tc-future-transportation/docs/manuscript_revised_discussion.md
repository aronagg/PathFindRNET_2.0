# Revised Discussion

The final evidence supports HG-SMG-TC as a structured, leakage-controlled
extension rather than a universally dominant clustering method. The strongest
result is improved maneuver-count alignment: original HG-aware selection improves
over untargeted selection, and HG-SMG-TC further reduces the aggregate target
error. This is consistent with the hypothesis that scene structure can make
unsupervised model selection more aligned with observed maneuver organization.

The independent reference metrics show a more nuanced picture. A5 slightly
improves mean NMI relative to A1, but mean macro F1 is lower and mean outlier
percentage is slightly higher. These differences are small in aggregate and vary
by scene and method. The manuscript should therefore separate target-count
alignment from label-agreement metrics and outlier behavior.

SE38th is scientifically useful because it exposes the boundary between
geometric endpoint submodes and semantic maneuvers. The original target estimator
counts supported geometric endpoint-region pairs. In favorable scenes these
align with semantic movements, but SE38th shows that they can over-segment a
physical approach. HG-SMG-TC addresses this by consolidating compatible
same-role micro-regions before constructing the maneuver graph. This supports a
methodological contribution, but not a claim that the estimator always recovers
true legal maneuver counts.

The baseline comparison also narrows the claims. Endpoint KMeans is strong in
several settings, confirming that much of the maneuver signal is already present
in entry and exit endpoints. HG-SMG-TC should therefore be positioned as a
selection and scene-structure framework, not as proof that complex trajectory
features always outperform endpoint features.

Homography contributes as a geometric support layer. It enables top-view
endpoint analysis and structure estimation, but the final clustering
representation remains camera isotropic shared-scale. The homography quality
analysis increases reproducibility and transparency, while the high endpoint
extrapolation fractions and manual point-pair dependence remain important
limitations.

Overall, the revised paper should claim that HG-SMG-TC provides a reproducible
and better controlled way to connect scene geometry with unsupervised trajectory
clustering. It should avoid claims of universal superiority, full ground truth,
or full-dataset generalization.
