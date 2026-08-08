# Final Scientific Interpretation

## Main Contribution

The revised contribution is a leakage-controlled trajectory-clustering evidence
package for five Bellevue intersections. The strongest methodological point is
not a single score or a universal clustering improvement, but the combination of
homography-guided maneuver-structure estimation, frozen split-aware selection,
and independent evaluation against exhaustive human-defined polygon-rule
reference labels.

## Original HG-Aware Method

Original HG-aware A1 improves target-count alignment relative to original
untargeted A0. In aggregate, A1 has mean observed target error
2.5333, compared with A0's
3.2000.
This supports the idea that maneuver-count awareness is useful, but it does not
establish broad superiority across every metric.

## What HG-SMG-TC Adds

HG-SMG-TC A5 replaces the point-target prior with a structured maneuver graph
and interval-aware selection. A5 has mean observed target error
2.1333, mean NMI 0.8386, and mean macro
F1 0.6723. The evidence supports A5 as a stronger and better
documented post-review extension, while still requiring trade-off language.

## Where It Helps Most

The method is most useful where the original target estimator is vulnerable to
semantic over-segmentation, especially SE38th. The final results and Task 07
diagnostics show that SE38th should be presented as a failure analysis and
methodological motivation, not hidden as an outlier.

## Where It Does Not Help

Endpoint-based KMeans is a strong baseline in several cases. Full resampled
trajectory features do not clearly dominate endpoint features in aggregate.
Therefore, the revised paper should not claim that HG-SMG-TC universally
outperforms every simple representation or every baseline metric.

## Baseline Evidence

The strongest endpoint baseline is `endpoint_camera_isotropic`. Its aggregate NMI is
0.7099, compared with A5's 0.8386. Its macro F1 is
0.5710, compared with A5's 0.6723. This is a
credible baseline and should be discussed directly.

## Recommended Abstract and Conclusion Claims

Use cautious wording:

- The study evaluates five Bellevue intersections, not the full dataset.
- Human-defined polygon-rule labels provide an independent reference relative
  to clustering outputs, but they are not per-trajectory manual ground truth.
- HG-aware and HG-SMG selection improve maneuver-count alignment on average.
- HG-SMG improves the evidence basis and mitigates semantic over-segmentation
  failure modes, but the results contain metric trade-offs.
- EMAS_HG is a task-specific score; independent validation comes from ARI, NMI,
  purity and mapped F1 against the polygon-rule reference.
