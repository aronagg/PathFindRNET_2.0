# HG-SMG-TC Preregistered Hypotheses

These hypotheses and metrics are frozen before the first HG-SMG-TC independent-test
execution. Scene is the experimental unit. Results will be shown per scene and method;
15 scene-method rows will not be described as 15 independent intersections.

## H1: SAC reduces micro-mode fragmentation

- Contrast: primary SAC versus frozen EMD micro-regions.
- Primary metrics: automatic-region-to-reference-approach completeness and number of
  automatic regions per dominant human-defined approach, evaluated only after outputs
  are frozen.
- Expected direction: completeness increases and fragmentation count decreases for
  entry and exit roles.

## H2: SMG reduces target over-segmentation without labels

- Contrast: A3 versus A1 target counts.
- Primary metric: scene-level absolute error against the independently observed test
  movement count; signed error is secondary.
- Expected direction: lower absolute target error.
- Safeguard: reference counts are evaluation-only and cannot change SAC, SMG, or the
  support-threshold heuristic.

## H3: UATP reduces single-K brittleness, especially for KMeans

- Contrast: A5 versus A3 and A10.
- Primary metrics: selected-K variation under preregistered development bootstrap,
  frequency of selection changes, and whether evaluated observed count lies inside the
  frozen interval.
- Expected direction: lower selected-K variability and fewer configuration changes;
  interval coverage is reported descriptively because only five scenes exist.

## H4: Full HG-SMG-TC improves independent structural agreement

- Contrast: A5 versus A1, paired within scene and method.
- Primary metrics: completeness, NMI, and mapped macro-F1.
- Guard metric: purity.
- Expected direction: positive scene-level median change for all three primary metrics.
  A purity decrease larger than 0.02 is defined a priori as practically material on
  the unit scale. This margin was fixed without consulting extension test outcomes.
- ARI, homogeneity, V-measure, weighted F1, noise percentage, internal metrics, and
  EMAS_HG are secondary or diagnostic.

## H5: Effects are not confined to SE38th

- Primary summary: per-scene A5-minus-A1 NMI and macro-F1 changes.
- Expected pattern: at least one non-SE38th scene improves in both metrics, and the
  five-scene table does not show an effect solely attributable to SE38th.
- No pooled trajectory-level significance test will substitute for scene variation.

## H6: Semantic targets are robust to resampling and calibration perturbation

- Primary metrics: UATP modal-target preservation probability, entropy, interval
  width, and overlap between the primary bootstrap distribution and Task-08-compatible
  +/-1, +/-2, +/-3, and +/-5 px calibration perturbation target distributions.
- Expected direction: higher modal preservation and lower entropy than the EMD point
  target diagnostic, without post hoc recalibration.
- The Task-08 matrices and frozen targets remain unchanged.

## H7: Main conclusions survive nuisance-parameter sensitivity

- Variants: heading windows 3/5/7, quantiles 0.90/0.95/0.975, camera-isotropic versus
  top-view heading, and UATP intervals 80/90/95%.
- Primary metrics: sign stability of A5-minus-A1 NMI and macro-F1, top-selection
  agreement, and Spearman rank correlation of candidate rankings.
- Expected direction: the primary qualitative signs persist across variants.
- Sensitivity variants may expose fragility but may never replace the primary protocol
  after extension test outcomes are visible.

## Decision and Reporting Policy

Hypotheses will be reported as supported, mixed, or unsupported. No pass threshold is
derived from previously seen independent-test values. With five scenes, descriptive
paired estimates and scene-level uncertainty take precedence over strong inferential
claims. Unfavorable scenes and the SE38th failure context remain visible.
