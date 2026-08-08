# Major Revision Change Summary

## Overall Repositioning

The manuscript has been rewritten around HG-SMG-TC: Homography-Guided Semantic
Maneuver Graph Trajectory Clustering. The revised framing replaces broad
universal-improvement claims with a leakage-controlled, evidence-based
methodological contribution.

## Major Changes

1. Added a split-aware evaluation protocol separating target-estimation,
   model-selection and independent-test data.
2. Added exhaustive human-defined polygon-rule-based reference labels for
   independent evaluation.
3. Formalized the original homography-guided target estimator and documented
   that `K_HG` can count geometric endpoint submodes.
4. Added SE38th failure analysis: frozen `K_HG=18` versus 9 independently
   observed semantic movements.
5. Added HG-SMG-TC as the revised central method, including SAC, SMG, UATP and
   PCMS.
6. Added independent-test comparison of original untargeted A0, original
   HG-aware A1 and HG-SMG-TC A5.
7. Added endpoint and resampled-trajectory baselines.
8. Added EMAS_HG-v1 equations, edge cases and weight-sensitivity analysis.
9. Added homography calibration, RANSAC settings, residuals, quality gate and
   perturbation sensitivity.
10. Added scene-level paired statistical analysis and removed strong
    population-level p-value framing.
11. Added final figure/table plan and manuscript-ready tables.
12. Added data/code availability and public-release checklist.

## Main Numerical Updates

- Original untargeted A0 mean observed target-count error: 3.2000.
- Original HG-aware A1 mean observed target-count error: 2.5333.
- HG-SMG-TC A5 mean observed target-count error: 2.1333.
- Original HG-aware A1 mean NMI: 0.8376.
- HG-SMG-TC A5 mean NMI: 0.8386.
- Original HG-aware A1 mean macro F1: 0.6838.
- HG-SMG-TC A5 mean macro F1: 0.6723.
- Original HG-aware A1 mean outlier percentage: 9.94%.
- HG-SMG-TC A5 mean outlier percentage: 10.34%.

## Claims Narrowed or Removed

- Removed any claim of universal superiority.
- Removed any implication that EMAS_HG is independent validation.
- Removed any implication that polygon-rule labels are fully manual
  per-trajectory ground truth.
- Narrowed `K_HG` to an automatically estimated observed geometric-maneuver
  target.
- Narrowed dataset claims to the five Bellevue intersections only.

## Remaining Manual Submission Work

- Insert final journal-formatted references.
- Fill section, page and line numbers in the reviewer response.
- Check final DOCX figure placement and table formatting.
- Confirm third-party imagery licensing before submission.
