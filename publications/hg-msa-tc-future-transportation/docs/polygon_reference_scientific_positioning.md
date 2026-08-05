# Scientific Positioning of the Polygon Reference

- The reference labels are exhaustive and deterministic over the canonical trajectory cohort.
- The region definitions and legal movement mappings were manually specified before independent-test clustering was evaluated.
- No clustering result, HG target, automatic OD assignment or EMAS score was used to generate the reference labels.
- The reference is based on camera-space entry and exit endpoint containment, not on manual inspection of every trajectory.
- Therefore it is a human-defined rule-based reference, not fully independent per-trajectory manual ground truth.
- It provides independent labels relative to the clustering output, but it shares endpoint information with the general maneuver-identification problem.
- Boundary, incomplete-track and polygon-definition uncertainty must be reported.

The primary scientific cohort remains the independent-test split. Labels for
`target_estimation` and `model_selection` document the full dataset and permit
diagnostics, but they must not alter frozen targets, configurations, thresholds,
EMAS weights, or candidate selection. The rule-generated reference measures
agreement with a manually specified movement codebook; it is not proof of lane-level
legality or complete physical trajectory correctness.

Manual review of the QA queue is a quality-control step only. It does not define the
reference labels and does not cherry-pick the scientific evaluation cohort.
