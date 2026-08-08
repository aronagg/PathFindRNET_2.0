# Revised Conclusion

This revised study introduces HG-SMG-TC, a homography-guided semantic maneuver
graph framework for unsupervised vehicle trajectory clustering at complex urban
intersections. The method combines endpoint micro-mode discovery, semantic
approach consolidation, maneuver-graph construction, bootstrap target
uncertainty and prior-constrained model selection. Its evaluation is
split-aware, uses persisted independent-test assignments, and compares outputs
against exhaustive human-defined polygon-rule-based reference labels.

The evidence supports cautious conclusions. Original HG-aware selection improves
target-count alignment over untargeted selection. HG-SMG-TC further improves mean
target-count alignment and slightly improves NMI relative to the original
HG-aware method, while showing trade-offs in macro F1 and outlier percentage.
The method is most informative in the SE38th case, where the original
homography-guided target estimator over-segments semantic maneuver structure.
HG-SMG-TC mitigates this through semantic consolidation, but it does not remove
all limitations.

The revised manuscript should conclude that homography-guided scene structure is
useful for unsupervised trajectory-clustering model selection when handled with
strict split control and independent reference evaluation. It should not claim
universal superiority, full manual ground truth, or generalization beyond the
five Bellevue scenes evaluated here.
