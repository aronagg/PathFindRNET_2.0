# Revised Introduction

Fixed traffic cameras provide large volumes of vehicle trajectory data for
traffic monitoring, autonomous-technology evaluation, and AI-based intersection
scene understanding. A recurring task is to group observed trajectories into
movement patterns such as left-turn, through and right-turn maneuvers. This task
is difficult because video trajectories are affected by perspective distortion,
occlusion, incomplete tracks, lane-level variation, detector noise and
scene-specific geometry.

Unsupervised trajectory clustering is attractive because many intersections do
not have dense manual trajectory labels. However, unsupervised clustering also
creates a model-selection problem: different algorithms and hyperparameters can
produce plausible but incompatible partitions. Internal clustering metrics alone
can prefer compact partitions that do not align with maneuver structure. A
method that uses scene structure must therefore be evaluated carefully to avoid
target leakage.

The revised manuscript centers on Homography-Guided Semantic Maneuver Graph
Trajectory Clustering (HG-SMG-TC). The method uses homography-transformed
endpoints to estimate endpoint micro-modes, consolidates compatible same-role
micro-modes into semantic approach supernodes, builds a semantic maneuver graph
from observed origin-destination support, and converts this structure into an
uncertainty-aware maneuver-count prior for model selection. The extension is not
presented as a universally superior clustering algorithm. It is presented as a
reproducible framework for separating geometric prior construction from
independent reference evaluation.

The empirical study uses only the five Bellevue intersections from the Traffic
Node Video Dataset. It does not claim validation on the full dataset. The
reference labels are exhaustive human-defined polygon-rule-based labels: the
scene regions and legal movement mapping were manually specified, and the labels
were then generated deterministically for every canonical trajectory. These
labels are independent of clustering outputs, HG targets, EMAS scores and
model-selection decisions, but they are not fully manual per-trajectory ground
truth.

The final evidence supports several narrowed claims. First, split-aware
evaluation and persisted assignments reduce leakage risk. Second, maneuver-count
aware selection improves target-count alignment on average. Third, HG-SMG-TC
provides a more structured and uncertainty-aware prior than the original
point-target HG-aware selection. Fourth, the results contain trade-offs: HG-SMG
improves mean target-count error and slightly improves NMI relative to the
original HG-aware method, but it does not improve every agreement or outlier
metric. Fifth, strong endpoint baselines show that the paper should not claim
universal superiority over simple trajectory representations.

The contributions are:

1. A leakage-controlled split-aware evaluation protocol for trajectory clustering.
2. Exhaustive human-defined polygon-rule-based reference labels for independent
   evaluation.
3. A homography-guided endpoint micro-mode discovery layer.
4. Semantic approach consolidation and semantic maneuver graph construction.
5. An uncertainty-aware maneuver-count prior and prior-constrained model
   selection.
6. Independent-test and baseline evaluation showing method trade-offs.
7. A reproducibility package including formulas, frozen protocols, calibration
   diagnostics, sensitivity analyses and public-release planning.
