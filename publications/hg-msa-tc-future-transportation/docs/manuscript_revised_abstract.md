# Revised Abstract

This study presents Homography-Guided Semantic Maneuver Graph Trajectory
Clustering (HG-SMG-TC), a leakage-controlled unsupervised workflow for clustering
vehicle trajectories from fixed traffic video sensors at complex urban
intersections. The method is evaluated on five Bellevue intersections from the
Traffic Node Video Dataset. The workflow separates three roles that are often
conflated in trajectory clustering: geometric scene-structure estimation,
unsupervised model selection, and independent reference evaluation. Homography is
used as a support layer for endpoint micro-mode discovery and maneuver-structure
estimation, while camera isotropic shared-scale coordinates remain the main
clustering representation. Endpoint micro-regions are consolidated into semantic
approach supernodes, aggregated into a semantic maneuver graph, and converted into
an uncertainty-aware maneuver-count prior for prior-constrained model selection.

The revised evaluation uses frozen target-estimation, model-selection and
independent-test splits. Independent-test assessment is performed only after
cluster assignments are persisted, using exhaustive human-defined
polygon-rule-based reference labels. Against the original untargeted selection,
the original HG-aware selection improves mean observed target-count error from
3.2000 to 2.5333. The HG-SMG-TC extension further reduces this error to 2.1333
and slightly increases mean NMI from 0.8376 to 0.8386 relative to the original
HG-aware method, while mean macro F1 changes from 0.6838 to 0.6723 and mean
outlier percentage changes from 9.94% to 10.34%. Baseline comparisons show that
simple endpoint KMeans can be strong, so the results are interpreted as
methodological trade-offs rather than universal dominance. The main contribution
is a reproducible, split-aware and reference-isolated framework for connecting
homography-derived scene structure with unsupervised trajectory-clustering model
selection.
