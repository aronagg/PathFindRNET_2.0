# Revised Related Work

## Vehicle Trajectory Clustering

Vehicle trajectory clustering has been studied for video, GPS and UAV-derived
trajectory data. Prior work covers endpoint-based origin-destination extraction,
shape-based similarity, density-based clustering, graph-based lane and movement
structure, and multi-criteria cluster ranking. This manuscript does not claim
novelty for trajectory clustering in general or for endpoint-based OD discovery
alone.

## Endpoint Regions and Turning Movements

Recent unsupervised endpoint-region methods identify persistent entry and exit
regions from vehicle trajectories and use them for turning-movement counts
[Rathore et al., 2026]. This directly overlaps with the endpoint-region discovery
component. HG-SMG-TC differs by using endpoint micro-modes as an intermediate
geometric layer that is consolidated into same-role semantic supernodes and then
used as an uncertainty-aware prior for clustering model selection.

## Trajectory Graphs and Lane-Level Structure

Graph-based and density-based methods have been used to infer lane-level road
structure and movement geometry from trajectories [Uduwaragoda et al., 2013;
Wang et al., 2017; Yuan et al., 2024; Wan et al., 2025]. These studies show that
geometric lane modes and semantic road approaches are distinct representational
levels. HG-SMG-TC does not reconstruct lane centerlines or a routable road
network. Its contribution is narrower: it consolidates endpoint micro-modes into
a semantic maneuver graph for clustering guidance.

## OD References and Clustering Evaluation

Trajectory clustering evaluation is sensitive to the reference definition. OD
endpoint references are useful but can blur the distinction between clustering,
classification and manual labeling [Rezaie and Saunier, 2021]. The revised
evaluation uses human-defined polygon-rule-based reference labels generated from
predefined entry and exit polygons. These labels are independent of cluster
assignments and model selection, but they are not fully manual per-trajectory
ground truth.

## Multi-Criteria Ranking and EMAS_HG

Multi-criteria ranking has been used for video trajectory clustering [Sekh et
al., 2020]. EMAS_HG in this work is a task-specific diagnostic score, not a
standard clustering-validity index and not the independent validation criterion.
Independent evidence is provided by ARI, NMI, purity and mapped F1 against the
polygon-rule reference.

## Novelty Boundary

The defensible novelty is the complete preregistered combination: homography
guided endpoint micro-mode discovery, label-free same-role semantic approach
consolidation, semantic maneuver graph aggregation, bootstrap uncertainty over
the observed maneuver target, prior-constrained unsupervised model selection, and
locked independent-test evaluation against an isolated rule-based reference.
Each constituent family has prior art and must be cited accordingly.
