# HG-SMG-TC Literature Register

Verified: 2026-08-08. Bibliographic metadata and scope were checked against the
linked publisher record or arXiv record. Preprints are explicitly separated from
peer-reviewed work.

## Positioning Rule

HG-SMG-TC does not claim novelty for endpoint-region discovery, OD extraction,
trajectory graphs, lane-mode extraction, or road-topology inference in isolation.
The proposed contribution is the preregistered coupling of endpoint micro-modes,
label-free semantic approach consolidation, a semantic maneuver graph, a bootstrap
target interval, and prior-constrained unsupervised model selection.

## Verified Register

### Rathore, Pattanaik, and Rathore (2026)

Parikshit Singh Rathore, Vishwajeet Pattanaik, and Punit Rathore. "Unsupervised
Detection of Entry and Exit Regions from Vehicle Trajectories for Camera-Agnostic
Turning Movement Counts." arXiv:2607.10949, 2026.
[arXiv record](https://arxiv.org/abs/2607.10949)

- Status: preprint, submitted 12 July 2026.
- Method: unsupervised discovery of persistent entry/exit polygons from trajectory
  endpoints, followed by point-in-polygon turning-movement assignment.
- Relevance: establishes endpoint-region discovery as prior art.
- Distinction: HG-SMG-TC starts from endpoint micro-modes and addresses their
  consolidation into approach-level supernodes plus uncertainty-aware clustering
  guidance; it does not claim the endpoint discovery step itself.

### Wan et al. (2025)

Chongshan Wan, Peng Yue, Can Yang, Chuanwei Cai, and Xiaoxue Liu. "Lane extraction
from trajectories at road intersections based on Graph Transformer Network."
*International Journal of Geographical Information Science* 39(4), 758-787, 2025.
DOI: [10.1080/13658816.2024.2433086](https://doi.org/10.1080/13658816.2024.2433086).

- Status: peer-reviewed journal article.
- Method: directional, shape, and distance relations form a trajectory graph; a
  Graph Transformer extracts lane representatives in complex intersections.
- Relevance: trajectory relation graphs and lane-level mode extraction are prior art.
- Distinction: HG-SMG-TC does not learn lane centerlines. It consolidates already
  detected geometric endpoint modes for maneuver-count priors used in clustering.

### Yuan et al. (2024)

Mengyue Yuan, Peng Yue, Can Yang, Jian Li, Kai Yan, Chuanwei Cai, and Chongshan Wan.
"Generating lane-level road networks from high-precision trajectory data with
lane-changing behavior analysis." *International Journal of Geographical Information
Science* 38(2), 243-273, 2024. DOI:
[10.1080/13658816.2023.2279977](https://doi.org/10.1080/13658816.2023.2279977).

- Status: peer-reviewed journal article.
- Method: principal curves, a lane-intersection graph, and trajectory-flow topology
  recover lane-level road networks while treating merge/diverge and lane changing.
- Relevance: demonstrates that lane/geometric modes and road-level semantic
  structure are different representational levels.
- Distinction: HG-SMG-TC estimates semantic approach supernodes and maneuver-count
  uncertainty; it does not reconstruct a lane-level road network.

### Wang et al. (2017)

Jing Wang, Chaoliang Wang, Xianfeng Song, and Venkatesh Raghavan. "Automatic
intersection and traffic rule detection by mining motor-vehicle GPS trajectories."
*Computers, Environment and Urban Systems* 64, 19-29, 2017. DOI:
[10.1016/j.compenvurbsys.2016.12.006](https://doi.org/10.1016/j.compenvurbsys.2016.12.006).

- Status: peer-reviewed journal article.
- Method: curved GPS trajectories, density processing, and spatial inference for
  intersection and traffic-rule detection.
- Relevance: intersection semantics and traffic-rule inference from trajectories are
  established topics.
- Distinction: HG-SMG-TC neither claims general traffic-rule discovery nor constructs
  a routable map; it produces a task-specific prior for video-trajectory clustering.

### Uduwaragoda, Perera, and Dias (2013)

E. R. I. A. C. Uduwaragoda, A. S. Perera, and S. A. D. Dias. "Generating Lane
Level Road Data from Vehicle Trajectories using Kernel Density Estimation." In
*2013 16th International IEEE Conference on Intelligent Transportation Systems
(ITSC)*, 384-391, 2013. DOI:
[10.1109/ITSC.2013.6728262](https://doi.org/10.1109/ITSC.2013.6728262).

- Status: peer-reviewed conference paper.
- Method: kernel-density processing of vehicle GPS trajectories to estimate lane
  centerline geometry.
- Relevance: historical prior art for lane-level geometric mode extraction.
- Distinction: HG-SMG-TC does not claim density-based lane reconstruction and uses
  camera trajectories for semantic target-prior construction.

### Rezaie and Saunier (2021)

Mohsen Rezaie and Nicolas Saunier. "Trajectory Clustering Performance Evaluation:
If we know the answer, it's not clustering." arXiv:2112.01570, 2021.
[arXiv record](https://arxiv.org/abs/2112.01570).

- Status: preprint/technical report, submitted 2 December 2021.
- Method: comparison of similarity, clustering, and evaluation methods on seven
  intersections; automatic OD endpoint references support label-based evaluation.
- Relevance: OD-derived references and the absence of a universally dominant
  clustering setup are direct methodological context.
- Distinction: HG-SMG-TC uses an exhaustive human-defined polygon-rule reference only
  after selection and keeps its graph-derived prior isolated from that reference.

### Sekh et al. (2020)

Arif Ahmed Sekh, Debi Prosad Dogra, Samarjit Kar, and Partha Pratim Roy. "Video
trajectory analysis using unsupervised clustering and multi-criteria ranking."
*Soft Computing* 24, 16643-16654, 2020. DOI:
[10.1007/s00500-020-04967-9](https://doi.org/10.1007/s00500-020-04967-9).

- Status: peer-reviewed journal article.
- Method: interpretable video-trajectory features, unsupervised clustering, and
  multi-criteria ranking.
- Relevance: unsupervised model ranking for video trajectories is prior art.
- Distinction: HG-SMG-TC's claimed contribution is a scene-structure-derived interval
  prior and frozen selection constraint, not multi-criteria ranking generally.

### Betru, Tran, and Ectors (2025)

Abel Betru, Thi Tran, and Wim Ectors. "Automating Composition of Origin-Destination
Flows of Intersections Based on UAV Data." *Procedia Computer Science* 257,
233-240, 2025. DOI:
[10.1016/j.procs.2025.03.032](https://doi.org/10.1016/j.procs.2025.03.032).

- Status: peer-reviewed proceedings article, open access.
- Method: GMM, DBSCAN, and HDBSCAN support automated OD-flow extraction from UAV
  trajectory data.
- Relevance: automated OD composition is prior art.
- Distinction: HG-SMG-TC does not claim OD counting alone; it aggregates micro-mode
  OD support over label-free semantic supernodes and propagates target uncertainty.

### Du, Liu, and Meng (2023)

Jiusheng Du, Xingwang Liu, and Chengyang Meng. "Road Intersection Extraction Based
on Low-Frequency Vehicle Trajectory Data." *Sustainability* 15(19), 14299, 2023.
DOI: [10.3390/su151914299](https://doi.org/10.3390/su151914299).

- Status: peer-reviewed journal article.
- Method: direction-aware filtering, CDC clustering, and DBSCAN-based estimation of
  intersection centers from low-frequency GNSS trajectories.
- Relevance: direction-aware intersection-geometry extraction is established.
- Distinction: HG-SMG-TC assumes a frozen scene center and homography, then uses
  directional geometry to consolidate endpoint micro-modes for clustering guidance.

## Conservative Novelty Statement

The literature supports novelty only at the level of the complete, preregistered
problem formulation: homography-guided endpoint micro-mode discovery is followed by
label-free directional-geometric approach consolidation, semantic OD graph
aggregation, bootstrap uncertainty over the observed maneuver target, and interval-
constrained unsupervised model selection evaluated against an isolated exhaustive
rule-based reference. Each constituent family has prior art and must be cited as such.
