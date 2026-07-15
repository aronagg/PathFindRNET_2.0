# Paper Results Text

The updated pipeline produced annotation-free dataset statistics for the Traffic Node Video Dataset 2.0 release. The analysis does not rely on manually annotated ground truth; instead, it evaluates practical trajectory usability through detection or tracked-observation volume, track persistence, continuity, interpolation demand, class consistency, trajectory roughness, clustering outlier behavior, and export coverage.

In the available new-release inputs, the script summarized `5` scenes, approximately `172261609.0` detection/tracked-observation rows, and `317854.0` trajectory tracks. These numbers should be reported as generated-output statistics, not as detector accuracy measures.

Compared with the previous release, the old-vs-new table reports absolute and relative changes for the main annotation-free indicators. The results indicate how the updated YOLO11x + Ultralytics/ByteTrack pipeline changes dataset volume, temporal persistence, continuity, and smoothness relative to the YOLOv7 + DeepSORT based release.

The annotation-free indicators show the practical usability of the generated trajectories, but they should be interpreted conservatively. They do not establish ground-truth detection accuracy on this dataset.
