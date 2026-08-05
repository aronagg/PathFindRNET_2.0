# Repository and Data Audit

## Scope

This audit supports the Future Transportation major revision for the HG-MSA-TC study.
It covers exactly five Bellevue scenes and records the repository state at commit
`1c82be2863679b021d35abfc932d4a7247554959`. The worktree already contained unrelated
tracked and untracked changes; this task did not alter them.

Terminology in this report:

- **Verified**: directly observed in code, configuration, file metadata, or generated
  data.
- **Inferred**: reproducible interpretation based on verified artifacts, but not stored
  explicitly in the source table.
- **Unresolved**: information or protocol choice that cannot be established without a
  separate decision or new data.

## High-Level Finding

**Verified.** The completed HG-MSA-TC runner estimates an automatic HG target, selects
clustering parameters against that target, and reports target-alignment terms including
EMAS_HG on the same trajectory sample. Its bootstrap repeats target estimation and
selection on the same 80% subsample. This creates the circularity raised by the
reviewer: target-alignment performance is not independent of the target used to select
the candidate.

**Protocol response.** This directory freezes three non-overlapping temporal subsets:

1. `target_estimation`: estimate the automatic observed maneuver target;
2. `model_selection`: select a candidate with the target fixed;
3. `independent_test`: final evaluation, including manual ground truth, after all
   decisions are frozen.

No algorithm, historical result, or manuscript artifact was changed in this task.

## Scene Mapping and Recording Structure

| Scene ID | Raw/interim recording prefix | Recordings with final trajectories | Final trajectories |
| --- | --- | ---: | ---: |
| `bellevue_116th_ne12th` | `Bellevue_116th_NE12th__...` | 21 | 2,321 |
| `bellevue_150th_newport` | `Bellevue_150th_Newport__...` | 24 | 9,448 |
| `bellevue_150th_eastgate` | `Bellevue_150th_Eastgate__...` | 24 | 26,266 |
| `bellevue_150th_se38th` | `Bellevue_150th_SE38th__...` | 23 | 9,060 |
| `bellevue_ne8th` | `Bellevue_Bellevue_NE8th__...` | 23 | 19,934 |

**Verified.** Raw videos are under `data/raw/<scene>/*.mp4`. Per-video tracker tables
are under `data/interim/<scene>/tracks_<video>.parquet`. Filenames encode local
recording start times from 2017-09-10 through 2017-09-11. Scene configs specify 30 fps.

**Verified.** Frame IDs restart at zero for each source recording. The merged
`trajectories.parquet` tables therefore contain overlapping frame ranges and cannot be
chronologically split by frame number alone.

**Verified.** `scripts/merge_tracks.py` sorts per-video shards lexicographically and
adds the previous global maximum plus one to each subsequent shard's local track IDs.
It does not persist a `video_id` column. The manifest generator reproduces this offset
rule from Parquet metadata. All 67,029 final trajectories map to exactly one shard.

**Inferred.** `source_recording_id`, `source_recording_track_id`, `start_time`, and
`end_time` in the manifest are reconstructed provenance. They are not columns in the
final feature source. Time is filename start plus frame/30 fps; no timezone is encoded.

## Data and Preprocessing Artifacts

### Raw and tracker data

| Layer | Path | Format and key fields | Role |
| --- | --- | --- | --- |
| Raw videos | `data/raw/<scene>/*.mp4` | MP4 | Camera source recordings. |
| Per-video tracking | `data/interim/<scene>/tracks_*.parquet` | `frame, track_id, cls, conf, cx, cy, w, h` | Detector/tracker output with source recording in filename. |
| Merged tracking | `data/interim/<scene>/tracks.parquet` when retained | Same Parquet schema | Multi-video table with offset IDs. |
| Point trajectories | `data/processed/<scene>/trajectories.parquet` | `track_id, frame, x, y, vx, vy, ax, ay` | Smoothed, long-form trajectories. |

Relevant code:

- `scripts/run_track.py`: detector/tracker execution.
- `scripts/merge_tracks.py`: per-video merge and track-ID offsetting.
- `scripts/build_trajectories.py`: vehicle class filter and trajectory build.
- `src/traffic/trajectories/build.py`: Savitzky-Golay smoothing and derivatives.
- `scripts/filter_fill_trajectories.py`: short-track filtering and gap filling.
- `src/traffic/trajectories/postprocess.py`: minimum 15 points, duplicate-frame mean,
  linear interpolation, maximum configured gap (default 120 inserted frames).

### Feature and final clustering input

| Layer | Path | Format and key fields |
| --- | --- | --- |
| Core feature layer | `data/processed/<scene>/feature_layers/features_core_raw.parquet` | One row/track; frame range, endpoints, path/displacement, straightness, speed/acceleration summaries. |
| Scene normalization layer | `.../feature_layers/features_scene_norm.parquet` | Endpoint normalization and normalized displacement. |
| Compatibility feature table | `data/processed/<scene>/features.parquet` | Combined one-row/track representation. |
| Threshold audit | `.../feature_analysis/feature_threshold_summary.json` | Configured thresholds and retention. |
| Final HG-MSA-TC input | `.../feature_analysis/features_trimmed_frame_disp_norm.parquet` | One row per retained trajectory; canonical cohort for this protocol. |

**Verified missing artifact.** Each `feature_layers_manifest.json` records
`data/processed/<scene>/trajectories_filtered_filled.parquet` as the source used to
build the active feature layers, and records `has_filled_col: true`. Those five cleaned
point-level files are no longer present at the current scene roots. They exist in some
legacy/sidecar locations, but those copies cannot be assumed byte-identical to the
declared active source. The active final feature inputs are present and checksummed;
byte-level revalidation of their cleaned point-level parent is currently blocked.

Relevant code:

- `scripts/gen_features.py`: feature layers and compatibility table.
- `scripts/analyze_features.py`: quantiles, candidate trims, and visual analysis.
- `scripts/apply_feature_trim.py`: deterministic trim output generation.
- `scripts/make_trimmed_features.py`: older frame-span trim utility.

**Verified filtering.** The active final files apply scene-level frame-span quantiles
0.20 to 0.90 plus `displacement_norm >= 0.6`. The threshold values were derived on the
full scene feature table before this revision protocol existed.

| Scene | Tracks in `trajectories.parquet` | Cleaned feature input | Frame-span range | Final retained | Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 116th/NE12th | 28,093 | 14,027 | 28 to 702 | 2,321 | 16.55% |
| 150th/Newport | 107,268 | 63,475 | 31 to 632 | 9,448 | 14.88% |
| 150th/Eastgate | 171,998 | 97,527 | 30 to 708.4 | 26,266 | 26.93% |
| 150th/SE38th | 107,728 | 64,764 | 31 to 707 | 9,060 | 13.99% |
| NE8th | 129,443 | 78,061 | 37 to 1,428 | 19,934 | 25.54% |

**Risk.** Although these filters do not use manual labels, their quantiles were computed
from all scene trajectories, including future split periods. This task freezes the
existing cohort because changing preprocessing is prohibited. The manuscript should
state this, and sensitivity to split-fitted preprocessing is a possible later analysis.

## Canonical Trajectory Unit and Schema

**Verified.** The completed HG-MSA-TC runner reads
`features_trimmed_frame_disp_norm.parquet` and uses `track_id`, `start_x`, `start_y`,
`end_x`, and `end_y`. The protocol therefore defines one canonical sample as one row
in this final file, not one point in `trajectories.parquet`.

**Verified identifiers.** Each source row has a unique merged integer `track_id` within
scene. The manifest preserves it as `original_track_id` and creates the globally unique
stable ID `<scene_id>:<track_id>`.

**Verified coordinate fields.** Camera endpoint fields are `start_x`, `start_y`,
`end_x`, and `end_y`. Full point trajectories use `x` and `y`. Homography output uses
`start_x_topview`, `start_y_topview`, `end_x_topview`, and `end_y_topview`.

**Verified frame/length fields.** Final features contain `frame_start`, `frame_end`,
`len_frames`, `frame_span`, and `duration_s`. Final cohort length statistics are:

| Scene | Minimum points | Median points | Maximum points |
| --- | ---: | ---: | ---: |
| 116th/NE12th | 44 | 223 | 703 |
| 150th/Newport | 32 | 245 | 633 |
| 150th/Eastgate | 33 | 226 | 709 |
| 150th/SE38th | 55 | 247 | 708 |
| NE8th | 41 | 272 | 1,429 |

**Verified class behavior.** `scripts/build_trajectories.py` keeps configured vehicle
labels `car`, `truck`, and `bus`, mapped to COCO IDs 2, 7, and 5. The exact class is then
dropped before trajectory construction. It cannot be reliably populated in the final
manifest and is left empty with status `not_retained_after_vehicle_class_filter`.

## Homography Components

| Component | Repository path | Function |
| --- | --- | --- |
| Calibration configs | `research_experiments/fov2026_trajectory_clustering/configs/hg_msa_tc_five_scene/homography_<scene>.json` | Camera-to-top-view matrices and calibration metadata. |
| Five-scene index | `.../configs/hg_msa_tc_five_scene/homography_five_scene_index.yaml` | Asset paths, quality status, and usability. |
| Point pairs | `.../outputs/hg_msa_tc_five_scene/calibration_inputs/<scene>/point_pair_template.csv` | Manually selected camera/top-view correspondences. |
| Calibration runner | `.../scripts/hg_msa_tc_five_scene/run_homography_calibration_five_scene.py` | `cv2.findHomography`, RANSAC threshold 10 px, quality metrics. |
| Top-view transformation | `run_hg_msa_tc_five_scene_pipeline.py` | Applies the matrix to trajectory endpoints. |

**Verified.** All five scenes are marked usable. Calibration uses 13 to 26 retained
point pairs depending on scene. Manual point selection is still required.

**Scene-specific manual decision.** NE8th homography estimation excludes manually
identified point-pair IDs 12, 13, 15, 21, and 23. The original 23-row CSV remains, while
the final 18-point calibration uses
`point_pair_template_excluding_problem_points_12_13_15_21_23.csv`.

## HG Target Estimation, Clustering, and EMAS_HG

Primary implementation:

`research_experiments/fov2026_trajectory_clustering/scripts/hg_msa_tc_five_scene/run_hg_msa_tc_five_scene_pipeline.py`

**Verified target estimator.** The runner:

1. transforms camera start/end points to top-view coordinates;
2. represents each endpoint by polar angle and a weighted radius term;
3. selects entry and exit KMeans region counts from `k=3..8` using silhouette, then
   Davies-Bouldin, then lower `k`;
4. counts observed entry-exit pairs;
5. evaluates support thresholds 0.001, 0.0025, 0.005, 0.01, and 0.02;
6. prefers coverage at least 0.90, minimizes adjacent-threshold target instability,
   then proximity to 0.005 support and higher coverage.

Historical declared targets are loaded only for the separately named diagnostic
`diagnostic_declared_vs_hg_target_five_scene`; they are not method input in this HG
runner. The completed automatic HG targets are 10, 12, 9, 12, and 10 in scene order.

**Verified candidate grids.** Methods are KMeans, HDBSCAN, and OPTICS:

- KMeans: `n_clusters=2..30`, `n_init=10`, `max_iter=300`.
- HDBSCAN: `min_cluster_size` in 80, 160, 320 and `min_samples` in 10, 20.
- OPTICS: `min_samples` in 40, 80; `xi` in 0.05, 0.07; `max_eps` from four
  nearest-neighbor distance quantiles computed on the current sample.

**Verified selection rules.** Untargeted selection sorts by highest silhouette, lowest
Davies-Bouldin, highest Calinski-Harabasz, lowest largest-cluster ratio, and fit time.
HG expected-aware selection first minimizes absolute cluster-count error to the HG
target; HDBSCAN/OPTICS then minimize outlier percentage; internal metrics follow.

**Verified EMAS_HG.** Its weighted terms are target alignment 0.50, non-outlier share
0.20, cluster balance 0.10, scaled silhouette 0.10, and transformed Davies-Bouldin
0.10. Because target alignment is both a primary selection criterion and half of
EMAS_HG, EMAS_HG on the same sample is not independent evidence.

**Verified sampling/seeds.** Constants are:

- `RANDOM_STATE = 20260702`;
- `MAX_EVAL_TRACKS = 6000`;
- internal metric sample maximum 3,000;
- target bootstraps 50;
- selection bootstraps 20;
- subsample fraction 0.80.

The main runner randomly samples up to 6,000 trajectories per scene before target
estimation and clustering. Bootstrap code re-estimates the target and re-runs search on
the same subsample. The incremental bootstrap script reuses candidate parameter lists
saved by the full run.

Other relevant fixed seeds are 42 in the normalization and compact-feature ablations,
20260702 in `run_stability_validation_msa_tc.py`, and 7 for several visualization-only
samples. Generic clustering methods default to 42 in
`src/traffic/cluster/methods.py` when no config value is supplied.

## Result, Figure, and Table Generation

Main result generation:

- `run_hg_msa_tc_five_scene_pipeline.py`: endpoint, target, selection, bootstrap, OD
  pseudo-reference, coordinate ablation, CSV, report, and figure outputs.
- `run_hg_msa_tc_selection_bootstrap.py`: incremental selection bootstrap CSV/report.
- `finalize_hg_msa_tc_five_scene_outputs.py` and
  `finalize_strict_quality_gate_after_point_pairs.py`: quality-gate/final status assets.

Publication asset generation:

- `generate_final_manuscript_hg_msa_tc.py`
- `generate_paper_revision_v2_assets.py`
- `generate_figure4_clean_same_type.py`
- `research_experiments/fov2026_trajectory_clustering/scripts/build_conference_presentation_pack.py`
- `build_final_manuscript_msa_tc.py`
- `build_fov2026_final_master_report.py`
- `build_paper_master_report.py`

Existing HG reports, result CSVs, and figures are under the corresponding
`reports/hg_msa_tc_five_scene`, `outputs/hg_msa_tc_five_scene`, and
`figures/hg_msa_tc_five_scene` directories. This task reads but does not regenerate or
edit them.

## Configuration and Environment

Scene configs are `configs/dataset/<scene>.yaml`; Hydra entry configuration is
`configs/defaults.yaml`; FoV scene and historical target registries are under
`research_experiments/fov2026_trajectory_clustering/configs/`.

**Verified ambiguity.** `configs/dataset/bellevue_150th_newport.yaml` has a `video`
value pointing to an external Eastgate file, while its `raw_dir`, scene ID, feature
source, and actual local raw files are Newport. Multi-video processing uses the scene
directories, but any single-video command that consumes this field directly is at risk
of reading the wrong scene.

Package definition: `pyproject.toml`, project `traffic-research==0.1.0`, Python >=3.10.
No lockfile or Conda environment file was found as the primary environment definition.

Detected runtime for this audit:

- Python 3.11.9 on Windows;
- numpy 2.2.6, pandas 2.3.3, pyarrow 22.0.0;
- scikit-learn 1.8.0, scipy 1.16.3, hdbscan 0.8.41;
- OpenCV 4.12.0.88, matplotlib 3.10.8;
- Hydra 1.3.2, OmegaConf 2.3.0, PyYAML 6.0.3;
- pytest 9.0.2, openpyxl 3.1.5.

## Leakage and Bias Risks

1. **Circular target evaluation:** the same automatic target currently drives candidate
   selection and target-alignment reporting on the same sample.
2. **Bootstrap non-independence:** target estimation and selection share each bootstrap
   sample; this measures resampling behavior, not independent target validity.
3. **OD pseudo-reference circularity:** OD pseudo-labels are derived from the same
   endpoint-region geometry and are not manual ground truth.
4. **Global preprocessing:** scene quantiles and normalized-displacement filters were
   derived before temporal splitting.
5. **Global normalization:** current coordinate normalization is fitted on whichever
   complete sample is passed to the runner. Future independent-test code must declare
   whether scaling is model-selection-fitted or transductive.
6. **Manual calibration:** homography point pairs and the NE8th exclusions are manual,
   scene-specific inputs. They do not use maneuver labels but must be disclosed.
7. **Post hoc historical targets:** manually declared values remain in older configs and
   reports. The HG runner uses them only diagnostically, but future code must preserve
   this separation.
8. **Figure/claim selection:** inspecting independent-test labels or metrics before all
   claims and display rules are frozen would leak evaluation information.

## Unresolved Questions

- The exact vehicle subclass cannot be recovered from final feature files without a
  separate audited reconstruction from upstream detection rows.
- The cleaned `trajectories_filtered_filled.parquet` parents named by the feature-layer
  manifests are absent from current scene roots, preventing byte-level lineage checks.
- Recording filename times have no explicit timezone or authoritative timestamp
  metadata.
- The final study must predeclare whether density methods are re-fit independently on
  `independent_test` (transductive clustering) or require an inductive assignment rule.
- OPTICS `max_eps` quantiles are sample-dependent; the revised runner must compute them
  only in the permitted model-selection stage and freeze resulting candidates.
- A split-aware preprocessing sensitivity analysis may be useful, but is prohibited in
  this task and is not required to use the frozen current cohort.
- Manual annotation guidelines, annotator training, inter-rater agreement, adjudication,
  and approach naming remain to be designed.

## Audit Conclusion

The repository contains sufficient identifiers and recording shards to define a
leakage-free temporal protocol without changing source data. The principal limitation
was lost `video_id` provenance in merged tables; the deterministic merge-offset rule
restores it unambiguously for every final trajectory. The new manifest and split remove
trajectory overlap between target estimation, model selection, and independent test,
but the scientific runner still needs a later split-aware implementation before revised
performance claims can be produced.
