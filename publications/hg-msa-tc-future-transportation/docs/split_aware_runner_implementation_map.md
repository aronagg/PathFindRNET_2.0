# Split-Aware Runner Implementation Map

## Reused Scientific Behavior

Source reference:
`research_experiments/fov2026_trajectory_clustering/scripts/hg_msa_tc_five_scene/run_hg_msa_tc_five_scene_pipeline.py`.

The publication runner preserves these definitions:

- homography transformation of camera start and end points;
- polar angle plus scaled radial endpoint representation;
- KMeans entry/exit region candidates from 3 through 8;
- OD support thresholds 0.1%, 0.25%, 0.5%, 1%, and 2%;
- target threshold preference and local-instability rule;
- camera isotropic shared-scale endpoint representation;
- KMeans, HDBSCAN, and OPTICS candidate grids;
- clustered-only silhouette, Davies-Bouldin, and Calinski-Harabasz metrics;
- quick score and `EMAS_HG` formulas;
- untargeted and HG-target-aware selection priorities.

These formulas were extracted into
`code/pipeline/hg_msa_tc_core.py`. The old full-data runner is imported only as a
scientific reference and is not executed by this protocol.

## Refactored Functions

- Target region selection now returns every entry and exit candidate rather than only
  the selected labels and summary.
- Target threshold estimation now records percentage and absolute support thresholds.
- Isotropic normalization can fit parameters or apply already frozen parameters.
- Candidate generation receives an explicit versioned config.
- OPTICS `max_eps` calculation receives only the current phase feature matrix.
- Selection keys are explicit and serialized for every candidate.
- Fit seeds are stored with every candidate and selected configuration.

## Publication-Specific Wrappers

- `split_aware_io.py` filters and verifies phase-authorized rows, validates checksums,
  rejects annotation inputs, records access logs, and enforces test unlocking.
- `run_split_aware_hg_msa_tc.py` implements benchmark, target, select, freeze, and test
  CLI phases.
- `configs/split_aware_runner.yaml` is the complete versioned scientific configuration.
- `tests/test_split_aware_runner.py` exercises phase guards, determinism, hash
  integrity, transductive test behavior, and known repository hazards.

## Differences from the Submitted Implementation

1. The submitted implementation used the complete cohort for both target estimation
   and model selection. The revision uses disjoint development subsets.
2. The former `MAX_EVAL_TRACKS=6000` fit cap is removed after a full-cohort benchmark.
3. Normalization parameters and OPTICS quantiles are fitted only on model selection.
4. Runtime is removed as a final tie-breaker because it is not deterministic. Canonical
   lexical parameter order replaces it.
5. Test evaluation is explicitly transductive and consistent across all three methods.
6. Target and selected outputs receive checksum provenance and immutable freezing.

## Known Repository Issues

- `configs/dataset/bellevue_150th_newport.yaml` contains a single-video field pointing
  to Eastgate. The publication runner resolves recording provenance from the canonical
  manifest and never relies on that field. A validator rejects cross-scene video paths.
- The five declared `trajectories_filtered_filled.parquet` parents are missing from
  their active scene roots. The runner uses the present checksummed final feature
  cohort and does not claim byte-level lineage to those absent files.

## Unresolved Methodological Questions

- Sensitivity to preprocessing thresholds fitted only on development data remains to
  be evaluated.
- Manual maneuver labels are not yet populated, so independent-test external validity
  cannot be measured in this task.
- The effect of homography calibration uncertainty on split-specific target estimates
  remains outside this runner task.
