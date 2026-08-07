# EMAS_HG-v1 Result Manifest

## Source and configuration

- Canonical implementation: `code/metrics/emas_hg.py`.
- Analysis runner: `code/run_emas_sensitivity.py`.
- Reporting runner: `code/run_emas_reporting.py`.
- Pre-specified weights: `configs/emas_weight_scenarios.yaml`.
- Immutable-input hashes: `results/emas/task_06_input_manifest.json`.
- Analysis counts and lock declarations: `results/emas/emas_analysis_manifest.json`.

## Reproduction

`emas_reproduction_check.csv` contains 345 development and independent-test rows,
their normalized components, stored and recomputed EMAS_HG, absolute difference and
pass status. It is the numerical compatibility record for EMAS_HG-v1.

## Sensitivity outputs

- `emas_local_weight_grid.csv`: 465 deterministic local vectors.
- `emas_global_weight_sample.csv`: 1,000 fixed-seed Dirichlet vectors.
- `emas_named_scenario_results.csv`: all 255 candidates under seven named scenarios.
- `emas_weight_grid_results.csv`: per-vector, per-scene-method top candidate, rank
  correlations and top-three overlap.
- `emas_rank_stability.csv`: top-rank preservation and distinct winners.
- `emas_candidate_score_sensitivity.csv`: per-candidate min/max/mean/SD/range.
- `emas_component_analysis.csv`: variance, saturation and component correlations.
- `emas_margin_analysis.csv`: original top-two margin, frozen-selection EMAS ranks and
  rank-reversal frequencies.

The sensitivity files contain development candidate metrics only. Independent-test
metrics appear only in the original-weight reproduction table. No alternative-weight
independent-test score is presented as validation.

## Figures

Five figures are available under `figures/emas/` in 300-dpi PNG and vector PDF:

1. named-scenario top-rank stability;
2. local target/outlier-weight stability heatmap;
3. rank-correlation distributions;
4. component correlation matrix;
5. original top-candidate margins.
