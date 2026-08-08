# Draft Point-by-Point Response to Reviewers

Dear Editor and Reviewers,

We thank the reviewers for the detailed critique. We have substantially revised
the manuscript. The revision is now centered on Homography-Guided Semantic
Maneuver Graph Trajectory Clustering (HG-SMG-TC), a leakage-controlled
post-review extension evaluated on the five Bellevue intersections. We narrowed
the claims, added independent rule-based reference evaluation, added baseline
comparisons, formalized EMAS_HG and homography calibration, and explicitly report
the SE38th target-estimation failure and resulting trade-offs.

Line references below are placeholders and should be completed after final DOCX
formatting.

## 1. Novelty and Scope

**Reviewer concern.** The original manuscript did not clearly distinguish its
novelty from existing endpoint, OD, trajectory graph and lane-extraction methods.

**Response.** We agree. The revised manuscript no longer claims novelty for
endpoint-region discovery, OD extraction, trajectory graphs, lane-mode extraction
or multi-criteria ranking in isolation. The contribution is now framed as the
complete HG-SMG-TC workflow: homography-guided endpoint micro-mode discovery,
semantic approach consolidation, semantic maneuver graph construction,
uncertainty-aware maneuver-count prior, prior-constrained model selection, and
locked independent evaluation.

**What changed.** We rewrote the Introduction, Related Work and Contribution
sections and integrated the verified literature register.

**Evidence.** `docs/literature_register_hg_smg.md`,
`docs/final_claim_audit.md`.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 2. Target Leakage

**Reviewer concern.** The original workflow could be interpreted as using
evaluation information during model selection.

**Response.** We revised the protocol to explicitly separate target-estimation,
model-selection and independent-test splits. Independent-test cluster assignments
are persisted before reference labels are read. The revised text states that no
targets, thresholds, weights or configurations are modified after independent
results are seen.

**What changed.** We added a split-aware evaluation subsection and cite the frozen
execution manifests.

**Evidence.** `docs/final_statistical_analysis.md`; independent-test run
manifests.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 3. Independent Reference Labels

**Reviewer concern.** The previous manuscript did not provide a sufficiently
independent reference for evaluating clustering outputs.

**Response.** We generated exhaustive human-defined polygon-rule-based reference
labels. The scene regions and legal movement mappings were manually specified
before independent-test clustering evaluation, and the labels were generated
deterministically for all canonical trajectories. We explicitly avoid calling the
reference fully manual per-trajectory ground truth.

**What changed.** We added a reference-label protocol section and coverage table.

**Evidence.** `docs/polygon_reference_scientific_positioning.md`;
`docs/manuscript_tables_final.md`.

**Caveat.** The reference is independent of clustering outputs but is still based
on endpoint containment.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 4. Target-Estimator Reproducibility and SE38th Failure

**Reviewer concern.** The target estimator needed reproducible details and
failure-mode analysis.

**Response.** We formalized the estimator, reproduced all frozen targets, and
diagnosed SE38th. The original `K_HG` should be understood as an automatically
estimated observed geometric-maneuver target, not a true semantic maneuver count.
SE38th has `K_HG=18` versus 9 independently observed semantic movements.

**What changed.** We added target-estimation equations, frozen scene parameters
and a SE38th failure paragraph.

**Evidence.** `docs/hg_target_estimator_scientific_interpretation.md`;
`docs/se38th_target_failure_analysis.md`.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 5. EMAS_HG Formula and Weights

**Reviewer concern.** EMAS_HG was not mathematically defined and the weight choice
was not justified.

**Response.** We now define all EMAS_HG-v1 components, transformations, clipping
rules and edge cases. Weight sensitivity was evaluated without retuning. The
reviewer-example weights preserved the original top candidate in 14 of 15
development scene-method groups; local-grid top-rank preservation was 93.88%.

**What changed.** We added an EMAS methods subsection and sensitivity summary.

**Evidence.** `docs/emas_hg_mathematical_definition.md`;
`docs/emas_weight_sensitivity_scientific_interpretation.md`.

**Caveat.** EMAS_HG is diagnostic and task-specific. It is not independent
validation.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 6. Homography Calibration and Quality

**Reviewer concern.** Homography calibration, RANSAC use, residuals and image
provenance were insufficiently documented.

**Response.** We audited and documented the homography implementation. The frozen
implementation uses OpenCV `findHomography` with RANSAC, 10 px threshold, 2,000
iterations and confidence 0.995. All frozen matrices reproduce exactly. Mean
all-point reprojection errors range from 8.3170 px to 13.9897 px. All scenes pass
the diagnostic quality gate, while high endpoint extrapolation remains a
limitation.

**What changed.** We added homography equations, calibration settings, quality
metrics and perturbation interpretation.

**Evidence.** `docs/homography_scientific_interpretation.md`;
`docs/homography_quality_gate.md`.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 7. Baseline Comparisons

**Reviewer concern.** The original baseline comparison was too weak.

**Response.** We added endpoint camera raw, endpoint camera isotropic and
resampled trajectory Euclidean baselines. The strongest endpoint baseline is
`endpoint_camera_isotropic`, with aggregate NMI 0.7099 and macro F1 0.5710.
HG-SMG-TC A5 has aggregate NMI 0.8386 and macro F1 0.6723, but endpoint KMeans is
strong in several cases.

**What changed.** We added baseline methods, results and a cautious discussion.

**Evidence.** `docs/baseline_scientific_interpretation.md`;
`results/final_synthesis/final_method_comparison.csv`.

**Caveat.** We do not claim universal superiority over all simple baselines.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 8. Statistical Dependence

**Reviewer concern.** The statistical analysis risked treating trajectories or
bootstrap rows as independent study replicates.

**Response.** The revised analysis treats the scene as the primary experimental
unit. Scene-level paired differences are summarized descriptively with bootstrap
confidence intervals over scenes. We avoid strong p-value claims because `n=5`
scenes is small.

**What changed.** We added a scene-level paired-analysis subsection.

**Evidence.** `docs/final_statistical_analysis.md`.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 9. Trade-Off Interpretation

**Reviewer concern.** The manuscript overclaimed improvement.

**Response.** We agree and have narrowed the claims. HG-SMG-TC A5 improves mean
observed target-count error from 2.5333 to 2.1333 relative to A1 and slightly
increases NMI from 0.8376 to 0.8386. It does not improve every metric: macro F1
changes from 0.6838 to 0.6723 and outlier percentage changes from 9.94% to
10.34%.

**What changed.** We rewrote the Results, Discussion and Conclusion to present
trade-offs.

**Evidence.** `docs/final_scientific_interpretation.md`;
`docs/final_claim_audit.md`.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 10. Data and Code Availability

**Reviewer concern.** Reproducibility and data availability were insufficient.

**Response.** We added a public-release plan covering code, configs, split
manifests, reference-label protocols, homography correspondences, compact result
tables and generated figures. Raw video and third-party map imagery
redistribution require license review.

**What changed.** We added Data and Code Availability and a public-release
checklist.

**Evidence.** `docs/public_release_final_checklist.md`;
`docs/manuscript_revised_data_availability.md`.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]

## 11. Figures, Tables and Terminology

**Reviewer concern.** Figures, tables and terminology needed revision.

**Response.** We created a final figure/table insertion plan and ready-to-copy
tables. We replaced ambiguous wording with precise terms such as
"human-defined polygon-rule-based reference labels", "automatically estimated
observed geometric-maneuver target", and "locked post-review extension
evaluation".

**What changed.** We added final tables, figure plan and terminology constraints.

**Evidence.** `docs/manuscript_tables_final.md`;
`docs/manuscript_figure_table_plan.md`.

**Manuscript location.** [Section X, Lines YY-ZZ to be filled after DOCX formatting]
