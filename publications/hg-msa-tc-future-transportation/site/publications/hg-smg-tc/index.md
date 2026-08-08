# From Geometric Endpoint Micro-Modes to Semantic Maneuver Graphs

**Homography-Guided Vehicle Trajectory Clustering at Complex Urban Intersections**

This page summarizes the Future Transportation revision package for
HG-SMG-TC: **Homography-Guided Semantic Maneuver Graph Trajectory Clustering**.

Expected public URL:

`https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/`

## Relation to PathFindRNET 2.0

PathFindRNET 2.0 provides the traffic-video trajectory extraction and dataset
context. This publication page links to a manuscript-specific reproducibility
package under:

[`publications/hg-msa-tc-future-transportation/`](../../../publications/hg-msa-tc-future-transportation/)

## Summary

HG-SMG-TC connects homography-guided endpoint micro-mode discovery with semantic
approach consolidation, semantic maneuver graph construction, uncertainty-aware
maneuver-count priors and prior-constrained model selection. The method is
evaluated on five Bellevue intersections using a leakage-controlled split
protocol and exhaustive human-defined polygon-rule-based reference labels.

## Dataset and Split Summary

The study uses five Bellevue scenes only:

- `bellevue_116th_ne12th`
- `bellevue_150th_newport`
- `bellevue_150th_eastgate`
- `bellevue_150th_se38th`
- `bellevue_ne8th`

The split roles are target estimation, model selection and locked independent
test. Independent-test labels are read only after cluster assignments are
persisted.

## Pipeline Overview

![HG-SMG-TC pipeline](../../../publications/hg-msa-tc-future-transportation/figures/final_synthesis/final_hg_smg_pipeline_schematic.png)

## Locked-Test Result Summary

| Method family | Target error | NMI | Macro F1 | Outlier % |
| --- | ---: | ---: | ---: | ---: |
| Original untargeted A0 | 3.2000 | 0.8155 | 0.6328 | 15.56 |
| Original HG-aware A1 | 2.5333 | 0.8376 | 0.6838 | 9.94 |
| HG-SMG-TC A5 | 2.1333 | 0.8386 | 0.6723 | 10.34 |
| Endpoint isotropic baseline | 3.4667 | 0.7099 | 0.5710 | 24.43 |
| Resampled trajectory baseline | 3.0000 | 0.6827 | 0.4752 | 25.02 |

The results support a conservative trade-off interpretation. HG-SMG-TC improves
mean observed target-count alignment and slightly improves NMI relative to the
original HG-aware method, but it does not improve every metric.

## Key Figures

![Target error](../../../publications/hg-msa-tc-future-transportation/figures/final_synthesis/final_target_count_error.png)

![Agreement metrics](../../../publications/hg-msa-tc-future-transportation/figures/final_synthesis/final_agreement_metrics.png)

![Baseline comparison](../../../publications/hg-msa-tc-future-transportation/figures/final_synthesis/final_baseline_comparison_summary.png)

## Reproducibility

Start with:

- [`README.md`](../../../publications/hg-msa-tc-future-transportation/README.md)
- [`REPRODUCIBILITY.md`](../../../publications/hg-msa-tc-future-transportation/REPRODUCIBILITY.md)
- [`DATA_AVAILABILITY.md`](../../../publications/hg-msa-tc-future-transportation/DATA_AVAILABILITY.md)

Core synthesis regeneration:

```powershell
.\publications\hg-msa-tc-future-transportation\scripts\reproduce_core_results.ps1
```

## Code and Archive

- GitHub repository: `https://github.com/aronagg/PathFindRNET_2.0`
- GitHub release tag: `[GitHub release tag to be added]`
- DOI archive: `[DOI to be added after archive deposit]`
- Archive URL: `[Archive URL to be added]`

## Citation

```text
PathFindRNET research team. From Geometric Endpoint Micro-Modes to Semantic
Maneuver Graphs: Homography-Guided Vehicle Trajectory Clustering at Complex
Urban Intersections. Future Transportation, revised manuscript package.
[DOI to be added after archive deposit]
```

## License and Provenance Notes

Raw videos are not redistributed by this publication package. Google-derived or
third-party map imagery is not redistributed unless licensing and attribution are
confirmed. Public materials should rely on source code, compact metrics,
calibration tables, generated figures and author-created diagrams.

[Back to PathFindRNET 2.0](../../../)
