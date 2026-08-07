# HG Target Estimator Failure Taxonomy

| Failure mode | Mechanism | Diagnostic signature | Current evidence | Consequence |
| --- | --- | --- | --- | --- |
| Semantic over-segmentation | One physical movement is represented by several geometric endpoint pairs. | Automatic target exceeds independent observed movements; repeated dominant manual mappings. | Strong at SE38th; limited non-semantic pairs at Newport. | KMeans may be forced to split semantic movements; density-method selection may favor fragmented solutions. |
| Semantic under-segmentation | Distinct movements share endpoint regions or fall below support. | Fewer automatic pairs than observed movements; mixed manual labels in one pair. | NE8th target 9 versus 10 observed; SE38th exit K=3 merges exits. | KMeans merges movements; density methods may select coarser settings. |
| Endpoint-region fragmentation | KMeans finds multiple internally separated modes within one approach. | Multiple automatic regions map to the same manual approach. | Four SE38th entry regions are dominated by approach A. | Multiplies OD combinations and inflates target. |
| Rare-movement suppression | A legitimate movement has support below threshold. | Pair disappears as threshold increases; observed movement lacks supported pair. | Plausible at Newport/NE8th; threshold curves quantify it. | Under-counted target and possible merged rare class. |
| Spurious low-support OD pair | Endpoint noise creates a small pair just above threshold. | Pair support near threshold; poor manual purity. | A minority of pairs; not the main SE38th cause. | Inflated target, particularly for KMeans. |
| Incomplete-track endpoint error | Track begins/ends inside the intersection or before the intended branch. | Short trajectories, invalid polygon status, mixed automatic pairs. | Present in a subset of SE38th rows; secondary. | Wrong entry/exit region and noisy OD support. |
| Homography-induced region distortion | Calibration error shifts or spreads transformed endpoints. | Region spread aligns with high reprojection uncertainty. | SE38th calibration is acceptable with caution; causality is not isolated here. | Can split or merge endpoint modes. |
| Threshold instability | Small threshold changes alter many supported pairs. | Large target range and low pair-set Jaccard near frozen threshold. | Moderate at Newport; limited locally at 116th/Eastgate/NE8th. | Target depends on rare-pair cutoff. |

These modes are not mutually exclusive. SE38th combines endpoint-region fragmentation,
exit-region merging, endpoint uncertainty, and some incomplete trajectories. The frozen
analysis supports the first two as the primary mechanisms.
