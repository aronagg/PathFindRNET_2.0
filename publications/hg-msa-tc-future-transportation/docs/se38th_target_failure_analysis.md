# SE38th Target Failure Analysis

## Frozen failure

SE38th produced `K_HG = 18`, while the independent polygon-rule reference contains
9 observed semantic movements. The frozen value is retained; this document diagnoses
the discrepancy and does not post hoc correct it.

## Why seven entry and three exit regions were selected

The selected counts are direct internal-metric optima over K=3..8:

| endpoint_role   |   n_regions |   silhouette |   davies_bouldin | selected   |
|:----------------|------------:|-------------:|-----------------:|:-----------|
| entry           |           3 |       0.6353 |           0.6635 | False      |
| entry           |           4 |       0.6604 |           0.4504 | False      |
| entry           |           5 |       0.7386 |           0.3688 | False      |
| entry           |           6 |       0.7581 |           0.3195 | False      |
| entry           |           7 |       0.7723 |           0.3571 | True       |
| entry           |           8 |       0.7538 |           0.4085 | False      |
| exit            |           3 |       0.6890 |           0.5868 | True       |
| exit            |           4 |       0.6286 |           0.6283 | False      |
| exit            |           5 |       0.6583 |           0.5279 | False      |
| exit            |           6 |       0.6815 |           0.4792 | False      |
| exit            |           7 |       0.6541 |           0.4858 | False      |
| exit            |           8 |       0.6675 |           0.4726 | False      |

Entry K=7 has the highest sampled silhouette (0.7723); exit K=3 has the highest exit
silhouette (0.6890). The selection rule has no road-branch consolidation constraint,
so internal feature-space separation can override one-region-per-physical-approach semantics.

## Automatic region to manual approach mapping

| endpoint_role   |   automatic_region_id | dominant_manual_approach   |   mapping_purity |   automatic_region_size | counts_by_manual_approach_json         |
|:----------------|----------------------:|:---------------------------|-----------------:|------------------------:|:---------------------------------------|
| entry           |                     0 | A                          |           1.0000 |                     778 | {"A": 777}                             |
| entry           |                     1 | D                          |           0.8190 |                     242 | {"A": 42, "D": 190}                    |
| entry           |                     2 | A                          |           1.0000 |                     369 | {"A": 369}                             |
| entry           |                     3 | C                          |           0.9122 |                     604 | {"A": 52, "C": 540}                    |
| entry           |                     4 | B                          |           0.6560 |                     125 | {"A": 43, "B": 82}                     |
| entry           |                     5 | A                          |           0.7619 |                     180 | {"A": 48, "B": 15}                     |
| entry           |                     6 | A                          |           1.0000 |                     184 | {"A": 182}                             |
| exit            |                     0 | G                          |           1.0000 |                    1194 | {"G": 1194}                            |
| exit            |                     1 | E                          |           0.5891 |                    1066 | {"E": 618, "F": 1, "G": 124, "H": 306} |
| exit            |                     2 | G                          |           0.7956 |                     222 | {"F": 37, "G": 144}                    |

Four entry regions (0, 2, 5, and 6) are dominated by manual approach A. This is direct
entry-region fragmentation. On the exit side, region 1 contains E, H, G, and one F
endpoint, while region 2 contains both G and F. Thus K=3 also merges physical exit
approaches. The failure combines entry over-segmentation with exit under-segmentation;
the entry split is the stronger driver of the high OD-pair count.

## Collapse of the 18 supported OD pairs

| automatic_od_pair   |   target_estimation_support_count |   target_estimation_support_percentage | empirical_target_reference_movement   |   empirical_target_reference_purity |   empirical_dominant_duplicate_group_size |   unique_target_reference_movement_count |
|:--------------------|----------------------------------:|---------------------------------------:|:--------------------------------------|------------------------------------:|------------------------------------------:|-----------------------------------------:|
| 0->0                |                               617 |                                24.8590 | bellevue_150th_se38th:A>G             |                              1.0000 |                                        10 |                                        1 |
| 3->1                |                               571 |                                23.0056 | bellevue_150th_se38th:C>E             |                              0.9264 |                                         1 |                                        4 |
| 2->0                |                               308 |                                12.4093 | bellevue_150th_se38th:A>G             |                              1.0000 |                                        10 |                                        1 |
| 6->1                |                               173 |                                 6.9702 | bellevue_150th_se38th:A>H             |                              0.9939 |                                         2 |                                        2 |
| 1->0                |                               123 |                                 4.9557 | bellevue_150th_se38th:D>G             |                              1.0000 |                                         1 |                                        1 |
| 0->2                |                               119 |                                 4.7945 | bellevue_150th_se38th:A>G             |                              1.0000 |                                        10 |                                        1 |
| 4->1                |                                91 |                                 3.6664 | bellevue_150th_se38th:B>E             |                              0.4835 |                                         2 |                                        3 |
| 5->0                |                                90 |                                 3.6261 | bellevue_150th_se38th:A>G             |                              0.9000 |                                        10 |                                        2 |
| 5->1                |                                75 |                                 3.0218 | bellevue_150th_se38th:B>E             |                              0.3333 |                                         2 |                                        4 |
| 1->2                |                                65 |                                 2.6189 | bellevue_150th_se38th:D>F             |                              0.9429 |                                         1 |                                        2 |
| 2->1                |                                60 |                                 2.4174 | bellevue_150th_se38th:A>G             |                              1.0000 |                                        10 |                                        1 |
| 1->1                |                                54 |                                 2.1757 | bellevue_150th_se38th:A>H             |                              0.7736 |                                         2 |                                        3 |
| 0->1                |                                42 |                                 1.6922 | bellevue_150th_se38th:A>G             |                              0.8919 |                                        10 |                                        2 |
| 4->0                |                                29 |                                 1.1684 | bellevue_150th_se38th:A>G             |                              1.0000 |                                        10 |                                        1 |
| 3->2                |                                17 |                                 0.6849 | bellevue_150th_se38th:C>F             |                              1.0000 |                                         1 |                                        1 |
| 3->0                |                                16 |                                 0.6446 | bellevue_150th_se38th:A>G             |                              1.0000 |                                        10 |                                        1 |
| 5->2                |                                15 |                                 0.6044 | bellevue_150th_se38th:A>G             |                              1.0000 |                                        10 |                                        1 |
| 6->0                |                                11 |                                 0.4432 | bellevue_150th_se38th:A>G             |                              1.0000 |                                        10 |                                        1 |

The 18 automatic pairs have only 7 distinct dominant target-reference movement labels;
11 pairs are duplicates under that dominant-label view
(`61.1%`). However, eight automatic
pairs contain more than one valid manual movement, so a one-to-one collapse is not
fully supported. Across all valid target-estimation trajectories in the 18 pairs,
10 manual movement
labels occur. The independent-test reference observes 9 movements. This distinction
shows that the estimator counts geometric endpoint submodes, not guaranteed semantic classes.

Only 1 of the 18 retained pairs has support below twice the frozen threshold;
the weighted fraction of trajectories with at most 30 canonical points is
`0.00%`. This diagnostic does not identify rare-pair support or short
tracks as a material explanation for the factor-of-two target error. Raising the
threshold alone also never reduces the diagnostic target to 9 over the pre-specified
0.05%-1.0% grid.

## Homography and lane-level interpretation

The frozen SE38th homography has all-point mean reprojection error
`10.25 px` and is classified
`acceptable_with_caution`. Homography uncertainty can broaden endpoint modes, but the
manual-approach mapping directly demonstrates repeated A regions even after calibration.
The separated geometric modes are consistent with lane-level or endpoint-position
substructure. They cannot be asserted to be lane-level maneuvers because no lane-marking
or mandatory-turn-arrow labels were used.

## Root-cause conclusion

SE38th is a semantic over-segmentation failure caused primarily by endpoint-region
granularity: one physical entry branch is resolved into several geometric modes, while
the exit partition simultaneously merges some semantic exits. Endpoint dispersion and
homography uncertainty remain plausible secondary contributors; the available short-track
diagnostic does not support incomplete tracks as a major cause. A future road-branch
consolidation stage could merge modes belonging to one physical approach, but it is not
applied to the current frozen results.
