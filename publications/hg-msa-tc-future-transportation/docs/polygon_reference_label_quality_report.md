# Polygon Reference Label Quality Report

Primary labels use the first and last finite point inside each canonical manifest interval and Shapely `covers`. Boundary-near diagnostics do not exclude labels unless an endpoint is covered by multiple polygons.

## Scene and Split Coverage

| scene_id                | split             |   total_trajectories |   valid_reference_labels |   valid_coverage_percentage |   entry_no_polygon_count |   exit_no_polygon_count |   multiple_polygon_count |   illegal_mapping_count |   missing_geometry_count |   boundary_near_1px_count |   boundary_near_3px_count |   boundary_near_5px_count |   boundary_near_10px_count |
|:------------------------|:------------------|---------------------:|-------------------------:|----------------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|--------------------------:|--------------------------:|--------------------------:|---------------------------:|
| bellevue_116th_ne12th   | target_estimation |                  723 |                      657 |                      90.871 |                        2 |                      64 |                        0 |                       0 |                        0 |                        24 |                        49 |                        89 |                        274 |
| bellevue_116th_ne12th   | model_selection   |                  634 |                      577 |                      91.009 |                        7 |                      50 |                        0 |                       0 |                        0 |                        14 |                        48 |                        83 |                        253 |
| bellevue_116th_ne12th   | independent_test  |                  964 |                      864 |                      89.627 |                       10 |                      91 |                        0 |                       0 |                        0 |                         6 |                        19 |                        61 |                        254 |
| bellevue_116th_ne12th   | total             |                 2321 |                     2098 |                      90.392 |                       19 |                     205 |                        0 |                       0 |                        0 |                        44 |                       116 |                       233 |                        781 |
| bellevue_150th_newport  | target_estimation |                 2423 |                     2327 |                      96.038 |                       90 |                       7 |                        0 |                       0 |                        0 |                        39 |                       121 |                       193 |                        374 |
| bellevue_150th_newport  | model_selection   |                 3101 |                     2967 |                      95.679 |                      124 |                      10 |                        0 |                       0 |                        0 |                        65 |                       219 |                       381 |                        625 |
| bellevue_150th_newport  | independent_test  |                 3924 |                     3852 |                      98.165 |                       56 |                      16 |                        0 |                       0 |                        0 |                        35 |                       171 |                       322 |                        751 |
| bellevue_150th_newport  | total             |                 9448 |                     9146 |                      96.804 |                      270 |                      33 |                        0 |                       0 |                        0 |                       139 |                       511 |                       896 |                       1750 |
| bellevue_150th_eastgate | target_estimation |                 8720 |                     8474 |                      97.179 |                      155 |                      95 |                        0 |                       0 |                        0 |                        40 |                       112 |                       216 |                        725 |
| bellevue_150th_eastgate | model_selection   |                 6791 |                     6615 |                      97.408 |                       81 |                     103 |                        0 |                       0 |                        0 |                        25 |                        85 |                       212 |                        761 |
| bellevue_150th_eastgate | independent_test  |                10755 |                    10586 |                      98.429 |                      108 |                      62 |                        0 |                       0 |                        0 |                        42 |                       197 |                       573 |                       2480 |
| bellevue_150th_eastgate | total             |                26266 |                    25675 |                      97.750 |                      344 |                     260 |                        0 |                       0 |                        0 |                       107 |                       394 |                      1001 |                       3966 |
| bellevue_150th_se38th   | target_estimation |                 2482 |                     2282 |                      91.942 |                      142 |                      58 |                        0 |                       0 |                        0 |                        17 |                       131 |                       399 |                       1037 |
| bellevue_150th_se38th   | model_selection   |                 2865 |                     2648 |                      92.426 |                      120 |                      97 |                        0 |                       0 |                        0 |                         9 |                        54 |                       177 |                        802 |
| bellevue_150th_se38th   | independent_test  |                 3713 |                     3422 |                      92.163 |                      180 |                     112 |                        0 |                       0 |                        0 |                        46 |                       459 |                      1303 |                       2184 |
| bellevue_150th_se38th   | total             |                 9060 |                     8352 |                      92.185 |                      442 |                     267 |                        0 |                       0 |                        0 |                        72 |                       644 |                      1879 |                       4023 |
| bellevue_ne8th          | target_estimation |                 5576 |                     4553 |                      81.654 |                       77 |                     979 |                        0 |                       0 |                        0 |                       226 |                       929 |                      1631 |                       3005 |
| bellevue_ne8th          | model_selection   |                 6321 |                     5854 |                      92.612 |                       86 |                     392 |                        0 |                       0 |                        0 |                       323 |                      1320 |                      1962 |                       3156 |
| bellevue_ne8th          | independent_test  |                 8037 |                     7510 |                      93.443 |                       56 |                     477 |                        0 |                       0 |                        0 |                       437 |                      1751 |                      2594 |                       3955 |
| bellevue_ne8th          | total             |                19934 |                    17917 |                      89.882 |                      219 |                    1848 |                        0 |                       0 |                        0 |                       986 |                      4000 |                      6187 |                      10116 |

## Exclusion Reasons

| scene_id                | split             | exclusion_reason   |   count |
|:------------------------|:------------------|:-------------------|--------:|
| bellevue_116th_ne12th   | target_estimation | valid              |     657 |
| bellevue_116th_ne12th   | target_estimation | exit_no_polygon    |      64 |
| bellevue_116th_ne12th   | target_estimation | entry_no_polygon   |       2 |
| bellevue_116th_ne12th   | model_selection   | valid              |     577 |
| bellevue_116th_ne12th   | model_selection   | exit_no_polygon    |      50 |
| bellevue_116th_ne12th   | model_selection   | entry_no_polygon   |       7 |
| bellevue_116th_ne12th   | independent_test  | valid              |     864 |
| bellevue_116th_ne12th   | independent_test  | exit_no_polygon    |      90 |
| bellevue_116th_ne12th   | independent_test  | entry_no_polygon   |      10 |
| bellevue_150th_newport  | target_estimation | valid              |    2327 |
| bellevue_150th_newport  | target_estimation | entry_no_polygon   |      90 |
| bellevue_150th_newport  | target_estimation | exit_no_polygon    |       6 |
| bellevue_150th_newport  | model_selection   | valid              |    2967 |
| bellevue_150th_newport  | model_selection   | entry_no_polygon   |     124 |
| bellevue_150th_newport  | model_selection   | exit_no_polygon    |      10 |
| bellevue_150th_newport  | independent_test  | valid              |    3852 |
| bellevue_150th_newport  | independent_test  | entry_no_polygon   |      56 |
| bellevue_150th_newport  | independent_test  | exit_no_polygon    |      16 |
| bellevue_150th_eastgate | target_estimation | valid              |    8474 |
| bellevue_150th_eastgate | target_estimation | exit_no_polygon    |      91 |
| bellevue_150th_eastgate | target_estimation | entry_no_polygon   |     155 |
| bellevue_150th_eastgate | model_selection   | valid              |    6615 |
| bellevue_150th_eastgate | model_selection   | exit_no_polygon    |      95 |
| bellevue_150th_eastgate | model_selection   | entry_no_polygon   |      81 |
| bellevue_150th_eastgate | independent_test  | valid              |   10586 |
| bellevue_150th_eastgate | independent_test  | exit_no_polygon    |      61 |
| bellevue_150th_eastgate | independent_test  | entry_no_polygon   |     108 |
| bellevue_150th_se38th   | target_estimation | valid              |    2282 |
| bellevue_150th_se38th   | target_estimation | entry_no_polygon   |     142 |
| bellevue_150th_se38th   | target_estimation | exit_no_polygon    |      58 |
| bellevue_150th_se38th   | model_selection   | valid              |    2648 |
| bellevue_150th_se38th   | model_selection   | exit_no_polygon    |      97 |
| bellevue_150th_se38th   | model_selection   | entry_no_polygon   |     120 |
| bellevue_150th_se38th   | independent_test  | valid              |    3422 |
| bellevue_150th_se38th   | independent_test  | exit_no_polygon    |     111 |
| bellevue_150th_se38th   | independent_test  | entry_no_polygon   |     180 |
| bellevue_ne8th          | target_estimation | valid              |    4553 |
| bellevue_ne8th          | target_estimation | exit_no_polygon    |     946 |
| bellevue_ne8th          | target_estimation | entry_no_polygon   |      77 |
| bellevue_ne8th          | model_selection   | valid              |    5854 |
| bellevue_ne8th          | model_selection   | exit_no_polygon    |     381 |
| bellevue_ne8th          | model_selection   | entry_no_polygon   |      86 |
| bellevue_ne8th          | independent_test  | valid              |    7510 |
| bellevue_ne8th          | independent_test  | exit_no_polygon    |     471 |
| bellevue_ne8th          | independent_test  | entry_no_polygon   |      56 |

## Per-Movement Counts

| scene_id                | split             | reference_movement_id       |   count |
|:------------------------|:------------------|:----------------------------|--------:|
| bellevue_116th_ne12th   | target_estimation | bellevue_116th_ne12th:B>H   |     207 |
| bellevue_116th_ne12th   | target_estimation | bellevue_116th_ne12th:D>G   |      39 |
| bellevue_116th_ne12th   | target_estimation | bellevue_116th_ne12th:C>E   |      66 |
| bellevue_116th_ne12th   | target_estimation | bellevue_116th_ne12th:D>F   |     247 |
| bellevue_116th_ne12th   | target_estimation | bellevue_116th_ne12th:D>E   |       6 |
| bellevue_116th_ne12th   | target_estimation | bellevue_116th_ne12th:A>G   |      73 |
| bellevue_116th_ne12th   | target_estimation | bellevue_116th_ne12th:A>H   |      11 |
| bellevue_116th_ne12th   | target_estimation | bellevue_116th_ne12th:A>F   |       4 |
| bellevue_116th_ne12th   | target_estimation | bellevue_116th_ne12th:C>H   |       4 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:D>F   |     167 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:D>G   |      63 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:B>H   |      95 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:C>E   |     139 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:A>G   |      74 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:A>H   |       5 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:D>E   |      17 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:C>H   |       7 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:A>F   |       9 |
| bellevue_116th_ne12th   | model_selection   | bellevue_116th_ne12th:B>E   |       1 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:D>F   |     204 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:B>H   |     223 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:A>F   |      45 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:A>G   |     158 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:D>G   |      34 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:D>E   |      29 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:A>H   |      40 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:C>E   |     116 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:B>E   |      10 |
| bellevue_116th_ne12th   | independent_test  | bellevue_116th_ne12th:C>H   |       5 |
| bellevue_150th_newport  | target_estimation | bellevue_150th_newport:C>E  |    1093 |
| bellevue_150th_newport  | target_estimation | bellevue_150th_newport:A>G  |     540 |
| bellevue_150th_newport  | target_estimation | bellevue_150th_newport:D>G  |     136 |
| bellevue_150th_newport  | target_estimation | bellevue_150th_newport:C>H  |     289 |
| bellevue_150th_newport  | target_estimation | bellevue_150th_newport:B>H  |     101 |
| bellevue_150th_newport  | target_estimation | bellevue_150th_newport:D>F  |     128 |
| bellevue_150th_newport  | target_estimation | bellevue_150th_newport:A>H  |      17 |
| bellevue_150th_newport  | target_estimation | bellevue_150th_newport:A>F  |      15 |
| bellevue_150th_newport  | target_estimation | bellevue_150th_newport:B>E  |       8 |
| bellevue_150th_newport  | model_selection   | bellevue_150th_newport:C>E  |    1122 |
| bellevue_150th_newport  | model_selection   | bellevue_150th_newport:A>G  |     757 |
| bellevue_150th_newport  | model_selection   | bellevue_150th_newport:C>H  |     357 |
| bellevue_150th_newport  | model_selection   | bellevue_150th_newport:D>G  |     325 |
| bellevue_150th_newport  | model_selection   | bellevue_150th_newport:D>F  |     178 |
| bellevue_150th_newport  | model_selection   | bellevue_150th_newport:B>H  |     144 |
| bellevue_150th_newport  | model_selection   | bellevue_150th_newport:A>H  |      34 |
| bellevue_150th_newport  | model_selection   | bellevue_150th_newport:A>F  |      44 |
| bellevue_150th_newport  | model_selection   | bellevue_150th_newport:B>E  |       6 |
| bellevue_150th_newport  | independent_test  | bellevue_150th_newport:D>F  |     313 |
| bellevue_150th_newport  | independent_test  | bellevue_150th_newport:C>E  |    1718 |
| bellevue_150th_newport  | independent_test  | bellevue_150th_newport:A>G  |     357 |
| bellevue_150th_newport  | independent_test  | bellevue_150th_newport:C>H  |    1081 |
| bellevue_150th_newport  | independent_test  | bellevue_150th_newport:D>G  |     185 |
| bellevue_150th_newport  | independent_test  | bellevue_150th_newport:B>H  |     137 |
| bellevue_150th_newport  | independent_test  | bellevue_150th_newport:A>F  |      42 |
| bellevue_150th_newport  | independent_test  | bellevue_150th_newport:B>E  |      12 |
| bellevue_150th_newport  | independent_test  | bellevue_150th_newport:A>H  |       7 |
| bellevue_150th_eastgate | target_estimation | bellevue_150th_eastgate:A>H |     640 |
| bellevue_150th_eastgate | target_estimation | bellevue_150th_eastgate:B>H |    1999 |
| bellevue_150th_eastgate | target_estimation | bellevue_150th_eastgate:A>G |     885 |
| bellevue_150th_eastgate | target_estimation | bellevue_150th_eastgate:D>F |    2148 |
| bellevue_150th_eastgate | target_estimation | bellevue_150th_eastgate:C>H |     262 |
| bellevue_150th_eastgate | target_estimation | bellevue_150th_eastgate:A>F |    1226 |
| bellevue_150th_eastgate | target_estimation | bellevue_150th_eastgate:D>G |     999 |
| bellevue_150th_eastgate | target_estimation | bellevue_150th_eastgate:C>E |     201 |
| bellevue_150th_eastgate | target_estimation | bellevue_150th_eastgate:B>E |     114 |
| bellevue_150th_eastgate | model_selection   | bellevue_150th_eastgate:A>G |     733 |
| bellevue_150th_eastgate | model_selection   | bellevue_150th_eastgate:A>F |    1067 |
| bellevue_150th_eastgate | model_selection   | bellevue_150th_eastgate:A>H |     637 |
| bellevue_150th_eastgate | model_selection   | bellevue_150th_eastgate:B>H |    1552 |
| bellevue_150th_eastgate | model_selection   | bellevue_150th_eastgate:D>F |    1513 |
| bellevue_150th_eastgate | model_selection   | bellevue_150th_eastgate:C>H |     324 |
| bellevue_150th_eastgate | model_selection   | bellevue_150th_eastgate:D>G |     427 |
| bellevue_150th_eastgate | model_selection   | bellevue_150th_eastgate:C>E |     265 |
| bellevue_150th_eastgate | model_selection   | bellevue_150th_eastgate:B>E |      97 |
| bellevue_150th_eastgate | independent_test  | bellevue_150th_eastgate:B>H |    4051 |
| bellevue_150th_eastgate | independent_test  | bellevue_150th_eastgate:A>F |    1111 |
| bellevue_150th_eastgate | independent_test  | bellevue_150th_eastgate:A>H |    1272 |
| bellevue_150th_eastgate | independent_test  | bellevue_150th_eastgate:D>F |    1667 |
| bellevue_150th_eastgate | independent_test  | bellevue_150th_eastgate:C>H |     850 |
| bellevue_150th_eastgate | independent_test  | bellevue_150th_eastgate:D>G |     355 |
| bellevue_150th_eastgate | independent_test  | bellevue_150th_eastgate:A>G |     613 |
| bellevue_150th_eastgate | independent_test  | bellevue_150th_eastgate:B>E |      99 |
| bellevue_150th_eastgate | independent_test  | bellevue_150th_eastgate:C>E |     568 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:C>E   |     516 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:A>G   |    1260 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:A>H   |     238 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:D>F   |      34 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:D>G   |     128 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:B>E   |      50 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:B>H   |      44 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:B>G   |       3 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:C>F   |       4 |
| bellevue_150th_se38th   | target_estimation | bellevue_150th_se38th:C>H   |       5 |
| bellevue_150th_se38th   | model_selection   | bellevue_150th_se38th:A>G   |     712 |
| bellevue_150th_se38th   | model_selection   | bellevue_150th_se38th:C>E   |    1565 |
| bellevue_150th_se38th   | model_selection   | bellevue_150th_se38th:A>H   |     101 |
| bellevue_150th_se38th   | model_selection   | bellevue_150th_se38th:B>E   |     110 |
| bellevue_150th_se38th   | model_selection   | bellevue_150th_se38th:C>F   |      13 |
| bellevue_150th_se38th   | model_selection   | bellevue_150th_se38th:D>G   |      53 |
| bellevue_150th_se38th   | model_selection   | bellevue_150th_se38th:D>F   |      48 |
| bellevue_150th_se38th   | model_selection   | bellevue_150th_se38th:B>H   |      39 |
| bellevue_150th_se38th   | model_selection   | bellevue_150th_se38th:C>H   |       7 |
| bellevue_150th_se38th   | independent_test  | bellevue_150th_se38th:A>G   |    1630 |
| bellevue_150th_se38th   | independent_test  | bellevue_150th_se38th:C>E   |    1117 |
| bellevue_150th_se38th   | independent_test  | bellevue_150th_se38th:A>H   |     182 |
| bellevue_150th_se38th   | independent_test  | bellevue_150th_se38th:D>F   |      56 |
| bellevue_150th_se38th   | independent_test  | bellevue_150th_se38th:B>E   |     204 |
| bellevue_150th_se38th   | independent_test  | bellevue_150th_se38th:D>G   |     141 |
| bellevue_150th_se38th   | independent_test  | bellevue_150th_se38th:B>H   |      70 |
| bellevue_150th_se38th   | independent_test  | bellevue_150th_se38th:C>F   |      12 |
| bellevue_150th_se38th   | independent_test  | bellevue_150th_se38th:C>H   |      10 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:D>F          |     768 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:D>G          |     155 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:B>H          |     764 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:C>E          |    1036 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:A>H          |     421 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:A>G          |    1020 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:C>F          |     258 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:B>E          |      54 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:C>H          |      66 |
| bellevue_ne8th          | target_estimation | bellevue_ne8th:D>E          |      11 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:D>F          |    1334 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:D>G          |     241 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:B>H          |    1044 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:C>E          |    1456 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:C>F          |     269 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:A>G          |    1043 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:B>E          |     117 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:C>H          |      19 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:A>H          |     297 |
| bellevue_ne8th          | model_selection   | bellevue_ne8th:D>E          |      34 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:C>E          |    1706 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:A>G          |    1868 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:C>F          |     279 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:D>F          |    1370 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:B>H          |    1203 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:A>H          |     701 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:D>G          |     222 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:B>E          |     109 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:C>H          |      19 |
| bellevue_ne8th          | independent_test  | bellevue_ne8th:D>E          |      33 |

## Recording-Level Coverage

| scene_id                | split             | recording_id                                 |   total_trajectories |   valid_reference_labels |   first_start_frame |   last_end_frame |   valid_coverage_percentage |
|:------------------------|:------------------|:---------------------------------------------|---------------------:|-------------------------:|--------------------:|-----------------:|----------------------------:|
| bellevue_116th_ne12th   | target_estimation | Bellevue_116th_NE12th__2017-09-10_19-08-25   |                  402 |                      349 |                   0 |           106404 |                      86.816 |
| bellevue_116th_ne12th   | target_estimation | Bellevue_116th_NE12th__2017-09-10_20-09-12   |                  101 |                       95 |                 322 |            43076 |                      94.059 |
| bellevue_116th_ne12th   | target_estimation | Bellevue_116th_NE12th__2017-09-10_21-08-54   |                  136 |                      131 |                   0 |            87300 |                      96.324 |
| bellevue_116th_ne12th   | target_estimation | Bellevue_116th_NE12th__2017-09-10_22-08-50   |                   15 |                       15 |                1167 |            13015 |                     100.000 |
| bellevue_116th_ne12th   | target_estimation | Bellevue_116th_NE12th__2017-09-10_23-08-29   |                   69 |                       67 |                1012 |           102674 |                      97.101 |
| bellevue_116th_ne12th   | model_selection   | Bellevue_116th_NE12th__2017-09-11_00-08-29   |                   20 |                       19 |                2164 |            76347 |                      95.000 |
| bellevue_116th_ne12th   | model_selection   | Bellevue_116th_NE12th__2017-09-11_01-08-29   |                   20 |                       17 |                1010 |            93174 |                      85.000 |
| bellevue_116th_ne12th   | model_selection   | Bellevue_116th_NE12th__2017-09-11_02-08-32   |                   13 |                       12 |                3096 |           100524 |                      92.308 |
| bellevue_116th_ne12th   | model_selection   | Bellevue_116th_NE12th__2017-09-11_03-08-30   |                   18 |                       15 |               28647 |           103125 |                      83.333 |
| bellevue_116th_ne12th   | model_selection   | Bellevue_116th_NE12th__2017-09-11_04-08-30   |                   47 |                       45 |               11090 |           103668 |                      95.745 |
| bellevue_116th_ne12th   | model_selection   | Bellevue_116th_NE12th__2017-09-11_05-08-39   |                   10 |                        9 |                 897 |            11628 |                      90.000 |
| bellevue_116th_ne12th   | model_selection   | Bellevue_116th_NE12th__2017-09-11_06-08-30   |                  420 |                      378 |                   0 |           106851 |                      90.000 |
| bellevue_116th_ne12th   | model_selection   | Bellevue_116th_NE12th__2017-09-11_07-08-32   |                   86 |                       82 |                   0 |            11779 |                      95.349 |
| bellevue_116th_ne12th   | independent_test  | Bellevue_116th_NE12th__2017-09-11_08-08-50   |                  180 |                      161 |                   0 |            16663 |                      89.444 |
| bellevue_116th_ne12th   | independent_test  | Bellevue_116th_NE12th__2017-09-11_09-08-31   |                  351 |                      310 |                   0 |            25926 |                      88.319 |
| bellevue_116th_ne12th   | independent_test  | Bellevue_116th_NE12th__2017-09-11_11-08-33   |                   84 |                       80 |                   0 |             9158 |                      95.238 |
| bellevue_116th_ne12th   | independent_test  | Bellevue_116th_NE12th__2017-09-11_12-08-33   |                   32 |                       30 |                   0 |             3261 |                      93.750 |
| bellevue_116th_ne12th   | independent_test  | Bellevue_116th_NE12th__2017-09-11_14-08-35   |                  166 |                      149 |                   0 |            15977 |                      89.759 |
| bellevue_116th_ne12th   | independent_test  | Bellevue_116th_NE12th__2017-09-11_15-08-36   |                   26 |                       23 |                   0 |             1853 |                      88.462 |
| bellevue_116th_ne12th   | independent_test  | Bellevue_116th_NE12th__2017-09-11_16-08-37   |                   18 |                       14 |                   0 |             1678 |                      77.778 |
| bellevue_116th_ne12th   | independent_test  | Bellevue_116th_NE12th__2017-09-11_17-08-39   |                  107 |                       97 |                   0 |             7417 |                      90.654 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-10_18-08-24  |                  470 |                      465 |                   0 |           107974 |                      98.936 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-10_19-08-24  |                  364 |                      356 |                   0 |           107973 |                      97.802 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-10_20-08-25  |                  298 |                      287 |                   0 |           106605 |                      96.309 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-10_21-08-28  |                  196 |                      193 |                 899 |           107177 |                      98.469 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-10_22-08-28  |                   65 |                       62 |                1380 |           107858 |                      95.385 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-10_23-08-29  |                   59 |                       58 |                 780 |           107609 |                      98.305 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-11_00-08-29  |                   18 |                       16 |                3775 |           104798 |                      88.889 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-11_01-08-29  |                   10 |                        9 |               13547 |            95354 |                      90.000 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-11_02-08-29  |                    3 |                        3 |               28384 |            66325 |                     100.000 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-11_03-08-29  |                    7 |                        5 |                3910 |            97007 |                      71.429 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-11_04-08-29  |                   18 |                       15 |                5621 |           107943 |                      83.333 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-11_05-08-29  |                   46 |                       39 |                2152 |           104815 |                      84.783 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-11_06-08-30  |                  231 |                      219 |                 621 |           107713 |                      94.805 |
| bellevue_150th_newport  | target_estimation | Bellevue_150th_Newport__2017-09-11_07-08-31  |                  638 |                      600 |                   0 |           107858 |                      94.044 |
| bellevue_150th_newport  | model_selection   | Bellevue_150th_Newport__2017-09-11_08-08-31  |                  853 |                      809 |                   0 |           107541 |                      94.842 |
| bellevue_150th_newport  | model_selection   | Bellevue_150th_Newport__2017-09-11_09-08-30  |                  544 |                      506 |                   0 |           107945 |                      93.015 |
| bellevue_150th_newport  | model_selection   | Bellevue_150th_Newport__2017-09-11_10-08-31  |                  421 |                      403 |                   0 |           107942 |                      95.724 |
| bellevue_150th_newport  | model_selection   | Bellevue_150th_Newport__2017-09-11_11-08-32  |                  411 |                      395 |                   0 |           107528 |                      96.107 |
| bellevue_150th_newport  | model_selection   | Bellevue_150th_Newport__2017-09-11_12-08-32  |                  477 |                      464 |                   0 |           107927 |                      97.275 |
| bellevue_150th_newport  | model_selection   | Bellevue_150th_Newport__2017-09-11_13-08-32  |                  395 |                      390 |                   0 |            95915 |                      98.734 |
| bellevue_150th_newport  | independent_test  | Bellevue_150th_Newport__2017-09-11_14-08-31  |                  614 |                      598 |                   0 |           107761 |                      97.394 |
| bellevue_150th_newport  | independent_test  | Bellevue_150th_Newport__2017-09-11_15-08-32  |                  978 |                      959 |                 582 |           107668 |                      98.057 |
| bellevue_150th_newport  | independent_test  | Bellevue_150th_Newport__2017-09-11_16-08-32  |                 1149 |                     1125 |                   0 |           107936 |                      97.911 |
| bellevue_150th_newport  | independent_test  | Bellevue_150th_Newport__2017-09-11_17-08-32  |                 1183 |                     1170 |                   0 |           107920 |                      98.901 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-10_18-08-24 |                  738 |                      706 |                   0 |           107577 |                      95.664 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-10_19-08-25 |                  789 |                      754 |                   0 |           107558 |                      95.564 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-10_20-08-25 |                  590 |                      570 |                1088 |           107783 |                      96.610 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-10_21-08-28 |                  441 |                      428 |                   0 |           107742 |                      97.052 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-10_22-08-28 |                  216 |                      211 |                   0 |           107904 |                      97.685 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-10_23-08-29 |                  123 |                      119 |                 101 |           106927 |                      96.748 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-11_00-08-29 |                   69 |                       65 |                3117 |           106510 |                      94.203 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-11_01-08-29 |                   46 |                       45 |                2410 |            94616 |                      97.826 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-11_02-08-29 |                   19 |                       19 |                  82 |           105381 |                     100.000 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-11_03-08-30 |                   37 |                       35 |                  13 |           106394 |                      94.595 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-11_04-08-29 |                   88 |                       86 |                 214 |           106953 |                      97.727 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-11_05-08-30 |                  272 |                      267 |                   0 |           107625 |                      98.162 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-11_06-08-30 |                  879 |                      868 |                   0 |           107675 |                      98.749 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-11_07-08-31 |                 1898 |                     1856 |                   0 |           107735 |                      97.787 |
| bellevue_150th_eastgate | target_estimation | Bellevue_150th_Eastgate__2017-09-11_08-08-31 |                 2515 |                     2445 |                   0 |           107822 |                      97.217 |
| bellevue_150th_eastgate | model_selection   | Bellevue_150th_Eastgate__2017-09-11_09-08-31 |                 1956 |                     1906 |                   0 |           107509 |                      97.444 |
| bellevue_150th_eastgate | model_selection   | Bellevue_150th_Eastgate__2017-09-11_10-08-31 |                 1498 |                     1451 |                   0 |           107854 |                      96.862 |
| bellevue_150th_eastgate | model_selection   | Bellevue_150th_Eastgate__2017-09-11_11-08-32 |                 1555 |                     1509 |                   0 |           107840 |                      97.042 |
| bellevue_150th_eastgate | model_selection   | Bellevue_150th_Eastgate__2017-09-11_12-08-32 |                 1782 |                     1749 |                   0 |           107861 |                      98.148 |
| bellevue_150th_eastgate | independent_test  | Bellevue_150th_Eastgate__2017-09-11_13-08-32 |                 1377 |                     1348 |                   0 |            96177 |                      97.894 |
| bellevue_150th_eastgate | independent_test  | Bellevue_150th_Eastgate__2017-09-11_14-08-31 |                 1824 |                     1804 |                   0 |           107882 |                      98.904 |
| bellevue_150th_eastgate | independent_test  | Bellevue_150th_Eastgate__2017-09-11_15-08-33 |                 2452 |                     2428 |                   0 |           107863 |                      99.021 |
| bellevue_150th_eastgate | independent_test  | Bellevue_150th_Eastgate__2017-09-11_16-08-33 |                 2638 |                     2617 |                   0 |           107864 |                      99.204 |
| bellevue_150th_eastgate | independent_test  | Bellevue_150th_Eastgate__2017-09-11_17-08-33 |                 2464 |                     2389 |                   0 |           107823 |                      96.956 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-10_18-08-24   |                  659 |                      611 |                   0 |           107880 |                      92.716 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-10_19-08-25   |                  586 |                      505 |                   0 |           107788 |                      86.177 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-10_20-08-25   |                  392 |                      362 |                   0 |           104086 |                      92.347 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-10_21-08-38   |                  293 |                      279 |                 105 |           107375 |                      95.222 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-10_22-08-28   |                  160 |                      155 |                   0 |           107999 |                      96.875 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-10_23-08-29   |                  113 |                      107 |                1045 |           107030 |                      94.690 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-11_00-08-29   |                   56 |                       51 |                2561 |           107161 |                      91.071 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-11_01-08-30   |                   32 |                       31 |                4297 |            93959 |                      96.875 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-11_02-08-29   |                   14 |                       14 |                 814 |           105428 |                     100.000 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-11_03-08-30   |                   22 |                       22 |                 822 |            99904 |                     100.000 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-11_04-08-29   |                   55 |                       48 |                5768 |           107780 |                      87.273 |
| bellevue_150th_se38th   | target_estimation | Bellevue_150th_SE38th__2017-09-11_05-08-30   |                  100 |                       97 |                 818 |           107730 |                      97.000 |
| bellevue_150th_se38th   | model_selection   | Bellevue_150th_SE38th__2017-09-11_06-08-32   |                  486 |                      445 |                 136 |           107412 |                      91.564 |
| bellevue_150th_se38th   | model_selection   | Bellevue_150th_SE38th__2017-09-11_07-08-31   |                  697 |                      670 |                   0 |            83269 |                      96.126 |
| bellevue_150th_se38th   | model_selection   | Bellevue_150th_SE38th__2017-09-11_08-09-36   |                  110 |                      108 |                 532 |            11681 |                      98.182 |
| bellevue_150th_se38th   | model_selection   | Bellevue_150th_SE38th__2017-09-11_09-08-31   |                  838 |                      759 |                   0 |            99785 |                      90.573 |
| bellevue_150th_se38th   | model_selection   | Bellevue_150th_SE38th__2017-09-11_10-08-33   |                  734 |                      666 |                 169 |           105123 |                      90.736 |
| bellevue_150th_se38th   | independent_test  | Bellevue_150th_SE38th__2017-09-11_11-08-34   |                  751 |                      672 |                   0 |           105643 |                      89.481 |
| bellevue_150th_se38th   | independent_test  | Bellevue_150th_SE38th__2017-09-11_12-08-38   |                  840 |                      752 |                   0 |           103117 |                      89.524 |
| bellevue_150th_se38th   | independent_test  | Bellevue_150th_SE38th__2017-09-11_13-08-32   |                  750 |                      697 |                   0 |            94337 |                      92.933 |
| bellevue_150th_se38th   | independent_test  | Bellevue_150th_SE38th__2017-09-11_14-08-32   |                  937 |                      882 |                   0 |           104996 |                      94.130 |
| bellevue_150th_se38th   | independent_test  | Bellevue_150th_SE38th__2017-09-11_16-08-35   |                  223 |                      221 |                   0 |            15370 |                      99.103 |
| bellevue_150th_se38th   | independent_test  | Bellevue_150th_SE38th__2017-09-11_17-08-45   |                  212 |                      198 |                   0 |            13283 |                      93.396 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-10_18-08-23 |                 1261 |                     1151 |                   0 |           107971 |                      91.277 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-10_19-08-24 |                 1309 |                      954 |                   0 |           107886 |                      72.880 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-10_20-08-24 |                  758 |                      634 |                   0 |           107932 |                      83.641 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-10_21-08-28 |                  491 |                      387 |                   0 |           107825 |                      78.819 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-10_22-08-28 |                  275 |                      223 |                   0 |           107985 |                      81.091 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-10_23-08-29 |                  189 |                      163 |                 139 |           107723 |                      86.243 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-11_00-08-29 |                  121 |                       99 |                 666 |           107464 |                      81.818 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-11_01-08-29 |                   51 |                       36 |                 894 |            95122 |                      70.588 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-11_02-08-29 |                   36 |                       28 |                 970 |           106617 |                      77.778 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-11_03-08-29 |                   63 |                       48 |                 210 |           107978 |                      76.190 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-11_04-08-29 |                  134 |                      110 |                   0 |           106828 |                      82.090 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-11_05-08-29 |                  307 |                      267 |                   7 |           107930 |                      86.971 |
| bellevue_ne8th          | target_estimation | Bellevue_Bellevue_NE8th__2017-09-11_06-08-29 |                  581 |                      453 |                  56 |           107975 |                      77.969 |
| bellevue_ne8th          | model_selection   | Bellevue_Bellevue_NE8th__2017-09-11_07-08-31 |                 1104 |                      950 |                   0 |           107889 |                      86.051 |
| bellevue_ne8th          | model_selection   | Bellevue_Bellevue_NE8th__2017-09-11_08-08-31 |                 1236 |                     1187 |                   0 |           107980 |                      96.036 |
| bellevue_ne8th          | model_selection   | Bellevue_Bellevue_NE8th__2017-09-11_09-08-30 |                 1218 |                     1139 |                   0 |           107983 |                      93.514 |
| bellevue_ne8th          | model_selection   | Bellevue_Bellevue_NE8th__2017-09-11_10-08-31 |                 1287 |                     1202 |                   0 |           107969 |                      93.395 |
| bellevue_ne8th          | model_selection   | Bellevue_Bellevue_NE8th__2017-09-11_11-08-31 |                 1476 |                     1376 |                   0 |           107972 |                      93.225 |
| bellevue_ne8th          | independent_test  | Bellevue_Bellevue_NE8th__2017-09-11_12-08-31 |                 1599 |                     1472 |                   0 |           107627 |                      92.058 |
| bellevue_ne8th          | independent_test  | Bellevue_Bellevue_NE8th__2017-09-11_13-08-32 |                 1378 |                     1268 |                   0 |            96267 |                      92.017 |
| bellevue_ne8th          | independent_test  | Bellevue_Bellevue_NE8th__2017-09-11_14-08-31 |                 1579 |                     1446 |                   0 |           107975 |                      91.577 |
| bellevue_ne8th          | independent_test  | Bellevue_Bellevue_NE8th__2017-09-11_15-08-32 |                 1727 |                     1649 |                   0 |           107979 |                      95.483 |
| bellevue_ne8th          | independent_test  | Bellevue_Bellevue_NE8th__2017-09-11_16-08-32 |                 1754 |                     1675 |                   0 |           107979 |                      95.496 |
