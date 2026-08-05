# Polygon Assignment Sensitivity Report

The primary published reference remains the canonical first/last finite-point assignment. Variants are deterministic diagnostics and were not selected using clustering, HG targets, OD pseudo-labels, or EMAS. The polygon buffer magnitude is 3 camera pixels.

## Scene-Level Stability

| scene_id                | variant             |   trajectories |   changed_count |   changed_percentage |   status_changed_count |   boundary_near_count |
|:------------------------|:--------------------|---------------:|----------------:|---------------------:|-----------------------:|----------------------:|
| bellevue_116th_ne12th   | median_first_last_3 |           2321 |              17 |               0.7324 |                     17 |                   781 |
| bellevue_150th_newport  | median_first_last_3 |           9448 |              52 |               0.5504 |                     52 |                  1750 |
| bellevue_150th_eastgate | median_first_last_3 |          26266 |              55 |               0.2094 |                     55 |                  3966 |
| bellevue_150th_se38th   | median_first_last_3 |           9060 |              30 |               0.3311 |                     30 |                  4023 |
| bellevue_ne8th          | median_first_last_3 |          19934 |             155 |               0.7776 |                    155 |                 10116 |
| bellevue_116th_ne12th   | median_first_last_5 |           2321 |              34 |               1.4649 |                     34 |                   781 |
| bellevue_150th_newport  | median_first_last_5 |           9448 |              77 |               0.8150 |                     77 |                  1750 |
| bellevue_150th_eastgate | median_first_last_5 |          26266 |             108 |               0.4112 |                    108 |                  3966 |
| bellevue_150th_se38th   | median_first_last_5 |           9060 |              75 |               0.8278 |                     75 |                  4023 |
| bellevue_ne8th          | median_first_last_5 |          19934 |             290 |               1.4548 |                    290 |                 10116 |
| bellevue_116th_ne12th   | polygon_inward_3px  |           2321 |              68 |               2.9298 |                     68 |                   781 |
| bellevue_150th_newport  | polygon_inward_3px  |           9448 |             350 |               3.7045 |                    350 |                  1750 |
| bellevue_150th_eastgate | polygon_inward_3px  |          26266 |             258 |               0.9823 |                    258 |                  3966 |
| bellevue_150th_se38th   | polygon_inward_3px  |           9060 |             560 |               6.1810 |                    560 |                  4023 |
| bellevue_ne8th          | polygon_inward_3px  |          19934 |            3213 |              16.1182 |                   3213 |                 10116 |
| bellevue_116th_ne12th   | polygon_outward_3px |           2321 |              45 |               1.9388 |                     45 |                   781 |
| bellevue_150th_newport  | polygon_outward_3px |           9448 |             160 |               1.6935 |                    160 |                  1750 |
| bellevue_150th_eastgate | polygon_outward_3px |          26266 |             128 |               0.4873 |                    128 |                  3966 |
| bellevue_150th_se38th   | polygon_outward_3px |           9060 |              77 |               0.8499 |                     77 |                  4023 |
| bellevue_ne8th          | polygon_outward_3px |          19934 |             687 |               3.4464 |                    687 |                 10116 |

## Status Transitions

| variant             | primary_reference_status   | variant_reference_status   |   count |
|:--------------------|:---------------------------|:---------------------------|--------:|
| median_first_last_3 | valid                      | valid                      |   63023 |
| median_first_last_3 | unassigned                 | unassigned                 |    3697 |
| median_first_last_3 | valid                      | unassigned                 |     165 |
| median_first_last_3 | unassigned                 | valid                      |     144 |
| median_first_last_5 | valid                      | valid                      |   62850 |
| median_first_last_5 | unassigned                 | unassigned                 |    3595 |
| median_first_last_5 | valid                      | unassigned                 |     338 |
| median_first_last_5 | unassigned                 | valid                      |     246 |
| polygon_inward_3px  | valid                      | valid                      |   58739 |
| polygon_inward_3px  | unassigned                 | unassigned                 |    3841 |
| polygon_inward_3px  | valid                      | unassigned                 |    4449 |
| polygon_outward_3px | valid                      | valid                      |   63188 |
| polygon_outward_3px | unassigned                 | unassigned                 |    2744 |
| polygon_outward_3px | unassigned                 | valid                      |    1097 |

## Movement-Level Stability

| scene_id                | variant             | primary_movement_id         |   trajectories |   stable_count |   stable_percentage |
|:------------------------|:--------------------|:----------------------------|---------------:|---------------:|--------------------:|
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:B>H   |            525 |            524 |             99.8095 |
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:D>G   |            136 |            136 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:C>E   |            321 |            317 |             98.7539 |
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:D>F   |            618 |            617 |             99.8382 |
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:D>E   |             52 |             52 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:A>G   |            305 |            304 |             99.6721 |
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:A>H   |             56 |             56 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:A>F   |             58 |             58 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:C>H   |             16 |             16 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_3 | bellevue_116th_ne12th:B>E   |             11 |             11 |            100.0000 |
| bellevue_150th_newport  | median_first_last_3 | bellevue_150th_newport:C>E  |           3933 |           3932 |             99.9746 |
| bellevue_150th_newport  | median_first_last_3 | bellevue_150th_newport:A>G  |           1654 |           1652 |             99.8791 |
| bellevue_150th_newport  | median_first_last_3 | bellevue_150th_newport:D>G  |            646 |            623 |             96.4396 |
| bellevue_150th_newport  | median_first_last_3 | bellevue_150th_newport:C>H  |           1727 |           1727 |            100.0000 |
| bellevue_150th_newport  | median_first_last_3 | bellevue_150th_newport:B>H  |            382 |            382 |            100.0000 |
| bellevue_150th_newport  | median_first_last_3 | bellevue_150th_newport:D>F  |            619 |            619 |            100.0000 |
| bellevue_150th_newport  | median_first_last_3 | bellevue_150th_newport:A>H  |             58 |             58 |            100.0000 |
| bellevue_150th_newport  | median_first_last_3 | bellevue_150th_newport:A>F  |            101 |            100 |             99.0099 |
| bellevue_150th_newport  | median_first_last_3 | bellevue_150th_newport:B>E  |             26 |             26 |            100.0000 |
| bellevue_150th_eastgate | median_first_last_3 | bellevue_150th_eastgate:A>H |           2549 |           2544 |             99.8038 |
| bellevue_150th_eastgate | median_first_last_3 | bellevue_150th_eastgate:B>H |           7602 |           7598 |             99.9474 |
| bellevue_150th_eastgate | median_first_last_3 | bellevue_150th_eastgate:A>G |           2231 |           2230 |             99.9552 |
| bellevue_150th_eastgate | median_first_last_3 | bellevue_150th_eastgate:D>F |           5328 |           5327 |             99.9812 |
| bellevue_150th_eastgate | median_first_last_3 | bellevue_150th_eastgate:C>H |           1436 |           1433 |             99.7911 |
| bellevue_150th_eastgate | median_first_last_3 | bellevue_150th_eastgate:A>F |           3404 |           3404 |            100.0000 |
| bellevue_150th_eastgate | median_first_last_3 | bellevue_150th_eastgate:D>G |           1781 |           1778 |             99.8316 |
| bellevue_150th_eastgate | median_first_last_3 | bellevue_150th_eastgate:C>E |           1034 |           1030 |             99.6132 |
| bellevue_150th_eastgate | median_first_last_3 | bellevue_150th_eastgate:B>E |            310 |            286 |             92.2581 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:C>E   |           3198 |           3191 |             99.7811 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:A>G   |           3602 |           3597 |             99.8612 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:A>H   |            521 |            521 |            100.0000 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:D>F   |            138 |            137 |             99.2754 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:D>G   |            322 |            317 |             98.4472 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:B>E   |            364 |            364 |            100.0000 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:B>H   |            153 |            153 |            100.0000 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:B>G   |              3 |              3 |            100.0000 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:C>F   |             29 |             29 |            100.0000 |
| bellevue_150th_se38th   | median_first_last_3 | bellevue_150th_se38th:C>H   |             22 |             22 |            100.0000 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:D>F          |           3472 |           3472 |            100.0000 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:D>G          |            618 |            615 |             99.5146 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:B>H          |           3011 |           3010 |             99.9668 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:C>E          |           4198 |           4173 |             99.4045 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:A>H          |           1419 |           1419 |            100.0000 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:A>G          |           3931 |           3898 |             99.1605 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:C>F          |            806 |            806 |            100.0000 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:B>E          |            280 |            274 |             97.8571 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:C>H          |            104 |            104 |            100.0000 |
| bellevue_ne8th          | median_first_last_3 | bellevue_ne8th:D>E          |             78 |             78 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:B>H   |            525 |            523 |             99.6190 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:D>G   |            136 |            136 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:C>E   |            321 |            314 |             97.8193 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:D>F   |            618 |            615 |             99.5146 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:D>E   |             52 |             52 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:A>G   |            305 |            301 |             98.6885 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:A>H   |             56 |             56 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:A>F   |             58 |             58 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:C>H   |             16 |             16 |            100.0000 |
| bellevue_116th_ne12th   | median_first_last_5 | bellevue_116th_ne12th:B>E   |             11 |             10 |             90.9091 |
| bellevue_150th_newport  | median_first_last_5 | bellevue_150th_newport:C>E  |           3933 |           3930 |             99.9237 |
| bellevue_150th_newport  | median_first_last_5 | bellevue_150th_newport:A>G  |           1654 |           1651 |             99.8186 |
| bellevue_150th_newport  | median_first_last_5 | bellevue_150th_newport:D>G  |            646 |            618 |             95.6656 |
| bellevue_150th_newport  | median_first_last_5 | bellevue_150th_newport:C>H  |           1727 |           1727 |            100.0000 |
| bellevue_150th_newport  | median_first_last_5 | bellevue_150th_newport:B>H  |            382 |            382 |            100.0000 |
| bellevue_150th_newport  | median_first_last_5 | bellevue_150th_newport:D>F  |            619 |            619 |            100.0000 |
| bellevue_150th_newport  | median_first_last_5 | bellevue_150th_newport:A>H  |             58 |             57 |             98.2759 |
| bellevue_150th_newport  | median_first_last_5 | bellevue_150th_newport:A>F  |            101 |            100 |             99.0099 |
| bellevue_150th_newport  | median_first_last_5 | bellevue_150th_newport:B>E  |             26 |             26 |            100.0000 |
| bellevue_150th_eastgate | median_first_last_5 | bellevue_150th_eastgate:A>H |           2549 |           2543 |             99.7646 |
| bellevue_150th_eastgate | median_first_last_5 | bellevue_150th_eastgate:B>H |           7602 |           7593 |             99.8816 |
| bellevue_150th_eastgate | median_first_last_5 | bellevue_150th_eastgate:A>G |           2231 |           2229 |             99.9104 |
| bellevue_150th_eastgate | median_first_last_5 | bellevue_150th_eastgate:D>F |           5328 |           5320 |             99.8498 |
| bellevue_150th_eastgate | median_first_last_5 | bellevue_150th_eastgate:C>H |           1436 |           1430 |             99.5822 |
| bellevue_150th_eastgate | median_first_last_5 | bellevue_150th_eastgate:A>F |           3404 |           3403 |             99.9706 |
| bellevue_150th_eastgate | median_first_last_5 | bellevue_150th_eastgate:D>G |           1781 |           1778 |             99.8316 |
| bellevue_150th_eastgate | median_first_last_5 | bellevue_150th_eastgate:C>E |           1034 |           1022 |             98.8395 |
| bellevue_150th_eastgate | median_first_last_5 | bellevue_150th_eastgate:B>E |            310 |            269 |             86.7742 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:C>E   |           3198 |           3175 |             99.2808 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:A>G   |           3602 |           3595 |             99.8057 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:A>H   |            521 |            519 |             99.6161 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:D>F   |            138 |            136 |             98.5507 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:D>G   |            322 |            313 |             97.2050 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:B>E   |            364 |            363 |             99.7253 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:B>H   |            153 |            153 |            100.0000 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:B>G   |              3 |              3 |            100.0000 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:C>F   |             29 |             29 |            100.0000 |
| bellevue_150th_se38th   | median_first_last_5 | bellevue_150th_se38th:C>H   |             22 |             22 |            100.0000 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:D>F          |           3472 |           3472 |            100.0000 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:D>G          |            618 |            614 |             99.3528 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:B>H          |           3011 |           3007 |             99.8672 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:C>E          |           4198 |           4143 |             98.6899 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:A>H          |           1419 |           1415 |             99.7181 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:A>G          |           3931 |           3856 |             98.0921 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:C>F          |            806 |            806 |            100.0000 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:B>E          |            280 |            269 |             96.0714 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:C>H          |            104 |            104 |            100.0000 |
| bellevue_ne8th          | median_first_last_5 | bellevue_ne8th:D>E          |             78 |             78 |            100.0000 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:B>H   |            525 |            523 |             99.6190 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:D>G   |            136 |            136 |            100.0000 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:C>E   |            321 |            297 |             92.5234 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:D>F   |            618 |            580 |             93.8511 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:D>E   |             52 |             51 |             98.0769 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:A>G   |            305 |            302 |             99.0164 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:A>H   |             56 |             56 |            100.0000 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:A>F   |             58 |             58 |            100.0000 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:C>H   |             16 |             16 |            100.0000 |
| bellevue_116th_ne12th   | polygon_inward_3px  | bellevue_116th_ne12th:B>E   |             11 |             11 |            100.0000 |
| bellevue_150th_newport  | polygon_inward_3px  | bellevue_150th_newport:C>E  |           3933 |           3931 |             99.9491 |
| bellevue_150th_newport  | polygon_inward_3px  | bellevue_150th_newport:A>G  |           1654 |           1651 |             99.8186 |
| bellevue_150th_newport  | polygon_inward_3px  | bellevue_150th_newport:D>G  |            646 |            427 |             66.0991 |
| bellevue_150th_newport  | polygon_inward_3px  | bellevue_150th_newport:C>H  |           1727 |           1725 |             99.8842 |
| bellevue_150th_newport  | polygon_inward_3px  | bellevue_150th_newport:B>H  |            382 |            334 |             87.4346 |
| bellevue_150th_newport  | polygon_inward_3px  | bellevue_150th_newport:D>F  |            619 |            563 |             90.9532 |
| bellevue_150th_newport  | polygon_inward_3px  | bellevue_150th_newport:A>H  |             58 |             58 |            100.0000 |
| bellevue_150th_newport  | polygon_inward_3px  | bellevue_150th_newport:A>F  |            101 |             86 |             85.1485 |
| bellevue_150th_newport  | polygon_inward_3px  | bellevue_150th_newport:B>E  |             26 |             21 |             80.7692 |
| bellevue_150th_eastgate | polygon_inward_3px  | bellevue_150th_eastgate:A>H |           2549 |           2536 |             99.4900 |
| bellevue_150th_eastgate | polygon_inward_3px  | bellevue_150th_eastgate:B>H |           7602 |           7564 |             99.5001 |
| bellevue_150th_eastgate | polygon_inward_3px  | bellevue_150th_eastgate:A>G |           2231 |           2222 |             99.5966 |
| bellevue_150th_eastgate | polygon_inward_3px  | bellevue_150th_eastgate:D>F |           5328 |           5328 |            100.0000 |
| bellevue_150th_eastgate | polygon_inward_3px  | bellevue_150th_eastgate:C>H |           1436 |           1409 |             98.1198 |
| bellevue_150th_eastgate | polygon_inward_3px  | bellevue_150th_eastgate:A>F |           3404 |           3329 |             97.7967 |
| bellevue_150th_eastgate | polygon_inward_3px  | bellevue_150th_eastgate:D>G |           1781 |           1777 |             99.7754 |
| bellevue_150th_eastgate | polygon_inward_3px  | bellevue_150th_eastgate:C>E |           1034 |            970 |             93.8104 |
| bellevue_150th_eastgate | polygon_inward_3px  | bellevue_150th_eastgate:B>E |            310 |            282 |             90.9677 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:C>E   |           3198 |           3190 |             99.7498 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:A>G   |           3602 |           3242 |             90.0056 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:A>H   |            521 |            399 |             76.5835 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:D>F   |            138 |            106 |             76.8116 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:D>G   |            322 |            295 |             91.6149 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:B>E   |            364 |            361 |             99.1758 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:B>H   |            153 |            153 |            100.0000 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:B>G   |              3 |              3 |            100.0000 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:C>F   |             29 |             21 |             72.4138 |
| bellevue_150th_se38th   | polygon_inward_3px  | bellevue_150th_se38th:C>H   |             22 |             22 |            100.0000 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:D>F          |           3472 |           2721 |             78.3698 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:D>G          |            618 |            193 |             31.2298 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:B>H          |           3011 |           2435 |             80.8701 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:C>E          |           4198 |           4171 |             99.3568 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:A>H          |           1419 |           1145 |             80.6906 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:A>G          |           3931 |           2847 |             72.4243 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:C>F          |            806 |            788 |             97.7667 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:B>E          |            280 |            274 |             97.8571 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:C>H          |            104 |             70 |             67.3077 |
| bellevue_ne8th          | polygon_inward_3px  | bellevue_ne8th:D>E          |             78 |             60 |             76.9231 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:B>H   |            525 |            525 |            100.0000 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:D>G   |            136 |            136 |            100.0000 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:C>E   |            321 |            321 |            100.0000 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:D>F   |            618 |            618 |            100.0000 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:D>E   |             52 |             52 |            100.0000 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:A>G   |            305 |            305 |            100.0000 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:A>H   |             56 |             56 |            100.0000 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:A>F   |             58 |             58 |            100.0000 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:C>H   |             16 |             16 |            100.0000 |
| bellevue_116th_ne12th   | polygon_outward_3px | bellevue_116th_ne12th:B>E   |             11 |             11 |            100.0000 |
| bellevue_150th_newport  | polygon_outward_3px | bellevue_150th_newport:C>E  |           3933 |           3933 |            100.0000 |
| bellevue_150th_newport  | polygon_outward_3px | bellevue_150th_newport:A>G  |           1654 |           1654 |            100.0000 |
| bellevue_150th_newport  | polygon_outward_3px | bellevue_150th_newport:D>G  |            646 |            646 |            100.0000 |
| bellevue_150th_newport  | polygon_outward_3px | bellevue_150th_newport:C>H  |           1727 |           1727 |            100.0000 |
| bellevue_150th_newport  | polygon_outward_3px | bellevue_150th_newport:B>H  |            382 |            382 |            100.0000 |
| bellevue_150th_newport  | polygon_outward_3px | bellevue_150th_newport:D>F  |            619 |            619 |            100.0000 |
| bellevue_150th_newport  | polygon_outward_3px | bellevue_150th_newport:A>H  |             58 |             58 |            100.0000 |
| bellevue_150th_newport  | polygon_outward_3px | bellevue_150th_newport:A>F  |            101 |            101 |            100.0000 |
| bellevue_150th_newport  | polygon_outward_3px | bellevue_150th_newport:B>E  |             26 |             26 |            100.0000 |
| bellevue_150th_eastgate | polygon_outward_3px | bellevue_150th_eastgate:A>H |           2549 |           2549 |            100.0000 |
| bellevue_150th_eastgate | polygon_outward_3px | bellevue_150th_eastgate:B>H |           7602 |           7602 |            100.0000 |
| bellevue_150th_eastgate | polygon_outward_3px | bellevue_150th_eastgate:A>G |           2231 |           2231 |            100.0000 |
| bellevue_150th_eastgate | polygon_outward_3px | bellevue_150th_eastgate:D>F |           5328 |           5328 |            100.0000 |
| bellevue_150th_eastgate | polygon_outward_3px | bellevue_150th_eastgate:C>H |           1436 |           1436 |            100.0000 |
| bellevue_150th_eastgate | polygon_outward_3px | bellevue_150th_eastgate:A>F |           3404 |           3404 |            100.0000 |
| bellevue_150th_eastgate | polygon_outward_3px | bellevue_150th_eastgate:D>G |           1781 |           1781 |            100.0000 |
| bellevue_150th_eastgate | polygon_outward_3px | bellevue_150th_eastgate:C>E |           1034 |           1034 |            100.0000 |
| bellevue_150th_eastgate | polygon_outward_3px | bellevue_150th_eastgate:B>E |            310 |            310 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:C>E   |           3198 |           3198 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:A>G   |           3602 |           3602 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:A>H   |            521 |            521 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:D>F   |            138 |            138 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:D>G   |            322 |            322 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:B>E   |            364 |            364 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:B>H   |            153 |            153 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:B>G   |              3 |              3 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:C>F   |             29 |             29 |            100.0000 |
| bellevue_150th_se38th   | polygon_outward_3px | bellevue_150th_se38th:C>H   |             22 |             22 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:D>F          |           3472 |           3472 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:D>G          |            618 |            618 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:B>H          |           3011 |           3011 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:C>E          |           4198 |           4198 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:A>H          |           1419 |           1419 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:A>G          |           3931 |           3931 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:C>F          |            806 |            806 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:B>E          |            280 |            280 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:C>H          |            104 |            104 |            100.0000 |
| bellevue_ne8th          | polygon_outward_3px | bellevue_ne8th:D>E          |             78 |             78 |            100.0000 |
