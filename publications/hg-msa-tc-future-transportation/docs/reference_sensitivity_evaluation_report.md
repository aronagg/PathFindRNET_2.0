# Reference Sensitivity Evaluation

The canonical first/last-point, unbuffered polygon reference remains primary. These analyses do not overwrite primary metrics and were not used to alter targets, configurations, or interpretation thresholds.

## Metric Changes from Primary Reference

| reference_variant   |   min_coverage_pct |   max_coverage_pct |   mean_abs_delta_ari |   mean_abs_delta_nmi |   mean_abs_delta_purity |   mean_abs_delta_macro_f1 |
|:--------------------|-------------------:|-------------------:|---------------------:|---------------------:|------------------------:|--------------------------:|
| median_first_last_3 |            89.4191 |            98.3264 |               0.0005 |               0.0003 |                  0.0004 |                    0.0003 |
| median_first_last_5 |            89.1079 |            98.2148 |               0.0009 |               0.0007 |                  0.0006 |                    0.0004 |
| polygon_inward_3px  |            75.8492 |            97.0804 |               0.0122 |               0.0059 |                  0.0041 |                    0.0048 |
| polygon_outward_3px |            90.2490 |            99.0061 |               0.0047 |               0.0018 |                  0.0019 |                    0.0008 |

## Strategy Ranking Counts

| reference_variant   | metric   |   hg_aware_higher |   equal |   untargeted_higher |
|:--------------------|:---------|------------------:|--------:|--------------------:|
| primary             | ari      |                 7 |       2 |                   6 |
| primary             | nmi      |                 7 |       2 |                   6 |
| primary             | purity   |                 7 |       3 |                   5 |
| primary             | macro_f1 |                 7 |       2 |                   6 |
| median_first_last_3 | ari      |                 7 |       2 |                   6 |
| median_first_last_3 | nmi      |                 7 |       2 |                   6 |
| median_first_last_3 | purity   |                 7 |       3 |                   5 |
| median_first_last_3 | macro_f1 |                 7 |       2 |                   6 |
| median_first_last_5 | ari      |                 7 |       2 |                   6 |
| median_first_last_5 | nmi      |                 7 |       2 |                   6 |
| median_first_last_5 | purity   |                 8 |       2 |                   5 |
| median_first_last_5 | macro_f1 |                 7 |       2 |                   6 |
| polygon_inward_3px  | ari      |                 7 |       2 |                   6 |
| polygon_inward_3px  | nmi      |                 7 |       2 |                   6 |
| polygon_inward_3px  | purity   |                 8 |       2 |                   5 |
| polygon_inward_3px  | macro_f1 |                 7 |       2 |                   6 |
| polygon_outward_3px | ari      |                 7 |       2 |                   6 |
| polygon_outward_3px | nmi      |                 7 |       2 |                   6 |
| polygon_outward_3px | purity   |                 7 |       3 |                   5 |
| polygon_outward_3px | macro_f1 |                 7 |       2 |                   6 |

## NE8th Diagnostic

| reference_variant   |   valid_coverage_pct |   mean_abs_delta_ari |   mean_abs_delta_macro_f1 |
|:--------------------|---------------------:|---------------------:|--------------------------:|
| median_first_last_3 |              93.5673 |               0.0008 |                    0.0002 |
| median_first_last_5 |              93.4802 |               0.0010 |                    0.0004 |
| polygon_inward_3px  |              75.8492 |               0.0273 |                    0.0084 |
| polygon_outward_3px |              97.3871 |               0.0160 |                    0.0012 |

Polygon buffering changes coverage more strongly than the 3/5-point endpoint variants, especially for NE8th. Therefore conclusions that change only under buffering must be presented as boundary-sensitive.
