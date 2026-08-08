# Revised Limitations

1. The study evaluates five Bellevue intersections only. It does not validate
   HG-SMG-TC across the full Traffic Node Video Dataset or across unrelated
   cities, camera types or traffic regimes.
2. The polygon-rule reference is exhaustive and independent of clustering
   outputs, but it is not fully manual per-trajectory ground truth. It is based
   on camera-space endpoint containment and manually defined movement mappings.
3. The frozen HG target estimator can count geometric endpoint submodes rather
   than semantic maneuvers. SE38th is the clearest failure case, with `K_HG=18`
   versus 9 independently observed semantic movements.
4. Homography calibration depends on manually selected point correspondences.
   All scenes pass the diagnostic quality gate, but endpoint extrapolation beyond
   the calibration hull is high.
5. EMAS_HG is heuristic and task-specific. It should not be interpreted as a
   universal clustering-validity index or as independent validation.
6. The HG-SMG-TC extension is a locked post-review extension evaluation, not a
   new pristine blind holdout. The original independent-test results were known
   before the extension was designed, although the extension choices were frozen
   before its first test execution.
7. The number of scenes is small. Scene-level paired analysis is descriptive and
   should not be presented as strong population-level inference.
8. Baseline comparisons show that simple endpoint methods, especially KMeans,
   can be strong. This limits any broad claim that graph-guided selection
   dominates simple coordinate representations.
9. Google-derived or map-derived imagery provenance requires manual publication
   review. Redistribution should be avoided unless licensing and attribution are
   confirmed.
10. Future work should prospectively test road-branch consolidation using lane
   markings, mandatory-turn arrows, map topology or additional external
   annotations.
