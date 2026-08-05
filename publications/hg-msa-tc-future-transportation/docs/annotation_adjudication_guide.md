# Adjudication Guide

1. Confirm that both first-pass exports are complete, immutable, and checksum-valid.
2. Run agreement analysis. Review missing labels before interpreting kappa.
3. Start the app with role `adjudicator`; it opens only disagreement or low-confidence
   cases and shows both original records side by side.
4. Inspect the full camera path, frozen scene guide, and source video where needed.
5. Enter an independent consensus label and a concise reason. Do not modify either
   original export.
6. Export consensus only after all queued cases are resolved and verify its checksum.
7. Generate the manual inventory from consensus. Rarity is derived by the configured
   threshold and must not be manually assigned.

Report exact maneuver-ID, entry, exit, validity, and coarse-type agreement; raw
percentage agreement; Cohen's kappa where mathematically defined; scene, movement,
and confidence strata; disagreement matrix; and missing-label counts. Synthetic test
metrics validate software only and are not scientific findings.
