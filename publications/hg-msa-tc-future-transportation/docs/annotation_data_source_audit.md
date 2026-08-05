# Annotation Data-Source Audit

## Decision

The annotation renderer uses the per-recording tracker shards
`data/interim/<scene>/tracks_<recording_id>.parquet`. Each shard contains `frame`,
`track_id`, `cx`, and `cy`, so it provides the complete source track as a
point-by-point camera-coordinate polyline. The canonical manifest maps
`<scene>:<merged_track_id>` to `source_recording_id` and
`source_recording_track_id`; the latter is the shard `track_id`. The paired source
video is `data/raw/<scene>/<recording_id>.mp4`.

The renderer crops the raw source track to the canonical manifest `start_frame` and
`end_frame`. This is necessary because the feature-analysis cohort may retain a
trimmed segment of a longer raw tracker history. It does not reconstruct points from
endpoints and does not alter source files.

## Scene Inventory

| Scene | Tracker shards | Tracker rows | Shard bytes | Raw videos | Video bytes | Ordered shard-checksum aggregate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bellevue_116th_ne12th` | 21 | 9,857,165 | 308,205,315 | 21 | 6,667,525,369 | `3885cee4cb3584a09658785efa9ad25271b4039070452bd28b003c336ddc9c9b` |
| `bellevue_150th_newport` | 24 | 32,501,895 | 989,411,907 | 24 | 16,949,183,860 | `53d0b5635aabb076d9b4698d5c35c7c4fc5eaeedf60f784364d969f06b96d19a` |
| `bellevue_150th_eastgate` | 24 | 42,725,664 | 1,260,606,084 | 24 | 15,614,842,713 | `ef206bd357f76750529acae6bd1f57af3c925429dd225e3461746da8ac3d989a` |
| `bellevue_150th_se38th` | 23 | 35,286,786 | 1,030,805,526 | 23 | 10,874,831,259 | `615dd5a36de93292a624a802d02722887d967038fbf8d208e9347841c11fcf53` |
| `bellevue_ne8th` | 23 | 51,890,099 | 1,510,809,841 | 23 | 15,250,118,484 | `c70eff3c8299be49f45324e4a0332da652c6900e426d04e33f40e9ad6f78135e` |

Every exact shard path, recording path, shard SHA-256, size, row count, and row-group
count is stored in `annotations/provenance/trajectory_source_catalog.csv` (catalog
SHA-256 at generation: `c71149a4bfdc7931bed6997a32ecadf82cd6766cb833b2f9cb2ca87b9c56a780`).
The aggregate values above hash the ordered sequence of shard SHA-256 strings; they
are inventory fingerprints, not replacements for the per-file checksums.

Example path pairs:

- 116th: `data/interim/bellevue_116th_ne12th/tracks_Bellevue_116th_NE12th__2017-09-10_19-08-25.parquet` and matching `.mp4` under `data/raw/bellevue_116th_ne12th/`.
- Newport: `data/interim/bellevue_150th_newport/tracks_Bellevue_150th_Newport__2017-09-10_18-08-24.parquet` and matching raw video.
- Eastgate: `data/interim/bellevue_150th_eastgate/tracks_Bellevue_150th_Eastgate__2017-09-10_18-08-24.parquet` and matching raw video.
- SE38th: `data/interim/bellevue_150th_se38th/tracks_Bellevue_150th_SE38th__2017-09-10_18-08-24.parquet` and matching raw video.
- NE8th: `data/interim/bellevue_ne8th/tracks_Bellevue_Bellevue_NE8th__2017-09-10_18-08-23.parquet` and matching raw video.

## Variant and Lineage Assessment

The selected shard geometry is raw tracker-centroid output in camera pixels. It is
not interpolated or independently smoothed by the annotation application. The root
`data/processed/<scene>/trajectories.parquet` files provide a merged, Savitzky-Golay
smoothed variant, but they are very large scene-level files and do not directly carry
recording identity. Declared cleaned/filled parent files are absent for some scenes;
they were not invented or silently substituted.

The canonical publication cohort comes from
`data/processed/<scene>/feature_analysis/features_trimmed_frame_disp_norm.parquet`.
Those one-row feature files do not hold full polylines. The manifest's verified merge
offset reverses the merged ID to the local recording track, and the manifest supplies
recording, frame range, and point-count provenance.

## Real-Data Preflight

All 27,393 independent-test IDs mapped to a source shard, a local track with at least
two finite points, and an existing source video. All canonical frame intervals were
covered. For 1,237 items, the raw source track extends beyond one or both canonical
boundaries; the renderer crops to the manifest interval. This is a documented lineage
difference, not a missing-polyline approximation. No blocker remains for rendering
the primary cohort.

Raw videos total about 60.8 GB. They are not copied. The catalog records exact path
and size; video SHA-256 is intentionally an on-demand policy before clip/frame use,
while trajectory shards were fully hashed during this audit.
