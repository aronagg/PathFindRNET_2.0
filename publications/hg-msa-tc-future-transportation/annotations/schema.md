# Blind Manual Annotation Schema

## Raw Record

Each append-only annotation revision contains:

| Column | Definition |
| --- | --- |
| `annotation_id` | UUID for this immutable revision. |
| `scene_id`, `trajectory_id`, `split`, `recording_id` | Frozen source identity. |
| `entry_approach`, `exit_approach` | Human choices from the frozen scene guide. |
| `manual_maneuver_id` | Deterministic `<scene>:<entry>><exit>` value. |
| `maneuver_type` | `straight`, `left`, `right`, `u_turn`, `other`, or `unknown`. |
| `validity` | `valid`, `ambiguous`, or `unusable`. |
| `confidence` | `high`, `medium`, or `low`. |
| `annotator_id`, `annotation_timestamp_utc` | Annotator and automatic UTC time. |
| `protocol_version` | Immutable annotation-protocol version. |
| `trajectory_source_checksum` | SHA-256 of the full-polyline source shard. |
| `rendering_version` | Camera renderer version. |
| `notes` | Optional manual note. |
| `revision_number`, `supersedes_annotation_id` | Non-destructive history. |

`rare_movement` is deliberately absent from human input. It can be derived after
consensus using a documented scene-level frequency threshold. No cluster, HG target,
automatic OD assignment, pseudo-label, metric, suggested label, or another person's
label is permitted in a blind queue or first-pass form.

## Storage

Annotator A, annotator B, and adjudication use separate SQLite databases. One active
revision per annotator/trajectory is enforced by a partial unique index; older
revisions remain stored. First-pass CSV and audit JSONL exports are immutable and have
SHA-256 sidecars. Adjudication stores both raw records as JSON plus a separate
consensus record and reason.

## Queue Contract

Queue files contain source/provenance fields only: queue identity and position, scene,
trajectory, split, recording, local source track, frame range, point count, source
paths/checksum, and video availability. Primary queue status remains
`locked_pending_protocol_freeze` until all five manually configured scene guides are
validated and `annotation_protocol_v1.yaml` is frozen.
