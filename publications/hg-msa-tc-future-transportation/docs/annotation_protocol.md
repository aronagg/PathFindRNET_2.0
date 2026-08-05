# Manual Ground-Truth Protocol

## Cohorts

Protocol development uses 100 deterministic, recording-round-robin
`target_estimation` trajectories per scene (500 total). The pilot supports camera
orientation, manual approach naming, written-rule refinement, UI testing, and
training. Pilot labels are never independent-test evidence.

The primary cohort is every `independent_test` trajectory: 27,393 items for annotator
A and the same 27,393 for annotator B. Their ordering seeds differ. Ordering uses
scene and recording only and does not use any inferred movement structure.

## Manual Scene Codebook

An authorized designer must enter neutral approach IDs, optional names, normalized
label and arrow positions, valid entry/exit sets, coarse maneuver mappings, and
ambiguity notes on a representative source frame. Approach counts and locations may
not be inferred from HG or clustering. All five guide YAML files must have
`status: ready_for_freeze` and valid guide images.

The app's `Freeze Protocol` action validates the five guides, the written guideline,
and the pre-existing scientific freeze manifest. It then writes immutable
`annotation_protocol_v1.yaml`, its SHA-256 sidecar, and a freeze report. Existing v1
files cannot be overwritten. Any later change requires an explicit new version.

## Labels

The human chooses entry, exit, coarse maneuver type, validity, confidence, and an
optional note. `manual_maneuver_id` is generated only from the scene and the two human
approach choices. `rare_movement` is not human input; it is derived after consensus
from a configurable scene-level frequency threshold. Full decision rules are in
`annotations/protocol/annotation_guideline.md`.

## Integrity

Every save is one SQLite transaction with a UTC timestamp, protocol version, source
checksum, renderer version, revision number, predecessor ID, and audit event. A new
revision deactivates but does not delete the previous revision. Exports are immutable
and checksummed. Hidden repeats are implemented only as a documented optional future
extension and are disabled by default.
