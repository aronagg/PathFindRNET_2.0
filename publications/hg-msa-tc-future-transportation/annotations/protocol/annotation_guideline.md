# Manual Trajectory Annotation Guideline

## Scope and Blinding

This protocol labels observed vehicle movements from full camera-coordinate
trajectories and source imagery. Do not consult homography targets, endpoint groups,
cluster assignments, pseudo-labels, model-selection results, or label frequencies.
The 500-item `target_estimation` pilot is for protocol development and training only.
It is excluded from independent-test metrics.

The scene guide defines neutral approach identifiers manually. Select entry and exit
from that frozen guide. The software then constructs
`<scene_id>:<entry_approach>><exit_approach>` deterministically. Never invent an
approach ID during primary annotation. The reserved `UNKNOWN` value is available only
when an entry or exit cannot be assigned; explain it in notes and normally use
`ambiguous` or `unusable` validity.

## Decision Order

1. Verify that the displayed full path and optional video identify one tracked object.
2. Decide validity: `valid`, `ambiguous`, or `unusable`.
3. Select the visually observed entry and exit approaches.
4. Select the coarse maneuver type: `straight`, `left`, `right`, `u_turn`, `other`,
   or `unknown`.
5. Record confidence and a short note where it helps later adjudication.

## Validity Rules

- **Complete and clearly interpretable:** `valid`; label entry, exit, and type.
- **Partial but still interpretable:** `valid` when both movement arms are reliable;
  otherwise `ambiguous`.
- **Boundary-truncated:** `ambiguous` if the missing segment makes entry or exit
  uncertain; it may remain `valid` when the movement is still visually decisive.
- **Fragmented path:** `ambiguous` when a single movement is likely but uncertain;
  `unusable` when continuity cannot be established.
- **Tracking ID switch:** `unusable` when two objects are joined or one object changes
  identity such that the movement cannot be trusted.
- **Physically implausible path:** check video. Use `unusable` for tracking artifacts;
  do not discard an unusual movement merely because it is uncommon.
- **Stopped or parked vehicle:** `unusable` when no intersection movement is observed.
- **Does not cross or meaningfully enter the intersection:** `unusable`, except a
  clearly observed driveway/service movement documented as `other`.
- **Unclear entry or unclear exit:** `ambiguous` if a plausible movement remains;
  `unusable` if an approach cannot be assigned at all.
- **Pedestrian, cyclist, or non-target object:** `unusable` for this vehicle study.
- **Duplicate or near-duplicate:** annotate the displayed item independently and note
  the suspected duplicate; do not remove it or infer its label from another item.
- **Unusual but valid maneuver:** `valid`; use the observed entry and exit. Rarity is
  derived only after consensus.
- **U-turn:** `valid` and `u_turn` when reversal to the same approach is visually clear.
- **Emergency/service access:** `valid` and usually `other` when physically observed;
  note the access type.
- **Parking or driveway movement:** `valid` and `other` only when the frozen scene guide
  includes a usable access identifier; otherwise `ambiguous` or `unusable`.
- **Illegal but visually observed movement:** label the observed movement as `valid`;
  legality is not inferred or judged here.

## Unknown, Ambiguous, and Unusable

`unknown` is a coarse `maneuver_type`: a trajectory can be valid enough to assign
entry/exit but its turn category may not fit or may remain unclear. `ambiguous` is a
validity judgment indicating competing interpretations that should be adjudicated.
`unusable` means the source evidence cannot support a reliable movement label.

## Confidence

- `high`: entry, exit, type, and object continuity are visually clear.
- `medium`: label is more likely than alternatives but one minor uncertainty remains.
- `low`: a best manual interpretation is recorded for adjudication, with a note.

## Independent and Adjudication Phases

Annotators A and B work independently and cannot inspect each other's records. Their
first-pass exports are immutable. Agreement is computed only after both exports are
complete. Adjudication opens disagreements and low-confidence cases, preserves both
raw labels, and stores a separate consensus decision with a reason.
