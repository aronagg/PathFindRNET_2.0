# Manual Maneuver Annotation Schema

## Purpose and Isolation Rule

This schema prepares independent trajectory-level ground truth for the five Bellevue
scenes. It does not define an annotation interface and contains no manual labels.

**The manually annotated labels are isolated from homography-guided target estimation
and clustering model selection. They are used only for independent final evaluation
and explicitly identified post hoc diagnostic analyses.**

Annotators may label all subsets for inter-annotator and diagnostic studies, but method
developers must keep all label columns inaccessible during `target_estimation` and
`model_selection`. The primary final performance analysis must be reported on
`independent_test` after method choices are frozen.

## Columns

| Column | Type | Required | Definition |
| --- | --- | --- | --- |
| `scene_id` | string | yes | Fixed Bellevue scene identifier from the manifest. |
| `trajectory_id` | string | yes | Stable scene-namespaced trajectory identifier. |
| `split` | enum | yes | `target_estimation`, `model_selection`, or `independent_test`. |
| `entry_approach` | string | after annotation | Entry arm/approach from a separately frozen scene guide. |
| `exit_approach` | string | after annotation | Exit arm/approach from a separately frozen scene guide. |
| `manual_maneuver_id` | string | after annotation | Scene-specific maneuver class ID from a frozen coding guide. |
| `maneuver_type` | enum | after annotation | Coarse movement category. |
| `validity` | enum | after annotation | Visual interpretability/usability judgment. |
| `rare_movement` | boolean | after annotation | Whether the movement is rare but valid. |
| `annotator_id` | string | after annotation | Pseudonymous annotator identifier. |
| `annotation_timestamp` | ISO-8601 string | after annotation | Time the judgment was saved. |
| `confidence` | enum | after annotation | Annotator confidence. |
| `notes` | string | optional | Concise ambiguity or quality note. |

## Allowed Values

### `maneuver_type`

- `straight`
- `left`
- `right`
- `u_turn`
- `other`
- `unknown`

### `validity`

- `valid`
- `ambiguous`
- `unusable`

### `confidence`

- `high`
- `medium`
- `low`

### `rare_movement`

- `true`
- `false`

## Annotation Rules to Freeze Before Labeling

1. Create a scene-specific approach naming guide independent of all cluster outputs.
2. Define how partial trajectories, tracking switches, parked vehicles, and boundary
   truncation are labeled.
3. Keep `unknown` distinct from `ambiguous` and `unusable`.
4. Record independent first-pass annotations before adjudication.
5. Preserve raw annotator judgments and write adjudicated labels to a separate,
   versioned file.
6. Never derive `manual_maneuver_id` from HG targets, cluster IDs, OD pseudo-labels,
   or expected cluster counts.

## Empty Template Guarantee

`annotation_template.csv` pre-populates only `scene_id`, `trajectory_id`, and `split`.
Every human annotation field is empty by construction and is checked by automated
tests.
