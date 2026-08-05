# Task 03 Execution Report

## Scope

- Branch: `feature/futuretransp-ground-truth-annotation`
- Base/final previous-task commit: `bad5f6a9aa43ee0fee16aa47e0ce3903ffec72d8`
- Frozen scientific configuration SHA-256:
  `829c95c4f0a012d08433536b22379afad4d5a79122e52f8fd6ac12817882167c`
- Fixed scenes: the five Bellevue scenes only.

## Implemented

The task added a blind Streamlit annotation UI, deterministic pilot and two-annotator
primary queues, manual scene-guide setup and immutable protocol freeze, full-polyline
camera rendering, optional video/frame assistance, append-only SQLite revisions,
audit and backup/export functions, agreement analysis, separate adjudication,
post-consensus inventory generation, preflight checks, documentation, and tests.

## Real-Data Preflight

- Pilot: 500 total, exactly 100 `target_estimation` items per scene.
- Primary A: 27,393 unique `independent_test` trajectories.
- Primary B: the same 27,393 in a different deterministic order.
- Full source polyline found: 27,393/27,393.
- Finite camera coordinates: 27,393/27,393.
- Canonical frame interval covered: 27,393/27,393.
- Source video found: 27,393/27,393.
- Raw track extends beyond canonical interval: 1,237; UI crops to manifest bounds.

The queue status is `locked_pending_protocol_freeze`. Draft guide frames and templates
exist, but approach IDs and mappings intentionally remain empty pending authorized
manual setup. Therefore no real annotation database, first-pass export, agreement
result, adjudication result, or inventory was generated.

## Scientific Isolation

No clustering, target estimation, pseudo-labeling, or independent-test evaluation ran.
No output was written under `results/independent_test`. HG targets, automatic OD
regions, clusters, selected algorithms, and metrics are absent from the blind queue
allowlist and UI. The pre-existing scientific protocol remains locked.

## Validation Status

- Publication tests: `45 passed`.
- Ruff lint over the annotation package and all publication tests: passed.
- Ruff format check over all new Python files: passed.
- Streamlit `AppTest` for protocol-designer mode: zero exceptions.
- PowerShell launcher parser: passed.
- Manual protocol freeze gate: correctly refused to freeze the five unconfigured
  draft guides.
- Five real, non-labeling pilot previews were rendered and visually inspected with
  preserved camera aspect ratio, complete polylines, direction arrows, and start/end
  markers. Preview files and the trajectory-level preflight table remain local
  generated artifacts and are excluded from Git/package payload.

The only workflow blocker is intentional: an authorized protocol designer must
manually define and verify the five approach codebooks. After that, freeze v1 and
regenerate the primary queues to change their status to `available`.
