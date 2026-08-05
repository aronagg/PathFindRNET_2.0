# Annotation User Guide

Run all commands from repository root in Windows PowerShell.

## Install Optional Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r publications\hg-msa-tc-future-transportation\requirements-annotation.txt
```

## Build and Validate Source Inputs

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py build-source-catalog
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py preflight
```

The preflight is read-only for repository data and logs access under
`annotations/provenance/`. It does not run clustering or evaluation.

## Create the Protocol Pilot

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py build-pilot-queue
```

This creates 100 `target_estimation` items per scene. Pilot work is training and
protocol development only.

After manually entering at least a draft approach set, start the isolated pilot UI:

```powershell
.\publications\hg-msa-tc-future-transportation\run_annotation_app.ps1 -Role protocol_pilot -Mode single -Port 8504
```

Its database is separate, ignored by Git, and excluded from all independent-test
metrics. The pilot can render before approaches exist, but cannot create a useful
movement label until the designer supplies manual approach IDs.

## Configure Scene Guides

Prepare neutral representative frames and empty guide templates:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py prepare-scene-guides
```

Start designer mode:

```powershell
.\publications\hg-msa-tc-future-transportation\run_annotation_app.ps1 -Role protocol_designer -Port 8501
```

Enter one neutral approach ID at a time. Use `Point` for a simple anchor or `Polygon`
to click the boundary vertices of a manually interpreted approach zone, then finish
the polygon with the check-mark button. The image toolbar supports fit-to-window and
25--250% display scaling. `entry`, `exit`, and `both` describe the visual role of the
zone. Review the editable table and all five generated guide images before saving.

The point or polygon is a visual codebook aid. It does not automatically assign an
approach to a trajectory and is not a source of pseudo-labels. No automatic endpoint,
HG-target, or cluster information may be consulted. Set a guide ready only after
manual verification.

## How the Ground Truth Is Produced

The frozen scene guide defines a shared manual vocabulary. For each trajectory, a
human annotator inspects the rendered camera-view path and, when needed, the source
video frames. The annotator manually records the entry approach, exit approach,
coarse maneuver type, validity, confidence, and an optional note. The application
derives `manual_maneuver_id` only from the scene and the two human approach choices.

Annotators A and B independently label the complete independent-test cohort in
different deterministic orders. Their first-pass exports are immutable. Agreement
is measured after both exports are complete; disagreements are resolved in the
separate adjudicator workflow without replacing either original decision. The
adjudicated record is the reference ground-truth label. Pilot labels, polygon
containment, clustering output, HG targets, and model predictions are not ground
truth.

## Freeze the Manual Protocol

Use the `Freeze Protocol` action in designer mode or run:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py freeze-protocol
```

The action fails while any guide remains a draft and refuses to overwrite v1. Verify
`annotation_protocol_v1.sha256` before proceeding.

## Generate Primary Queues

After freeze, regenerate both queues so their status changes from locked to available:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py build-primary-queues
```

Each queue must contain all 27,393 independent-test trajectories exactly once. A and
B have different deterministic orders and no label columns.

## Start Independent Annotation

Use separate ports and database files:

```powershell
.\publications\hg-msa-tc-future-transportation\run_annotation_app.ps1 -Role annotator_A -Mode single -Port 8501
.\publications\hg-msa-tc-future-transportation\run_annotation_app.ps1 -Role annotator_B -Mode single -Port 8502
```

The command field supports a complete keyboard submission:
`entry exit type validity confidence`, for example `A B s v h`. Type keys are
`s/l/r/u/o/x`; validity keys are `v/a/u`; confidence keys are `h/m/l`. Press Enter to
save and move forward. Grid mode requires a visible `Reviewed` check before manual
selection. It never auto-selects or suggests a class.

Optional source clips are generated on demand under the ignored cache. The embedded
player supports play/pause; the manual frame control steps one frame at a time.

## Back Up Databases

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py backup annotator_A
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py backup annotator_B
```

Each command uses SQLite backup and runs `PRAGMA integrity_check`.

## Export Immutable First Passes

Run only after each queue is complete:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py export-first-pass annotator_A
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py export-first-pass annotator_B
```

Existing exports are never overwritten. Each CSV has a SHA-256 sidecar and each
SQLite audit log is exported as JSONL.

## Agreement and Adjudication

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py agreement
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py init-database adjudication
.\publications\hg-msa-tc-future-transportation\run_annotation_app.ps1 -Role adjudicator -Port 8503
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py export-consensus
```

Generate the post-consensus inventory with a documented threshold, for example 1%:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py inventory --rare-threshold 0.01
```

## Verify Checksums and Progress

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py verify-checksum publications\hg-msa-tc-future-transportation\annotations\exports\independent_test_annotator_A.csv
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py progress annotator_A --elapsed-hours 8.5
```

Never copy raw videos into Git, commit real SQLite databases/exports, unlock scientific
test clustering, or use manual labels to revise the frozen method.
