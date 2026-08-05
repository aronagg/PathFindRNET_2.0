# Polygon-Rule Reference Generation Protocol

## Scope and terminology

This protocol creates exhaustive **human-defined polygon-rule-based reference
labels** for the fixed 67,029-trajectory canonical cohort and exactly five Bellevue
scenes. The polygons and twelve legal entry-to-exit mappings per scene were designed
manually before independent-test clustering evaluation. The generator does not use
HG targets, cluster IDs, pseudo-reference labels, automatic OD assignments, EMAS
scores, or model-selection metrics.

The output is not fully independent per-trajectory manual annotation. It is a frozen,
deterministic application of human-defined camera-space endpoint rules.

## Frozen protocol

Each scene guide must contain four unique entry polygons, four unique exit polygons,
twelve unique legal mappings, a maneuver type for every mapping, and scene-specific
ambiguity notes. The freeze command validates these requirements and rejects derived
cluster/HG/EMAS fields. Version 1 cannot be overwritten.

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py freeze
```

## Endpoint and containment rule

The entry endpoint is the first finite `(cx, cy)` point and the exit endpoint is the
last finite point inside the canonical manifest frame interval. Points outside that
interval are not read into the endpoint decision. No smoothing changes the primary
endpoints.

Normalized guide polygons are scaled to the 1280 x 720 representative-frame space.
Containment uses Shapely `Polygon.covers(Point)`. Zero matches are unassigned; more
than one match is ambiguous. Polygon order and nearest-centroid or nearest-polygon
fallbacks never resolve the primary label. Boundary distance and nearest polygon are
retained only for diagnostics.

## Movement and status rule

A row is `valid` only when exactly one entry and one exit polygon match and that pair
exists in the frozen legal mapping. Its ID is `<scene_id>:<entry_id>><exit_id>`.
Every other canonical trajectory remains in the dataset with an explicit status and
reason. No row is silently discarded.

## Reproduction commands

Run from the repository root after installing the dedicated dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r publications\hg-msa-tc-future-transportation\requirements-reference-labels.txt
```

Then run:

```powershell
# Generate all labels and split exports
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py generate

# Generate legal/observed inventories and quality summaries
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py inventories

# Run endpoint and polygon sensitivity diagnostics
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py sensitivity

# Generate the deterministic QA queue and QC figures
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\reference_labels\cli.py figures

# Run all publication tests
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
```

`target_estimation` and `model_selection` labels are diagnostics/data-publication
outputs only. Future primary evaluation must load valid rows exclusively from
`independent_test_reference_labels.csv`. This task does not execute or unlock that
clustering run.
