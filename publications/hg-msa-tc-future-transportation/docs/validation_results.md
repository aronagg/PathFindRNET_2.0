# Evaluation Protocol Validation Results

Validation command:

```powershell
.\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
```

Result on 2026-08-04:

```text
............                                                             [100%]
12 passed in 67.76s (0:01:07)
```

Validated properties:

- deterministic scene-namespaced trajectory IDs and feature fingerprints;
- deterministic complete-recording split assignment;
- no trajectory ID in more than one subset;
- no exact trajectory fingerprint in more than one subset;
- all five required scenes in every subset;
- per-scene proportions within 6 percentage points of 30/30/40;
- stable manifest and split ordering;
- required manifest, split, and annotation columns;
- all human annotation fields empty;
- current source files match recorded SHA-256 checksums;
- source-file checksums remain unchanged during a full manifest rebuild.

The full manifest was rebuilt during the test session rather than testing only cached
CSV content.
