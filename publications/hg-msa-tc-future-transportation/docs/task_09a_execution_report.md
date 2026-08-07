# Task 09A Execution Report

## Scope

- Branch: `feature/futuretransp-hg-smg-preregistration`
- Exact base: `8bfb4826725ea5c4c04042c037b521edcf216ec4`
- Deliverable: HG-SMG-TC protocol/preregistration freeze only
- HG-SMG-TC implementation executed: no
- New scientific result generated: no
- Independent-test reference or metric used for design: no

## Principal Decisions

- Primary SAC is directional-geometric and uses top-view side/bearing plus directed
  five-point camera-isotropic heading.
- OD-profile similarity is diagnostic-only in primary SAC.
- SAC uses 500 deterministic descriptor bootstraps, a 0.95 self-consistency quantile,
  max-normalized pair compatibility, and complete-link consolidation.
- SMG reuses the frozen support-threshold grid and selection heuristic.
- UATP uses 500 recording-aware full-pipeline bootstraps and a primary 90% interval.
- PCMS uses distance to the interval before the exact frozen tie-break sequence.
- A0-A10, H1-H7, seeds, edge cases, and sensitivity-only variants are frozen.

Protocol SHA-256:
`2e892f25b1c1770a55fa29e36206b634f424f9ff2508bf045c92d388e69b4be6`.

## Holdout Audit

All 115 raw recordings from the five fixed Bellevue scenes are represented in the
67,029-trajectory canonical manifest. No pristine unused same-scene recording exists.
No holdout was fabricated.

## Commands and Verification

```powershell
git -c safe.directory=E:/Education/Doktori_iskola/research/FinalPhase/PathFindRNET_2.0 switch -c feature/futuretransp-hg-smg-preregistration
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/reproduction/cli.py validate-preregistration
./.venv/Scripts/python.exe -m pytest publications/hg-msa-tc-future-transportation/tests -q
./.venv/Scripts/python.exe -m ruff check publications/hg-msa-tc-future-transportation/code publications/hg-msa-tc-future-transportation/tests
```

- Protocol validator: passed; 11 planned ablations, 0 extension outputs.
- Pytest: `127 passed in 69.12s`.
- Ruff: `All checks passed!`.

## Scientific Boundary

This is an explicitly post-review extension motivated by known failure analysis. It
does not claim blind prospective conception. Its safeguard is that all extension
choices are frozen before the first locked post-review extension evaluation. Task-08
homographies, targets, model selections, labels, and independent assignments remain
unchanged.
