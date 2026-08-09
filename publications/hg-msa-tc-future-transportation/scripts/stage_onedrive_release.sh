#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-.}"
STAGING_ROOT="${STAGING_ROOT:?Set STAGING_ROOT to a local OneDrive-synced folder}"
INCLUDE_PRIORITIES="${INCLUDE_PRIORITIES:-${PRIORITY_LEVEL:-P0}}"
DRY_RUN="${DRY_RUN:-1}"

python - "$SOURCE_ROOT" "$STAGING_ROOT" "$INCLUDE_PRIORITIES" "$DRY_RUN" <<'PY'
import csv
import shutil
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
staging_root = Path(sys.argv[2]).resolve()
priorities = {p.strip() for p in sys.argv[3].split(",") if p.strip()}
dry_run = sys.argv[4] not in {"0", "false", "False", "no", "No"}
manifest = source_root / "publications/hg-msa-tc-future-transportation/docs/onedrive_upload_manifest_prioritized.csv"
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
selected = [r for r in rows if r["upload_priority"] in priorities and r["upload_priority"] != "P3"]
missing = []
copied = []
if not dry_run:
    staging_root.mkdir(parents=True, exist_ok=True)
    (staging_root / "00_RELEASE_MANIFESTS").mkdir(parents=True, exist_ok=True)
for row in selected:
    src = source_root / row["repository_relative_path"]
    dst = staging_root / row["onedrive_target_relative_path"]
    if not src.exists():
        missing.append(row)
        continue
    if dry_run:
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        continue
    shutil.copy2(src, dst)
    copied.append(row)
print(f"Selected files: {len(selected)}")
print(f"Missing sources: {len(missing)}")
print(f"Copied files: {len(copied)}")
print(f"DryRun: {dry_run}")
PY
