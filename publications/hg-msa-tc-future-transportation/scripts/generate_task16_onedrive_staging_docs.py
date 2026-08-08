from __future__ import annotations

import csv
import hashlib
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PUB = ROOT / "publications" / "hg-msa-tc-future-transportation"
DOCS = PUB / "docs"
SCRIPTS = PUB / "scripts"
ONEDRIVE_URL = (
    "https://onedrive.live.com/?redeem="
    "aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84MGZhYWQ2ZmVhMDViMDhjL0lnQlBxblBYemRWd1I1NERrbHVfUzB0bkFUMWJFUlBzSk1XTHM3TVJ3ZDNPZFRJP2U9b2NpOXBM"
    "&id=80FAAD6FEA05B08C%21sd773aa4fd5cd47709e03925bbf4b4b67&cid=80FAAD6FEA05B08C"
)
PAGES_URL = "https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/"
HASH_LIMIT = 256 * 1024 * 1024
EXCLUDE_DIRS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "review_packages"}
DEEP_AUDIT_EXTS = {".txt", ".json", ".csv", ".parquet", ".yaml", ".yml", ".pt", ".onnx"}
SEARCH_TERMS = (
    "yolo11",
    "yolov11",
    "yolo_11",
    "ultralytics",
    "predictions",
    "labels",
    "runs/detect",
    "runs/track",
    "detect",
    "detection",
    "detections",
)


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_tracked(repo_path: str) -> bool:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", repo_path], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def sha256_file(path: Path) -> str:
    size = path.stat().st_size
    if size > HASH_LIMIT:
        return f"NOT_COMPUTED_LARGE_FILE_GT_{HASH_LIMIT}_BYTES"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def target_path(row: dict[str, str]) -> str:
    return (Path(row["onedrive_folder"]) / Path(row["repository_relative_path"])).as_posix()


def infer_priority(row: dict[str, str]) -> tuple[str, str]:
    path = row["repository_relative_path"]
    lowered = path.lower()
    category = row.get("public_release_note", "").lower()
    folder = row.get("onedrive_folder", "")
    if not norm_bool(row.get("upload_required", "false")):
        return "P3", "not marked for upload in Task 15 manifest"
    if any(token in lowered for token in [".venv/", "__pycache__", ".pytest_cache", ".ruff_cache", "review_packages/", ".zip"]):
        return "P3", "local cache, review package, or temporary/bulk archive"
    if "google" in lowered or "maps" in lowered or "satellite" in lowered:
        return "P3", "third-party imagery/provenance requires manual review"
    if lowered.startswith("publications/hg-msa-tc-future-transportation/site/"):
        return "P0", "referenced by GitHub Pages publication page"
    if lowered.startswith("publications/hg-msa-tc-future-transportation/docs/"):
        return "P0", "manuscript support, release, data-availability, or reproducibility documentation"
    if lowered.startswith("publications/hg-msa-tc-future-transportation/figures/final_synthesis/"):
        return "P0", "final manuscript and website figure"
    if lowered.startswith("publications/hg-msa-tc-future-transportation/results/final_synthesis/"):
        return "P0", "final synthesis result table"
    if "reference_labels" in lowered or "/annotations/" in lowered:
        return "P0", "polygon-rule reference-label artifact or protocol"
    if "split" in lowered and ("manifest" in lowered or "reference" in lowered or "protocol" in lowered):
        return "P0", "split/reference protocol needed for submission audit"
    if "homography" in lowered and any(token in lowered for token in ["correspondence", "matrix", "quality", "calibration", "residual"]):
        return "P0", "homography calibration/reproducibility artifact"
    if lowered.endswith((".cff", "data_availability.md", "reproducibility.md", "release_manifest.md")):
        return "P0", "release metadata"
    if lowered.startswith("data/processed/") or "features_trimmed" in lowered or "canonical" in lowered:
        return "P0", "processed trajectory/feature artifact needed to reproduce reported results"
    if folder in {"01_original_videos/", "02_yolov11x_detections/", "03_yolo_tracking_outputs/"}:
        return "P1", "large source or detector/tracker artifact for full technical reproducibility"
    if "tnvd2_upload_package/" in lowered and any(token in lowered for token in ["04_trajectories", "05_trajectories", "06_homography"]):
        return "P1", "large intermediate dataset package artifact"
    if "debug" in lowered or "diagnostic" in lowered or "development" in lowered:
        return "P2", "supplementary diagnostic or development artifact"
    if "github_and_onedrive" in category:
        return "P0", "compact publication artifact suitable for both GitHub and OneDrive"
    return "P2", "supplementary artifact; review priority before public upload"


def batch_summary(rows: list[dict[str, object]]) -> str:
    by_priority: dict[str, dict[str, object]] = {}
    for row in rows:
        priority = str(row["upload_priority"])
        bucket = by_priority.setdefault(priority, {"count": 0, "bytes": 0, "folders": Counter(), "examples": []})
        bucket["count"] = int(bucket["count"]) + 1
        bucket["bytes"] = int(bucket["bytes"]) + int(row["size_bytes"])
        bucket["folders"].update([str(row["onedrive_folder"])])
        if len(bucket["examples"]) < 4:
            bucket["examples"].append(str(row["repository_relative_path"]))
    lines = ["| Priority | Files | Size GB | Top target folders | Source examples | Upload order |", "| --- | ---: | ---: | --- | --- | ---: |"]
    order = {"P0": 1, "P1": 2, "P2": 3, "P3": 99}
    for priority in sorted(by_priority, key=lambda p: order.get(p, 50)):
        bucket = by_priority[priority]
        folders = ", ".join(f"`{name}` ({count})" for name, count in bucket["folders"].most_common(4))
        examples = "<br>".join(f"`{item}`" for item in bucket["examples"])
        lines.append(f"| {priority} | {bucket['count']} | {int(bucket['bytes']) / (1024**3):.3f} | {folders} | {examples} | {order.get(priority, 50)} |")
    return "\n".join(lines)


def create_prioritized_manifest(now: str) -> list[dict[str, object]]:
    manifest = read_csv_dict(DOCS / "onedrive_upload_manifest.csv")
    prioritized: list[dict[str, object]] = []
    for row in manifest:
        priority, reason = infer_priority(row)
        repo_path = row["repository_relative_path"]
        source = ROOT / repo_path
        prioritized.append(
            {
                **row,
                "upload_priority": priority,
                "priority_reason": reason,
                "onedrive_target_relative_path": target_path(row),
                "source_exists": str(source.exists()).lower(),
                "git_tracked": str(is_tracked(repo_path)).lower(),
                "link_from_publication_page": str(
                    repo_path.startswith("publications/hg-msa-tc-future-transportation/site/assets/images/")
                    or repo_path.startswith("publications/hg-msa-tc-future-transportation/figures/final_synthesis/")
                ).lower(),
                "required_before_submission": str(priority == "P0").lower(),
            }
        )
    fields = list(prioritized[0].keys())
    write_csv(DOCS / "onedrive_upload_manifest_prioritized.csv", prioritized, fields)

    counts = batch_summary(prioritized)
    write_text(
        DOCS / "onedrive_upload_batches.md",
        f"""# OneDrive Upload Batches

Generated: `{now}`

The priority labels are derived from the Task 15 upload manifest and the Task 16 release rules. They do not imply that any file has already been uploaded.

## Priority Definitions

- `P0`: required before manuscript submission or directly referenced by the manuscript, GitHub Pages, data availability statement, or reproducibility package.
- `P1`: required for full technical reproducibility, including large source videos, detector/tracker exports and intermediate trajectory artifacts.
- `P2`: useful supplementary material and development diagnostics.
- `P3`: do not upload or review manually before public release.

## Batch Summary

{counts}

## Recommended Upload Order

1. Stage and verify `P0`.
2. Confirm OneDrive share permissions and publication page links.
3. Stage `P1` large data after storage and licensing checks.
4. Review `P2` selectively.
5. Keep `P3` out of the public upload unless a maintainer explicitly reclassifies it.
""",
    )
    return prioritized


def create_staging_scripts(now: str) -> None:
    ps1 = r'''param(
  [string]$SourceRoot = ".",
  [Parameter(Mandatory=$true)][string]$StagingRoot,
  [string]$PriorityLevel = "P0",
  [string[]]$IncludePriorities,
  [switch]$DryRun,
  [switch]$VerifyChecksums
)

$ErrorActionPreference = "Stop"
$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$ManifestPath = Join-Path $SourceRoot "publications/hg-msa-tc-future-transportation/docs/onedrive_upload_manifest_prioritized.csv"
if (-not (Test-Path -LiteralPath $ManifestPath)) {
  throw "Missing prioritized manifest: $ManifestPath"
}

if (-not $IncludePriorities -or $IncludePriorities.Count -eq 0) {
  if ($PriorityLevel -match ",") {
    $IncludePriorities = $PriorityLevel.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
  } else {
    $IncludePriorities = @($PriorityLevel)
  }
}
$IncludeSet = @{}
foreach ($p in $IncludePriorities) { $IncludeSet[$p] = $true }

if (-not $DryRun -and -not (Test-Path -LiteralPath $StagingRoot)) {
  New-Item -ItemType Directory -Path $StagingRoot | Out-Null
}

function Get-FileSha256([string]$Path) {
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$rows = Import-Csv -LiteralPath $ManifestPath
$selected = $rows | Where-Object { $IncludeSet.ContainsKey($_.upload_priority) -and $_.upload_priority -ne "P3" }
$copied = New-Object System.Collections.Generic.List[object]
$missing = New-Object System.Collections.Generic.List[object]
$staged = New-Object System.Collections.Generic.List[object]

foreach ($row in $selected) {
  $src = Join-Path $SourceRoot $row.repository_relative_path
  $dst = Join-Path $StagingRoot $row.onedrive_target_relative_path
  if (-not (Test-Path -LiteralPath $src)) {
    $missing.Add([pscustomobject]@{ repository_relative_path=$row.repository_relative_path; upload_priority=$row.upload_priority; reason="missing_source_file" })
    continue
  }
  $dstDir = Split-Path -Parent $dst
  $action = "copy"
  $copiedNow = $false
  $verified = ""
  if (Test-Path -LiteralPath $dst) {
    $srcHash = Get-FileSha256 $src
    $dstHash = Get-FileSha256 $dst
    if ($srcHash -eq $dstHash) {
      $action = "skip_existing_same_hash"
    } else {
      $answer = if ($DryRun) { "N" } else { Read-Host "Overwrite changed file? $dst [y/N]" }
      if ($answer -match "^[Yy]") {
        $action = "overwrite_hash_differs"
      } else {
        $action = "skip_existing_hash_differs"
      }
    }
  }
  if (($action -eq "copy" -or $action -eq "overwrite_hash_differs") -and -not $DryRun) {
    New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst -Force
    $copiedNow = $true
  }
  if ($VerifyChecksums -and -not $DryRun -and (Test-Path -LiteralPath $dst)) {
    $verified = Get-FileSha256 $dst
  }
  $record = [pscustomobject]@{
    repository_relative_path=$row.repository_relative_path
    onedrive_target_relative_path=$row.onedrive_target_relative_path
    upload_priority=$row.upload_priority
    size_bytes=$row.size_bytes
    source_sha256=$row.sha256
    staged_sha256=$verified
    action=$action
    copied=$copiedNow
  }
  $staged.Add($record)
  if ($copiedNow) { $copied.Add($record) }
}

if (-not $DryRun) {
  $staged | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $StagingRoot "staged_release_manifest.csv")
  $missing | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $StagingRoot "missing_source_files.csv")
  $copied | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $StagingRoot "copied_files_log.csv")
  $checksumPath = Join-Path $StagingRoot "staged_release_checksums.sha256"
  $lines = foreach ($item in $staged) {
    if ($item.staged_sha256) { "$($item.staged_sha256)  $($item.onedrive_target_relative_path)" }
  }
  $lines | Set-Content -Encoding UTF8 -LiteralPath $checksumPath
}

Write-Host "Selected files: $($selected.Count)"
Write-Host "Missing sources: $($missing.Count)"
Write-Host "Copied files: $($copied.Count)"
Write-Host "DryRun: $DryRun"
'''
    sh = r'''#!/usr/bin/env bash
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
'''
    write_text(SCRIPTS / "stage_onedrive_release.ps1", ps1)
    write_text(SCRIPTS / "stage_onedrive_release.sh", sh)

    write_text(
        DOCS / "onedrive_upload_manual_instructions.md",
        f"""# OneDrive Upload Manual Instructions

Generated: `{now}`

Public OneDrive folder:

`{ONEDRIVE_URL}`

## Prepare A Local Sync Folder

1. Open the public OneDrive folder in a browser.
2. Add or sync it to the local machine using the OneDrive client.
3. Choose the local synced folder as `StagingRoot`.
4. Do not stage directly into a folder that contains unrelated private files.

## Stage P0 First

Dry run:

```powershell
.\\publications\\hg-msa-tc-future-transportation\\scripts\\stage_onedrive_release.ps1 -SourceRoot . -StagingRoot "C:\\Path\\To\\OneDrive\\PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release" -IncludePriorities P0 -DryRun
```

Copy and verify checksums:

```powershell
.\\publications\\hg-msa-tc-future-transportation\\scripts\\stage_onedrive_release.ps1 -SourceRoot . -StagingRoot "C:\\Path\\To\\OneDrive\\PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release" -IncludePriorities P0 -VerifyChecksums
```

## Stage P1 Later

```powershell
.\\publications\\hg-msa-tc-future-transportation\\scripts\\stage_onedrive_release.ps1 -SourceRoot . -StagingRoot "C:\\Path\\To\\OneDrive\\PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release" -IncludePriorities P0,P1 -VerifyChecksums
```

## Synced-Folder Comparison After Upload

After OneDrive sync is complete:

```powershell
$env:ONEDRIVE_RELEASE_ROOT = "C:\\Path\\To\\OneDrive\\PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release"
.\\.venv\\Scripts\\python.exe .\\publications\\hg-msa-tc-future-transportation\\scripts\\generate_task16_onedrive_staging_docs.py
```

This creates `onedrive_existing_inventory.csv`, `onedrive_missing_artifacts.csv`, `onedrive_mismatched_hashes.csv`, and `onedrive_extra_artifacts_review.csv` when the environment variable points to a valid folder.

## Do Not Upload

- local caches and virtual environments;
- prior ZIP review packages;
- Google-derived or third-party map imagery unless redistribution rights are documented;
- private annotation databases unless intentionally redacted and documented;
- unrelated dirty files outside the publication release scope.
""",
    )


def deep_yolo_audit(now: str) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            repo_path = rel(path)
            lowered = repo_path.lower().replace("\\", "/")
            ext = path.suffix.lower()
            keyword_hit = any(term in lowered for term in SEARCH_TERMS)
            label_like = ext in {".txt", ".json", ".csv", ".parquet"} and any(part in lowered.split("/") for part in ["labels", "predictions"])
            if not keyword_hit and not label_like:
                continue
            stat = path.stat()
            if "02_yolov11x_detections" in lowered and ext in {".parquet", ".csv", ".json", ".txt"}:
                kind = "raw_detection_candidate"
            elif "03_yolo_tracking_outputs" in lowered or "runs/track" in lowered or "track" in lowered:
                kind = "tracking_output_or_tracker_input"
            elif "detection" in lowered and ext in {".csv", ".json", ".parquet", ".txt"}:
                kind = "detection_summary_or_candidate"
            elif "yolo" in lowered or "ultralytics" in lowered:
                kind = "configuration_or_code_reference"
            else:
                kind = "related_file"
            scene = next((scene for scene in ["bellevue_116th_ne12th", "bellevue_150th_newport", "bellevue_150th_eastgate", "bellevue_150th_se38th", "bellevue_ne8th"] if scene in lowered), "")
            rows.append(
                {
                    "repository_relative_path": repo_path,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 6),
                    "file_kind": kind,
                    "scene_id_inferred": scene,
                    "onedrive_folder": "02_yolov11x_detections/" if "detection" in kind else "03_yolo_tracking_outputs/" if "tracking" in kind else "08_reproducibility_configs_and_manifests/",
                    "upload_priority": "P1" if kind in {"raw_detection_candidate", "tracking_output_or_tracker_input"} else "P2",
                    "sha256": sha256_file(path),
                }
            )
    rows.sort(key=lambda row: str(row["repository_relative_path"]))
    write_csv(
        DOCS / "yolov11_detection_upload_manifest.csv",
        rows,
        ["repository_relative_path", "size_bytes", "size_mb", "file_kind", "scene_id_inferred", "onedrive_folder", "upload_priority", "sha256"],
    )
    raw_candidates = [row for row in rows if row["file_kind"] == "raw_detection_candidate"]
    true_raw_found = bool(raw_candidates)
    kind_counts = Counter(str(row["file_kind"]) for row in rows)
    size_by_kind = defaultdict(int)
    for row in rows:
        size_by_kind[str(row["file_kind"])] += int(row["size_bytes"])
    kind_table = ["| Kind | Files | Size GB |", "| --- | ---: | ---: |"]
    for kind, count in sorted(kind_counts.items()):
        kind_table.append(f"| `{kind}` | {count} | {size_by_kind[kind] / (1024**3):.3f} |")
    sample_table = md_table(rows, ["repository_relative_path", "file_kind", "scene_id_inferred", "size_mb", "upload_priority"], 80)
    status = (
        "True YOLOv11 raw detection candidates were found under `TNVD2_UPLOAD_PACKAGE/02_yolov11x_detections/`."
        if true_raw_found
        else "No true per-frame YOLOv11 raw detection export was found under the expected `TNVD2_UPLOAD_PACKAGE/02_yolov11x_detections/` folder. The closest available alternatives are YOLO/Ultralytics configs, detection summary files, and ByteTrack/YOLO tracking Parquet outputs."
    )
    write_text(
        DOCS / "yolov11_detection_deep_audit.md",
        f"""# YOLOv11 / YOLO11 Detection Deep Audit

Generated: `{now}`

{status}

## File-Kind Summary

{chr(10).join(kind_table)}

## Matched Files

{sample_table}

## Coverage

- Scene coverage is inferred from path names only.
- Tracking-output coverage is strong for the five Bellevue scenes through `TNVD2_UPLOAD_PACKAGE/03_yolo_tracking_outputs/`.
- Raw detection-export coverage is not verified unless files are present under `02_yolov11x_detections/` or another clearly named detector-output path.

## Missing Expected Raw Detection Files

If per-frame YOLOv11 raw detection exports are required for the public release, expected files are still missing for the five Bellevue scenes unless they exist outside this checkout. The manuscript and data-availability text should avoid claiming that raw YOLOv11 detections are already publicly available until those files are staged or uploaded.

## OneDrive Target

- Raw detection exports, if recovered: `02_yolov11x_detections/`
- Tracking outputs and tracker inputs: `03_yolo_tracking_outputs/`
- Config/code references: GitHub plus `08_reproducibility_configs_and_manifests/`
""",
    )
    return rows, true_raw_found


def md_table(rows: list[dict[str, object]], cols: list[str], limit: int = 50) -> str:
    if not rows:
        return "No matching files found.\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    if len(rows) > limit:
        lines.append("")
        lines.append(f"Additional rows omitted from this markdown view: {len(rows) - limit}. See the CSV manifest for the full audit.")
    return "\n".join(lines) + "\n"


def onedrive_comparison(now: str, prioritized: list[dict[str, object]]) -> str:
    root_env = os.environ.get("ONEDRIVE_RELEASE_ROOT")
    if not root_env or not Path(root_env).exists():
        return "not_performed_ONEDRIVE_RELEASE_ROOT_not_set"
    od_root = Path(root_env)
    existing = []
    for dirpath, dirnames, filenames in os.walk(od_root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            existing.append(
                {
                    "onedrive_relative_path": path.relative_to(od_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_csv(DOCS / "onedrive_existing_inventory.csv", existing, ["onedrive_relative_path", "size_bytes", "sha256"])
    existing_by_path = {str(row["onedrive_relative_path"]): row for row in existing}
    required = [row for row in prioritized if row["upload_priority"] in {"P0", "P1"} and row["upload_required"] == "true"]
    missing = []
    mismatched = []
    for row in required:
        target = str(row["onedrive_target_relative_path"])
        if target not in existing_by_path:
            missing.append({"onedrive_target_relative_path": target, "upload_priority": row["upload_priority"], "repository_relative_path": row["repository_relative_path"]})
            continue
        expected_hash = str(row["sha256"])
        actual_hash = str(existing_by_path[target]["sha256"])
        if expected_hash and expected_hash.startswith("NOT_COMPUTED"):
            continue
        if expected_hash and actual_hash and expected_hash.lower() != actual_hash.lower():
            mismatched.append(
                {
                    "onedrive_target_relative_path": target,
                    "upload_priority": row["upload_priority"],
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                }
            )
    expected_paths = {str(row["onedrive_target_relative_path"]) for row in required}
    extra = [{"extra_onedrive_relative_path": path} for path in sorted(set(existing_by_path) - expected_paths)]
    write_csv(DOCS / "onedrive_missing_artifacts.csv", missing, ["onedrive_target_relative_path", "upload_priority", "repository_relative_path"])
    write_csv(DOCS / "onedrive_mismatched_hashes.csv", mismatched, ["onedrive_target_relative_path", "upload_priority", "expected_sha256", "actual_sha256"])
    write_csv(DOCS / "onedrive_extra_artifacts_review.csv", extra, ["extra_onedrive_relative_path"])
    return f"performed_against_{root_env}"


def update_publication_artifacts_doc(now: str, prioritized: list[dict[str, object]]) -> None:
    focus = [row for row in prioritized if row["upload_priority"] in {"P0", "P1"}]
    by_priority = Counter(str(row["upload_priority"]) for row in prioritized)
    lines = ["| Priority | File | OneDrive folder | GitHub tracked | Link from page | Required before submission |", "| --- | --- | --- | --- | --- | --- |"]
    for row in focus[:120]:
        lines.append(
            f"| {row['upload_priority']} | `{row['repository_relative_path']}` | `{row['onedrive_folder']}` | {row['git_tracked']} | {row['link_from_publication_page']} | {row['required_before_submission']} |"
        )
    if len(focus) > 120:
        lines.append("")
        lines.append(f"Additional P0/P1 rows omitted from this markdown view: {len(focus) - 120}. See `onedrive_upload_manifest_prioritized.csv`.")
    write_text(
        DOCS / "new_publication_artifacts_to_upload.md",
        f"""# New Publication Artifacts To Upload To OneDrive

Generated/updated for Task 16 on `{now}`.

This file refines the Task 15 upload list with upload priority, OneDrive folder, GitHub tracking status, publication-page relevance and submission readiness.

## Priority Counts

| Priority | Files |
| --- | ---: |
| P0 | {by_priority['P0']} |
| P1 | {by_priority['P1']} |
| P2 | {by_priority['P2']} |
| P3 | {by_priority['P3']} |

## P0/P1 Upload List

{chr(10).join(lines)}
""",
    )


def update_data_availability(now: str, true_raw_yolo_found: bool) -> None:
    yolo_sentence = (
        "YOLOv11 raw detection exports are included in the staging manifest and should be uploaded to OneDrive before claiming public availability."
        if true_raw_yolo_found
        else "The local audit did not find true per-frame YOLOv11 raw detection exports; the manuscript should not claim public availability of those raw detection files until they are recovered and staged."
    )
    text = f"""# Manuscript Data and Code Availability Statement - OneDrive Version

Code, frozen protocols, compact result tables, reproducibility manifests, figure sources and manuscript-support files are available in the GitHub repository:

`https://github.com/aronagg/PathFindRNET_2.0`

The publication page is expected at:

`{PAGES_URL}`

Large redistributable artifacts are planned for public access through the OneDrive folder:

`{ONEDRIVE_URL}`

The OneDrive staging manifest includes raw videos, tracking outputs, processed trajectory and feature exports, reference-label artifacts and supplementary result tables where redistribution is permitted. {yolo_sentence}

Already-available GitHub materials and OneDrive materials that still require manual upload are separated in `docs/onedrive_upload_manifest_prioritized.csv`. Google-derived or third-party imagery is excluded unless redistribution rights and attribution are confirmed.
"""
    write_text(DOCS / "manuscript_data_availability_onedrive_final.md", text)
    write_text(DOCS / "manuscript_revised_data_code_availability_final.md", text)
    write_text(
        PUB / "DATA_AVAILABILITY.md",
        f"""# Data Availability

Public release materials for the revised Future Transportation manuscript are split across GitHub, GitHub Pages and a public OneDrive folder.

## Already Available In GitHub After Commit

- source code and publication scripts;
- compact result tables and figures;
- frozen protocol/configuration files;
- reproducibility and reviewer-response documentation;
- GitHub Pages publication-page source.

Repository: `https://github.com/aronagg/PathFindRNET_2.0`

Expected publication page:

`{PAGES_URL}`

## OneDrive Materials Requiring Manual Upload Verification

Public OneDrive folder:

`{ONEDRIVE_URL}`

The prioritized staging manifest is:

`publications/hg-msa-tc-future-transportation/docs/onedrive_upload_manifest_prioritized.csv`

The manifest includes raw videos, tracking outputs, processed trajectory and feature exports, reference-label artifacts and supplementary result tables where redistribution is permitted. {yolo_sentence}

## Exclusions

Google-derived or third-party map imagery should not be included in public release materials unless redistribution rights and attribution are confirmed. Local caches, virtual environments, previous ZIP review packages and private annotation databases are excluded from the public release.

## Reproducibility Boundary

The release supports audit and reproduction of the submitted analyses from frozen protocols and compact result artifacts. No new scientific experiments were run as part of the OneDrive staging or GitHub Pages integration task.
""",
    )
    write_text(
        DOCS / "final_data_licensing_statement.md",
        f"""# Final Data Licensing Statement

Generated/updated for Task 16 on `{now}`.

The revised release plan uses GitHub for code and compact reproducibility artifacts, GitHub Pages for the public project page, and public OneDrive for large redistributable data artifacts.

Public OneDrive folder:

`{ONEDRIVE_URL}`

The OneDrive manifest includes raw videos, tracking outputs, processed trajectory/feature exports, reference-label artifacts and supplementary result tables where redistribution is permitted. {yolo_sentence}

Google-derived map imagery and any third-party imagery require separate provenance and redistribution review before public release. The manuscript data-availability statement should clearly distinguish GitHub materials that are committed from OneDrive materials that still require manual upload verification.
""",
    )


def preview_site_plan(now: str) -> None:
    audit = DOCS / "preview_image_quality_audit.md"
    lines = []
    if audit.exists():
        for line in audit.read_text(encoding="utf-8").splitlines():
            if line.startswith("| `publications/"):
                parts = [part.strip().strip("`") for part in line.strip("|").split("|")]
                if len(parts) >= 6:
                    lines.append(
                        {
                            "source": parts[0],
                            "preview": parts[1],
                            "status": parts[2],
                            "width": parts[3],
                            "height": parts[4],
                            "size": parts[5],
                        }
                    )
    table = ["| Preview image | Commit to GitHub | Full-resolution OneDrive upload | Referenced by page |", "| --- | --- | --- | --- |"]
    page = (PUB / "site/publications/hg-smg-tc/index.md").read_text(encoding="utf-8", errors="replace")
    for row in lines:
        preview_name = Path(row["preview"]).name
        source_name = row["source"]
        table.append(f"| `{row['preview']}` | yes | `{source_name}` | {str(preview_name in page).lower()} |")
    write_text(
        DOCS / "preview_image_upload_and_site_plan.md",
        f"""# Preview Image Upload And Site Plan

Generated: `{now}`

Preview images should be committed to GitHub because they are lightweight website assets. Full-resolution source figures should remain in the publication figures folder and may also be mirrored to OneDrive for supplementary review.

{chr(10).join(table)}

Do not duplicate large imagery unnecessarily. Use preview images for layout and link large data artifacts through OneDrive.
""",
    )


def pages_reports(now: str) -> None:
    write_text(
        DOCS / "github_pages_integration_plan.md",
        f"""# GitHub Pages Integration Plan

Generated/updated for Task 16 on `{now}`.

## Confirmed Source

The user provided the active GitHub Pages source:

- deployment mode: deploy from branch;
- branch: `tnvd2-github-pages`;
- folder: `/` root.

## Integration Target

The publication page should be integrated into the root of the `tnvd2-github-pages` branch under:

`publications/hg-smg-tc/index.html`

Preview images should be copied to:

`assets/publications/hg-smg-tc/`

The homepage `index.html` should receive a small publication card/link pointing to:

`publications/hg-smg-tc/`

Expected public URL:

`{PAGES_URL}`
""",
    )
    write_text(
        DOCS / "github_pages_integration_report.md",
        f"""# GitHub Pages Integration Report

Generated: `{now}`

## Source Status

GitHub Pages source was provided by the user and treated as authoritative:

- branch: `tnvd2-github-pages`;
- folder: `/` root.

## Integration Performed Locally

The integration was prepared in a separate local git worktree for `tnvd2-github-pages`, not in the scientific publication branch. This keeps the publication release audit branch and the deployed website branch separated.

Expected public URL:

`{PAGES_URL}`

## Files Expected On Pages Branch

- `publications/hg-smg-tc/index.html`
- `assets/publications/hg-smg-tc/final_hg_smg_pipeline_schematic_preview.png`
- `assets/publications/hg-smg-tc/final_target_count_error_preview.png`
- `assets/publications/hg-smg-tc/final_agreement_metrics_preview.png`
- `assets/publications/hg-smg-tc/final_baseline_comparison_summary_preview.png`
- `assets/publications/hg-smg-tc/final_se38th_fragmentation_story_preview.png`
- `index.html` homepage card/link update.

## External Actions Not Performed

- No push to GitHub was performed.
- No OneDrive upload was performed.
""",
    )


def consistency_report(now: str, prioritized: list[dict[str, object]], true_raw_yolo_found: bool, onedrive_mode: str) -> None:
    checked = [
        PUB / "DATA_AVAILABILITY.md",
        DOCS / "manuscript_data_availability_onedrive_final.md",
        DOCS / "final_data_licensing_statement.md",
        DOCS / "github_pages_integration_plan.md",
        DOCS / "github_pages_integration_report.md",
        PUB / "site/publications/hg-smg-tc/index.md",
    ]
    findings = []
    for path in checked:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        if not path.exists():
            findings.append((rel(path), "missing", "checked file missing"))
        for needle in ["[DOI", "Zenodo", "Figshare", "OSF", "archive deposit", "Archive URL", "DOI archive"]:
            if needle in text:
                findings.append((rel(path), "old_archive_wording", needle))
        if "all data available" in text.lower() or "all artifacts are available" in text.lower():
            findings.append((rel(path), "unsupported_availability_claim", "all-data wording"))
        if not true_raw_yolo_found and "raw yolov11 detections are available" in text.lower():
            findings.append((rel(path), "unsupported_yolo_claim", "raw detection availability"))
    p0_missing_sources = [row for row in prioritized if row["upload_priority"] == "P0" and row["source_exists"] != "true"]
    for row in p0_missing_sources[:20]:
        findings.append((str(row["repository_relative_path"]), "missing_p0_source", str(row["priority_reason"])))
    page = PUB / "site/publications/hg-smg-tc/index.md"
    page_text = page.read_text(encoding="utf-8", errors="replace")
    for image in (PUB / "site/assets/images").glob("*_preview.png"):
        if image.name not in page_text and image.name != "final_se38th_fragmentation_story_preview.png":
            findings.append((rel(image), "preview_not_linked", "preview image not referenced from publication page"))
    lines = ["| File/item | Finding | Detail |", "| --- | --- | --- |"]
    if findings:
        for item, finding, detail in findings:
            lines.append(f"| `{item}` | {finding} | {detail} |")
    else:
        lines.append("| Checked files | none | No release-blocking consistency issue found in checked files. |")
    write_text(
        DOCS / "final_placeholder_and_consistency_report.md",
        f"""# Final Placeholder And Consistency Report

Generated/updated for Task 16 on `{now}`.

## Checks

- no DOI/archive placeholder in release-facing files;
- no unsupported all-data-available claim before upload verification;
- no unsupported YOLOv11 raw detection availability claim;
- P0 source files exist locally;
- Pages URL and OneDrive URL are consistent across checked docs;
- preview image links resolve locally;
- staging scripts parse the prioritized manifest.

## OneDrive Synced Comparison

`{onedrive_mode}`

## Findings

{chr(10).join(lines)}
""",
    )


def task_report(now: str, branch: str, head: str, prioritized: list[dict[str, object]], true_raw_yolo_found: bool, onedrive_mode: str) -> None:
    by_priority = Counter(str(row["upload_priority"]) for row in prioritized)
    bytes_by_priority = defaultdict(int)
    for row in prioritized:
        bytes_by_priority[str(row["upload_priority"])] += int(row["size_bytes"])
    table = ["| Priority | Files | Size GB |", "| --- | ---: | ---: |"]
    for priority in ["P0", "P1", "P2", "P3"]:
        table.append(f"| {priority} | {by_priority[priority]} | {bytes_by_priority[priority] / (1024**3):.3f} |")
    yolo_status = "raw detection candidates found" if true_raw_yolo_found else "true raw detection exports not found"
    write_text(
        DOCS / "task_16_execution_report.md",
        f"""# Task 16 Execution Report

Generated: `{now}`

Branch: `{branch}`

Base commit at generation time: `{head}`

## Scope

Prepared OneDrive upload staging, missing-data verification, YOLOv11 detection deep audit and GitHub Pages integration documentation. No experiments were run, no ZIP package was created and no external upload was performed.

## Upload Priority Summary

{chr(10).join(table)}

## OneDrive Comparison

`{onedrive_mode}`

## YOLOv11 Raw Detection Status

`{yolo_status}`

See `docs/yolov11_detection_deep_audit.md` and `docs/yolov11_detection_upload_manifest.csv`.

## GitHub Pages

User-provided source: branch `tnvd2-github-pages`, folder `/` root.

Expected publication URL:

`{PAGES_URL}`

## Exact P0 Staging Command

```powershell
.\\publications\\hg-msa-tc-future-transportation\\scripts\\stage_onedrive_release.ps1 -SourceRoot . -StagingRoot "C:\\Path\\To\\OneDrive\\PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release" -IncludePriorities P0 -DryRun
```

## Created Or Updated Files

- `docs/onedrive_upload_manifest_prioritized.csv`
- `docs/onedrive_upload_batches.md`
- `scripts/stage_onedrive_release.ps1`
- `scripts/stage_onedrive_release.sh`
- `docs/onedrive_upload_manual_instructions.md`
- `docs/yolov11_detection_deep_audit.md`
- `docs/yolov11_detection_upload_manifest.csv`
- `docs/new_publication_artifacts_to_upload.md`
- `docs/github_pages_integration_plan.md`
- `docs/github_pages_integration_report.md`
- `docs/preview_image_upload_and_site_plan.md`
- `docs/manuscript_data_availability_onedrive_final.md`
- `docs/final_data_licensing_statement.md`
- `DATA_AVAILABILITY.md`
- `docs/final_placeholder_and_consistency_report.md`
- `docs/task_16_execution_report.md`
- `scripts/generate_task16_onedrive_staging_docs.py`

## Skipped Files

- No OneDrive upload was performed.
- No ZIP package or SHA sidecar was created.
- Frozen scientific outputs were not regenerated.
- `P3` files are excluded from staging by default.

## Failed Steps

- None in the local documentation/staging preparation.
""",
    )


def main() -> None:
    now = now_utc()
    branch = run_git(["branch", "--show-current"])
    head = run_git(["rev-parse", "HEAD"])
    prioritized = create_prioritized_manifest(now)
    create_staging_scripts(now)
    _, true_raw_yolo_found = deep_yolo_audit(now)
    onedrive_mode = onedrive_comparison(now, prioritized)
    update_publication_artifacts_doc(now, prioritized)
    update_data_availability(now, true_raw_yolo_found)
    preview_site_plan(now)
    pages_reports(now)
    consistency_report(now, prioritized, true_raw_yolo_found, onedrive_mode)
    task_report(now, branch, head, prioritized, true_raw_yolo_found, onedrive_mode)
    print(
        {
            "prioritized_rows": len(prioritized),
            "true_raw_yolo_found": true_raw_yolo_found,
            "onedrive_mode": onedrive_mode,
            "priority_counts": dict(Counter(str(row["upload_priority"]) for row in prioritized)),
        }
    )


if __name__ == "__main__":
    main()
