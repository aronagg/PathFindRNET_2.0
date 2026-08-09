from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PUB = ROOT / "publications" / "hg-msa-tc-future-transportation"
DOCS = PUB / "docs"
ONEDRIVE_ROOT = Path(r"C:\Users\aggko\OneDrive\aggaron\Education\PhD\Traffic_Node_Video_Dataset_2_0")
ONEDRIVE_URL = (
    "https://onedrive.live.com/?redeem="
    "aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84MGZhYWQ2ZmVhMDViMDhjL0lnQlBxblBYemRWd1I1NERrbHVfUzB0bkFUMWJFUlBzSk1XTHM3TVJ3ZDNPZFRJP2U9b2NpOXBM"
    "&id=80FAAD6FEA05B08C%21sd773aa4fd5cd47709e03925bbf4b4b67&cid=80FAAD6FEA05B08C"
)
PAGES_URL = "https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/"
HASH_LIMIT = 256 * 1024 * 1024
SCAN_ROOTS = [
    Path("data"),
    Path("TNVD2_UPLOAD_PACKAGE"),
    Path("Traffic_Node_Video_Dataset_2_0_2026"),
    Path("traffic_node_dataset_2_0_starter_pack"),
    Path("publications/hg-msa-tc-future-transportation"),
]
EXCLUDE_DIRS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "review_packages"}
EXCLUDE_EXTS = {".zip"}
AUDIT_EXTS = {".csv", ".json", ".jsonl", ".txt", ".parquet", ".feather", ".pkl", ".pickle", ".npz", ".npy", ".yaml", ".yml"}
TERMS = (
    "yolo11",
    "yolov11",
    "yolo_11",
    "yolo8",
    "yolov8",
    "ultralytics",
    "detect",
    "detection",
    "detections",
    "prediction",
    "predictions",
    "labels",
    "runs/detect",
    "runs/track",
    "track",
    "tracks",
    "tracking",
    "bytetrack",
    "deepsort",
    "object",
    "bbox",
    "boxes",
    "confidence",
    "class",
    "frame",
    "trajectory",
    "trajectories",
)
SCENES = [
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
]
VOLATILE_VERIFICATION_DOCS = {
    "final_placeholder_and_consistency_report.md",
    "interim_detection_tracking_inventory.csv",
    "local_release_artifact_inventory.csv",
    "new_publication_artifacts_to_upload.md",
    "onedrive_folder_structure_proposal.md",
    "onedrive_existing_inventory.csv",
    "onedrive_missing_artifacts.csv",
    "onedrive_mismatched_hashes.csv",
    "onedrive_extra_artifacts_review.csv",
    "onedrive_staging_verification_report.md",
    "onedrive_upload_batches.md",
    "onedrive_upload_manifest.csv",
    "onedrive_upload_manifest_prioritized.csv",
    "preview_image_quality_audit.md",
    "task_17_execution_report.md",
    "yolov11_detection_deep_audit.md",
    "yolov11_detection_upload_manifest.csv",
    "yolov11_raw_detection_file_locations.md",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    size = path.stat().st_size
    if size > HASH_LIMIT:
        return f"NOT_COMPUTED_LARGE_FILE_GT_{HASH_LIMIT}_BYTES"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_tracked(path: str) -> bool:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def parquet_columns(path: Path) -> list[str]:
    if path.suffix.lower() != ".parquet":
        return []
    try:
        import pyarrow.parquet as pq

        return [str(name) for name in pq.read_schema(path).names]
    except Exception:
        return []


def infer_scene(repo_path: str) -> str:
    lowered = repo_path.lower()
    for scene in SCENES:
        if scene in lowered:
            return scene
    aliases = {
        "bellevue_116th_ne12th": ["bellevue_116th_ne12th", "116th_ne12th", "116th_ne12"],
        "bellevue_150th_newport": ["bellevue_150th_newport", "150th_newport"],
        "bellevue_150th_eastgate": ["bellevue_150th_eastgate", "150th_eastgate"],
        "bellevue_150th_se38th": ["bellevue_150th_se38th", "150th_se38th"],
        "bellevue_ne8th": ["bellevue_ne8th", "bellevue_ne8th", "ne8th"],
    }
    for scene, terms in aliases.items():
        if any(term in lowered for term in terms):
            return scene
    return ""


def classify_interim(repo_path: str, columns: list[str]) -> tuple[str, bool, str, str]:
    lowered = repo_path.lower().replace("\\", "/")
    name = Path(lowered).name
    cols = {col.lower() for col in columns}
    has_detector_cols = bool(cols & {"bbox", "boxes", "x1", "y1", "x2", "y2", "confidence", "conf", "class", "class_id", "frame"})
    has_track_cols = bool(cols & {"track_id", "trajectory_id"})
    if "reference_labels" in lowered or "requirements-reference-labels" in lowered:
        return "schema/metadata only", False, "P0", "05_reference_labels_and_protocols/"
    if "schema" in lowered:
        return "schema/metadata only", False, "P2", "08_reproducibility_configs_and_manifests/"
    if "detection_statistics" in lowered or "statistics" in lowered and "detect" in lowered:
        return "detector summary/statistics", False, "P2", "02_yolov11x_detections/"
    if name.startswith("labels_") or "cluster_id" in cols:
        return "trajectory/intermediate output", False, "P2", "04_processed_trajectories_and_features/"
    if "trajectory" in lowered or "trajectories" in lowered:
        if "processed" in lowered:
            return "processed trajectory", False, "P0", "04_processed_trajectories_and_features/"
        return "trajectory/intermediate output", False, "P1", "04_processed_trajectories_and_features/"
    if "track" in lowered or "bytetrack" in lowered or has_track_cols:
        return "tracker output", False, "P1", "03_yolo_tracking_outputs/"
    raw_label_path = "/labels/" in lowered and ("yolo" in lowered or "runs/" in lowered)
    if ("detections" in lowered or "/detect" in lowered or "prediction" in lowered or raw_label_path) and not any(
        token in lowered for token in ["track", "trajectory", "trajectories"]
    ):
        return "true raw per-frame detector output", True, "P1", "02_yolov11x_detections/"
    if has_detector_cols and not has_track_cols and "track" not in lowered:
        return "true raw per-frame detector output", True, "P1", "02_yolov11x_detections/"
    if "yolo" in lowered or "ultralytics" in lowered:
        return "schema/metadata only", False, "P2", "08_reproducibility_configs_and_manifests/"
    return "unclear", False, "P2", "99_misc_review/"


def should_audit(repo_path: str, path: Path) -> bool:
    lowered = repo_path.lower().replace("\\", "/")
    if path.suffix.lower() not in AUDIT_EXTS:
        return False
    if lowered.startswith("data/interim/"):
        return True
    return any(term in lowered for term in TERMS)


def scan_interim_and_detection_outputs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for scan_root in SCAN_ROOTS:
        base = ROOT / scan_root
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() in EXCLUDE_EXTS:
                    continue
                repo_path = rel(path)
                if repo_path in seen or not should_audit(repo_path, path):
                    continue
                seen.add(repo_path)
                cols = parquet_columns(path)
                artifact_type, raw_yolo, priority, folder = classify_interim(repo_path, cols)
                stat = path.stat()
                rows.append(
                    {
                        "repository_relative_path": repo_path,
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 6),
                        "scene_id_inferred": infer_scene(repo_path),
                        "detected_file_type": artifact_type,
                        "likely_raw_yolov11_detection_output": str(raw_yolo).lower(),
                        "should_upload_to_onedrive": str(priority != "P3").lower(),
                        "recommended_priority": priority,
                        "recommended_onedrive_folder": folder,
                        "parquet_columns_sample": ";".join(cols[:20]),
                        "sha256": sha256_file(path),
                    }
                )
    rows.sort(key=lambda row: str(row["repository_relative_path"]))
    return rows


def category_for(repo_path: str) -> str:
    lowered = repo_path.lower().replace("\\", "/")
    if lowered.startswith("data/raw/") or "/01_original_videos/" in lowered:
        return "raw_video"
    if lowered.startswith("data/interim/") or "/03_yolo_tracking_outputs/" in lowered or "track" in lowered:
        return "tracking_output"
    if "detection" in lowered or "detections" in lowered or "yolo" in lowered:
        return "yolov11_detection_or_detection_statistics"
    if "data/processed" in lowered or "features" in lowered or "trajectory" in lowered:
        return "processed_trajectory_or_feature"
    if "/annotations/" in lowered or "reference_label" in lowered or "polygon" in lowered:
        return "reference_label_or_annotation_protocol"
    if "/figures/" in lowered or "/site/assets/images/" in lowered or lowered.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        return "figure_or_website_asset"
    if "/results/" in lowered or "/docs/" in lowered:
        return "publication_result_or_documentation"
    if "/configs/" in lowered or "protocol" in lowered or "manifest" in lowered:
        return "configuration_or_manifest"
    return "other"


def folder_for(category: str, repo_path: str) -> str:
    lowered = repo_path.lower().replace("\\", "/")
    if lowered.startswith("data/interim/"):
        return "03_yolo_tracking_outputs/"
    return {
        "raw_video": "01_original_videos/",
        "yolov11_detection_or_detection_statistics": "02_yolov11x_detections/",
        "tracking_output": "03_yolo_tracking_outputs/",
        "processed_trajectory_or_feature": "04_processed_trajectories_and_features/",
        "reference_label_or_annotation_protocol": "05_reference_labels_and_protocols/",
        "publication_result_or_documentation": "06_publication_results_and_docs/",
        "figure_or_website_asset": "07_figures_and_website_assets/",
        "configuration_or_manifest": "08_reproducibility_configs_and_manifests/",
    }.get(category, "99_misc_review/")


def action_for(category: str, repo_path: str) -> str:
    lowered = repo_path.lower()
    if category == "raw_video":
        return "onedrive_large_data_only; exclude from git package"
    if "google" in lowered or "maps" in lowered or "satellite" in lowered:
        return "review_provenance_before_public_redistribution"
    if category in {"tracking_output", "processed_trajectory_or_feature", "yolov11_detection_or_detection_statistics"}:
        return "onedrive_data_release; compact summaries in github where appropriate"
    if category in {"publication_result_or_documentation", "figure_or_website_asset", "configuration_or_manifest", "reference_label_or_annotation_protocol"}:
        return "github_and_onedrive"
    return "review_before_release"


def scan_release_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scan_root in SCAN_ROOTS:
        base = ROOT / scan_root
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() in EXCLUDE_EXTS:
                    continue
                repo_path = rel(path)
                category = category_for(repo_path)
                stat = path.stat()
                rows.append(
                    {
                        "repository_relative_path": repo_path,
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 6),
                        "extension": path.suffix.lower(),
                        "category": category,
                        "release_action": action_for(category, repo_path),
                        "onedrive_folder": folder_for(category, repo_path),
                        "sha256": sha256_file(path),
                        "last_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                        .replace(microsecond=0)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    }
                )
    rows.sort(key=lambda row: str(row["repository_relative_path"]))
    return rows


def target_relative_path(row: dict[str, object]) -> str:
    source = str(row["repository_relative_path"])
    folder = str(row["onedrive_folder"])
    return (Path(folder) / source).as_posix()


def priority_for(row: dict[str, object], audit_by_path: dict[str, dict[str, object]]) -> tuple[str, str]:
    path = str(row["repository_relative_path"])
    lowered = path.lower().replace("\\", "/")
    category = str(row["category"])
    if any(token in lowered for token in [".venv/", "__pycache__", ".pytest_cache", ".ruff_cache", "review_packages/", ".zip"]):
        return "P3", "cache, review package, prior archive, or temporary output"
    if "google" in lowered or "maps" in lowered or "satellite" in lowered:
        return "P3", "third-party imagery/provenance requires manual review"
    if Path(lowered).name in VOLATILE_VERIFICATION_DOCS:
        return "P2", "dynamic staging verification/report artifact; keep in GitHub, not required in P0 OneDrive staging"
    audit = audit_by_path.get(path)
    if audit:
        return str(audit["recommended_priority"]), f"Task 17 interim/detection audit: {audit['detected_file_type']}"
    if lowered.startswith("publications/hg-msa-tc-future-transportation/site/"):
        return "P0", "GitHub Pages source or preview asset"
    if lowered.startswith("publications/hg-msa-tc-future-transportation/docs/"):
        return "P0", "final manuscript/reviewer/release support documentation"
    if lowered.startswith("publications/hg-msa-tc-future-transportation/scripts/"):
        return "P0", "release/reproducibility script"
    if lowered.startswith("publications/hg-msa-tc-future-transportation/figures/final_synthesis/"):
        return "P0", "final manuscript figure"
    if lowered.startswith("publications/hg-msa-tc-future-transportation/results/final_synthesis/"):
        return "P0", "final manuscript result table"
    if "reference_labels" in lowered or "/annotations/" in lowered:
        return "P0", "polygon-rule reference label or protocol"
    if "split" in lowered and ("manifest" in lowered or "reference" in lowered or "protocol" in lowered):
        return "P0", "split/reference protocol"
    if "homography" in lowered and any(token in lowered for token in ["correspondence", "matrix", "quality", "calibration", "residual"]):
        return "P0", "homography numeric/provenance artifact"
    if lowered.startswith("data/processed/") or "features_trimmed" in lowered:
        return "P0", "processed trajectory/feature artifact needed for reported-result reproduction"
    if category == "raw_video":
        return "P1", "raw video for full technical reproducibility"
    if category == "tracking_output":
        return "P1", "tracking/intermediate output for trajectory rebuild"
    if category == "yolov11_detection_or_detection_statistics":
        return "P1", "detector-related artifact for full technical reproducibility or detector audit"
    if "diagnostic" in lowered or "development" in lowered or "debug" in lowered:
        return "P2", "supplementary diagnostic/development artifact"
    if category in {"publication_result_or_documentation", "configuration_or_manifest", "figure_or_website_asset"}:
        return "P0", "compact publication/release artifact"
    return "P2", "supplementary or unclear public-release artifact"


def write_manifest_outputs(inventory: list[dict[str, object]], audit_rows: list[dict[str, object]], now: str) -> list[dict[str, object]]:
    write_csv(DOCS / "local_release_artifact_inventory.csv", inventory, list(inventory[0].keys()))
    upload_rows = []
    for row in inventory:
        upload_required = str(row["release_action"]).startswith("onedrive") or str(row["release_action"]) == "github_and_onedrive"
        upload_rows.append(
            {
                "repository_relative_path": row["repository_relative_path"],
                "onedrive_folder": row["onedrive_folder"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "upload_required": str(upload_required).lower(),
                "priority": "high" if upload_required else "review",
                "public_release_note": row["release_action"],
            }
        )
    write_csv(DOCS / "onedrive_upload_manifest.csv", upload_rows, list(upload_rows[0].keys()))

    audit_by_path = {str(row["repository_relative_path"]): row for row in audit_rows}
    prioritized = []
    for row in inventory:
        priority, reason = priority_for(row, audit_by_path)
        upload_required = priority != "P3"
        prioritized.append(
            {
                "repository_relative_path": row["repository_relative_path"],
                "onedrive_folder": row["onedrive_folder"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "upload_required": str(upload_required).lower(),
                "priority": "high" if priority in {"P0", "P1"} else "medium" if priority == "P2" else "review",
                "public_release_note": row["release_action"],
                "upload_priority": priority,
                "priority_reason": reason,
                "onedrive_target_relative_path": target_relative_path(row),
                "source_exists": str((ROOT / str(row["repository_relative_path"])).exists()).lower(),
                "git_tracked": str(git_tracked(str(row["repository_relative_path"]))).lower(),
                "link_from_publication_page": str(
                    str(row["repository_relative_path"]).startswith("publications/hg-msa-tc-future-transportation/site/assets/images/")
                    or str(row["repository_relative_path"]).startswith("publications/hg-msa-tc-future-transportation/figures/final_synthesis/")
                ).lower(),
                "required_before_submission": str(priority == "P0").lower(),
            }
        )
    write_csv(DOCS / "onedrive_upload_manifest_prioritized.csv", prioritized, list(prioritized[0].keys()))

    write_text(DOCS / "onedrive_upload_batches.md", batch_report(prioritized, now))
    write_text(DOCS / "new_publication_artifacts_to_upload.md", publication_artifact_report(prioritized, now))
    write_text(DOCS / "onedrive_folder_structure_proposal.md", folder_structure_report(inventory, prioritized, now))
    return prioritized


def priority_summary(rows: list[dict[str, object]]) -> tuple[Counter[str], dict[str, int]]:
    counts: Counter[str] = Counter()
    sizes: dict[str, int] = defaultdict(int)
    for row in rows:
        priority = str(row["upload_priority"])
        counts[priority] += 1
        sizes[priority] += int(row["size_bytes"])
    return counts, sizes


def batch_report(rows: list[dict[str, object]], now: str) -> str:
    counts, sizes = priority_summary(rows)
    lines = ["| Priority | Files | Size GB | Upload order |", "| --- | ---: | ---: | ---: |"]
    for order, priority in enumerate(["P0", "P1", "P2", "P3"], start=1):
        lines.append(f"| {priority} | {counts[priority]} | {sizes[priority] / (1024**3):.3f} | {order if priority != 'P3' else 'do not upload'} |")
    by_folder: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_folder[str(row["upload_priority"])][str(row["onedrive_folder"])] += 1
    folder_lines = ["| Priority | Target folder | Files |", "| --- | --- | ---: |"]
    for priority in ["P0", "P1", "P2", "P3"]:
        for folder, count in by_folder[priority].most_common():
            folder_lines.append(f"| {priority} | `{folder}` | {count} |")
    return f"""# OneDrive Upload Batches

Generated/updated for Task 17 on `{now}`.

## Priority Summary

{chr(10).join(lines)}

## Folder Summary

{chr(10).join(folder_lines)}

## Upload Order

1. Stage and verify `P0` only before manuscript submission.
2. Stage `P1` later only after explicit approval, because it contains raw video, tracking/intermediate artifacts and full rebuild inputs.
3. Review `P2` selectively.
4. Keep `P3` out of the public OneDrive release unless manually reclassified.
"""


def publication_artifact_report(rows: list[dict[str, object]], now: str) -> str:
    counts, _ = priority_summary(rows)
    selected = [row for row in rows if row["upload_priority"] in {"P0", "P1"}]
    table = ["| Priority | File | OneDrive folder | GitHub tracked | Link from page | Required before submission |", "| --- | --- | --- | --- | --- | --- |"]
    for row in selected[:150]:
        table.append(
            f"| {row['upload_priority']} | `{row['repository_relative_path']}` | `{row['onedrive_folder']}` | {row['git_tracked']} | {row['link_from_publication_page']} | {row['required_before_submission']} |"
        )
    if len(selected) > 150:
        table.append("")
        table.append(f"Additional P0/P1 rows omitted from this markdown view: {len(selected) - 150}. See `onedrive_upload_manifest_prioritized.csv`.")
    return f"""# New Publication Artifacts To Upload To OneDrive

Generated/updated for Task 17 on `{now}`.

## Priority Counts

| Priority | Files |
| --- | ---: |
| P0 | {counts['P0']} |
| P1 | {counts['P1']} |
| P2 | {counts['P2']} |
| P3 | {counts['P3']} |

## P0/P1 Upload List

{chr(10).join(table)}
"""


def folder_structure_report(inventory: list[dict[str, object]], prioritized: list[dict[str, object]], now: str) -> str:
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for row in inventory:
        item = by_category[str(row["category"])]
        item["files"] += 1
        item["bytes"] += int(row["size_bytes"])
    cat_lines = ["| Category | Files | Size GB |", "| --- | ---: | ---: |"]
    for category in sorted(by_category):
        item = by_category[category]
        cat_lines.append(f"| `{category}` | {item['files']} | {item['bytes'] / (1024**3):.3f} |")
    counts, sizes = priority_summary(prioritized)
    return f"""# Public OneDrive Folder Structure Proposal

Generated/updated for Task 17 on `{now}`.

Public OneDrive folder:

`{ONEDRIVE_URL}`

Local staging root:

`{ONEDRIVE_ROOT}`

## Proposed Structure

```text
Traffic_Node_Video_Dataset_2_0/
  00_RELEASE_MANIFESTS/
  01_original_videos/
  02_yolov11x_detections/
  03_yolo_tracking_outputs/
  04_processed_trajectories_and_features/
  05_reference_labels_and_protocols/
  06_publication_results_and_docs/
  07_figures_and_website_assets/
  08_reproducibility_configs_and_manifests/
  09_licenses_and_provenance/
  99_misc_review/
```

## Inventory Summary

{chr(10).join(cat_lines)}

## Priority Summary

| Priority | Files | Size GB |
| --- | ---: | ---: |
| P0 | {counts['P0']} | {sizes['P0'] / (1024**3):.3f} |
| P1 | {counts['P1']} | {sizes['P1'] / (1024**3):.3f} |
| P2 | {counts['P2']} | {sizes['P2'] / (1024**3):.3f} |
| P3 | {counts['P3']} | {sizes['P3'] / (1024**3):.3f} |
"""


def write_audit_reports(audit_rows: list[dict[str, object]], now: str) -> bool:
    fields = [
        "repository_relative_path",
        "size_bytes",
        "size_mb",
        "scene_id_inferred",
        "detected_file_type",
        "likely_raw_yolov11_detection_output",
        "should_upload_to_onedrive",
        "recommended_priority",
        "recommended_onedrive_folder",
        "parquet_columns_sample",
        "sha256",
    ]
    write_csv(DOCS / "interim_detection_tracking_inventory.csv", audit_rows, fields)
    write_csv(DOCS / "yolov11_detection_upload_manifest.csv", audit_rows, fields)
    raw_found = any(str(row["likely_raw_yolov11_detection_output"]) == "true" for row in audit_rows)
    by_type = Counter(str(row["detected_file_type"]) for row in audit_rows)
    size_by_type: dict[str, int] = defaultdict(int)
    by_scene = Counter(str(row["scene_id_inferred"]) or "unknown" for row in audit_rows)
    for row in audit_rows:
        size_by_type[str(row["detected_file_type"])] += int(row["size_bytes"])
    type_lines = ["| Type | Files | Size GB |", "| --- | ---: | ---: |"]
    for kind, count in sorted(by_type.items()):
        type_lines.append(f"| `{kind}` | {count} | {size_by_type[kind] / (1024**3):.3f} |")
    scene_lines = ["| Scene/coverage inferred from path | Files |", "| --- | ---: |"]
    for scene, count in sorted(by_scene.items()):
        scene_lines.append(f"| `{scene}` | {count} |")
    samples = ["| Path | Type | Scene | Raw YOLOv11? | Priority | Target folder |", "| --- | --- | --- | --- | --- | --- |"]
    for row in audit_rows[:120]:
        samples.append(
            f"| `{row['repository_relative_path']}` | {row['detected_file_type']} | `{row['scene_id_inferred']}` | {row['likely_raw_yolov11_detection_output']} | {row['recommended_priority']} | `{row['recommended_onedrive_folder']}` |"
        )
    if len(audit_rows) > 120:
        samples.append("")
        samples.append(f"Additional rows omitted from this markdown view: {len(audit_rows) - 120}. See `interim_detection_tracking_inventory.csv`.")
    conclusion = (
        "True raw per-frame YOLOv11 detector outputs were found and should be staged as P1."
        if raw_found
        else "No true raw per-frame YOLOv11 detector outputs were found. The `data/interim` files are classified as tracker/intermediate outputs, primarily `tracks_*.parquet` files."
    )
    report = f"""# YOLOv11 / YOLO11 Detection Deep Audit

Generated/updated for Task 17 on `{now}`.

{conclusion}

## File Type Summary

{chr(10).join(type_lines)}

## Scene/Recording Coverage

{chr(10).join(scene_lines)}

## Matched Files

{chr(10).join(samples)}

## Interpretation

- `data/interim/` contains five-scene tracking/intermediate Parquet outputs.
- The local audit found detector statistics/schema files, but not a complete per-frame raw YOLOv11 detection export.
- Data availability wording should not claim raw YOLOv11 detections are publicly available unless those files are later recovered and staged.
- Tracking/intermediate outputs are recommended as `P1`, not `P0`, because Task 17 requested P0 staging only.
"""
    write_text(DOCS / "yolov11_detection_deep_audit.md", report)
    return raw_found


def update_stage_scripts() -> None:
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
$ManifestDir = Join-Path $StagingRoot "00_RELEASE_MANIFESTS"
if (-not $DryRun -and -not (Test-Path -LiteralPath $ManifestDir)) {
  New-Item -ItemType Directory -Path $ManifestDir -Force | Out-Null
}

function Get-FileSha256([string]$Path) {
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$rows = Import-Csv -LiteralPath $ManifestPath
$selected = $rows | Where-Object { $IncludeSet.ContainsKey($_.upload_priority) -and $_.upload_priority -ne "P3" }
$p3Selected = $rows | Where-Object { $IncludeSet.ContainsKey($_.upload_priority) -and $_.upload_priority -eq "P3" }
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
    if ($DryRun) {
      $action = "skip_existing_dry_run_no_hash"
    } else {
      $srcHash = Get-FileSha256 $src
      $dstHash = Get-FileSha256 $dst
      if ($srcHash -eq $dstHash) {
        $action = "skip_existing_same_hash"
      } else {
        $answer = Read-Host "Overwrite changed file? $dst [y/N]"
        if ($answer -match "^[Yy]") {
          $action = "overwrite_hash_differs"
        } else {
          $action = "skip_existing_hash_differs"
        }
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

$totalBytes = 0
foreach ($item in $selected) { $totalBytes += [int64]$item.size_bytes }
$copiedBytes = 0
foreach ($item in $copied) { $copiedBytes += [int64]$item.size_bytes }

if (-not $DryRun) {
  $staged | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $ManifestDir "staged_release_manifest.csv")
  $missing | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $ManifestDir "missing_source_files.csv")
  $copied | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $ManifestDir "copied_files_log.csv")
  $checksumPath = Join-Path $ManifestDir "staged_release_checksums.sha256"
  $lines = foreach ($item in $staged) {
    if ($item.staged_sha256) { "$($item.staged_sha256)  $($item.onedrive_target_relative_path)" }
  }
  $lines | Set-Content -Encoding UTF8 -LiteralPath $checksumPath
  @"
# Staging Summary

Generated: $(Get-Date -Format o)

Source root: $SourceRoot

Staging root: $StagingRoot

Included priorities: $($IncludePriorities -join ",")

Selected files: $($selected.Count)

Selected size GB: $([math]::Round($totalBytes / 1GB, 3))

Missing sources: $($missing.Count)

Copied files: $($copied.Count)

Copied size GB: $([math]::Round($copiedBytes / 1GB, 3))

P3 selected: $($p3Selected.Count)

Cloud sync status: manual OneDrive verification still required.
"@ | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $ManifestDir "staging_summary.md")
  @"
# PathFindRNET 2.0 Public Release Folder

This local OneDrive-synced folder stages public release artifacts for PathFindRNET 2.0 and the Future Transportation HG-SMG-TC manuscript package.

GitHub repository:
https://github.com/aronagg/PathFindRNET_2.0

GitHub Pages publication page:
https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/

Priority folders:

- P0: manuscript-submission release materials, compact reproducibility artifacts, final result tables, final figures, reference-label/protocol artifacts and minimal processed data needed for reported-result reproduction.
- P1: full technical reproducibility materials such as raw videos, tracking outputs and intermediate data needed to rebuild trajectories.
- P2: supplementary diagnostics.

Not redistributed unless separately reviewed:

- Google-derived or third-party map imagery;
- local caches and virtual environments;
- prior ZIP review packages;
- private annotation databases;
- unavailable raw YOLOv11 detector exports.

Citation/contact placeholders should be updated after final publication.
"@ | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $StagingRoot "README_PUBLIC_RELEASE.md")
}

Write-Host "Selected files: $($selected.Count)"
Write-Host "Selected size GB: $([math]::Round($totalBytes / 1GB, 3))"
Write-Host "Missing sources: $($missing.Count)"
Write-Host "P3 selected: $($p3Selected.Count)"
Write-Host "Copied files: $($copied.Count)"
Write-Host "Copied size GB: $([math]::Round($copiedBytes / 1GB, 3))"
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
'''
    write_text(PUB / "scripts" / "stage_onedrive_release.ps1", ps1)
    write_text(PUB / "scripts" / "stage_onedrive_release.sh", sh)


def availability_text(raw_found: bool, p0_staged: bool, now: str) -> str:
    yolo_sentence = (
        "True raw YOLOv11 detector exports were found by the Task 17 audit and are assigned to P1 for later staging."
        if raw_found
        else "The Task 17 audit did not find true raw per-frame YOLOv11 detector exports; therefore the release does not claim that raw detector exports are available."
    )
    p0_sentence = (
        "P0 release materials have been staged locally into the OneDrive-synced folder; cloud-sync completion still requires manual verification."
        if p0_staged
        else "P0 release materials are prepared in the staging manifest but local copy/cloud-sync completion still requires verification."
    )
    return f"""# Manuscript Data and Code Availability Statement - OneDrive Version

Generated/updated for Task 17 on `{now}`.

Code, frozen protocols, compact result tables, reproducibility manifests, figure sources and manuscript-support files are available in the GitHub repository:

`https://github.com/aronagg/PathFindRNET_2.0`

The publication page is expected at:

`{PAGES_URL}`

Large redistributable artifacts are managed through the public OneDrive folder:

`{ONEDRIVE_URL}`

{p0_sentence}

The OneDrive manifest includes raw videos, tracking/intermediate outputs, processed trajectory and feature exports, reference-label artifacts and supplementary result tables where redistribution is permitted. {yolo_sentence}

GitHub materials and OneDrive-staged materials are separated in `docs/onedrive_upload_manifest_prioritized.csv`. Google-derived or third-party imagery is excluded unless redistribution rights and attribution are confirmed.
"""


def update_availability_docs(raw_found: bool, p0_staged: bool, now: str) -> None:
    text = availability_text(raw_found, p0_staged, now)
    write_text(DOCS / "manuscript_data_availability_onedrive_final.md", text)
    write_text(DOCS / "manuscript_revised_data_code_availability_final.md", text)
    write_text(PUB / "DATA_AVAILABILITY.md", text.replace("# Manuscript Data and Code Availability Statement - OneDrive Version", "# Data Availability"))
    write_text(
        DOCS / "final_data_licensing_statement.md",
        f"""# Final Data Licensing Statement

Generated/updated for Task 17 on `{now}`.

The release plan uses GitHub for code and compact reproducibility artifacts, GitHub Pages for the public project page, and public OneDrive for large redistributable data artifacts.

Public OneDrive folder:

`{ONEDRIVE_URL}`

P0 materials are staged locally in:

`{ONEDRIVE_ROOT}`

Cloud-sync completion still requires manual verification in OneDrive. Raw videos and tracking/intermediate outputs are classified as P1 and were not staged by Task 17. Google-derived map imagery and third-party imagery require separate provenance and redistribution review before public release. The Task 17 audit did not find true raw per-frame YOLOv11 detector exports unless otherwise stated in `docs/yolov11_detection_deep_audit.md`.
""",
    )


def compare_onedrive(prioritized: list[dict[str, object]], now: str) -> tuple[int, int, int, bool]:
    existing = []
    for dirpath, dirnames, filenames in os.walk(ONEDRIVE_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            existing.append(
                {
                    "onedrive_relative_path": path.relative_to(ONEDRIVE_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_csv(DOCS / "onedrive_existing_inventory.csv", existing, ["onedrive_relative_path", "size_bytes", "sha256"])
    existing_by_path = {str(row["onedrive_relative_path"]).lower(): row for row in existing}
    p0_rows = [row for row in prioritized if row["upload_priority"] == "P0"]
    missing = []
    mismatched = []
    for row in p0_rows:
        target = str(row["onedrive_target_relative_path"])
        target_key = target.lower()
        if target_key not in existing_by_path:
            missing.append({"onedrive_target_relative_path": target, "upload_priority": "P0", "repository_relative_path": row["repository_relative_path"]})
            continue
        expected = str(row["sha256"])
        actual = str(existing_by_path[target_key]["sha256"])
        if expected.startswith("NOT_COMPUTED"):
            continue
        if expected.lower() != actual.lower():
            mismatched.append(
                {
                    "onedrive_target_relative_path": target,
                    "upload_priority": "P0",
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )
    expected_p0 = {str(row["onedrive_target_relative_path"]).lower() for row in p0_rows}
    extra = [{"extra_onedrive_relative_path": str(row["onedrive_relative_path"])} for key, row in sorted(existing_by_path.items()) if key not in expected_p0]
    write_csv(DOCS / "onedrive_missing_artifacts.csv", missing, ["onedrive_target_relative_path", "upload_priority", "repository_relative_path"])
    write_csv(DOCS / "onedrive_mismatched_hashes.csv", mismatched, ["onedrive_target_relative_path", "upload_priority", "expected_sha256", "actual_sha256"])
    write_csv(DOCS / "onedrive_extra_artifacts_review.csv", extra, ["extra_onedrive_relative_path"])
    p0_complete = len(missing) == 0 and len(mismatched) == 0
    write_text(
        DOCS / "onedrive_staging_verification_report.md",
        f"""# OneDrive Staging Verification Report

Generated: `{now}`

Local OneDrive root:

`{ONEDRIVE_ROOT}`

## P0 Verification

- P0 manifest rows: {len(p0_rows)}
- Missing P0 files: {len(missing)}
- Mismatched P0 hashes: {len(mismatched)}
- Extra files in local OneDrive tree: {len(extra)}
- P0 complete locally: {str(p0_complete).lower()}

Cloud-sync status still requires manual verification in OneDrive.
""",
    )
    return len(missing), len(mismatched), len(extra), p0_complete


def final_reports(
    now: str,
    prioritized: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    raw_found: bool,
    p0_missing: int | None,
    p0_mismatched: int | None,
    p0_complete: bool,
) -> None:
    counts, sizes = priority_summary(prioritized)
    type_counts = Counter(str(row["detected_file_type"]) for row in audit_rows)
    type_lines = ["| Type | Files |", "| --- | ---: |"]
    for kind, count in sorted(type_counts.items()):
        type_lines.append(f"| `{kind}` | {count} |")
    priority_lines = ["| Priority | Files | Size GB |", "| --- | ---: | ---: |"]
    for priority in ["P0", "P1", "P2", "P3"]:
        priority_lines.append(f"| {priority} | {counts[priority]} | {sizes[priority] / (1024**3):.3f} |")
    findings = []
    for path in [
        PUB / "DATA_AVAILABILITY.md",
        DOCS / "manuscript_data_availability_onedrive_final.md",
        DOCS / "final_data_licensing_statement.md",
    ]:
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in ["[DOI", "Zenodo", "Figshare", "OSF", "archive deposit", "Archive URL", "DOI archive"]:
            if needle in text:
                findings.append((rel(path), "old_archive_wording", needle))
        if "raw yolov11 detections are available" in text.lower():
            findings.append((rel(path), "unsupported_yolo_claim", "raw YOLOv11 availability"))
    finding_lines = ["| Item | Finding | Detail |", "| --- | --- | --- |"]
    if findings:
        for item, finding, detail in findings:
            finding_lines.append(f"| `{item}` | {finding} | {detail} |")
    else:
        finding_lines.append("| Checked files | none | No old archive placeholders or unsupported raw YOLOv11 availability claim found. |")
    write_text(
        DOCS / "final_placeholder_and_consistency_report.md",
        f"""# Final Placeholder And Consistency Report

Generated/updated for Task 17 on `{now}`.

## Findings

{chr(10).join(finding_lines)}

## P0 OneDrive Verification

- Missing P0 files: {p0_missing if p0_missing is not None else 'not yet compared'}
- Mismatched P0 hashes: {p0_mismatched if p0_mismatched is not None else 'not yet compared'}
- P0 complete locally: {str(p0_complete).lower()}
- Cloud sync: manual verification still required.
""",
    )
    write_text(
        DOCS / "task_17_execution_report.md",
        f"""# Task 17 Execution Report

Generated: `{now}`

Branch: `{run_git(['branch', '--show-current'])}`

Base commit at generation time: `{run_git(['rev-parse', 'HEAD'])}`

## data/interim Detection Audit Result

True raw YOLOv11 detections found: `{str(raw_found).lower()}`

## Detection/Tracking/Intermediate Files Found

{chr(10).join(type_lines)}

## Updated Priority Summary

{chr(10).join(priority_lines)}

## OneDrive Staging

Local OneDrive root used:

`{ONEDRIVE_ROOT}`

P0 missing after local comparison: `{p0_missing if p0_missing is not None else 'not compared'}`

P0 mismatched after local comparison: `{p0_mismatched if p0_mismatched is not None else 'not compared'}`

P0 complete locally: `{str(p0_complete).lower()}`

Cloud sync status: manual verification still required in OneDrive.

## Data Availability Status

The wording distinguishes local P0 staging from cloud-sync verification and does not claim raw YOLOv11 detector exports are available when they were not found.

## Checks

- Python compile and Ruff should be run for changed Python scripts.
- PowerShell parser check should be run for `stage_onedrive_release.ps1`.
- Dry-run staging should be run before actual P0 copy.
- P0 staging verification should be run after actual copy.
- No experiments were run by this documentation/audit script.
""",
    )


def main() -> None:
    now = now_utc()
    # Refresh Task 15-style inventory first so Task 17 sees current docs/scripts.
    subprocess.check_call([sys.executable, str(PUB / "scripts" / "generate_task15_onedrive_release_docs.py")], cwd=ROOT)
    audit_rows = scan_interim_and_detection_outputs()
    raw_found = write_audit_reports(audit_rows, now)
    inventory = scan_release_inventory()
    prioritized = write_manifest_outputs(inventory, audit_rows, now)
    update_stage_scripts()
    update_availability_docs(raw_found, p0_staged=False, now=now)
    final_reports(now, prioritized, audit_rows, raw_found, None, None, False)
    print(
        {
            "audit_rows": len(audit_rows),
            "true_raw_yolov11_found": raw_found,
            "priority_counts": dict(priority_summary(prioritized)[0]),
        }
    )


if __name__ == "__main__":
    main()
