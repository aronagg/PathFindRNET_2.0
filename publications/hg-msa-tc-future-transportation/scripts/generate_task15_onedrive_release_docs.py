from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PUB = ROOT / "publications" / "hg-msa-tc-future-transportation"
DOCS = PUB / "docs"
SITE = PUB / "site"
ASSETS = SITE / "assets" / "images"

ONEDRIVE_URL = (
    "https://onedrive.live.com/?redeem="
    "aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84MGZhYWQ2ZmVhMDViMDhjL0lnQlBxblBYemRWd1I1NERrbHVfUzB0bkFUMWJFUlBzSk1XTHM3TVJ3ZDNPZFRJP2U9b2NpOXBM"
    "&id=80FAAD6FEA05B08C%21sd773aa4fd5cd47709e03925bbf4b4b67&cid=80FAAD6FEA05B08C"
)

HASH_LIMIT = 256 * 1024 * 1024
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "review_packages"}
EXCLUDE_EXT = {".zip"}
SCAN_ROOTS = [
    Path("data"),
    Path("TNVD2_UPLOAD_PACKAGE"),
    Path("Traffic_Node_Video_Dataset_2_0_2026"),
    Path("publications/hg-msa-tc-future-transportation"),
    Path("traffic_node_dataset_2_0_starter_pack/tnvd2_starter_pack/github_pages"),
]


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    size = path.stat().st_size
    if size > HASH_LIMIT:
        return f"NOT_COMPUTED_LARGE_FILE_GT_{HASH_LIMIT}_BYTES"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_for(repo_path: str) -> str:
    lowered = repo_path.lower()
    name = Path(lowered).name
    if lowered.startswith("data/raw/") or "/01_original_videos/" in lowered:
        return "raw_video"
    if "02_yolov11x_detections" in lowered or "detection" in name:
        return "yolov11_detection_or_detection_statistics"
    if "03_yolo_tracking_outputs" in lowered or "/tracks" in lowered or "tracking" in lowered:
        return "tracking_output"
    if "data/processed" in lowered or "05_trajectories" in lowered or "features" in lowered or "trajectory" in lowered:
        return "processed_trajectory_or_feature"
    if "/annotations/" in lowered or "reference_label" in lowered or "polygon" in lowered:
        return "reference_label_or_annotation_protocol"
    if "/results/" in lowered or "/reports/" in lowered or "/docs/" in lowered:
        return "publication_result_or_documentation"
    if "/figures/" in lowered or "/site/assets/images/" in lowered or lowered.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        return "figure_or_website_asset"
    if "/configs/" in lowered or "protocol" in lowered or "manifest" in lowered:
        return "configuration_or_manifest"
    return "other"


def onedrive_folder_for(category: str) -> str:
    folders = {
        "raw_video": "01_original_videos/",
        "yolov11_detection_or_detection_statistics": "02_yolov11x_detections/",
        "tracking_output": "03_yolo_tracking_outputs/",
        "processed_trajectory_or_feature": "04_processed_trajectories_and_features/",
        "reference_label_or_annotation_protocol": "05_reference_labels_and_protocols/",
        "publication_result_or_documentation": "06_publication_results_and_docs/",
        "figure_or_website_asset": "07_figures_and_website_assets/",
        "configuration_or_manifest": "08_reproducibility_configs_and_manifests/",
    }
    return folders.get(category, "99_misc_review/")


def release_action_for(category: str, repo_path: str) -> str:
    lowered = repo_path.lower()
    if category == "raw_video":
        return "onedrive_large_data_only; exclude from git package"
    if "google" in lowered or "maps" in lowered or "satellite" in lowered:
        return "review_provenance_before_public_redistribution"
    if category in {
        "publication_result_or_documentation",
        "figure_or_website_asset",
        "configuration_or_manifest",
        "reference_label_or_annotation_protocol",
    }:
        return "github_and_onedrive"
    if category in {
        "yolov11_detection_or_detection_statistics",
        "tracking_output",
        "processed_trajectory_or_feature",
    }:
        return "onedrive_data_release; compact summaries in github where appropriate"
    return "review_before_release"


def md_table(items: list[dict[str, object]], cols: list[str], limit: int = 50) -> str:
    if not items:
        return "No matching files found.\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for item in items[:limit]:
        lines.append("| " + " | ".join(str(item.get(col, "")) for col in cols) + " |")
    if len(items) > limit:
        lines.append("")
        lines.append(f"Additional rows omitted from this markdown view: {len(items) - limit}. See the CSV manifest for the complete list.")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def scan_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scan_root in SCAN_ROOTS:
        base = ROOT / scan_root
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() in EXCLUDE_EXT:
                    continue
                repo_path = rel(path)
                if any(part in EXCLUDE_DIRS for part in Path(repo_path).parts):
                    continue
                stat = path.stat()
                category = category_for(repo_path)
                rows.append(
                    {
                        "repository_relative_path": repo_path,
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 6),
                        "extension": path.suffix.lower(),
                        "category": category,
                        "release_action": release_action_for(category, repo_path),
                        "onedrive_folder": onedrive_folder_for(category),
                        "sha256": sha256_file(path),
                        "last_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                        .replace(microsecond=0)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    }
                )
    rows.sort(key=lambda row: str(row["repository_relative_path"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_table(rows: list[dict[str, object]]) -> str:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row["category"])
        item = summary.setdefault(category, {"files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += int(row["size_bytes"])
    lines = ["| Category | Files | Size GB |", "| --- | ---: | ---: |"]
    for category in sorted(summary):
        item = summary[category]
        lines.append(f"| `{category}` | {item['files']} | {item['bytes'] / (1024**3):.3f} |")
    return "\n".join(lines)


def create_preview_images(now: str) -> list[dict[str, object]]:
    selected = [
        PUB / "figures/final_synthesis/final_hg_smg_pipeline_schematic.png",
        PUB / "figures/final_synthesis/final_target_count_error.png",
        PUB / "figures/final_synthesis/final_agreement_metrics.png",
        PUB / "figures/final_synthesis/final_baseline_comparison_summary.png",
        PUB / "figures/final_synthesis/final_se38th_fragmentation_story.png",
    ]
    try:
        from PIL import Image

        pil_available = True
    except Exception:
        Image = None
        pil_available = False

    ASSETS.mkdir(parents=True, exist_ok=True)
    previews: list[dict[str, object]] = []
    for source in selected:
        if not source.exists():
            previews.append({"source": rel(source), "status": "missing", "preview": "", "width": "", "height": "", "size_bytes": ""})
            continue
        output = ASSETS / f"{source.stem}_preview.png"
        if pil_available:
            assert Image is not None
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((1200, 800))
                image.save(output, optimize=True)
                width, height = image.size
        else:
            shutil.copy2(source, output)
            width = "copied_original_no_pil"
            height = "copied_original_no_pil"
        previews.append(
            {
                "source": rel(source),
                "status": "created",
                "preview": rel(output),
                "width": width,
                "height": height,
                "size_bytes": output.stat().st_size,
            }
        )

    lines = [
        "| Source | Preview | Status | Width | Height | Size bytes |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in previews:
        lines.append(
            f"| `{row['source']}` | `{row['preview']}` | {row['status']} | {row['width']} | {row['height']} | {row['size_bytes']} |"
        )
    write_text(
        DOCS / "preview_image_quality_audit.md",
        f"""# Preview Image Quality Audit

Generated: `{now}`

Preview images were generated from existing publication figures for GitHub Pages. No scientific figure content was changed.

{chr(10).join(lines)}

## Quality Notes

- The preview set uses final synthesis figures only.
- The files are resized for web display and should not replace the publication-quality source figures under `figures/`.
- If journal production requires higher resolution, use the original figure files rather than the `_preview.png` copies.
""",
    )
    return previews


def main() -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    branch = run_git(["branch", "--show-current"])
    head = run_git(["rev-parse", "HEAD"])

    DOCS.mkdir(parents=True, exist_ok=True)
    rows = scan_inventory()
    inventory_fields = [
        "repository_relative_path",
        "size_bytes",
        "size_mb",
        "extension",
        "category",
        "release_action",
        "onedrive_folder",
        "sha256",
        "last_modified_utc",
    ]
    write_csv(DOCS / "local_release_artifact_inventory.csv", rows, inventory_fields)

    upload_rows = []
    for row in rows:
        action = str(row["release_action"])
        upload_required = action.startswith("onedrive") or action == "github_and_onedrive"
        upload_rows.append(
            {
                "repository_relative_path": row["repository_relative_path"],
                "onedrive_folder": row["onedrive_folder"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "upload_required": str(upload_required).lower(),
                "priority": "high"
                if row["category"]
                in {"raw_video", "processed_trajectory_or_feature", "reference_label_or_annotation_protocol", "publication_result_or_documentation"}
                else "medium",
                "public_release_note": action,
            }
        )
    write_csv(
        DOCS / "onedrive_upload_manifest.csv",
        upload_rows,
        ["repository_relative_path", "onedrive_folder", "size_bytes", "sha256", "upload_required", "priority", "public_release_note"],
    )

    onedrive_root = os.environ.get("ONEDRIVE_RELEASE_ROOT")
    onedrive_mode = "mode_b_manifest_only_no_local_onedrive_sync_root"
    if onedrive_root and Path(onedrive_root).exists():
        od_root = Path(onedrive_root)
        od_rows = []
        for dirpath, dirnames, filenames in os.walk(od_root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                od_rows.append(
                    {
                        "onedrive_relative_path": path.relative_to(od_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        write_csv(DOCS / "onedrive_existing_inventory.csv", od_rows, ["onedrive_relative_path", "size_bytes", "sha256"])
        expected = {
            (Path(str(row["onedrive_folder"])) / Path(str(row["repository_relative_path"])).name).as_posix()
            for row in upload_rows
            if row["upload_required"] == "true"
        }
        existing = {str(row["onedrive_relative_path"]) for row in od_rows}
        write_csv(
            DOCS / "onedrive_missing_artifacts.csv",
            [{"expected_onedrive_relative_path": value} for value in sorted(expected - existing)],
            ["expected_onedrive_relative_path"],
        )
        write_csv(
            DOCS / "onedrive_extra_artifacts_review.csv",
            [{"extra_onedrive_relative_path": value} for value in sorted(existing - expected)],
            ["extra_onedrive_relative_path"],
        )
        onedrive_mode = f"mode_a_compared_with_local_sync_root:{onedrive_root}"

    category_summary = summary_table(rows)
    yolo = [row for row in rows if row["category"] == "yolov11_detection_or_detection_statistics"]
    tracking = [row for row in rows if row["category"] == "tracking_output"]
    raw_videos = [row for row in rows if row["category"] == "raw_video"]
    processed = [row for row in rows if row["category"] == "processed_trajectory_or_feature"]
    figures = [row for row in rows if row["category"] == "figure_or_website_asset"]
    publication_docs = [row for row in rows if row["category"] == "publication_result_or_documentation"]
    reference_labels = [row for row in rows if row["category"] == "reference_label_or_annotation_protocol"]

    write_text(
        DOCS / "yolov11_raw_detection_file_locations.md",
        f"""# YOLOv11 Raw Detection File Locations

Generated: `{now}`

This audit searched the local checkout for YOLOv11/Yolov11x detection files and adjacent tracking outputs. It did not run detection or tracking.

## Detection-Oriented Files

{md_table(yolo, ['repository_relative_path', 'size_mb', 'sha256'], 80)}

## Adjacent Tracking Outputs

{md_table(tracking, ['repository_relative_path', 'size_mb', 'sha256'], 80)}

## Interpretation

- The most explicit public-package location is `TNVD2_UPLOAD_PACKAGE/02_yolov11x_detections/`.
- Detection-statistics files were also found through names containing `detection`.
- Tracking outputs under `TNVD2_UPLOAD_PACKAGE/03_yolo_tracking_outputs/` are adjacent downstream artifacts, not raw detector outputs.
- If additional raw YOLOv11 per-frame detection exports exist outside this checkout, upload them to the OneDrive folder `02_yolov11x_detections/` and add them to `docs/onedrive_upload_manifest.csv` before public release.
""",
    )

    write_text(
        DOCS / "onedrive_folder_structure_proposal.md",
        f"""# Public OneDrive Folder Structure Proposal

Generated: `{now}`

Public OneDrive folder:

`{ONEDRIVE_URL}`

## Proposed Structure

```text
PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release/
  00_README_AND_MANIFESTS/
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

## Mapping From Local Inventory

{category_summary}

## Required Upload Policy

- Keep GitHub focused on code, compact CSV summaries, documentation, reproducibility manifests and manuscript-support material.
- Put large raw videos, detector/tracking exports and large trajectory data in OneDrive.
- Do not upload Google-derived rasters or third-party map screenshots unless redistribution rights and attribution are confirmed.
- Keep raw video and large data out of Git history.
- Update `docs/onedrive_upload_manifest.csv` after any manual upload or rename.
""",
    )

    write_text(
        DOCS / "new_publication_artifacts_to_upload.md",
        f"""# New Publication Artifacts To Upload To OneDrive

Generated: `{now}`

This list is derived from `docs/onedrive_upload_manifest.csv`. It is a release-planning list only; no upload was performed.

## High-Priority Groups

| Group | Local evidence | Suggested OneDrive folder |
| --- | ---: | --- |
| Raw videos | {len(raw_videos)} files | `01_original_videos/` |
| YOLOv11 detection/statistics files | {len(yolo)} files | `02_yolov11x_detections/` |
| YOLO tracking outputs | {len(tracking)} files | `03_yolo_tracking_outputs/` |
| Processed trajectories/features | {len(processed)} files | `04_processed_trajectories_and_features/` |
| Reference labels/protocols | {len(reference_labels)} files | `05_reference_labels_and_protocols/` |
| Publication results/docs | {len(publication_docs)} files | `06_publication_results_and_docs/` |
| Figures/site assets | {len(figures)} files | `07_figures_and_website_assets/` |

## Notes

- The full itemized upload list is `docs/onedrive_upload_manifest.csv`.
- Large files have size metadata but may not have local SHA-256 hashes if they exceed the configured audit limit of {HASH_LIMIT} bytes.
- Publication result CSVs, reference-label summaries and final figures should also remain in GitHub when compact enough.
- OneDrive should be treated as the public large-artifact mirror, not as evidence that files were generated without the repository provenance records.
""",
    )

    previews = create_preview_images(now)

    data_availability = f"""# Data Availability

Public release materials for the revised Future Transportation manuscript are split across GitHub, GitHub Pages and a public OneDrive folder.

Public OneDrive folder:

`{ONEDRIVE_URL}`

## GitHub Repository

The GitHub repository should contain source code, compact result tables, frozen protocol files, reproducibility manifests, manuscript-support documentation and website source files.

Repository: `https://github.com/aronagg/PathFindRNET_2.0`

Expected publication page after GitHub Pages integration:

`https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/`

## OneDrive Large-Artifact Release

The OneDrive folder is the planned public location for large artifacts that are unsuitable for Git history, including raw videos, YOLOv11 detection exports, tracking outputs and large trajectory-level exports where redistributable.

The local upload manifest is:

`publications/hg-msa-tc-future-transportation/docs/onedrive_upload_manifest.csv`

## Exclusions

The publication package does not redistribute raw data for which licensing or privacy review is unresolved. Google-derived or third-party map imagery should not be included in public release materials unless redistribution rights and attribution are confirmed.

## Reproducibility Boundary

The release supports audit and reproduction of the submitted analyses from frozen protocols and compact result artifacts. No new scientific experiments were run as part of the OneDrive release audit.
"""
    write_text(PUB / "DATA_AVAILABILITY.md", data_availability)

    write_text(
        DOCS / "final_data_licensing_statement.md",
        f"""# Final Data Licensing Statement

The revised release plan uses GitHub for code and compact reproducibility artifacts, GitHub Pages for the public project page, and public OneDrive for large redistributable data artifacts.

Public OneDrive folder:

`{ONEDRIVE_URL}`

Raw videos, YOLOv11 detection outputs, tracking outputs and large trajectory-level exports should be provided through OneDrive only when redistribution is permitted. Google-derived map imagery and any third-party imagery require separate provenance and redistribution review before public release.

The manuscript data-availability statement should describe the OneDrive folder as the public large-artifact location and should not claim a permanent archive identifier that has not been assigned.
""",
    )

    manuscript_availability = f"""# Manuscript Data and Code Availability Statement - OneDrive Version

Code, frozen protocols, compact result tables, reproducibility manifests, figure sources and manuscript-support files are available in the GitHub repository:

`https://github.com/aronagg/PathFindRNET_2.0`

A project page is planned at:

`https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/`

Large redistributable artifacts are planned for public access through the OneDrive folder:

`{ONEDRIVE_URL}`

The large-artifact release is intended to include raw videos, YOLOv11 detection exports, tracking outputs, processed trajectory and feature exports, reference-label artifacts and supplementary result tables where redistribution is permitted. Third-party or Google-derived imagery is excluded unless redistribution rights and attribution are confirmed. The release manifest and local upload plan are provided in `docs/local_release_artifact_inventory.csv` and `docs/onedrive_upload_manifest.csv`.
"""
    write_text(DOCS / "manuscript_revised_data_code_availability_final.md", manuscript_availability)
    write_text(DOCS / "manuscript_data_availability_onedrive_final.md", manuscript_availability)

    write_text(
        DOCS / "public_release_scope.md",
        f"""# Public Release Scope

Generated/updated for Task 15 on `{now}`.

## In Scope For GitHub

- Source code and publication-specific scripts.
- Frozen protocol/configuration files.
- Compact CSV summaries, manifests and checksums.
- Manuscript-support documentation and reviewer-response material.
- Public website source under `site/`.
- Web-optimized preview images under `site/assets/images/`.

## In Scope For OneDrive

- Raw video files, if redistribution is permitted.
- YOLOv11 detection exports.
- Tracking outputs.
- Large processed trajectory and feature exports.
- Reference-label artifacts that are too large for Git.
- Supplementary full-resolution figures and tables where appropriate.

Public OneDrive folder:

`{ONEDRIVE_URL}`

## Excluded Or Requires Manual Review

- Google-derived and third-party map imagery unless redistribution rights and attribution are confirmed.
- Local virtual environments, caches and generated ZIP review packages.
- Manual SQLite annotation databases unless intentionally redacted and documented.
- Any file containing private absolute paths or credentials.
""",
    )

    write_text(
        DOCS / "release_tagging_plan.md",
        f"""# Release Tagging Plan

Generated/updated for Task 15 on `{now}`.

## Recommended GitHub Tag

`futuretransp-hg-smg-tc-v1.0`

## Release Contents

The GitHub release should point to:

- the repository source tree;
- `publications/hg-msa-tc-future-transportation/README.md`;
- `publications/hg-msa-tc-future-transportation/DATA_AVAILABILITY.md`;
- the GitHub Pages publication page;
- the public OneDrive folder for large redistributable artifacts.

Public OneDrive folder:

`{ONEDRIVE_URL}`

## Tagging Steps

```powershell
git status --short
git tag -a futuretransp-hg-smg-tc-v1.0 -m "Future Transportation HG-SMG-TC reproducibility release"
git push origin futuretransp-hg-smg-tc-v1.0
```

Do not create or advertise a release tag until the manuscript-support package, GitHub Pages page and OneDrive upload manifest have been reviewed.
""",
    )

    write_text(
        DOCS / "final_submission_checklist.md",
        f"""# Final Submission Checklist

Generated/updated for Task 15 on `{now}`.

## Manuscript Package

- [ ] Final DOCX/PDF exported from the revised manuscript.
- [ ] Figure and table numbering checked against manuscript text.
- [ ] Reviewer response line/page placeholders completed after final formatting.
- [ ] Data and code availability statement uses GitHub, GitHub Pages and OneDrive links.
- [ ] No unassigned permanent archive identifier is claimed.

## Public Release

- [ ] GitHub release tag selected and reviewed.
- [ ] GitHub Pages source location confirmed by repository maintainer.
- [ ] Proposed page under `site/publications/hg-smg-tc/index.md` integrated into the active Pages source.
- [ ] Public OneDrive folder populated according to `docs/onedrive_upload_manifest.csv`.
- [ ] OneDrive share permissions checked in a private browser session.
- [ ] Large raw videos and detector/tracking exports kept out of Git history.
- [ ] Google-derived or third-party imagery excluded unless redistribution rights are documented.

## Consistency Checks

- [ ] `DATA_AVAILABILITY.md` matches the manuscript statement.
- [ ] `CITATION.cff` does not claim unavailable identifiers.
- [ ] `docs/final_placeholder_and_consistency_report.md` reviewed.
- [ ] Website preview images render on the deployed page.
""",
    )

    write_text(
        DOCS / "public_data_archive_plan.md",
        f"""# Public OneDrive Data Release Plan

Generated/updated for Task 15 on `{now}`.

The release plan for this manuscript is GitHub plus GitHub Pages plus public OneDrive. OneDrive is used for large redistributable artifacts that are not appropriate for Git history.

Public OneDrive folder:

`{ONEDRIVE_URL}`

## Required Manifests

- `docs/local_release_artifact_inventory.csv`
- `docs/onedrive_upload_manifest.csv`
- `docs/new_publication_artifacts_to_upload.md`
- `docs/yolov11_raw_detection_file_locations.md`

## Upload Groups

1. `01_original_videos/`
2. `02_yolov11x_detections/`
3. `03_yolo_tracking_outputs/`
4. `04_processed_trajectories_and_features/`
5. `05_reference_labels_and_protocols/`
6. `06_publication_results_and_docs/`
7. `07_figures_and_website_assets/`
8. `08_reproducibility_configs_and_manifests/`
9. `09_licenses_and_provenance/`

## Manual Checks Before Public Sharing

- Verify that each uploaded item is listed in `docs/onedrive_upload_manifest.csv`.
- Confirm share permissions from a browser session that is not signed in.
- Exclude Google-derived or third-party map rasters unless redistribution is documented.
- Keep generated ZIP review packages out of the public data folder unless explicitly intended.
""",
    )

    readme_path = PUB / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    old = """## DOI Placeholder

No DOI is claimed yet. Use this placeholder until an archive deposit exists:

`[DOI to be added after archive deposit]`"""
    new = f"""## Public OneDrive Data Release

Large redistributable artifacts are planned for public access through OneDrive:

`{ONEDRIVE_URL}`

The local upload manifest is `docs/onedrive_upload_manifest.csv`. The release does not claim an unavailable permanent archive identifier."""
    readme_text = readme_text.replace(old, new)
    readme_text = readme_text.replace(
        "public-release, data-availability and GitHub Pages planning files.",
        "public-release, OneDrive data-availability and GitHub Pages planning files.",
    )
    write_text(readme_path, readme_text)

    write_text(
        SITE / "publications" / "hg-smg-tc" / "index.md",
        f"""# From Geometric Endpoint Micro-Modes to Semantic Maneuver Graphs

**Homography-Guided Vehicle Trajectory Clustering at Complex Urban Intersections**

This page summarizes the Future Transportation revision package for HG-SMG-TC: **Homography-Guided Semantic Maneuver Graph Trajectory Clustering**.

Expected public URL after GitHub Pages integration:

`https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/`

## Release Links

- GitHub repository: `https://github.com/aronagg/PathFindRNET_2.0`
- Publication package: `publications/hg-msa-tc-future-transportation/`
- Public OneDrive data folder: `{ONEDRIVE_URL}`

## Summary

HG-SMG-TC connects homography-guided endpoint micro-mode discovery with semantic approach consolidation, semantic maneuver graph construction, uncertainty-aware maneuver-count priors and prior-constrained model selection. The method is evaluated on five Bellevue intersections using a leakage-controlled split protocol and exhaustive human-defined polygon-rule-based reference labels.

## Dataset and Split Summary

The study uses five Bellevue scenes only:

- `bellevue_116th_ne12th`
- `bellevue_150th_newport`
- `bellevue_150th_eastgate`
- `bellevue_150th_se38th`
- `bellevue_ne8th`

The split roles are target estimation, model selection and locked independent test. Independent-test labels are read only after cluster assignments are persisted.

## Pipeline Overview

![HG-SMG-TC pipeline](../../assets/images/final_hg_smg_pipeline_schematic_preview.png)

## Locked-Test Result Summary

| Method family | Target error | NMI | Macro F1 | Outlier % |
| --- | ---: | ---: | ---: | ---: |
| Original untargeted A0 | 3.2000 | 0.8155 | 0.6328 | 15.56 |
| Original HG-aware A1 | 2.5333 | 0.8376 | 0.6838 | 9.94 |
| HG-SMG-TC A5 | 2.1333 | 0.8386 | 0.6723 | 10.34 |
| Endpoint isotropic baseline | 3.4667 | 0.7099 | 0.5710 | 24.43 |
| Resampled trajectory baseline | 3.0000 | 0.6827 | 0.4752 | 25.02 |

The results support a conservative trade-off interpretation. HG-SMG-TC improves mean observed target-count alignment and slightly improves NMI relative to the original HG-aware method, but it does not improve every metric.

## Key Figures

![Target error](../../assets/images/final_target_count_error_preview.png)

![Agreement metrics](../../assets/images/final_agreement_metrics_preview.png)

![Baseline comparison](../../assets/images/final_baseline_comparison_summary_preview.png)

![SE38th fragmentation](../../assets/images/final_se38th_fragmentation_story_preview.png)

## Reproducibility

Start with:

- `publications/hg-msa-tc-future-transportation/README.md`
- `publications/hg-msa-tc-future-transportation/REPRODUCIBILITY.md`
- `publications/hg-msa-tc-future-transportation/DATA_AVAILABILITY.md`

Core synthesis regeneration:

```powershell
.\\publications\\hg-msa-tc-future-transportation\\scripts\\reproduce_core_results.ps1
```

## License and Provenance Notes

Raw videos and large detector/tracking artifacts belong in the public OneDrive data folder, not in Git history. Google-derived or third-party map imagery should not be redistributed unless licensing and attribution are confirmed. Public materials should rely on source code, compact metrics, calibration tables, generated figures and author-created diagrams.

[Back to PathFindRNET 2.0](../../../)
""",
    )

    write_text(
        DOCS / "github_pages_integration_plan.md",
        f"""# GitHub Pages Integration Plan

Generated/updated for Task 15 on `{now}`.

## Source Detection

The active GitHub Pages source could not be safely determined from this checkout. No root Pages source was modified.

Observed state:

- no root `_config.yml` was found;
- no root `mkdocs.yml` was found;
- no `.github/workflows/` Pages workflow was found;
- the local `gh-pages` branch was not present.

## Proposed Page

The publication page is staged at:

`publications/hg-msa-tc-future-transportation/site/publications/hg-smg-tc/index.md`

Web preview images are staged at:

`publications/hg-msa-tc-future-transportation/site/assets/images/`

Expected public URL after integration:

`https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/`

## Manual Integration Steps

1. Confirm the repository Pages source in GitHub settings or Actions.
2. Copy or route `site/publications/hg-smg-tc/index.md` into the active Pages source.
3. Copy `site/assets/images/` into the corresponding public assets folder.
4. Verify relative image links after deployment.
5. Add the OneDrive folder link to the deployed page.
""",
    )

    root_readme_proposal = f"""# Root README Publication Section Proposal

Generated/updated for Task 15 on `{now}`.

Add the following section to the repository root README after the general project overview.

```markdown
## Future Transportation HG-SMG-TC Revision Package

The revised Future Transportation manuscript support package is available under:

`publications/hg-msa-tc-future-transportation/`

It contains frozen protocols, compact results, reproducibility documentation, revised manuscript support files and GitHub Pages source for the HG-SMG-TC study on five Bellevue intersections.

Public page after deployment:

`https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/`

Large redistributable artifacts are planned for public access through OneDrive:

`{ONEDRIVE_URL}`
```
"""
    write_text(DOCS / "root_readme_update_proposal.md", root_readme_proposal)
    write_text(ROOT / "README_publication_section_draft.md", root_readme_proposal)

    checked_files = [
        PUB / "DATA_AVAILABILITY.md",
        PUB / "README.md",
        PUB / "CITATION.cff",
        DOCS / "final_data_licensing_statement.md",
        DOCS / "manuscript_revised_data_code_availability_final.md",
        DOCS / "public_release_scope.md",
        DOCS / "release_tagging_plan.md",
        DOCS / "final_submission_checklist.md",
        DOCS / "manuscript_data_availability_onedrive_final.md",
        SITE / "publications" / "hg-smg-tc" / "index.md",
    ]
    findings: list[tuple[str, str, str]] = []
    for path in checked_files:
        if not path.exists():
            findings.append((rel(path), "missing", "required file missing"))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in ["[DOI", "Zenodo", "Figshare", "OSF", "archive deposit", "Archive URL"]:
            if needle in text:
                findings.append((rel(path), "old_archive_wording", needle))
        if "[GitHub release tag to be added]" in text:
            findings.append((rel(path), "manual_placeholder", "GitHub release tag"))
    finding_lines = ["| File | Type | Detail |", "| --- | --- | --- |"]
    if findings:
        for file_name, finding_type, detail in findings:
            finding_lines.append(f"| `{file_name}` | {finding_type} | {detail} |")
    else:
        finding_lines.append("| All checked files | none | No old archive placeholders found in checked release-facing files. |")
    write_text(
        DOCS / "final_placeholder_and_consistency_report.md",
        f"""# Final Placeholder And Consistency Report

Generated/updated for Task 15 on `{now}`.

## Checked Files

{chr(10).join(f'- `{rel(path)}`' for path in checked_files)}

## Findings

{chr(10).join(finding_lines)}

## Remaining Manual Checks

- Confirm the final GitHub release tag before submission.
- Confirm the deployed GitHub Pages URL after the site source is connected.
- Confirm OneDrive share permissions from a browser session that is not signed in.
- Confirm that third-party imagery is excluded or separately licensed before public release.
""",
    )

    created_or_updated = [
        "scripts/generate_task15_onedrive_release_docs.py",
        "docs/local_release_artifact_inventory.csv",
        "docs/onedrive_upload_manifest.csv",
        "docs/onedrive_folder_structure_proposal.md",
        "docs/yolov11_raw_detection_file_locations.md",
        "docs/new_publication_artifacts_to_upload.md",
        "docs/preview_image_quality_audit.md",
        "DATA_AVAILABILITY.md",
        "docs/final_data_licensing_statement.md",
        "docs/manuscript_revised_data_code_availability_final.md",
        "docs/public_release_scope.md",
        "docs/release_tagging_plan.md",
        "docs/final_submission_checklist.md",
        "docs/manuscript_data_availability_onedrive_final.md",
        "docs/public_data_archive_plan.md",
        "README.md",
        "docs/root_readme_update_proposal.md",
        "../README_publication_section_draft.md",
        "site/publications/hg-smg-tc/index.md",
        "docs/github_pages_integration_plan.md",
        "docs/final_placeholder_and_consistency_report.md",
        "docs/task_15_execution_report.md",
    ]
    created_or_updated.extend(str(Path("site/assets/images") / Path(str(row["preview"])).name) for row in previews if row["preview"])
    preview_lines = [
        "| Source | Preview | Status | Width | Height | Size bytes |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in previews:
        preview_lines.append(
            f"| `{row['source']}` | `{row['preview']}` | {row['status']} | {row['width']} | {row['height']} | {row['size_bytes']} |"
        )
    write_text(
        DOCS / "task_15_execution_report.md",
        f"""# Task 15 Execution Report

Generated: `{now}`

Branch: `{branch}`

Base/final commit at generation time: `{head}`

## Scope

Task 15 replaced the earlier permanent-archive-oriented release wording with a GitHub + GitHub Pages + public OneDrive release plan. No experiments were run and no frozen scientific outputs were modified intentionally.

Public OneDrive folder:

`{ONEDRIVE_URL}`

## OneDrive Audit Mode

`{onedrive_mode}`

If this is mode B, the task produced a local upload manifest but did not compare against a synced OneDrive directory because `ONEDRIVE_RELEASE_ROOT` was not set to a valid local folder.

## Inventory Summary

{category_summary}

## Preview Images

{chr(10).join(preview_lines)}

## Created Or Updated Files

{chr(10).join(f'- `{path}`' for path in created_or_updated)}

## Skipped Files

- No OneDrive existing/missing/extra comparison CSVs were created unless a valid `ONEDRIVE_RELEASE_ROOT` was present.
- No ZIP package or SHA-256 sidecar was created because Task 15 explicitly did not request a ZIP.
- Raw videos and large source datasets were inventoried but not copied or modified.
- The active repository GitHub Pages source was not modified because it could not be safely determined from this checkout.

## Failed Steps

- None at generation time. External upload and deployed Pages verification remain manual release steps.

## Commands

```powershell
git switch -c feature/futuretransp-onedrive-release-and-submission
.\\.venv\\Scripts\\python.exe .\\publications\\hg-msa-tc-future-transportation\\scripts\\generate_task15_onedrive_release_docs.py
```
""",
    )

    print(
        json.dumps(
            {
                "inventory_rows": len(rows),
                "upload_rows": len(upload_rows),
                "onedrive_mode": onedrive_mode,
                "preview_created": sum(1 for row in previews if row["status"] == "created"),
                "docs_written": len(created_or_updated),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
