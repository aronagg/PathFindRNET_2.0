"""Build the deterministic, recording-block evaluation split and audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_trajectory_manifest import (  # noqa: E402
    MANIFEST_COLUMNS,
    PROTOCOL_VERSION,
    SCENES,
    default_output_path as default_manifest_path,
    default_repo_root,
    sha256_file,
)


SPLIT_NAMES = ("target_estimation", "model_selection", "independent_test")
SPLIT_PROPORTIONS = {
    "target_estimation": 0.30,
    "model_selection": 0.30,
    "independent_test": 0.40,
}
RANDOM_SEED = 20260702
SPLIT_COLUMNS = (
    "split_protocol_version",
    "scene_id",
    "trajectory_id",
    "split",
    "source_recording_id",
    "source_recording_start_time",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "number_of_points",
    "validity_under_existing_preprocessing_pipeline",
    "filtering_status",
    "data_fingerprint",
    "exact_duplicate_count",
    "exact_duplicate_group",
    "possible_near_duplicate",
    "near_duplicate_candidate_count",
    "near_duplicate_group",
    "split_assignment_basis",
    "split_adjustment_reason",
)
ANNOTATION_COLUMNS = (
    "scene_id",
    "trajectory_id",
    "split",
    "entry_approach",
    "exit_approach",
    "manual_maneuver_id",
    "maneuver_type",
    "validity",
    "rare_movement",
    "annotator_id",
    "annotation_timestamp",
    "confidence",
    "notes",
)


def publication_root(repo_root: Path) -> Path:
    return repo_root / "publications/hg-msa-tc-future-transportation"


def default_paths(repo_root: Path) -> dict[str, Path]:
    root = publication_root(repo_root)
    return {
        "manifest": default_manifest_path(repo_root),
        "split": root / "data/splits/evaluation_split.csv",
        "config": root / "configs/evaluation_split.yaml",
        "report": root / "docs/evaluation_split_report.md",
        "annotation": root / "annotations/annotation_template.csv",
    }


def choose_recording_boundaries(scene: pd.DataFrame) -> tuple[int, int]:
    recordings = (
        scene.groupby(["source_recording_start_time", "source_recording_id"], sort=True)
        .size()
        .reset_index(name="n_trajectories")
        .sort_values(["source_recording_start_time", "source_recording_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(recordings) < 3:
        raise ValueError(
            f"Scene {scene['scene_id'].iloc[0]} has {len(recordings)} recording blocks; at least 3 are required"
        )
    cumulative = recordings["n_trajectories"].cumsum().to_numpy()
    total = int(cumulative[-1])
    target_first = SPLIT_PROPORTIONS["target_estimation"] * total
    target_second = (
        SPLIT_PROPORTIONS["target_estimation"] + SPLIT_PROPORTIONS["model_selection"]
    ) * total
    candidates: list[tuple[float, float, int, int]] = []
    for first_end in range(0, len(recordings) - 2):
        for second_end in range(first_end + 1, len(recordings) - 1):
            first_error = abs(float(cumulative[first_end]) - target_first)
            second_error = abs(float(cumulative[second_end]) - target_second)
            candidates.append((first_error + second_error, max(first_error, second_error), first_end, second_end))
    _, _, first_end, second_end = min(candidates)
    return first_end, second_end


def _recording_assignments(scene: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    recordings = (
        scene.groupby(["source_recording_start_time", "source_recording_id"], sort=True)
        .size()
        .reset_index(name="n_trajectories")
        .sort_values(["source_recording_start_time", "source_recording_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    first_end, second_end = choose_recording_boundaries(scene)
    assignments: dict[str, str] = {}
    split_values: list[str] = []
    for index, recording_id in enumerate(recordings["source_recording_id"]):
        if index <= first_end:
            split = "target_estimation"
        elif index <= second_end:
            split = "model_selection"
        else:
            split = "independent_test"
        assignments[str(recording_id)] = split
        split_values.append(split)
    recordings["split"] = split_values
    return assignments, recordings


def build_split(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if tuple(manifest["scene_id"].drop_duplicates()) != SCENES:
        raise ValueError("Manifest scene order or membership differs from the required five scenes")
    if not manifest["validity_under_existing_preprocessing_pipeline"].astype(bool).all():
        raise ValueError("The canonical split input must contain only final pipeline-valid trajectories")

    result = manifest.copy()
    result["split"] = ""
    result["split_assignment_basis"] = "complete_source_recording_temporal_block"
    result["split_adjustment_reason"] = ""
    recording_tables: dict[str, pd.DataFrame] = {}
    for scene_id in SCENES:
        scene_mask = result["scene_id"] == scene_id
        assignments, recordings = _recording_assignments(result.loc[scene_mask])
        result.loc[scene_mask, "split"] = result.loc[scene_mask, "source_recording_id"].map(assignments)
        recording_tables[scene_id] = recordings

    # Exact copies must never cross subsets. When a duplicate group straddles a
    # temporal boundary, all members are conservatively moved to its earliest split.
    rank = {name: index for index, name in enumerate(SPLIT_NAMES)}
    duplicate_rows = result[result["exact_duplicate_count"] > 1]
    for fingerprint, group in duplicate_rows.groupby("data_fingerprint", sort=False):
        splits = group["split"].drop_duplicates().tolist()
        if len(splits) <= 1:
            continue
        chosen = min(splits, key=rank.__getitem__)
        indexes = group.index
        result.loc[indexes, "split"] = chosen
        result.loc[indexes, "split_assignment_basis"] = "exact_duplicate_group"
        result.loc[indexes, "split_adjustment_reason"] = (
            f"duplicate_group_{fingerprint[:12]}_kept_in_earliest_split"
        )

    result.insert(0, "split_protocol_version", PROTOCOL_VERSION)
    return result.loc[:, SPLIT_COLUMNS].reset_index(drop=True), recording_tables


def _split_config(
    repo_root: Path,
    manifest_path: Path,
    split: pd.DataFrame,
    recordings: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    boundaries: dict[str, Any] = {}
    for scene_id in SCENES:
        scene_rows = split[split["scene_id"] == scene_id]
        table = recordings[scene_id]
        subset_rows = {}
        for split_name in SPLIT_NAMES:
            subset = scene_rows[scene_rows["split"] == split_name]
            subset_recordings = table[table["split"] == split_name]
            subset_rows[split_name] = {
                "recording_ids": subset_recordings["source_recording_id"].astype(str).tolist(),
                "recording_start": str(subset_recordings["source_recording_start_time"].iloc[0]),
                "recording_end": str(subset_recordings["source_recording_start_time"].iloc[-1]),
                "boundary_after_recording_id": str(subset_recordings["source_recording_id"].iloc[-1]),
                "n_recordings": int(len(subset_recordings)),
                "n_trajectories": int(len(subset)),
                "proportion": float(len(subset) / len(scene_rows)),
            }
        boundaries[scene_id] = subset_rows

    source_checksums = {
        scene_id: {
            "path": str(split.loc[split["scene_id"] == scene_id, "trajectory_id"].iloc[0]).split(":")[0],
            "sha256": "",
        }
        for scene_id in SCENES
    }
    manifest = pd.read_csv(manifest_path, usecols=["scene_id", "source_file", "source_file_checksum_sha256"])
    for scene_id in SCENES:
        row = manifest[manifest["scene_id"] == scene_id].iloc[0]
        source_checksums[scene_id] = {
            "path": str(row["source_file"]),
            "sha256": str(row["source_file_checksum_sha256"]),
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scene_list": list(SCENES),
        "canonical_trajectory_cohort": (
            "rows in feature_analysis/features_trimmed_frame_disp_norm.parquet; "
            "one row per pipeline-valid trajectory"
        ),
        "split_proportions": SPLIT_PROPORTIONS,
        "split_strategy": (
            "chronological complete-recording blocks; earliest target_estimation, "
            "middle model_selection, latest independent_test"
        ),
        "boundary_optimization": (
            "choose two complete-recording boundaries minimizing deviation from cumulative 30% and 60%"
        ),
        "temporal_boundaries": boundaries,
        "random_seed": RANDOM_SEED,
        "random_seed_usage": "recorded for protocol compatibility; not used by this deterministic split",
        "duplicate_policy": (
            "exact fingerprint groups are kept in one subset; a group crossing a provisional temporal "
            "boundary is assigned to its earliest provisional subset"
        ),
        "near_duplicate_policy": "report candidates; do not remove or reassign automatically",
        "missing_timestamp_fallback": (
            "stop with an error; do not silently randomize. A future explicit fallback may order by "
            "recording ID and frame range only after documented review"
        ),
        "manual_label_policy": (
            "manual labels are forbidden during homography-guided target estimation and clustering "
            "model selection; they are reserved for independent final evaluation and explicitly "
            "identified post hoc diagnostics"
        ),
        "manifest_path": manifest_path.resolve().relative_to(repo_root.resolve()).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "source_data_checksums": source_checksums,
    }


def _cross_split_group_count(df: pd.DataFrame, group_column: str) -> int:
    groups = df[df[group_column].astype(str) != ""].groupby(group_column)["split"].nunique()
    return int((groups > 1).sum())


def markdown_table(df: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(render(row[column]) for column in df.columns) + " |")
    return "\n".join(lines)


def build_split_report(split: pd.DataFrame, config: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for scene_id in SCENES:
        scene = split[split["scene_id"] == scene_id]
        for split_name in SPLIT_NAMES:
            subset = scene[scene["split"] == split_name]
            lengths = subset["number_of_points"]
            rows.append(
                {
                    "scene": scene_id,
                    "subset": split_name,
                    "n": len(subset),
                    "percent": 100.0 * len(subset) / len(scene),
                    "recordings": subset["source_recording_id"].nunique(),
                    "time_start": subset["start_time"].min(),
                    "time_end": subset["end_time"].max(),
                    "frame_min": subset["start_frame"].min(),
                    "frame_max": subset["end_frame"].max(),
                    "length_median": lengths.median(),
                    "length_min": lengths.min(),
                    "length_max": lengths.max(),
                    "duplicate_rows": int((subset["exact_duplicate_count"] > 1).sum()),
                    "missing_id": int(subset["trajectory_id"].isna().sum() + (subset["trajectory_id"] == "").sum()),
                    "missing_time_or_frame": int(
                        subset[["start_time", "end_time", "start_frame", "end_frame"]].isna().any(axis=1).sum()
                    ),
                    "filtering_status": "; ".join(
                        f"{key}={value}"
                        for key, value in subset["filtering_status"].value_counts().items()
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    display = summary.copy()
    display["percent"] = display["percent"].map(lambda value: f"{value:.2f}")
    display["length_median"] = display["length_median"].map(lambda value: f"{value:.1f}")
    table = markdown_table(display)

    total_by_split = (
        split.groupby("split", sort=False)
        .agg(n=("trajectory_id", "size"), scenes=("scene_id", "nunique"))
        .reindex(SPLIT_NAMES)
        .reset_index()
    )
    total_by_split["percent"] = 100.0 * total_by_split["n"] / len(split)
    total_by_split["percent"] = total_by_split["percent"].map(lambda value: f"{value:.2f}")
    total_table = markdown_table(total_by_split)
    exact_cross = _cross_split_group_count(split, "exact_duplicate_group")
    near_cross = _cross_split_group_count(split, "near_duplicate_group")
    adjusted = int((split["split_adjustment_reason"] != "").sum())

    lines = [
        "# Leakage-Free Evaluation Split Report",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`.",
        "",
        "## Protocol",
        "",
        "The split uses complete source-recording blocks in chronological order for each scene. "
        "The earliest block is reserved for homography-guided target estimation, the middle block "
        "for candidate/model selection, and the latest block for independent testing. Frame IDs "
        "reset in each hourly recording, so frame-only ordering would not be temporally valid.",
        "",
        "The requested 30/30/40 proportions are targets rather than row-level cut points. Boundaries "
        "are selected jointly to minimize count deviation while keeping recordings intact.",
        "",
        "## Scene and Subset Counts",
        "",
        table,
        "",
        "## Totals",
        "",
        total_table,
        "",
        "## Leakage Checks",
        "",
        f"- Unique trajectory IDs: `{split['trajectory_id'].nunique():,}` of `{len(split):,}` rows.",
        f"- Exact fingerprint groups crossing subsets: `{exact_cross}`.",
        f"- Rows reassigned to keep an exact duplicate group together: `{adjusted}`.",
        f"- Approximate near-duplicate candidate groups crossing subsets: `{near_cross}`.",
        f"- Missing trajectory IDs: `{int(split['trajectory_id'].isna().sum()):,}`.",
        f"- Missing time/frame rows: `{int(split[['start_time','end_time','start_frame','end_frame']].isna().any(axis=1).sum()):,}`.",
        "",
        "## Filtering Status",
        "",
        "The canonical cohort consists only of rows retained in "
        "`features_trimmed_frame_disp_norm.parquet`. Consequently every split row is marked "
        "pipeline-valid with the same final filtering status. Earlier rejected trajectories are "
        "outside this evaluation cohort and are summarized in the repository audit.",
        "",
        "## Imbalance and Boundary Effects",
        "",
        "Count deviations from 30/30/40 reflect preservation of complete hourly recording blocks. "
        "This is preferable to cutting a recording or randomly mixing trajectories from the same "
        "time period across protocol stages. All five scenes occur in all three subsets.",
        "",
        "## Issues Before Manual Annotation",
        "",
        "- Exact vehicle subclass is unavailable in the final processed trajectory schema; the "
        "upstream class filter retained car, bus, and truck detections but discarded class metadata.",
        "- Source recording provenance is reconstructed from the verified `merge_tracks.py` ID-offset "
        "rule because the merged trajectory table does not retain `video_id`.",
        "- Recording timestamps are inferred from filenames at 30 fps and are local wall-clock values; "
        "the repository contains no explicit timezone field.",
        "- Approximate near-duplicate candidates must be reviewed if any cross subsets; they are not "
        "automatically removed because geometric similarity can represent distinct vehicles.",
        "- Annotation must begin only after freezing this versioned split and checksum set.",
        "",
        "## Manual-Label Isolation",
        "",
        "The manually annotated labels are isolated from homography-guided target estimation and "
        "clustering model selection. They are used only for independent final evaluation and "
        "explicitly identified post hoc diagnostic analyses.",
        "",
        "The `independent_test` labels must remain inaccessible to all target estimation, hyperparameter "
        "search, model selection, threshold setting, and stopping decisions.",
        "",
        "## Configuration Trace",
        "",
        f"- Manifest checksum: `{config['manifest_sha256']}`.",
        f"- Random seed recorded but unused by splitting: `{config['random_seed']}`.",
        "- Full recording lists and temporal boundaries are stored in `configs/evaluation_split.yaml`.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    repo_root: Path,
    manifest_path: Path,
    split: pd.DataFrame,
    recordings: dict[str, pd.DataFrame],
    paths: dict[str, Path],
) -> dict[str, Any]:
    paths["split"].parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(paths["split"], index=False, lineterminator="\n")

    config = _split_config(repo_root, manifest_path, split, recordings)
    paths["config"].parent.mkdir(parents=True, exist_ok=True)
    with paths["config"].open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False, width=100)

    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(build_split_report(split, config), encoding="utf-8", newline="\n")

    annotation = split[["scene_id", "trajectory_id", "split"]].copy()
    for column in ANNOTATION_COLUMNS[3:]:
        annotation[column] = ""
    paths["annotation"].parent.mkdir(parents=True, exist_ok=True)
    annotation.loc[:, ANNOTATION_COLUMNS].to_csv(
        paths["annotation"], index=False, lineterminator="\n"
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--annotation-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    paths = default_paths(repo_root)
    overrides = {
        "manifest": args.manifest,
        "split": args.split_output,
        "config": args.config_output,
        "report": args.report_output,
        "annotation": args.annotation_output,
    }
    for key, value in overrides.items():
        if value is not None:
            paths[key] = value.resolve()
    if not paths["manifest"].exists():
        raise FileNotFoundError(
            f"Manifest not found: {paths['manifest']}. Run build_trajectory_manifest.py first."
        )

    manifest = pd.read_csv(paths["manifest"], keep_default_na=False)
    split, recordings = build_split(manifest)
    write_outputs(repo_root, paths["manifest"], split, recordings, paths)
    print(f"Wrote {len(split):,} split assignments to {paths['split']}")
    counts = split.groupby(["scene_id", "split"], sort=False).size().unstack(fill_value=0)
    print(counts.to_string())
    print(f"Split checksum: {hashlib.sha256(paths['split'].read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
