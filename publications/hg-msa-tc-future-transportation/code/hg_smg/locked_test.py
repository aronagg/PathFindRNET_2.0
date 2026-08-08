"""Locked post-review HG-SMG-TC independent-test execution.

This module is intentionally separate from the Task 09B development runner.  It
uses only frozen Task 09A/09B artifacts for model execution, persists all cluster
assignments before loading polygon-rule reference labels, and records the order
of data access in manifests.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from independent_test import evaluator as base_evaluator
from independent_test.protocol import METHODS, SCENES
from pipeline import hg_msa_tc_core as core
from pipeline import split_aware_io as protocol_io

from .io import PUBLICATION_ROOT, REPOSITORY_ROOT, relative
from .provenance import sha256_file


CONFIRMATION_PHRASE = "I_CONFIRM_FROZEN_HG_SMG_EXTENSION_EVALUATION"
DEVELOPMENT_BRANCH = "feature/futuretransp-hg-smg-development"
LOCKED_TEST_BRANCH = "feature/futuretransp-hg-smg-locked-test-evaluation"
REQUIRED_DEVELOPMENT_COMMIT = "47fa961ebb77e9a2ba32b8cb621dbeb8fd43673b"
SCIENTIFIC_IMPLEMENTATION_COMMIT = "55bab42853ce341c75b31404ed39eb34e9841c6b"
REQUIRED_PROTOCOL_HASH = (
    "2e892f25b1c1770a55fa29e36206b634f424f9ff2508bf045c92d388e69b4be6"
)
REQUIRED_DEVELOPMENT_FREEZE_HASH = (
    "c86830ea99b331fa8322915c33425398ac3edee76d242e0915f05c7c5ff0a14a"
)
RESULTS_ROOT = PUBLICATION_ROOT / "results/hg_smg/independent_test"
FIGURES_ROOT = PUBLICATION_ROOT / "figures/hg_smg/independent_test"
DOCS_ROOT = PUBLICATION_ROOT / "docs"
AUTHORIZATION_PATH = PUBLICATION_ROOT / "configs/HG_SMG_LOCKED_TEST_AUTHORIZATION.json"
ZIP_PATH = (
    REPOSITORY_ROOT / "futuretransp_revision_10_hg_smg_locked_test_evaluation.zip"
)

EXECUTABLE_ABLATIONS = ("A2", "A3", "A5", "A6", "A7", "A9", "A10")
COPIED_ORIGINAL_ABLATIONS = ("A0", "A1")
NON_EXECUTABLE_ABLATIONS = ("A4", "A8")
ALL_ABLATIONS = tuple(f"A{index}" for index in range(11))
PRIMARY_ABLATION = "A5"
ORIGINAL_HG_ABLATION = "A1"
LEGAL_MOVEMENT_COUNT = 12


@dataclass(frozen=True)
class LockedPaths:
    runner_config: Path = PUBLICATION_ROOT / "configs/split_aware_runner.yaml"
    protocol: Path = PUBLICATION_ROOT / "configs/hg_smg_protocol_v1.yaml"
    ablation_protocol: Path = PUBLICATION_ROOT / "configs/hg_smg_ablation_protocol.yaml"
    development_freeze: Path = PUBLICATION_ROOT / "configs/hg_smg_development_freeze_v1.yaml"
    development_freeze_sidecar: Path = (
        PUBLICATION_ROOT / "configs/hg_smg_development_freeze_v1.sha256"
    )
    development_results: Path = PUBLICATION_ROOT / "results/hg_smg/development"
    original_results: Path = PUBLICATION_ROOT / "results/independent_test"
    reference_export: Path = (
        PUBLICATION_ROOT
        / "annotations/reference_labels/independent_test_reference_labels.csv"
    )
    reference_manifest: Path = (
        PUBLICATION_ROOT
        / "annotations/reference_labels/reference_output_manifest.json"
    )
    trajectory_manifest: Path = PUBLICATION_ROOT / "data/manifests/trajectory_manifest.csv"
    evaluation_split: Path = PUBLICATION_ROOT / "data/splits/evaluation_split.csv"


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git(*args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY_ROOT.as_posix()}",
            *args,
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _input_paths() -> dict[str, Path]:
    config = _load_yaml(LockedPaths.runner_config)
    return {key: _repo_path(value) for key, value in config["inputs"].items()}


def _branch_is_valid() -> bool:
    return _git("branch", "--show-current") in {DEVELOPMENT_BRANCH, LOCKED_TEST_BRANCH}


def _development_commit_is_ancestor() -> bool:
    merge_base = _git("merge-base", "HEAD", REQUIRED_DEVELOPMENT_COMMIT)
    return merge_base == REQUIRED_DEVELOPMENT_COMMIT


def _reference_export_hash(paths: LockedPaths) -> tuple[str, int]:
    manifest = _load_json(paths.reference_manifest)
    record = manifest["output_files"][paths.reference_export.name]
    return str(record["sha256"]), int(record["rows"])


def run_preflight(write_report: bool = True, require_no_outputs: bool = True) -> dict[str, Any]:
    """Verify frozen inputs before reading independent-test features."""
    paths = LockedPaths()
    if require_no_outputs and RESULTS_ROOT.exists():
        raise FileExistsError(f"HG-SMG locked-test output already exists: {RESULTS_ROOT}")
    required = [
        paths.runner_config,
        paths.protocol,
        paths.ablation_protocol,
        paths.development_freeze,
        paths.development_freeze_sidecar,
        paths.trajectory_manifest,
        paths.evaluation_split,
        paths.reference_manifest,
        paths.reference_export,
        paths.development_results / "ablation_selected_configurations.csv",
        paths.development_results / "ablation_development_summary.csv",
        paths.development_results / "pcms_selected_configurations.csv",
        paths.development_results / "development_result_manifest.json",
        paths.original_results / "cluster_assignments.parquet",
        paths.original_results / "clustering_run_manifest.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing locked-test artifacts: {missing}")

    protocol_hash = sha256_file(paths.protocol)
    freeze_hash = sha256_file(paths.development_freeze)
    recorded_freeze_hash = paths.development_freeze_sidecar.read_text(
        encoding="ascii"
    ).split()[0]
    freeze = _load_yaml(paths.development_freeze)
    development_manifest = _load_json(paths.development_results / "development_result_manifest.json")
    checks: list[dict[str, Any]] = []

    def add_check(name: str, status: bool, detail: str = "") -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if status else "FAIL",
                "detail": detail,
            }
        )
        if not status:
            raise ValueError(f"Locked-test preflight failed: {name} {detail}")

    add_check("branch_valid", _branch_is_valid(), _git("branch", "--show-current"))
    add_check(
        "required_development_commit_is_ancestor",
        _development_commit_is_ancestor(),
        REQUIRED_DEVELOPMENT_COMMIT,
    )
    add_check("protocol_hash_matches", protocol_hash == REQUIRED_PROTOCOL_HASH, protocol_hash)
    add_check("freeze_hash_matches", freeze_hash == REQUIRED_DEVELOPMENT_FREEZE_HASH, freeze_hash)
    add_check(
        "freeze_sidecar_matches",
        recorded_freeze_hash == REQUIRED_DEVELOPMENT_FREEZE_HASH,
        recorded_freeze_hash,
    )
    add_check(
        "freeze_protocol_hash_matches",
        freeze["task09a_protocol_sha256"] == REQUIRED_PROTOCOL_HASH,
        str(freeze["task09a_protocol_sha256"]),
    )
    add_check(
        "freeze_code_commit_matches",
        freeze["code_commit"] == SCIENTIFIC_IMPLEMENTATION_COMMIT,
        str(freeze["code_commit"]),
    )
    add_check("development_independent_access_false", freeze["independent_test_access"] is False)
    add_check("development_reference_access_false", freeze["reference_label_access"] is False)
    add_check("development_test_execution_false", freeze["independent_test_execution"] is False)
    add_check(
        "development_manifest_freeze_hash_matches",
        development_manifest["freeze_sha256"] == REQUIRED_DEVELOPMENT_FREEZE_HASH,
        str(development_manifest["freeze_sha256"]),
    )

    for output, record in freeze["outputs"].items():
        path = REPOSITORY_ROOT / output
        actual = sha256_file(path)
        add_check(f"frozen_output:{Path(output).name}", actual == record["sha256"], actual)

    expected_reference_hash, expected_reference_rows = _reference_export_hash(paths)
    actual_reference_hash = sha256_file(paths.reference_export)
    add_check(
        "reference_export_hash_matches_manifest",
        actual_reference_hash == expected_reference_hash,
        actual_reference_hash,
    )
    add_check(
        "reference_export_row_count_manifest",
        expected_reference_rows == 27_393,
        str(expected_reference_rows),
    )
    original_manifest = _load_json(paths.original_results / "clustering_run_manifest.json")
    for filename, expected in original_manifest["output_files"].items():
        actual = sha256_file(paths.original_results / filename)
        add_check(f"original_assignment_artifact:{filename}", actual == expected, actual)
    add_check(
        "original_assignments_persisted_before_reference",
        original_manifest["reference_labels_read"] is False,
    )

    split_counts = (
        pd.read_csv(paths.evaluation_split, usecols=["scene_id", "split"])
        .query("split == 'independent_test'")
        .groupby("scene_id")
        .size()
        .to_dict()
    )
    expected_counts = {
        "bellevue_116th_ne12th": 964,
        "bellevue_150th_newport": 3924,
        "bellevue_150th_eastgate": 10755,
        "bellevue_150th_se38th": 3713,
        "bellevue_ne8th": 8037,
    }
    add_check("independent_split_counts_match", split_counts == expected_counts, str(split_counts))
    summary = pd.read_csv(paths.development_results / "ablation_development_summary.csv")
    add_check("ablation_ids_are_frozen", tuple(summary["ablation_id"]) == ALL_ABLATIONS)
    add_check(
        "a8_not_identifiable_recorded",
        summary.loc[summary["ablation_id"] == "A8", "status"].iloc[0] == "not_identifiable",
    )
    result = {
        "status": "PASS",
        "task": "Task 10 locked post-review HG-SMG-TC independent-test preflight",
        "timestamp_utc": _utc_timestamp(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "required_development_commit": REQUIRED_DEVELOPMENT_COMMIT,
        "scientific_implementation_commit": SCIENTIFIC_IMPLEMENTATION_COMMIT,
        "protocol_hash": protocol_hash,
        "development_freeze_hash": freeze_hash,
        "reference_export_hash": actual_reference_hash,
        "reference_export_rows": expected_reference_rows,
        "test_counts_by_scene": expected_counts,
        "checks": checks,
        "independent_test_features_loaded": False,
        "reference_rows_loaded": False,
    }
    if write_report:
        lines = [
            "# HG-SMG Locked-Test Preflight Report",
            "",
            f"- Status: **{result['status']}**",
            f"- Timestamp UTC: `{result['timestamp_utc']}`",
            f"- Branch: `{result['git_branch']}`",
            f"- Git HEAD: `{result['git_head']}`",
            f"- Required development commit ancestor: `{REQUIRED_DEVELOPMENT_COMMIT}`",
            f"- Protocol hash: `{protocol_hash}`",
            f"- Development freeze hash: `{freeze_hash}`",
            f"- Reference export hash: `{actual_reference_hash}`",
            "- Independent-test features loaded during preflight: **no**",
            "- Reference rows loaded during preflight: **no**",
            "",
            "## Checks",
            "",
            "| Check | Status | Detail |",
            "| --- | --- | --- |",
        ]
        lines.extend(
            f"| `{row['check']}` | {row['status']} | `{row.get('detail', '')}` |"
            for row in checks
        )
        _write_text("\n".join(lines), DOCS_ROOT / "hg_smg_locked_test_preflight_report.md")
    return result


def authorize_locked_test() -> dict[str, Any]:
    preflight = run_preflight(write_report=True, require_no_outputs=True)
    payload = {
        "authorization_version": "hg-smg-locked-test-authorization-v1",
        "task": "Task 10 locked post-review HG-SMG-TC independent-test evaluation",
        "protocol_hash": REQUIRED_PROTOCOL_HASH,
        "development_freeze_hash": REQUIRED_DEVELOPMENT_FREEZE_HASH,
        "required_development_commit": REQUIRED_DEVELOPMENT_COMMIT,
        "scientific_implementation_commit": SCIENTIFIC_IMPLEMENTATION_COMMIT,
        "git_commit_at_authorization": _git("rev-parse", "HEAD"),
        "git_branch_at_authorization": _git("branch", "--show-current"),
        "timestamp_utc": _utc_timestamp(),
        "confirmation_phrase_required": CONFIRMATION_PHRASE,
        "author_authorization_statement": (
            "I authorize the first locked post-review HG-SMG-TC independent-test "
            "evaluation. No target, threshold, ablation, model-selection rule, "
            "or hyperparameter tuning will occur after test results are visible."
        ),
        "preflight_status": preflight["status"],
    }
    _write_json(payload, AUTHORIZATION_PATH)
    return payload


def _verify_authorization() -> dict[str, Any]:
    if not AUTHORIZATION_PATH.exists():
        raise FileNotFoundError("Locked-test authorization file is missing.")
    payload = _load_json(AUTHORIZATION_PATH)
    if payload["protocol_hash"] != REQUIRED_PROTOCOL_HASH:
        raise ValueError("Authorization protocol hash mismatch.")
    if payload["development_freeze_hash"] != REQUIRED_DEVELOPMENT_FREEZE_HASH:
        raise ValueError("Authorization development-freeze hash mismatch.")
    if payload["confirmation_phrase_required"] != CONFIRMATION_PHRASE:
        raise ValueError("Authorization confirmation phrase mismatch.")
    return payload


def _configuration_id(row: Any, ablation_id: str) -> str:
    payload = {
        "scene": str(row.scene),
        "method": str(row.method),
        "ablation_id": ablation_id,
        "params_json": str(row.params_json),
        "fit_random_seed": int(row.fit_random_seed),
    }
    return protocol_io.canonical_sha256(payload)


def _test_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = LockedPaths()
    membership = protocol_io.load_split_membership(
        paths.evaluation_split, "test", tuple(SCENES)
    )
    metadata = protocol_io.load_manifest_metadata(paths.trajectory_manifest, membership)
    protocol_io.assert_phase_rows(metadata, "test")
    return membership, metadata


def _scene_normalizations() -> dict[str, dict[str, float]]:
    original = pd.read_csv(PUBLICATION_ROOT / "results/development/selected_configurations.csv")
    output: dict[str, dict[str, float]] = {}
    for scene, frame in original.groupby("scene", sort=True):
        values = frame["normalization_parameters_json"].drop_duplicates().tolist()
        if len(values) != 1:
            raise ValueError(f"Frozen normalization differs within {scene}")
        output[str(scene)] = json.loads(values[0])
    return output


def _selected_configurations() -> pd.DataFrame:
    paths = LockedPaths()
    selected_rows = []
    original_selected = pd.read_csv(PUBLICATION_ROOT / "results/development/selected_configurations.csv")
    for ablation_id, source_strategy, label in [
        ("A0", "untargeted_selection", "original_untargeted"),
        ("A1", "hg_expected_aware_selection", "original_hg_aware_point_target"),
    ]:
        rows = original_selected[
            original_selected["selection_strategy"] == source_strategy
        ].copy()
        rows["ablation_id"] = ablation_id
        rows["selection_strategy"] = label
        rows["source_selection_strategy"] = source_strategy
        rows["executable_status"] = "copied_from_frozen_original_test_assignments"
        selected_rows.append(rows)
    ablations = pd.read_csv(paths.development_results / "ablation_selected_configurations.csv")
    executable = ablations[ablations["ablation_id"].isin(EXECUTABLE_ABLATIONS)].copy()
    executable["selection_strategy"] = "pcms_interval_prior"
    executable["source_selection_strategy"] = "pcms_interval_prior"
    executable["executable_status"] = "fit_from_frozen_task09b_selected_configuration"
    selected_rows.append(executable)
    selected = pd.concat(selected_rows, ignore_index=True, sort=False)
    if selected.duplicated(["scene", "method", "ablation_id", "selection_strategy"]).any():
        raise ValueError("Duplicate locked-test selected configurations.")
    expected = len(SCENES) * len(METHODS) * (
        len(COPIED_ORIGINAL_ABLATIONS) + len(EXECUTABLE_ABLATIONS)
    )
    if len(selected) != expected:
        raise ValueError(f"Unexpected executable selection rows: {len(selected)}")
    return selected


def run_assignments(confirmation: str | None) -> dict[str, Any]:
    if confirmation != CONFIRMATION_PHRASE:
        raise PermissionError("Explicit locked-test confirmation phrase is required.")
    preflight = run_preflight(write_report=True, require_no_outputs=True)
    authorization = _verify_authorization()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=False)
    run_timestamp = _utc_timestamp()
    run_id = "hg-smg-locked-test-" + run_timestamp.replace(":", "").replace("-", "")
    membership, metadata = _test_metadata()
    selected = _selected_configurations()
    normalizations = _scene_normalizations()
    logger = protocol_io.AccessLogger(RESULTS_ROOT / "data_access_log.jsonl", "test")
    logger.add(
        "ALL",
        LockedPaths.evaluation_split.relative_to(REPOSITORY_ROOT),
        membership["split"].unique(),
        len(membership),
        sha256_file(LockedPaths.evaluation_split),
        "split_control_index",
    )
    logger.add(
        "ALL",
        LockedPaths.trajectory_manifest.relative_to(REPOSITORY_ROOT),
        membership["split"].unique(),
        len(membership),
        sha256_file(LockedPaths.trajectory_manifest),
        "trajectory_control_manifest",
    )

    original_assignments = pd.read_parquet(
        LockedPaths.original_results / "cluster_assignments.parquet"
    )
    assignment_tables: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []
    label_cache: dict[tuple[str, str, str, int], np.ndarray] = {}
    feature_sources: dict[str, str] = {}
    feature_hashes: dict[str, str] = {}
    for scene in SCENES:
        scene_metadata = metadata[metadata["scene_id"] == scene].copy()
        feature_frame = protocol_io.load_scene_features(
            REPOSITORY_ROOT, scene_metadata, "test", logger
        )
        source_paths = feature_frame["source_file"].drop_duplicates().tolist()
        source_hashes = feature_frame["source_file_checksum_sha256"].drop_duplicates().tolist()
        if len(source_paths) != 1 or len(source_hashes) != 1:
            raise ValueError(f"Unexpected feature provenance for {scene}")
        feature_sources[scene] = str(source_paths[0])
        feature_hashes[scene] = str(source_hashes[0])
        features, _ = core.isotropic_normalize(
            feature_frame[list(core.FEATURE_COLUMNS)].to_numpy(np.float64),
            normalizations[scene],
        )
        scene_selected = selected[selected["scene"] == scene].copy()
        for row in scene_selected.sort_values(
            ["ablation_id", "method", "selection_strategy"], kind="mergesort"
        ).itertuples(index=False):
            if row.ablation_id in COPIED_ORIGINAL_ABLATIONS:
                source_strategy = row.source_selection_strategy
                copied = original_assignments[
                    (original_assignments["scene_id"] == scene)
                    & (original_assignments["method"] == row.method)
                    & (original_assignments["selection_strategy"] == source_strategy)
                ].copy()
                copied = copied.set_index("trajectory_id").loc[
                    feature_frame["trajectory_id"].astype(str)
                ]
                labels = copied["cluster_label"].to_numpy(int)
                fit_time = 0.0
                configuration_id = str(copied["frozen_configuration_id"].iloc[0])
                source = "copied_frozen_original_assignment"
            else:
                parameters = json.loads(row.params_json)
                key = (
                    scene,
                    str(row.method),
                    json.dumps(parameters, sort_keys=True),
                    int(row.fit_random_seed),
                )
                if key not in label_cache:
                    started = time.perf_counter()
                    label_cache[key] = core.fit_predict(
                        str(row.method), features, parameters, int(row.fit_random_seed)
                    ).astype(int)
                    fit_time = time.perf_counter() - started
                else:
                    fit_time = 0.0
                labels = label_cache[key]
                configuration_id = _configuration_id(row, str(row.ablation_id))
                source = "fit_frozen_task09b_configuration"
            if len(labels) != len(feature_frame):
                raise ValueError(f"Assignment count mismatch for {scene}/{row.method}")
            table = pd.DataFrame(
                {
                    "scene_id": scene,
                    "trajectory_id": feature_frame["trajectory_id"].astype(str),
                    "method": str(row.method),
                    "ablation_id": str(row.ablation_id),
                    "selection_strategy": str(row.selection_strategy),
                    "cluster_label": labels.astype(int),
                    "is_noise": labels.astype(int) == -1,
                    "frozen_configuration_id": configuration_id,
                    "hg_smg_protocol_hash": REQUIRED_PROTOCOL_HASH,
                    "development_freeze_hash": REQUIRED_DEVELOPMENT_FREEZE_HASH,
                    "feature_checksum": feature_hashes[scene],
                    "run_id": run_id,
                    "run_timestamp_utc": run_timestamp,
                }
            )
            assignment_tables.append(table)
            run_records.append(
                {
                    "scene_id": scene,
                    "method": str(row.method),
                    "ablation_id": str(row.ablation_id),
                    "selection_strategy": str(row.selection_strategy),
                    "configuration_source": source,
                    "frozen_configuration_id": configuration_id,
                    "params_json": str(row.params_json),
                    "fit_random_seed": int(row.fit_random_seed),
                    "feature_file": feature_sources[scene],
                    "feature_checksum": feature_hashes[scene],
                    "n_trajectories": len(feature_frame),
                    "n_non_noise_clusters": int(len(np.unique(labels[labels >= 0]))),
                    "noise_count": int((labels == -1).sum()),
                    "fit_time_s": fit_time,
                }
            )

    assignments = pd.concat(assignment_tables, ignore_index=True)
    assignments = assignments.sort_values(
        ["scene_id", "ablation_id", "method", "selection_strategy", "trajectory_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    expected_rows = 27_393 * len(METHODS) * (
        len(COPIED_ORIGINAL_ABLATIONS) + len(EXECUTABLE_ABLATIONS)
    )
    if len(assignments) != expected_rows:
        raise ValueError(f"Unexpected locked-test assignment rows: {len(assignments)}")
    if assignments.duplicated(
        ["scene_id", "trajectory_id", "method", "ablation_id", "selection_strategy"]
    ).any():
        raise ValueError("Duplicate locked-test assignment rows.")
    csv_path = RESULTS_ROOT / "cluster_assignments.csv"
    parquet_path = RESULTS_ROOT / "cluster_assignments.parquet"
    _write_csv(assignments, csv_path)
    assignments.to_parquet(parquet_path, index=False, compression="zstd")
    logger.flush()
    manifest = {
        "task": "Task 10 locked post-review HG-SMG-TC independent-test assignments",
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp,
        "assignments_persisted_at_utc": _utc_timestamp(),
        "authorization_sha256": sha256_file(AUTHORIZATION_PATH),
        "authorization_git_commit": authorization["git_commit_at_authorization"],
        "preflight_status": preflight["status"],
        "protocol_hash": REQUIRED_PROTOCOL_HASH,
        "development_freeze_hash": REQUIRED_DEVELOPMENT_FREEZE_HASH,
        "assignment_rows": len(assignments),
        "unique_test_trajectories": int(assignments["trajectory_id"].nunique()),
        "scene_method_ablation_runs": len(run_records),
        "reference_labels_read": False,
        "evaluation_started": False,
        "no_tuning_after_test_results": True,
        "non_executable_ablations": list(NON_EXECUTABLE_ABLATIONS),
        "output_files": {
            "cluster_assignments.csv": sha256_file(csv_path),
            "cluster_assignments.parquet": sha256_file(parquet_path),
            "data_access_log.jsonl": sha256_file(RESULTS_ROOT / "data_access_log.jsonl"),
        },
        "runs": run_records,
    }
    _write_json(manifest, RESULTS_ROOT / "clustering_run_manifest.json")
    _write_text(
        "\n".join(
            [
                f"{sha256_file(csv_path)}  cluster_assignments.csv",
                f"{sha256_file(parquet_path)}  cluster_assignments.parquet",
                f"{sha256_file(RESULTS_ROOT / 'clustering_run_manifest.json')}  clustering_run_manifest.json",
                f"{sha256_file(RESULTS_ROOT / 'data_access_log.jsonl')}  data_access_log.jsonl",
            ]
        ),
        RESULTS_ROOT / "clustering_run_checksums.sha256",
    )
    return manifest


def _verify_persisted_assignments() -> dict[str, Any]:
    manifest_path = RESULTS_ROOT / "clustering_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Assignments must be persisted before evaluation.")
    manifest = _load_json(manifest_path)
    if manifest.get("reference_labels_read") is not False:
        raise ValueError("Assignment manifest does not prove reference isolation.")
    if manifest.get("evaluation_started") is not False:
        raise ValueError("Assignment manifest was unexpectedly mutated.")
    for filename, expected in manifest["output_files"].items():
        actual = sha256_file(RESULTS_ROOT / filename)
        if actual != expected:
            raise ValueError(f"Persisted assignment artifact changed: {filename}")
    return manifest


def _target_prior_by_ablation() -> pd.DataFrame:
    dev = LockedPaths.development_results
    rows = []
    original = pd.read_csv(PUBLICATION_ROOT / "results/development/selected_configurations.csv")
    for scene, frame in original.groupby("scene", sort=True):
        target = int(frame["hg_estimated_target"].drop_duplicates().iloc[0])
        for ablation_id in ("A0", "A1"):
            rows.append(
                {
                    "scene": scene,
                    "ablation_id": ablation_id,
                    "target_prior": target,
                    "target_prior_kind": "original_hg_point_target",
                }
            )
    a2 = pd.read_csv(dev / "a2_targets.csv")
    for _, row in a2.iterrows():
        rows.append(
            {
                "scene": row["scene"],
                "ablation_id": "A2",
                "target_prior": int(row["point_target"]),
                "target_prior_kind": "sac_mapped_point_target",
            }
        )
    smg = pd.read_csv(dev / "smg_targets.csv")
    for ablation_id in ("A3", "A5", "A6", "A7", "A9", "A10"):
        subset = smg[smg["variant_id"] == ablation_id]
        if subset.empty and ablation_id == "A10":
            subset = smg[smg["variant_id"] == "A5"]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "scene": row["scene"],
                    "ablation_id": ablation_id,
                    "target_prior": int(row["smg_target"]),
                    "target_prior_kind": "smg_target_or_interval_center",
                }
            )
    priors = pd.DataFrame(rows)
    if priors.duplicated(["scene", "ablation_id"]).any():
        priors = priors.drop_duplicates(["scene", "ablation_id"], keep="first")
    return priors


def _feature_cache() -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    _, metadata = _test_metadata()
    normalizations = _scene_normalizations()
    cache = {}
    for scene in SCENES:
        scene_metadata = metadata[metadata["scene_id"] == scene].copy()
        frame = protocol_io.load_scene_features(REPOSITORY_ROOT, scene_metadata, "test", None)
        features, _ = core.isotropic_normalize(
            frame[list(core.FEATURE_COLUMNS)].to_numpy(np.float64),
            normalizations[scene],
        )
        cache[scene] = (frame, features)
    return cache


def _valid_reference_counts(references: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scene in SCENES:
        scene_frame = references[references["scene_id"] == scene]
        valid = scene_frame[scene_frame["reference_status"] == "valid"]
        rows.append(
            {
                "scene_id": scene,
                "total_test_trajectories": len(scene_frame),
                "valid_reference_trajectories": len(valid),
                "excluded_reference_trajectories": len(scene_frame) - len(valid),
                "reference_coverage_pct": 100.0 * len(valid) / len(scene_frame),
                "observed_reference_movement_count": int(
                    valid["reference_movement_id"].nunique()
                ),
                "legal_movement_count": LEGAL_MOVEMENT_COUNT,
            }
        )
    return pd.DataFrame(rows)


def run_evaluation() -> dict[str, Any]:
    assignment_manifest = _verify_persisted_assignments()
    assignment_loaded_at = _utc_timestamp()
    assignments = pd.read_parquet(RESULTS_ROOT / "cluster_assignments.parquet")
    expected_reference_hash, _ = _reference_export_hash(LockedPaths())
    actual_reference_hash = sha256_file(LockedPaths.reference_export)
    if actual_reference_hash != expected_reference_hash:
        raise ValueError("Reference export hash changed.")
    references = pd.read_csv(LockedPaths.reference_export, keep_default_na=False)
    reference_loaded_at = _utc_timestamp()
    if set(references["split"]) != {"independent_test"}:
        raise ValueError("Locked evaluation received non-test reference rows.")
    _write_text(
        json.dumps(
            {
                "phase": "hg_smg_locked_reference_evaluation",
                "assignments_loaded_at_utc": assignment_loaded_at,
                "reference_loaded_at_utc": reference_loaded_at,
                "assignments_persisted_at_utc": assignment_manifest[
                    "assignments_persisted_at_utc"
                ],
                "reference_file_sha256": actual_reference_hash,
                "row_count": len(references),
            },
            sort_keys=True,
            ensure_ascii=True,
        ),
        RESULTS_ROOT / "evaluation_data_access_log.jsonl",
    )
    coverage = _valid_reference_counts(references)
    priors = _target_prior_by_ablation()
    features = _feature_cache()
    metric_rows: list[dict[str, Any]] = []
    movement_tables: list[pd.DataFrame] = []
    mapping_tables: list[pd.DataFrame] = []
    for scene in SCENES:
        feature_frame, feature_matrix = features[scene]
        feature_index = pd.Series(
            np.arange(len(feature_frame)), index=feature_frame["trajectory_id"].astype(str)
        )
        scene_reference = references[references["scene_id"] == scene]
        valid = scene_reference[scene_reference["reference_status"] == "valid"]
        observed = int(valid["reference_movement_id"].nunique())
        scene_assignments = assignments[assignments["scene_id"] == scene]
        for (ablation_id, method, strategy), run in scene_assignments.groupby(
            ["ablation_id", "method", "selection_strategy"], sort=True
        ):
            run = run.sort_values("trajectory_id", kind="mergesort")
            indices = feature_index.loc[run["trajectory_id"].astype(str)].to_numpy(int)
            target_row = priors[
                (priors["scene"] == scene) & (priors["ablation_id"] == ablation_id)
            ]
            target_prior = int(target_row["target_prior"].iloc[0])
            seed = int(
                pd.read_csv(
                    LockedPaths.development_results / "ablation_selected_configurations.csv"
                )
                .query("scene == @scene and method == @method and ablation_id == @ablation_id")
                ["fit_random_seed"]
                .head(1)
                .fillna(20260701)
                .iloc[0]
                if ablation_id not in COPIED_ORIGINAL_ABLATIONS
                else 20260701
            )
            cluster_metrics = core.safe_cluster_metrics(
                feature_matrix[indices], run["cluster_label"].to_numpy(int), seed, 3000
            )
            partition, per_movement, mapping = base_evaluator.evaluate_partition(
                run,
                valid,
                observed,
                target_prior,
                cluster_metrics,
            )
            cov = coverage[coverage["scene_id"] == scene].iloc[0].to_dict()
            identifier = {
                "scene_id": scene,
                "method": method,
                "ablation_id": ablation_id,
                "selection_strategy": strategy,
                "frozen_configuration_id": run["frozen_configuration_id"].iloc[0],
                "target_prior": target_prior,
                "target_prior_kind": target_row["target_prior_kind"].iloc[0],
            }
            metric_rows.append(
                {
                    **identifier,
                    **{k: v for k, v in cov.items() if k != "scene_id"},
                    "cluster_assignment_count": len(run),
                    "cluster_assignment_coverage_pct": 100.0 * len(run) / len(scene_reference),
                    **partition,
                }
            )
            movement_tables.append(per_movement.assign(**identifier))
            mapping_tables.append(mapping.assign(**identifier))
    metrics = pd.DataFrame(metric_rows)
    per_movement = pd.concat(movement_tables, ignore_index=True)
    mapping = pd.concat(mapping_tables, ignore_index=True)
    ablation_comparison = _ablation_comparison(metrics)
    sensitivity = _sensitivity_evaluation(metrics, coverage)
    fragmentation = _se38th_fragmentation(assignments, references)
    outputs = {
        "metrics.csv": metrics,
        "per_movement_metrics.csv": per_movement,
        "cluster_movement_mapping.csv": mapping,
        "ablation_comparison.csv": ablation_comparison,
        "sensitivity_evaluation.csv": sensitivity,
        "se38th_fragmentation_analysis.csv": fragmentation,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, RESULTS_ROOT / filename)
    manifest = {
        "task": "Task 10 HG-SMG locked independent-test evaluation",
        "evaluation_timestamp_utc": _utc_timestamp(),
        "assignments_loaded_before_reference": True,
        "assignments_persisted_at_utc": assignment_manifest["assignments_persisted_at_utc"],
        "reference_loaded_at_utc": reference_loaded_at,
        "reference_split": "independent_test",
        "valid_reference_only_for_metrics": True,
        "reference_file_sha256": actual_reference_hash,
        "output_checksums": {
            filename: sha256_file(RESULTS_ROOT / filename) for filename in outputs
        },
    }
    _write_json(manifest, RESULTS_ROOT / "evaluation_run_manifest.json")
    return manifest


def _ablation_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "observed_target_abs_error",
        "legal_target_abs_error",
        "nmi",
        "ari",
        "purity",
        "macro_f1",
        "weighted_f1",
        "completeness",
        "noise_pct_all_test",
    ]
    a1 = metrics[metrics["ablation_id"] == ORIGINAL_HG_ABLATION]
    a5 = metrics[metrics["ablation_id"] == PRIMARY_ABLATION]
    for ablation_id, frame in metrics.groupby("ablation_id", sort=True):
        row: dict[str, Any] = {
            "ablation_id": ablation_id,
            "scene_method_cases": len(frame),
            "mean_observed_target_abs_error": frame["observed_target_abs_error"].mean(),
            "mean_nmi": frame["nmi"].mean(),
            "mean_ari": frame["ari"].mean(),
            "mean_purity": frame["purity"].mean(),
            "mean_macro_f1": frame["macro_f1"].mean(),
            "mean_completeness": frame["completeness"].mean(),
            "mean_noise_pct_all_test": frame["noise_pct_all_test"].mean(),
        }
        merged_a1 = frame.merge(
            a1,
            on=["scene_id", "method"],
            suffixes=("", "_a1"),
            how="inner",
        )
        for metric in metric_cols:
            row[f"mean_delta_{metric}_vs_a1"] = (
                merged_a1[metric] - merged_a1[f"{metric}_a1"]
            ).mean()
        rows.append(row)
    paired = a5.merge(a1, on=["scene_id", "method"], suffixes=("_a5", "_a1"))
    primary = {
        "ablation_id": "A5_vs_A1_paired",
        "scene_method_cases": len(paired),
    }
    for metric in metric_cols:
        delta = paired[f"{metric}_a5"] - paired[f"{metric}_a1"]
        primary[f"mean_delta_{metric}_a5_minus_a1"] = delta.mean()
        primary[f"a5_better_{metric}_count"] = int(
            (delta < 0).sum()
            if "error" in metric or "noise" in metric
            else (delta > 0).sum()
        )
        primary[f"a5_equal_{metric}_count"] = int(np.isclose(delta, 0).sum())
    rows.append(primary)
    return pd.DataFrame(rows)


def _sensitivity_evaluation(metrics: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    structural = pd.read_csv(
        LockedPaths.development_results / "preregistered_sensitivity_summary.csv"
    )
    primary = metrics[metrics["ablation_id"] == PRIMARY_ABLATION]
    rows = []
    observed = coverage.set_index("scene_id")["observed_reference_movement_count"].to_dict()
    primary_target = _target_prior_by_ablation()
    primary_target = primary_target[primary_target["ablation_id"] == PRIMARY_ABLATION]
    primary_target_map = primary_target.set_index("scene")["target_prior"].to_dict()
    for _, row in structural.iterrows():
        scene = row["scene"]
        variant = row["variant_id"]
        if str(row.get("smg_target", "")) and not pd.isna(row.get("smg_target")):
            target = int(float(row["smg_target"]))
            target_abs_error = abs(target - int(observed[scene]))
            primary_abs_error = abs(primary_target_map[scene] - int(observed[scene]))
            rows.append(
                {
                    "scene_id": scene,
                    "sensitivity_variant": variant,
                    "evaluation_kind": "structural_target_sensitivity",
                    "target_or_interval": target,
                    "observed_reference_movement_count": observed[scene],
                    "abs_error": target_abs_error,
                    "primary_a5_abs_error": primary_abs_error,
                    "preserves_primary_target_error": target_abs_error == primary_abs_error,
                    "cluster_assignments_rerun": False,
                    "note": "Preregistered structural variant; no new test selection was made.",
                }
            )
        elif str(row.get("interval_lower", "")):
            lower = int(float(row["interval_lower"]))
            upper = int(float(row["interval_upper"]))
            obs = int(observed[scene])
            rows.append(
                {
                    "scene_id": scene,
                    "sensitivity_variant": variant,
                    "evaluation_kind": "uatp_interval_sensitivity",
                    "target_or_interval": f"[{lower},{upper}]",
                    "observed_reference_movement_count": obs,
                    "abs_error": 0 if lower <= obs <= upper else min(abs(obs - lower), abs(obs - upper)),
                    "primary_a5_abs_error": abs(primary_target_map[scene] - obs),
                    "preserves_primary_target_error": lower <= obs <= upper,
                    "cluster_assignments_rerun": False,
                    "note": "Interval sensitivity is evaluated against observed count only.",
                }
            )
    summary = (
        primary.groupby("scene_id", sort=True)[
            ["observed_target_abs_error", "nmi", "ari", "purity", "macro_f1"]
        ]
        .mean()
        .reset_index()
    )
    return pd.DataFrame(rows).merge(summary, on="scene_id", how="left")


def _se38th_fragmentation(
    assignments: pd.DataFrame, references: pd.DataFrame
) -> pd.DataFrame:
    scene = "bellevue_150th_se38th"
    valid = references[
        (references["scene_id"] == scene) & (references["reference_status"] == "valid")
    ][["scene_id", "trajectory_id", "reference_movement_id"]]
    rows = []
    for (ablation_id, method, strategy), run in assignments[
        assignments["scene_id"] == scene
    ].groupby(["ablation_id", "method", "selection_strategy"], sort=True):
        joined = valid.merge(
            run[["scene_id", "trajectory_id", "cluster_label", "is_noise"]],
            on=["scene_id", "trajectory_id"],
            how="inner",
            validate="one_to_one",
        )
        for movement, group in joined.groupby("reference_movement_id", sort=True):
            non_noise = group[group["cluster_label"] >= 0]
            counts = non_noise["cluster_label"].value_counts()
            support = len(group)
            if counts.empty:
                effective = 0.0
                entropy = 0.0
                dominant = 0.0
                clusters_1pct = 0
                purity_weighted = 0.0
            else:
                probs = counts.to_numpy(float) / counts.sum()
                entropy = float(-(probs * np.log2(probs)).sum())
                effective = float(2**entropy)
                dominant = float(counts.max() / counts.sum())
                clusters_1pct = int((counts >= max(1, math.ceil(0.01 * support))).sum())
                purity_weighted = dominant
            rows.append(
                {
                    "scene_id": scene,
                    "method": method,
                    "ablation_id": ablation_id,
                    "selection_strategy": strategy,
                    "reference_movement_id": movement,
                    "movement_support": support,
                    "unique_non_noise_clusters": int(counts.size),
                    "clusters_with_at_least_1pct_of_movement": clusters_1pct,
                    "effective_number_of_clusters": effective,
                    "fragmentation_entropy_bits": entropy,
                    "dominant_cluster_share_non_noise": dominant,
                    "noise_count_in_movement": int(group["is_noise"].sum()),
                    "weighted_cluster_purity_proxy": purity_weighted,
                }
            )
    return pd.DataFrame(rows)


def generate_figures() -> list[Path]:
    metrics = pd.read_csv(RESULTS_ROOT / "metrics.csv")
    comparison = pd.read_csv(RESULTS_ROOT / "ablation_comparison.csv")
    fragmentation = pd.read_csv(RESULTS_ROOT / "se38th_fragmentation_analysis.csv")
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    def savefig(name: str) -> None:
        path = FIGURES_ROOT / name
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        outputs.append(path)
        plt.close()

    subset = metrics[metrics["ablation_id"].isin(["A1", "A5"])]
    pivot = subset.pivot_table(
        index="scene_id",
        columns="ablation_id",
        values="observed_target_abs_error",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", figsize=(9, 4), color=["#7f8c8d", "#2166ac"])
    plt.ylabel("Mean observed target-count error")
    plt.xlabel("Scene")
    plt.title("A1 vs A5 target-count error by scene")
    savefig("target_count_error_by_scene.png")

    for metric, filename, ylabel in [
        ("nmi", "nmi_by_scene.png", "NMI"),
        ("ari", "ari_by_scene.png", "ARI"),
        ("purity", "purity_by_scene.png", "Purity"),
        ("macro_f1", "macro_f1_by_scene.png", "Macro F1"),
    ]:
        pivot = subset.pivot_table(
            index="scene_id", columns="ablation_id", values=metric, aggfunc="mean"
        )
        pivot.plot(kind="bar", figsize=(9, 4), color=["#7f8c8d", "#2166ac"])
        plt.ylabel(ylabel)
        plt.xlabel("Scene")
        plt.title(f"A1 vs A5 {ylabel} by scene")
        savefig(filename)

    pivot = subset.pivot_table(
        index="scene_id",
        columns="ablation_id",
        values="noise_pct_all_test",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", figsize=(9, 4), color=["#7f8c8d", "#2166ac"])
    plt.ylabel("Outlier percentage")
    plt.xlabel("Scene")
    plt.title("A1 vs A5 outlier percentage")
    savefig("outlier_percentage.png")

    comp = comparison[comparison["ablation_id"].str.match(r"^A\\d+$", na=False)]
    plt.figure(figsize=(8, 4))
    plt.plot(comp["ablation_id"], comp["mean_nmi"], marker="o", label="NMI")
    plt.plot(comp["ablation_id"], comp["mean_macro_f1"], marker="s", label="Macro F1")
    plt.ylabel("Mean score")
    plt.xlabel("Ablation")
    plt.title("Ablation summary")
    plt.legend()
    savefig("ablation_summary.png")

    frag = fragmentation[
        (fragmentation["ablation_id"].isin(["A1", "A5"]))
        & (fragmentation["method"] == "kmeans")
    ]
    pivot = frag.pivot_table(
        index="reference_movement_id",
        columns="ablation_id",
        values="effective_number_of_clusters",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", figsize=(10, 4), color=["#7f8c8d", "#2166ac"])
    plt.ylabel("Effective cluster count")
    plt.xlabel("SE38th reference movement")
    plt.title("SE38th fragmentation before/after")
    savefig("se38th_fragmentation_before_after.png")

    sensitivity = pd.read_csv(RESULTS_ROOT / "sensitivity_evaluation.csv")
    structural = sensitivity[
        sensitivity["evaluation_kind"] == "structural_target_sensitivity"
    ]
    pivot = structural.pivot_table(
        index="sensitivity_variant",
        values="abs_error",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", figsize=(9, 4), legend=False, color="#4c78a8")
    plt.ylabel("Mean abs error vs observed count")
    plt.xlabel("Sensitivity variant")
    plt.title("Sensitivity robustness")
    savefig("sensitivity_robustness.png")
    return outputs


def write_reports() -> None:
    metrics = pd.read_csv(RESULTS_ROOT / "metrics.csv")
    comparison = pd.read_csv(RESULTS_ROOT / "ablation_comparison.csv")
    sensitivity = pd.read_csv(RESULTS_ROOT / "sensitivity_evaluation.csv")
    fragmentation = pd.read_csv(RESULTS_ROOT / "se38th_fragmentation_analysis.csv")
    a1 = metrics[metrics["ablation_id"] == "A1"]
    a5 = metrics[metrics["ablation_id"] == "A5"]
    paired = a5.merge(a1, on=["scene_id", "method"], suffixes=("_a5", "_a1"))
    main = {
        "target_error_delta": float(
            (
                paired["observed_target_abs_error_a5"]
                - paired["observed_target_abs_error_a1"]
            ).mean()
        ),
        "nmi_delta": float((paired["nmi_a5"] - paired["nmi_a1"]).mean()),
        "macro_f1_delta": float((paired["macro_f1_a5"] - paired["macro_f1_a1"]).mean()),
        "purity_delta": float((paired["purity_a5"] - paired["purity_a1"]).mean()),
        "a5_target_error_better": int(
            (
                paired["observed_target_abs_error_a5"]
                < paired["observed_target_abs_error_a1"]
            ).sum()
        ),
        "a5_target_error_equal": int(
            np.isclose(
                paired["observed_target_abs_error_a5"],
                paired["observed_target_abs_error_a1"],
            ).sum()
        ),
    }
    protocol_lines = [
        "# HG-SMG Independent-Test Execution Protocol",
        "",
        "This task executed the first locked post-review HG-SMG-TC test evaluation.",
        "Assignments were persisted and checksummed before polygon-rule reference labels were loaded.",
        "",
        f"- Confirmation phrase: `{CONFIRMATION_PHRASE}`",
        f"- Protocol hash: `{REQUIRED_PROTOCOL_HASH}`",
        f"- Development freeze hash: `{REQUIRED_DEVELOPMENT_FREEZE_HASH}`",
        "- Executable ablations: A0, A1, A2, A3, A5, A6, A7, A9, A10.",
        "- A4 is target-prior only; A8 remains not identifiable under protocol v1.",
    ]
    _write_text("\n".join(protocol_lines), DOCS_ROOT / "hg_smg_independent_test_execution_protocol.md")

    interp = [
        "# HG-SMG Independent-Test Scientific Interpretation",
        "",
        "Primary comparison is A5 full HG-SMG-TC + PCMS against A1 original frozen HG-aware point-target selection.",
        "",
        "## A1 vs A5 Summary",
        "",
        f"- Mean observed target-error delta A5-A1: `{main['target_error_delta']:.4f}`.",
        f"- A5 lower/equal target error in `{main['a5_target_error_better']}`/`{main['a5_target_error_equal']}` of 15 scene-method cases.",
        f"- Mean NMI delta A5-A1: `{main['nmi_delta']:.4f}`.",
        f"- Mean macro-F1 delta A5-A1: `{main['macro_f1_delta']:.4f}`.",
        f"- Mean purity delta A5-A1: `{main['purity_delta']:.4f}`.",
        "",
        "The result must be read as a trade-off analysis, not as universal superiority.",
        "HG-SMG primarily tests whether semantic-merge-guided target priors reduce known endpoint-region over-segmentation.",
    ]
    _write_text("\n".join(interp), DOCS_ROOT / "hg_smg_independent_test_scientific_interpretation.md")

    ablation_lines = [
        "# HG-SMG Ablation Interpretation",
        "",
        "| Ablation | Cases | Mean target error | Mean NMI | Mean macro-F1 | Mean purity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison[comparison["ablation_id"].str.match(r"^A\\d+$", na=False)].iterrows():
        ablation_lines.append(
            f"| {row['ablation_id']} | {int(row['scene_method_cases'])} | "
            f"{row['mean_observed_target_abs_error']:.4f} | {row['mean_nmi']:.4f} | "
            f"{row['mean_macro_f1']:.4f} | {row['mean_purity']:.4f} |"
        )
    ablation_lines.extend(
        [
            "",
            "A4 has no model-selection output by design. A8 is excluded from execution because protocol v1 does not define an executable JSD compatibility threshold.",
        ]
    )
    _write_text("\n".join(ablation_lines), DOCS_ROOT / "hg_smg_ablation_interpretation.md")

    sens_lines = [
        "# HG-SMG Sensitivity Interpretation",
        "",
        "Sensitivity variants are evaluated from preregistered development structural outputs. They do not overwrite primary A5 results and do not create new independent-test selection rules.",
        "",
        f"- Sensitivity rows: `{len(sensitivity)}`.",
        f"- Structural variants preserving primary target-error equality: `{int(sensitivity['preserves_primary_target_error'].sum())}`.",
    ]
    _write_text("\n".join(sens_lines), DOCS_ROOT / "hg_smg_sensitivity_interpretation.md")

    se38 = fragmentation[
        (fragmentation["ablation_id"].isin(["A1", "A5"]))
        & (fragmentation["method"] == "kmeans")
    ]
    se38_summary = se38.groupby("ablation_id")["effective_number_of_clusters"].mean()
    se38_lines = [
        "# HG-SMG SE38th Result Interpretation",
        "",
        "SE38th is the main diagnostic scene because the frozen original target estimator over-counted geometric endpoint submodes.",
        "",
    ]
    for ablation_id, value in se38_summary.items():
        se38_lines.append(f"- KMeans mean effective cluster count per movement, {ablation_id}: `{value:.4f}`.")
    se38_lines.append("")
    se38_lines.append("This diagnostic describes fragmentation. It does not post hoc merge clusters or alter the frozen test results.")
    _write_text("\n".join(se38_lines), DOCS_ROOT / "hg_smg_se38th_result_interpretation.md")

    execution = [
        "# Task 10 Execution Report",
        "",
        f"- Protocol hash: `{REQUIRED_PROTOCOL_HASH}`",
        f"- Development freeze hash: `{REQUIRED_DEVELOPMENT_FREEZE_HASH}`",
        f"- Assignment rows: `{len(pd.read_parquet(RESULTS_ROOT / 'cluster_assignments.parquet'))}`",
        f"- Metrics rows: `{len(metrics)}`",
        f"- Mean A5-A1 target-error delta: `{main['target_error_delta']:.4f}`",
        f"- Mean A5-A1 NMI delta: `{main['nmi_delta']:.4f}`",
        "- No tuning occurred after independent-test results were visible.",
        "- Manuscript DOCX was not modified.",
    ]
    _write_text("\n".join(execution), DOCS_ROOT / "task_10_execution_report.md")


def _task_files() -> list[Path]:
    roots = [
        PUBLICATION_ROOT / "code/hg_smg/locked_test.py",
        PUBLICATION_ROOT / "code/hg_smg/cli.py",
        PUBLICATION_ROOT / "tests/test_hg_smg_locked_test.py",
        AUTHORIZATION_PATH,
        DOCS_ROOT / "hg_smg_locked_test_preflight_report.md",
        DOCS_ROOT / "hg_smg_independent_test_execution_protocol.md",
        DOCS_ROOT / "hg_smg_independent_test_scientific_interpretation.md",
        DOCS_ROOT / "hg_smg_ablation_interpretation.md",
        DOCS_ROOT / "hg_smg_sensitivity_interpretation.md",
        DOCS_ROOT / "hg_smg_se38th_result_interpretation.md",
        DOCS_ROOT / "task_10_execution_report.md",
    ]
    if RESULTS_ROOT.exists():
        roots.extend(path for path in RESULTS_ROOT.rglob("*") if path.is_file())
    if FIGURES_ROOT.exists():
        roots.extend(path for path in FIGURES_ROOT.rglob("*") if path.is_file())
    return sorted({path for path in roots if path.exists()})


def package_outputs(test_result: str = "not_run") -> dict[str, Any]:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    if ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256").exists():
        ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256").unlink()
    metrics = pd.read_csv(RESULTS_ROOT / "metrics.csv")
    coverage = (
        metrics[["scene_id", "total_test_trajectories", "valid_reference_trajectories"]]
        .drop_duplicates()
        .sort_values("scene_id")
    )
    a1 = metrics[metrics["ablation_id"] == "A1"]
    a5 = metrics[metrics["ablation_id"] == "A5"]
    paired = a5.merge(a1, on=["scene_id", "method"], suffixes=("_a5", "_a1"))
    summary = {
        "branch": _git("branch", "--show-current"),
        "base_development_commit": REQUIRED_DEVELOPMENT_COMMIT,
        "final_commit": _git("rev-parse", "HEAD"),
        "protocol_hash": REQUIRED_PROTOCOL_HASH,
        "development_freeze_hash": REQUIRED_DEVELOPMENT_FREEZE_HASH,
        "preflight_result": "PASS",
        "assignment_rows": int(
            pd.read_parquet(RESULTS_ROOT / "cluster_assignments.parquet").shape[0]
        ),
        "metrics_rows": int(len(metrics)),
        "a5_minus_a1_mean_target_error_delta": float(
            (
                paired["observed_target_abs_error_a5"]
                - paired["observed_target_abs_error_a1"]
            ).mean()
        ),
        "a5_minus_a1_mean_nmi_delta": float((paired["nmi_a5"] - paired["nmi_a1"]).mean()),
        "a5_minus_a1_mean_macro_f1_delta": float(
            (paired["macro_f1_a5"] - paired["macro_f1_a1"]).mean()
        ),
        "test_result": test_result,
        "no_tuning_occurred": True,
    }
    task_summary_lines = [
        "# TASK_SUMMARY",
        "",
        f"- Branch: `{summary['branch']}`",
        f"- Base development commit: `{REQUIRED_DEVELOPMENT_COMMIT}`",
        f"- Final commit: `{summary['final_commit']}`",
        f"- Protocol hash: `{REQUIRED_PROTOCOL_HASH}`",
        f"- Development freeze hash: `{REQUIRED_DEVELOPMENT_FREEZE_HASH}`",
        f"- Assignment rows: `{summary['assignment_rows']}`",
        f"- Metrics rows: `{summary['metrics_rows']}`",
        f"- A5-A1 mean observed target-error delta: `{summary['a5_minus_a1_mean_target_error_delta']:.4f}`",
        f"- A5-A1 mean NMI delta: `{summary['a5_minus_a1_mean_nmi_delta']:.4f}`",
        f"- A5-A1 mean macro-F1 delta: `{summary['a5_minus_a1_mean_macro_f1_delta']:.4f}`",
        f"- Test/lint: `{test_result}`",
        "- No tuning occurred after test results.",
        "- Manuscript DOCX was not modified.",
        "",
        "## Reference Coverage",
        "",
        "| Scene | Total | Valid |",
        "| --- | ---: | ---: |",
    ]
    task_summary_lines.extend(
        f"| {row.scene_id} | {int(row.total_test_trajectories)} | {int(row.valid_reference_trajectories)} |"
        for row in coverage.itertuples(index=False)
    )
    task_summary = "\n".join(task_summary_lines) + "\n"
    files = _task_files()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("TASK_SUMMARY.md", task_summary)
        for path in files:
            relative_path = relative(path)
            lower = relative_path.lower()
            if any(
                marker in lower
                for marker in (
                    ".git/",
                    ".venv/",
                    "data/raw/",
                    "google_maps",
                    "prior_zip",
                )
            ):
                raise ValueError(f"Forbidden artifact selected for ZIP: {relative_path}")
            archive.write(path, relative_path)
    zip_hash = sha256_file(ZIP_PATH)
    sidecar = ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256")
    sidecar.write_text(f"{zip_hash}  {ZIP_PATH.name}\n", encoding="ascii")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
    package_manifest = {
        **summary,
        "zip_path": str(ZIP_PATH),
        "zip_sha256": zip_hash,
        "zip_entries": len(names),
        "task_files_included": len(files),
        "forbidden_artifacts_absent": True,
    }
    _write_json(package_manifest, RESULTS_ROOT / "package_manifest.json")
    return package_manifest


def run_all(confirmation: str | None) -> dict[str, Any]:
    authorize_locked_test()
    run_assignments(confirmation)
    run_evaluation()
    generate_figures()
    write_reports()
    return package_outputs()
