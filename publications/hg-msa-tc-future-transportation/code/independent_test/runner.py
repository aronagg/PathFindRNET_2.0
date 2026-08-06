"""Persist frozen independent-test cluster assignments without reading references."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import hg_msa_tc_core as core
from pipeline import split_aware_io as protocol_io

from .protocol import METHODS, SCENES, STRATEGIES, Paths, run_preflight, verify_unlock


ASSIGNMENT_COLUMNS = (
    "scene_id",
    "trajectory_id",
    "method",
    "selection_strategy",
    "cluster_label",
    "is_noise",
    "frozen_configuration_id",
    "frozen_protocol_hash",
    "feature_checksum",
    "run_id",
    "run_timestamp_utc",
)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _configuration_id(row: Any) -> str:
    payload = {
        "scene": str(row.scene),
        "method": str(row.method),
        "selection_strategy": str(row.selection_strategy),
        "params_json": str(row.params_json),
        "fit_random_seed": int(row.fit_random_seed),
        "normalization_parameters_json": str(row.normalization_parameters_json),
    }
    return protocol_io.canonical_sha256(payload)


def _load_test_metadata(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = protocol_io.load_split_membership(paths.evaluation_split, "test", SCENES)
    metadata = protocol_io.load_manifest_metadata(paths.trajectory_manifest, membership)
    protocol_io.assert_phase_rows(metadata, "test")
    return membership, metadata


def run_frozen_clustering(paths: Paths, explicit_confirmation: bool) -> dict[str, Any]:
    """Run all 30 frozen configurations; this module has no reference-label import."""
    preflight = run_preflight(paths, write_report=False)
    unlock = verify_unlock(paths, preflight, explicit_confirmation)
    if paths.results.exists():
        raise FileExistsError(
            f"Single-use independent-test output directory already exists: {paths.results}"
        )
    paths.results.mkdir(parents=True, exist_ok=False)

    run_timestamp = protocol_io.utc_timestamp()
    run_id = (
        "independent-test-v1-"
        + run_timestamp.replace(":", "").replace("+00:00", "Z").replace("-", "")
        + "-"
        + preflight["frozen_protocol_hash"][:12]
    )
    membership, metadata = _load_test_metadata(paths)
    selected = pd.read_csv(paths.selected_configurations)
    logger = protocol_io.AccessLogger(paths.results / "data_access_log.jsonl", "test")
    logger.add(
        "ALL",
        paths.evaluation_split.relative_to(paths.repo_root),
        membership["split"].unique(),
        len(membership),
        protocol_io.sha256_file(paths.evaluation_split),
        "split_control_index",
    )
    logger.add(
        "ALL",
        paths.trajectory_manifest.relative_to(paths.repo_root),
        membership["split"].unique(),
        len(membership),
        protocol_io.sha256_file(paths.trajectory_manifest),
        "trajectory_control_manifest",
    )

    assignment_tables: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []
    for scene in SCENES:
        scene_metadata = metadata[metadata["scene_id"] == scene].copy()
        frame = protocol_io.load_scene_features(paths.repo_root, scene_metadata, "test", logger)
        source_paths = frame["source_file"].drop_duplicates().tolist()
        source_hashes = frame["source_file_checksum_sha256"].drop_duplicates().tolist()
        if len(source_paths) != 1 or len(source_hashes) != 1:
            raise ValueError(f"Unexpected feature provenance for {scene}")
        scene_rows = selected[selected["scene"] == scene].copy()
        if len(scene_rows) != len(METHODS) * len(STRATEGIES):
            raise ValueError(f"Frozen configuration count changed for {scene}")
        normalization_values = scene_rows["normalization_parameters_json"].unique()
        if len(normalization_values) != 1:
            raise ValueError(f"Frozen normalization differs within {scene}")
        values = frame[list(core.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        features, applied_normalization = core.isotropic_normalize(
            values, json.loads(normalization_values[0])
        )
        for row in scene_rows.sort_values(
            ["method", "selection_strategy"], kind="mergesort"
        ).itertuples(index=False):
            parameters = json.loads(row.params_json)
            started = time.perf_counter()
            labels = core.fit_predict(
                str(row.method), features, parameters, int(row.fit_random_seed)
            )
            fit_time = time.perf_counter() - started
            if len(labels) != len(frame):
                raise ValueError(f"Assignment count changed for {scene}/{row.method}")
            configuration_id = _configuration_id(row)
            assignment_tables.append(
                pd.DataFrame(
                    {
                        "scene_id": scene,
                        "trajectory_id": frame["trajectory_id"].astype(str),
                        "method": str(row.method),
                        "selection_strategy": str(row.selection_strategy),
                        "cluster_label": labels.astype(int),
                        "is_noise": labels.astype(int) == -1,
                        "frozen_configuration_id": configuration_id,
                        "frozen_protocol_hash": preflight["frozen_protocol_hash"],
                        "feature_checksum": str(source_hashes[0]),
                        "run_id": run_id,
                        "run_timestamp_utc": run_timestamp,
                    }
                )
            )
            run_records.append(
                {
                    "scene_id": scene,
                    "method": str(row.method),
                    "selection_strategy": str(row.selection_strategy),
                    "frozen_configuration_id": configuration_id,
                    "params_json": str(row.params_json),
                    "fit_random_seed": int(row.fit_random_seed),
                    "normalization_parameters": applied_normalization,
                    "feature_file": str(source_paths[0]),
                    "feature_checksum": str(source_hashes[0]),
                    "n_trajectories": len(frame),
                    "n_non_noise_clusters": int(len(np.unique(labels[labels >= 0]))),
                    "noise_count": int((labels == -1).sum()),
                    "fit_time_s": fit_time,
                }
            )

    assignments = pd.concat(assignment_tables, ignore_index=True)
    assignments = assignments.sort_values(
        ["scene_id", "method", "selection_strategy", "trajectory_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(assignments) != 27_393 * len(METHODS) * len(STRATEGIES):
        raise ValueError(f"Unexpected assignment rows: {len(assignments)}")
    if assignments.duplicated(["scene_id", "trajectory_id", "method", "selection_strategy"]).any():
        raise ValueError("Duplicate independent-test assignments detected.")
    if set(assignments.columns) != set(ASSIGNMENT_COLUMNS):
        raise ValueError("Assignment schema changed.")

    csv_path = paths.results / "cluster_assignments.csv"
    parquet_path = paths.results / "cluster_assignments.parquet"
    _write_csv(assignments[list(ASSIGNMENT_COLUMNS)], csv_path)
    assignments[list(ASSIGNMENT_COLUMNS)].to_parquet(parquet_path, index=False, compression="zstd")
    logger.flush()
    persisted_at = protocol_io.utc_timestamp()
    manifest = {
        "task_name": "future-transportation-independent-test-evaluation-v1",
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp,
        "assignments_persisted_at_utc": persisted_at,
        "frozen_protocol_hash": preflight["frozen_protocol_hash"],
        "frozen_protocol_file_sha256": preflight["frozen_protocol_file_sha256"],
        "selected_configurations_sha256": protocol_io.sha256_file(paths.selected_configurations),
        "unlock_sha256": protocol_io.sha256_file(paths.unlock),
        "unlock_git_commit": unlock["git_commit"],
        "test_split": "independent_test",
        "transductive_fit": True,
        "reference_labels_read": False,
        "evaluation_started": False,
        "target_recomputed": False,
        "hyperparameters_recomputed": False,
        "normalization_refitted": False,
        "assignment_rows": len(assignments),
        "unique_test_trajectories": assignments["trajectory_id"].nunique(),
        "scene_method_strategy_runs": len(run_records),
        "output_files": {
            "cluster_assignments.csv": protocol_io.sha256_file(csv_path),
            "cluster_assignments.parquet": protocol_io.sha256_file(parquet_path),
            "data_access_log.jsonl": protocol_io.sha256_file(
                paths.results / "data_access_log.jsonl"
            ),
        },
        "runs": run_records,
    }
    manifest_path = paths.results / "clustering_run_manifest.json"
    protocol_io.write_json_atomic(manifest_path, manifest)
    sidecar_lines = [
        f"{protocol_io.sha256_file(csv_path)}  {csv_path.name}",
        f"{protocol_io.sha256_file(parquet_path)}  {parquet_path.name}",
        f"{protocol_io.sha256_file(manifest_path)}  {manifest_path.name}",
        f"{protocol_io.sha256_file(paths.results / 'data_access_log.jsonl')}  data_access_log.jsonl",
    ]
    protocol_io.write_text_atomic(
        paths.results / "clustering_run_checksums.sha256",
        "\n".join(sidecar_lines) + "\n",
    )
    return manifest
