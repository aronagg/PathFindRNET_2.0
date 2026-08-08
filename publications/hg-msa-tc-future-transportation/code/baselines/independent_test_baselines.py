"""Deterministic baseline comparisons on the frozen independent-test split."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from independent_test import evaluator as reference_evaluator
from independent_test.protocol import METHODS, SCENES
from pipeline import hg_msa_tc_core as core
from pipeline import split_aware_io as split_io


PUBLICATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PUBLICATION_ROOT.parents[1]
RESULTS_ROOT = PUBLICATION_ROOT / "results/baselines/independent_test"
FIGURES_ROOT = PUBLICATION_ROOT / "figures/baselines/independent_test"
DOCS_ROOT = PUBLICATION_ROOT / "docs"
TASK10_COMMIT = "2819638724f8b41232ca4879c246f16a6c2a1c9d"
BASELINE_VERSION = "futuretransp-baseline-comparison-v1"
LEGAL_MOVEMENT_COUNT = 12


BASELINES = (
    "endpoint_camera_raw",
    "endpoint_camera_isotropic",
    "resampled_trajectory_euclidean",
)
DTW_STATUS = "skipped_full_pairwise_infeasible"
RESAMPLED_POINT_COUNT = 20


@dataclass(frozen=True)
class Paths:
    trajectory_manifest: Path = PUBLICATION_ROOT / "data/manifests/trajectory_manifest.csv"
    evaluation_split: Path = PUBLICATION_ROOT / "data/splits/evaluation_split.csv"
    reference_export: Path = (
        PUBLICATION_ROOT
        / "annotations/reference_labels/independent_test_reference_labels.csv"
    )
    reference_manifest: Path = (
        PUBLICATION_ROOT
        / "annotations/reference_labels/reference_output_manifest.json"
    )
    original_metrics: Path = PUBLICATION_ROOT / "results/independent_test/independent_test_metrics.csv"
    hg_smg_metrics: Path = PUBLICATION_ROOT / "results/hg_smg/independent_test/metrics.csv"


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_head() -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY_ROOT.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def git_branch() -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY_ROOT.as_posix()}",
            "branch",
            "--show-current",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = split_io.hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_text(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def preflight(require_no_outputs: bool = True) -> dict[str, Any]:
    paths = Paths()
    if require_no_outputs and RESULTS_ROOT.exists():
        raise FileExistsError(f"Baseline output directory already exists: {RESULTS_ROOT}")
    required = [
        paths.trajectory_manifest,
        paths.evaluation_split,
        paths.reference_export,
        paths.reference_manifest,
        paths.original_metrics,
        paths.hg_smg_metrics,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing baseline inputs: {missing}")
    merge_base = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY_ROOT.as_posix()}",
            "merge-base",
            "HEAD",
            TASK10_COMMIT,
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()
    if merge_base != TASK10_COMMIT:
        raise RuntimeError("Task 11 branch is not based on the Task 10 final commit.")
    reference_manifest = json.loads(paths.reference_manifest.read_text(encoding="utf-8"))
    reference_record = reference_manifest["output_files"][paths.reference_export.name]
    if split_io.sha256_file(paths.reference_export) != reference_record["sha256"]:
        raise RuntimeError("Independent-test reference export checksum changed.")
    split_counts = (
        pd.read_csv(paths.evaluation_split, usecols=["scene_id", "split"])
        .query("split == 'independent_test'")
        .groupby("scene_id")
        .size()
        .to_dict()
    )
    payload = {
        "status": "PASS",
        "task": "Task 11 baseline comparison preflight",
        "timestamp_utc": utc_timestamp(),
        "git_head": git_head(),
        "git_branch": git_branch(),
        "task10_commit_is_ancestor": True,
        "reference_export_sha256": reference_record["sha256"],
        "reference_export_rows": int(reference_record["rows"]),
        "split_counts_by_scene": split_counts,
        "reference_rows_loaded": False,
        "baseline_assignments_exist": False,
    }
    write_text(
        [
            "# Baseline Preflight",
            "",
            f"- Status: **{payload['status']}**",
            f"- Git HEAD: `{payload['git_head']}`",
            f"- Reference export SHA-256: `{payload['reference_export_sha256']}`",
            "- Reference rows loaded during preflight: **no**",
        ],
        DOCS_ROOT / "baseline_preflight_report.md",
    )
    return payload


def test_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = Paths()
    membership = split_io.load_split_membership(
        paths.evaluation_split, "test", tuple(SCENES)
    )
    metadata = split_io.load_manifest_metadata(paths.trajectory_manifest, membership)
    split_io.assert_phase_rows(metadata, "test")
    return membership, metadata


def feature_matrix_endpoint(frame: pd.DataFrame, isotropic: bool) -> tuple[np.ndarray, dict[str, Any]]:
    values = frame[list(core.FEATURE_COLUMNS)].to_numpy(np.float64)
    if isotropic:
        features, parameters = core.isotropic_normalize(values)
        return features, {"normalization": "camera_isotropic_shared_scale", **parameters}
    return values, {"normalization": "raw_camera_pixels"}


def _trajectory_path(scene: str) -> Path:
    return REPOSITORY_ROOT / "data/processed" / scene / "trajectories.parquet"


def _resample_points(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == 0:
        return np.full((count, 2), np.nan)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    old_t = np.linspace(0.0, 1.0, len(points))
    new_t = np.linspace(0.0, 1.0, count)
    return np.column_stack(
        [
            np.interp(new_t, old_t, points[:, 0]),
            np.interp(new_t, old_t, points[:, 1]),
        ]
    )


def feature_matrix_resampled(scene: str, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    track_ids = frame["original_track_id"].astype(int).tolist()
    trajectory_path = _trajectory_path(scene)
    trajectories = pd.read_parquet(
        trajectory_path,
        columns=["track_id", "frame", "x", "y"],
        filters=[("track_id", "in", track_ids)],
    )
    feature_source = REPOSITORY_ROOT / str(frame["source_file"].drop_duplicates().iloc[0])
    intervals = pd.read_parquet(
        feature_source,
        columns=["track_id", "frame_start", "frame_end"],
        filters=[("track_id", "in", track_ids)],
    )
    intervals = intervals.rename(
        columns={"track_id": "original_track_id", "frame_start": "start_frame", "frame_end": "end_frame"}
    )
    joined = trajectories.merge(
        intervals,
        left_on="track_id",
        right_on="original_track_id",
        how="inner",
        validate="many_to_one",
    )
    joined = joined[
        (joined["frame"] >= joined["start_frame"])
        & (joined["frame"] <= joined["end_frame"])
    ].sort_values(["track_id", "frame"], kind="mergesort")
    rows = []
    for track_id in track_ids:
        group = joined[joined["track_id"] == track_id]
        points = group[["x", "y"]].to_numpy(np.float64)
        rows.append(_resample_points(points, RESAMPLED_POINT_COUNT).reshape(-1))
    values = np.vstack(rows)
    # Shared x/y scale over every sampled coordinate preserves image aspect ratio.
    x_columns = np.arange(0, values.shape[1], 2)
    y_columns = np.arange(1, values.shape[1], 2)
    x_min = float(np.nanmin(values[:, x_columns]))
    y_min = float(np.nanmin(values[:, y_columns]))
    scale = max(
        float(np.nanmax(values[:, x_columns]) - x_min),
        float(np.nanmax(values[:, y_columns]) - y_min),
        1e-12,
    )
    normalized = values.copy()
    normalized[:, x_columns] = (normalized[:, x_columns] - x_min) / scale
    normalized[:, y_columns] = (normalized[:, y_columns] - y_min) / scale
    provenance = {
        "normalization": "resampled_camera_isotropic_shared_scale",
        "resampled_points": RESAMPLED_POINT_COUNT,
        "feature_dimension": int(normalized.shape[1]),
        "trajectory_source": trajectory_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "trajectory_source_sha256": split_io.sha256_file(trajectory_path),
        "feature_memory_bytes": int(normalized.nbytes),
    }
    return normalized, provenance


def baseline_features(scene: str, frame: pd.DataFrame, baseline_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    if baseline_id == "endpoint_camera_raw":
        return feature_matrix_endpoint(frame, isotropic=False)
    if baseline_id == "endpoint_camera_isotropic":
        return feature_matrix_endpoint(frame, isotropic=True)
    if baseline_id == "resampled_trajectory_euclidean":
        return feature_matrix_resampled(scene, frame)
    raise ValueError(f"Unknown baseline: {baseline_id}")


def deterministic_grid(method: str, features: np.ndarray, seed: int) -> list[dict[str, Any]]:
    n_rows = len(features)
    if method == "kmeans":
        return [
            {"n_clusters": k, "n_init": 10, "max_iter": 300}
            for k in (8, 10, 12, 14, 18)
            if k < n_rows
        ]
    if method == "hdbscan":
        sizes = sorted({max(30, int(0.01 * n_rows)), max(80, int(0.03 * n_rows))})
        return [
            {"min_cluster_size": size, "min_samples": samples}
            for size in sizes
            for samples in (10, 25)
            if size < n_rows and samples < n_rows
        ]
    if method == "optics":
        eps_values = core.optics_eps_grid(features, seed, [0.03, 0.08], 2000)
        return [
            {"min_samples": samples, "xi": 0.05, "max_eps": eps}
            for samples in (25, 50)
            for eps in eps_values
            if samples < n_rows
        ]
    raise ValueError(method)


def selection_key(row: pd.Series, method: str) -> tuple[Any, ...]:
    def finite(value: Any, fallback: float, invert: bool = False) -> float:
        if pd.isna(value):
            return fallback
        numeric = float(value)
        return -numeric if invert else numeric

    key: list[Any] = []
    if method in {"hdbscan", "optics"}:
        key.append(finite(row["pct_outliers"], 100.0))
    key.extend(
        [
            finite(row["silhouette_clustered_only"], 1.0, invert=True),
            finite(row["davies_bouldin_clustered_only"], np.inf),
            finite(row["calinski_harabasz_clustered_only"], np.inf, invert=True),
            finite(row["largest_cluster_ratio"], 1.0),
            str(row["params_json"]),
        ]
    )
    return tuple(key)


def select_trial(trials: pd.DataFrame, method: str) -> pd.Series:
    valid = trials[trials["error"] == ""].copy()
    if valid.empty:
        raise RuntimeError(f"No successful baseline trial for {method}")
    keys = valid.apply(lambda row: selection_key(row, method), axis=1)
    selected_index = min(keys.index, key=lambda index: keys.loc[index])
    selected = valid.loc[selected_index].copy()
    selected["selection_rule"] = (
        "density methods minimize outlier percentage, then maximize silhouette, "
        "minimize Davies-Bouldin, maximize Calinski-Harabasz, minimize largest "
        "cluster ratio, then lexical parameters; KMeans omits the outlier term."
    )
    selected["selection_key_json"] = json.dumps(list(keys.loc[selected_index]), default=str)
    return selected


def run_candidate_trials(
    scene: str,
    baseline_id: str,
    method: str,
    features: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rows = []
    labels_by_trial: dict[int, np.ndarray] = {}
    for trial_index, parameters in enumerate(deterministic_grid(method, features, seed), start=1):
        fit_seed = seed + trial_index
        started = time.perf_counter()
        try:
            labels = core.fit_predict(method, features, parameters, fit_seed)
            metrics = core.label_statistics(
                features,
                labels,
                LEGAL_MOVEMENT_COUNT,
                time.perf_counter() - started,
                fit_seed,
                3000,
            )
            labels_by_trial[trial_index] = labels.astype(int)
            error = ""
        except Exception as exc:
            metrics = {
                "n_total": len(features),
                "n_clusters": 0,
                "cluster_count_error": LEGAL_MOVEMENT_COUNT,
                "n_outliers": np.nan,
                "pct_outliers": np.nan,
                "largest_cluster_ratio": np.nan,
                "fit_time_s": time.perf_counter() - started,
                "hg_estimated_target": LEGAL_MOVEMENT_COUNT,
                "silhouette_clustered_only": np.nan,
                "davies_bouldin_clustered_only": np.nan,
                "calinski_harabasz_clustered_only": np.nan,
                "quick_score": np.nan,
                "EMAS_HG": np.nan,
            }
            error = repr(exc)
        rows.append(
            {
                "scene_id": scene,
                "baseline_id": baseline_id,
                "method": method,
                "trial_index": trial_index,
                "fit_random_seed": fit_seed,
                "params_json": json.dumps(parameters, sort_keys=True),
                **metrics,
                "error": error,
            }
        )
    trials = pd.DataFrame(rows)
    selected = select_trial(trials, method)
    return trials, labels_by_trial[int(selected["trial_index"])], selected


def run_assignments() -> dict[str, Any]:
    preflight(require_no_outputs=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=False)
    run_id = "baseline-independent-test-" + utc_timestamp().replace(":", "").replace("-", "")
    membership, metadata = test_metadata()
    logger = split_io.AccessLogger(RESULTS_ROOT / "data_access_log.jsonl", "test")
    logger.add(
        "ALL",
        Paths.evaluation_split.relative_to(REPOSITORY_ROOT),
        membership["split"].unique(),
        len(membership),
        split_io.sha256_file(Paths.evaluation_split),
        "split_control_index",
    )
    assignment_tables = []
    trial_tables = []
    selected_rows = []
    runtime_rows = []
    for scene_index, scene in enumerate(SCENES):
        scene_metadata = metadata[metadata["scene_id"] == scene].copy()
        feature_frame = split_io.load_scene_features(
            REPOSITORY_ROOT, scene_metadata, "test", logger
        )
        for baseline_index, baseline_id in enumerate(BASELINES):
            started_feature = time.perf_counter()
            features, provenance = baseline_features(scene, feature_frame, baseline_id)
            feature_time = time.perf_counter() - started_feature
            for method_index, method in enumerate(METHODS):
                seed = 20261101 + scene_index * 1000 + baseline_index * 100 + method_index * 10
                trials, labels, selected = run_candidate_trials(
                    scene, baseline_id, method, features, seed
                )
                trial_tables.append(trials)
                selected_rows.append(selected.to_dict())
                configuration_id = split_io.canonical_sha256(
                    {
                        "baseline_id": baseline_id,
                        "scene": scene,
                        "method": method,
                        "params_json": selected["params_json"],
                        "fit_random_seed": int(selected["fit_random_seed"]),
                        "selection_rule": selected["selection_rule"],
                    }
                )
                assignment_tables.append(
                    pd.DataFrame(
                        {
                            "scene_id": scene,
                            "trajectory_id": feature_frame["trajectory_id"].astype(str),
                            "baseline_id": baseline_id,
                            "method": method,
                            "cluster_label": labels.astype(int),
                            "is_noise": labels.astype(int) == -1,
                            "baseline_configuration_id": configuration_id,
                            "baseline_version": BASELINE_VERSION,
                            "feature_provenance_json": json.dumps(provenance, sort_keys=True),
                            "run_id": run_id,
                            "run_timestamp_utc": utc_timestamp(),
                        }
                    )
                )
                runtime_rows.append(
                    {
                        "scene_id": scene,
                        "baseline_id": baseline_id,
                        "method": method,
                        "feature_build_time_s": feature_time,
                        "selected_trial_fit_time_s": float(selected["fit_time_s"]),
                        "candidate_count": len(trials),
                        "feature_rows": len(features),
                        "feature_columns": features.shape[1],
                        "feature_memory_bytes": int(features.nbytes),
                    }
                )
    logger.flush()
    assignments = pd.concat(assignment_tables, ignore_index=True).sort_values(
        ["scene_id", "baseline_id", "method", "trajectory_id"], kind="mergesort"
    )
    expected_rows = 27_393 * len(BASELINES) * len(METHODS)
    if len(assignments) != expected_rows:
        raise RuntimeError(f"Unexpected baseline assignment count: {len(assignments)}")
    if assignments.duplicated(["scene_id", "trajectory_id", "baseline_id", "method"]).any():
        raise RuntimeError("Duplicate baseline assignments detected.")
    assignments.to_parquet(RESULTS_ROOT / "baseline_assignments.parquet", index=False, compression="zstd")
    write_csv(pd.concat(trial_tables, ignore_index=True), RESULTS_ROOT / "baseline_candidate_trials.csv")
    write_csv(pd.DataFrame(selected_rows), RESULTS_ROOT / "baseline_selected_configurations.csv")
    write_csv(pd.DataFrame(runtime_rows), RESULTS_ROOT / "baseline_runtime_summary.csv")
    manifest = {
        "task": "Task 11 baseline assignments",
        "run_id": run_id,
        "assignment_rows": len(assignments),
        "assignments_persisted_at_utc": utc_timestamp(),
        "reference_labels_read": False,
        "evaluation_started": False,
        "baseline_ids": list(BASELINES),
        "methods": list(METHODS),
        "dtw_frechet_status": DTW_STATUS,
        "output_checksums": {
            "baseline_assignments.parquet": split_io.sha256_file(
                RESULTS_ROOT / "baseline_assignments.parquet"
            ),
            "baseline_candidate_trials.csv": split_io.sha256_file(
                RESULTS_ROOT / "baseline_candidate_trials.csv"
            ),
            "baseline_selected_configurations.csv": split_io.sha256_file(
                RESULTS_ROOT / "baseline_selected_configurations.csv"
            ),
            "baseline_runtime_summary.csv": split_io.sha256_file(
                RESULTS_ROOT / "baseline_runtime_summary.csv"
            ),
        },
    }
    write_json(manifest, RESULTS_ROOT / "baseline_assignment_manifest.json")
    return manifest


def _verify_assignments_persisted() -> dict[str, Any]:
    manifest_path = RESULTS_ROOT / "baseline_assignment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Baseline assignments must be persisted before evaluation.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["reference_labels_read"] is not False:
        raise RuntimeError("Baseline assignment manifest does not prove label isolation.")
    for filename, expected in manifest["output_checksums"].items():
        if split_io.sha256_file(RESULTS_ROOT / filename) != expected:
            raise RuntimeError(f"Baseline output changed before evaluation: {filename}")
    return manifest


def reference_coverage(references: pd.DataFrame) -> pd.DataFrame:
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
                "observed_reference_movement_count": int(valid["reference_movement_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def run_evaluation() -> dict[str, Any]:
    assignment_manifest = _verify_assignments_persisted()
    assignments_loaded_at = utc_timestamp()
    assignments = pd.read_parquet(RESULTS_ROOT / "baseline_assignments.parquet")
    reference_hash = split_io.sha256_file(Paths.reference_export)
    references = pd.read_csv(Paths.reference_export, keep_default_na=False)
    reference_loaded_at = utc_timestamp()
    if set(references["split"]) != {"independent_test"}:
        raise RuntimeError("Baseline evaluation received non-test references.")
    write_json(
        {
            "phase": "baseline_reference_evaluation",
            "assignments_persisted_at_utc": assignment_manifest["assignments_persisted_at_utc"],
            "assignments_loaded_at_utc": assignments_loaded_at,
            "reference_loaded_at_utc": reference_loaded_at,
            "reference_file_sha256": reference_hash,
            "reference_rows": len(references),
        },
        RESULTS_ROOT / "baseline_evaluation_access_log.json",
    )
    coverage = reference_coverage(references)
    metric_rows = []
    movement_tables = []
    mapping_tables = []
    selected = pd.read_csv(RESULTS_ROOT / "baseline_selected_configurations.csv")
    for scene in SCENES:
        scene_reference = references[references["scene_id"] == scene]
        valid = scene_reference[scene_reference["reference_status"] == "valid"]
        observed = int(valid["reference_movement_id"].nunique())
        scene_assignments = assignments[assignments["scene_id"] == scene]
        for (baseline_id, method), run in scene_assignments.groupby(["baseline_id", "method"], sort=True):
            row = selected[
                (selected["scene_id"] == scene)
                & (selected["baseline_id"] == baseline_id)
                & (selected["method"] == method)
            ].iloc[0]
            partition, per_movement, mapping = reference_evaluator.evaluate_partition(
                run,
                valid,
                observed,
                LEGAL_MOVEMENT_COUNT,
                {
                    "silhouette_clustered_only": float(row["silhouette_clustered_only"]),
                    "davies_bouldin_clustered_only": float(row["davies_bouldin_clustered_only"]),
                    "calinski_harabasz_clustered_only": float(row["calinski_harabasz_clustered_only"]),
                },
            )
            cov = coverage[coverage["scene_id"] == scene].iloc[0].to_dict()
            identifier = {
                "scene_id": scene,
                "baseline_id": baseline_id,
                "method": method,
                "baseline_configuration_id": run["baseline_configuration_id"].iloc[0],
            }
            metric_rows.append(
                {
                    **identifier,
                    **{key: value for key, value in cov.items() if key != "scene_id"},
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
    write_csv(metrics, RESULTS_ROOT / "baseline_metrics.csv")
    write_csv(per_movement, RESULTS_ROOT / "baseline_per_movement_metrics.csv")
    write_csv(mapping, RESULTS_ROOT / "baseline_cluster_mapping.csv")
    comparison = comparison_to_hg_smg(metrics)
    write_csv(comparison, RESULTS_ROOT / "baseline_comparison_to_hg_smg.csv")
    oracle = oracle_diagnostic(references)
    write_csv(oracle, RESULTS_ROOT / "oracle_reference_diagnostic.csv")
    manifest = {
        "task": "Task 11 baseline reference evaluation",
        "evaluation_timestamp_utc": utc_timestamp(),
        "assignments_loaded_before_reference": True,
        "assignments_persisted_at_utc": assignment_manifest["assignments_persisted_at_utc"],
        "reference_loaded_at_utc": reference_loaded_at,
        "valid_reference_only_for_metrics": True,
        "output_checksums": {
            name: split_io.sha256_file(RESULTS_ROOT / name)
            for name in [
                "baseline_metrics.csv",
                "baseline_per_movement_metrics.csv",
                "baseline_cluster_mapping.csv",
                "baseline_comparison_to_hg_smg.csv",
                "oracle_reference_diagnostic.csv",
            ]
        },
    }
    write_json(manifest, RESULTS_ROOT / "baseline_evaluation_manifest.json")
    return manifest


def oracle_diagnostic(references: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scene in SCENES:
        scene_frame = references[references["scene_id"] == scene]
        valid = scene_frame[scene_frame["reference_status"] == "valid"]
        observed = int(valid["reference_movement_id"].nunique())
        rows.append(
            {
                "scene_id": scene,
                "diagnostic_id": "oracle_polygon_reference",
                "diagnostic_type": "oracle/reference diagnostic, not an unsupervised baseline",
                "valid_reference_trajectories": len(valid),
                "observed_reference_movement_count": observed,
                "non_noise_cluster_count_implied": observed,
                "observed_target_abs_error": 0,
                "ari": 1.0,
                "nmi": 1.0,
                "purity": 1.0,
                "macro_f1": 1.0,
            }
        )
    return pd.DataFrame(rows)


def comparison_to_hg_smg(metrics: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_csv(Paths.original_metrics)
    hg_smg = pd.read_csv(Paths.hg_smg_metrics)
    references = []
    references.append(
        original[original["selection_strategy"] == "untargeted_selection"].assign(
            comparator_id="original_untargeted"
        )
    )
    references.append(
        original[original["selection_strategy"] == "hg_expected_aware_selection"].assign(
            comparator_id="original_hg_aware_A1"
        )
    )
    references.append(
        hg_smg[hg_smg["ablation_id"] == "A5"].assign(
            comparator_id="hg_smg_A5"
        )
    )
    reference_metrics = pd.concat(references, ignore_index=True)
    metric_names = [
        "observed_target_abs_error",
        "legal_target_abs_error",
        "ari",
        "nmi",
        "purity",
        "macro_f1",
        "weighted_f1",
        "noise_pct_all_test",
    ]
    rows = []
    for comparator, comp in reference_metrics.groupby("comparator_id", sort=True):
        merged = metrics.merge(
            comp[["scene_id", "method", *metric_names]],
            on=["scene_id", "method"],
            suffixes=("_baseline", "_comparator"),
            how="inner",
        )
        for baseline_id, frame in merged.groupby("baseline_id", sort=True):
            row = {
                "baseline_id": baseline_id,
                "comparator_id": comparator,
                "scene_method_cases": len(frame),
            }
            for metric in metric_names:
                delta = frame[f"{metric}_baseline"] - frame[f"{metric}_comparator"]
                row[f"mean_delta_{metric}_baseline_minus_comparator"] = float(delta.mean())
                row[f"median_delta_{metric}_baseline_minus_comparator"] = float(delta.median())
            rows.append(row)
    return pd.DataFrame(rows)


def generate_figures() -> list[Path]:
    metrics = pd.read_csv(RESULTS_ROOT / "baseline_metrics.csv")
    hg = pd.read_csv(Paths.hg_smg_metrics)
    a5 = hg[hg["ablation_id"] == "A5"].copy()
    a5["baseline_id"] = "hg_smg_A5"
    combined = pd.concat(
        [
            metrics[["scene_id", "method", "baseline_id", "ari", "nmi", "purity", "macro_f1", "observed_target_abs_error", "noise_pct_all_test"]],
            a5[["scene_id", "method", "baseline_id", "ari", "nmi", "purity", "macro_f1", "observed_target_abs_error", "noise_pct_all_test"]],
        ],
        ignore_index=True,
    )
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(name: str) -> None:
        path = FIGURES_ROOT / name
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        outputs.append(path)
        plt.close()

    summary = combined.groupby("baseline_id")[["ari", "nmi", "purity", "macro_f1"]].mean()
    summary.plot(kind="bar", figsize=(9, 4))
    plt.ylabel("Mean score")
    plt.title("Baseline vs HG-SMG agreement metrics")
    save("baseline_agreement_metrics.png")

    target = combined.groupby("baseline_id")["observed_target_abs_error"].mean()
    target.plot(kind="bar", figsize=(8, 4), color="#4c78a8")
    plt.ylabel("Mean observed target-count error")
    plt.title("Observed target-count error comparison")
    save("baseline_target_error.png")

    outliers = combined.groupby("baseline_id")["noise_pct_all_test"].mean()
    outliers.plot(kind="bar", figsize=(8, 4), color="#f58518")
    plt.ylabel("Mean outlier percentage")
    plt.title("Outlier percentage comparison")
    save("baseline_outlier_percentage.png")

    se38 = combined[combined["scene_id"] == "bellevue_150th_se38th"]
    se38.pivot_table(index="baseline_id", columns="method", values="nmi", aggfunc="mean").plot(
        kind="bar", figsize=(9, 4)
    )
    plt.ylabel("NMI")
    plt.title("SE38th baseline comparison")
    save("baseline_se38th_comparison.png")

    trade = combined.groupby("baseline_id")[["observed_target_abs_error", "nmi"]].mean()
    plt.figure(figsize=(7, 5))
    plt.scatter(trade["observed_target_abs_error"], trade["nmi"])
    for label, row in trade.iterrows():
        plt.annotate(label, (row["observed_target_abs_error"], row["nmi"]), fontsize=8)
    plt.xlabel("Mean observed target-count error")
    plt.ylabel("Mean NMI")
    plt.title("Target agreement vs NMI trade-off")
    save("baseline_tradeoff_target_error_nmi.png")
    return outputs


def write_reports() -> None:
    metrics = pd.read_csv(RESULTS_ROOT / "baseline_metrics.csv")
    comparison = pd.read_csv(RESULTS_ROOT / "baseline_comparison_to_hg_smg.csv")
    runtime = pd.read_csv(RESULTS_ROOT / "baseline_runtime_summary.csv")
    oracle = pd.read_csv(RESULTS_ROOT / "oracle_reference_diagnostic.csv")
    summary = metrics.groupby("baseline_id")[
        ["observed_target_abs_error", "ari", "nmi", "purity", "macro_f1", "noise_pct_all_test"]
    ].agg(["mean", "median"])
    write_text(
        [
            "# Baseline Method Definitions",
            "",
            "All baselines are deterministic, reviewer-focused comparisons on the frozen independent-test split.",
            "",
            "## Endpoint-only baseline",
            "Uses raw camera-space `start_x,start_y,end_x,end_y` endpoint coordinates.",
            "",
            "## Isotropic endpoint baseline",
            "Uses the same endpoint coordinates after camera-space shared-scale isotropic normalization fitted without labels.",
            "",
            "## Resampled-trajectory Euclidean baseline",
            f"Resamples each canonical trajectory interval to `{RESAMPLED_POINT_COUNT}` points, flattens x/y coordinates, and applies a shared camera-space x/y scale.",
            "",
            "## Selection",
            "Candidate grids are small and deterministic. Selection uses only internal clustering metrics before any reference labels are loaded.",
        ],
        DOCS_ROOT / "baseline_method_definitions.md",
    )
    write_text(
        [
            "# Baseline Feasibility Report",
            "",
            f"- DTW/Frechet status: `{DTW_STATUS}`.",
            "- Full pairwise trajectory distances over 27,393 test trajectories would require approximately 750 million pairwise comparisons before method-specific clustering.",
            "- A subsampled DTW/Frechet result would not be directly comparable to the full-cohort independent-test metrics, so it is documented as infeasible for this compact package.",
            "- Oracle polygon grouping is reported separately as a reference diagnostic, not as an unsupervised baseline.",
        ],
        DOCS_ROOT / "baseline_feasibility_report.md",
    )
    result_lines = [
        "# Baseline Independent-Test Results",
        "",
        "## Aggregate Baseline Summary",
        "",
        summary.to_markdown(),
        "",
        "## Runtime Summary",
        "",
        runtime.groupby("baseline_id")[["feature_build_time_s", "selected_trial_fit_time_s", "feature_memory_bytes"]].mean().to_markdown(),
    ]
    write_text(result_lines, DOCS_ROOT / "baseline_independent_test_results.md")
    interp = [
        "# Baseline Scientific Interpretation",
        "",
        "Simple endpoint baselines are strong for several scenes because the maneuver structure is largely endpoint-defined.",
        "The resampled trajectory baseline tests whether full shape adds value beyond endpoints; results should be read scene-by-scene rather than as universal dominance.",
        "HG-SMG-TC remains most relevant where endpoint-region over-segmentation affects target selection, especially SE38th KMeans.",
        "",
        "## Comparison to HG-SMG",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Oracle Diagnostic",
        "",
        oracle.to_markdown(index=False),
    ]
    write_text(interp, DOCS_ROOT / "baseline_scientific_interpretation.md")
    execution = [
        "# Task 11 Execution Report",
        "",
        f"- Branch: `{git_branch()}`",
        f"- Git HEAD: `{git_head()}`",
        f"- Assignment rows: `{len(pd.read_parquet(RESULTS_ROOT / 'baseline_assignments.parquet'))}`",
        f"- Metrics rows: `{len(metrics)}`",
        f"- DTW/Frechet: `{DTW_STATUS}`",
        "- Reference labels were loaded only after assignment persistence.",
        "- No frozen result, protocol hash, model-selection configuration, reference label, homography, target output, or manuscript DOCX was modified.",
    ]
    write_text(execution, DOCS_ROOT / "task_11_execution_report.md")


def run_all() -> dict[str, Any]:
    run_assignments()
    evaluation = run_evaluation()
    figures = generate_figures()
    write_reports()
    return {
        "evaluation": evaluation,
        "figures": [str(path) for path in figures],
        "metrics": str(RESULTS_ROOT / "baseline_metrics.csv"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("assign")
    sub.add_parser("evaluate")
    sub.add_parser("figures")
    sub.add_parser("reports")
    sub.add_parser("all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        print(preflight()["status"])
    elif args.command == "assign":
        print(run_assignments()["assignment_rows"])
    elif args.command == "evaluate":
        print(run_evaluation()["evaluation_timestamp_utc"])
    elif args.command == "figures":
        print(len(generate_figures()))
    elif args.command == "reports":
        write_reports()
        print("reports")
    elif args.command == "all":
        print(run_all()["metrics"])


if __name__ == "__main__":
    main()
