"""Leakage-controlled, split-aware HG-MSA-TC publication runner.

Usage from the repository root::

    python publications/hg-msa-tc-future-transportation/code/pipeline/run_split_aware_hg_msa_tc.py target
    python publications/hg-msa-tc-future-transportation/code/pipeline/run_split_aware_hg_msa_tc.py select
    python publications/hg-msa-tc-future-transportation/code/pipeline/run_split_aware_hg_msa_tc.py freeze
    python publications/hg-msa-tc-future-transportation/code/pipeline/run_split_aware_hg_msa_tc.py test

The real test command remains locked unless an external unlock artifact and an
explicit CLI confirmation are both present. The synthetic test path does not read
repository trajectories.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import hg_msa_tc_core as core  # noqa: E402
import split_aware_io as protocol_io  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
PUBLICATION_ROOT = REPO_ROOT / "publications" / "hg-msa-tc-future-transportation"
DEFAULT_CONFIG = PUBLICATION_ROOT / "configs" / "split_aware_runner.yaml"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def load_config(path: Path) -> dict[str, Any]:
    config = protocol_io.load_yaml(path)
    scenes = tuple(config.get("scenes", []))
    required_scenes = (
        "bellevue_116th_ne12th",
        "bellevue_150th_newport",
        "bellevue_150th_eastgate",
        "bellevue_150th_se38th",
        "bellevue_ne8th",
    )
    if scenes != required_scenes:
        raise ValueError(
            f"Runner must use exactly the five fixed Bellevue scenes: {scenes}"
        )
    if tuple(config["coordinate_representation"]["feature_columns"]) != core.FEATURE_COLUMNS:
        raise ValueError("The frozen feature column order has changed.")
    return config


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    development = resolve_repo_path(config["outputs"]["development_directory"])
    return {
        "development": development,
        "access_log": resolve_repo_path(config["outputs"]["access_log"]),
        "target_estimates": development / "target_estimates.csv",
        "target_candidates": development / "target_candidates.csv",
        "target_region_candidates": development / "target_region_candidates.csv",
        "target_od_counts": development / "target_od_support_counts.csv",
        "target_provenance": development / "target_estimation_provenance.json",
        "selection_candidates": development / "model_selection_candidates.csv",
        "selected_configurations": development / "selected_configurations.csv",
        "selection_provenance": development / "model_selection_provenance.json",
        "benchmark": development / "sampling_benchmark.csv",
        "frozen_protocol": resolve_repo_path(config["outputs"]["frozen_protocol"]),
        "frozen_manifest": resolve_repo_path(config["outputs"]["frozen_manifest"]),
        "frozen_report": PUBLICATION_ROOT / "docs" / "frozen_protocol_report.md",
    }


def input_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: resolve_repo_path(value) for key, value in config["inputs"].items()
    }


def guard_development_mutation(
    config: dict[str, Any], paths: dict[str, Path], force_new_version: bool
) -> None:
    frozen_path = paths["frozen_protocol"]
    if not frozen_path.exists():
        return
    frozen = protocol_io.load_yaml(frozen_path)
    protocol_io.validate_frozen_payload(frozen)
    if not force_new_version:
        raise PermissionError(
            "The development protocol is frozen. Use a new versioned config and "
            "--force-new-protocol-version; existing frozen outputs are immutable."
        )
    if config["protocol_version"] == frozen.get("protocol_version"):
        raise PermissionError(
            "--force-new-protocol-version requires a different protocol_version."
        )
    frozen_development = frozen.get("paths", {}).get("development_directory")
    current_development = relative(paths["development"])
    if frozen_development == current_development:
        raise PermissionError(
            "A new protocol version must use a new development output directory."
        )


def load_phase_metadata(
    config: dict[str, Any], phase: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = input_paths(config)
    scenes = tuple(config["scenes"])
    membership = protocol_io.load_split_membership(
        paths["evaluation_split"], phase, scenes
    )
    metadata = protocol_io.load_manifest_metadata(
        paths["trajectory_manifest"], membership
    )
    protocol_io.assert_phase_rows(metadata, phase)
    return membership, metadata


def add_control_access_records(
    logger: protocol_io.AccessLogger,
    config: dict[str, Any],
    membership: pd.DataFrame,
) -> None:
    paths = input_paths(config)
    observed = membership["split"].unique()
    logger.add(
        "ALL",
        Path(relative(paths["evaluation_split"])),
        observed,
        len(membership),
        protocol_io.sha256_file(paths["evaluation_split"]),
        "split_control_index",
    )
    logger.add(
        "ALL",
        Path(relative(paths["trajectory_manifest"])),
        observed,
        len(membership),
        protocol_io.sha256_file(paths["trajectory_manifest"]),
        "trajectory_control_manifest",
    )


def homography_path(config: dict[str, Any], scene: str) -> Path:
    directory = resolve_repo_path(
        config["inputs"]["homography_config_directory"]
    )
    return directory / f"homography_{scene}.json"


def load_homography(config: dict[str, Any], scene: str) -> np.ndarray:
    path = homography_path(config, scene)
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["homography_camera_to_topview"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid homography shape for {scene}: {matrix.shape}")
    return matrix


def validate_homography_index(config: dict[str, Any]) -> None:
    index_path = input_paths(config)["homography_index"]
    index = protocol_io.load_yaml(index_path)
    rows = {row["scene"]: row for row in index["scenes"]}
    if tuple(rows) != tuple(config["scenes"]):
        raise ValueError("Homography index does not match the five fixed scenes.")
    unusable = [scene for scene, row in rows.items() if not bool(row.get("usable"))]
    if unusable:
        raise ValueError(f"Unusable homographies in frozen scene set: {unusable}")
    for scene in config["scenes"]:
        if not homography_path(config, scene).exists():
            raise FileNotFoundError(homography_path(config, scene))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def target_phase(config: dict[str, Any], paths: dict[str, Path]) -> None:
    validate_homography_index(config)
    membership, metadata = load_phase_metadata(config, "target")
    logger = protocol_io.AccessLogger(paths["access_log"], "target")
    add_control_access_records(logger, config, membership)
    summaries: list[dict[str, Any]] = []
    threshold_tables: list[pd.DataFrame] = []
    region_tables: list[pd.DataFrame] = []
    od_tables: list[pd.DataFrame] = []
    scene_provenance: dict[str, Any] = {}
    random_seed = int(config["random_seed"])
    target_config = config["target_estimation"]

    for scene_index, scene in enumerate(config["scenes"]):
        scene_metadata = metadata[metadata["scene_id"] == scene].copy()
        features = protocol_io.load_scene_features(
            REPO_ROOT, scene_metadata, "target", logger
        )
        endpoints = core.transform_camera_endpoints(
            features, load_homography(config, scene)
        )
        seed = random_seed + scene_index * 1000
        summary, threshold_candidates, region_candidates, od_counts = (
            core.estimate_hg_target(
                scene,
                endpoints,
                seed,
                [int(value) for value in target_config["region_counts"]],
                [float(value) for value in target_config["support_thresholds"]],
                int(target_config["region_metric_sample_size"]),
            )
        )
        summaries.append(summary)
        threshold_tables.append(threshold_candidates)
        region_tables.append(region_candidates)
        od_counts.insert(0, "scene", scene)
        od_tables.append(od_counts)
        source_path = REPO_ROOT / scene_metadata["source_file"].iloc[0]
        scene_provenance[scene] = {
            "number_of_trajectories": int(len(features)),
            "split": "target_estimation",
            "sampling_applied": False,
            "feature_file": relative(source_path),
            "feature_file_sha256": protocol_io.sha256_file(source_path),
            "homography_file": relative(homography_path(config, scene)),
            "homography_file_sha256": protocol_io.sha256_file(
                homography_path(config, scene)
            ),
            "random_seed": seed,
        }

    estimates = pd.DataFrame(summaries)
    estimates = estimates.sort_values("scene", key=lambda series: series.map(
        {scene: index for index, scene in enumerate(config["scenes"])}
    ))
    write_csv(estimates, paths["target_estimates"])
    write_csv(pd.concat(threshold_tables, ignore_index=True), paths["target_candidates"])
    write_csv(
        pd.concat(region_tables, ignore_index=True),
        paths["target_region_candidates"],
    )
    write_csv(pd.concat(od_tables, ignore_index=True), paths["target_od_counts"])
    logger.flush()

    inputs = input_paths(config)
    provenance = {
        "protocol_version": config["protocol_version"],
        "implementation_version": core.IMPLEMENTATION_VERSION,
        "phase": "target",
        "allowed_split": "target_estimation",
        "forbidden_inputs": [
            "model_selection feature rows",
            "independent_test feature rows",
            "manual annotations",
            "historical declared targets",
            "previous full-data selected results",
        ],
        "created_at_utc": protocol_io.utc_timestamp(),
        "sampling_applied": False,
        "sampling_limit": None,
        "scene_provenance": scene_provenance,
        "input_checksums": {
            relative(inputs["trajectory_manifest"]): protocol_io.sha256_file(
                inputs["trajectory_manifest"]
            ),
            relative(inputs["evaluation_split"]): protocol_io.sha256_file(
                inputs["evaluation_split"]
            ),
            relative(inputs["homography_index"]): protocol_io.sha256_file(
                inputs["homography_index"]
            ),
        },
        "output_checksums": {
            relative(path): protocol_io.sha256_file(path)
            for path in (
                paths["target_estimates"],
                paths["target_candidates"],
                paths["target_region_candidates"],
                paths["target_od_counts"],
            )
        },
    }
    protocol_io.write_json_atomic(paths["target_provenance"], provenance)
    print(estimates[["scene", "hg_estimated_target", "n_trajectories"]].to_string(index=False))


def verify_output_checksum(provenance: dict[str, Any], path: Path) -> None:
    expected = provenance["output_checksums"].get(relative(path))
    if expected is None:
        raise ValueError(f"Output is not covered by provenance: {path}")
    actual = protocol_io.sha256_file(path)
    if actual != expected:
        raise ValueError(f"Output checksum mismatch for {path}")


def json_selection_key(key: tuple[Any, ...]) -> str:
    values = []
    for value in key:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            values.append("Infinity" if value > 0 else "-Infinity")
        elif isinstance(value, np.generic):
            values.append(value.item())
        else:
            values.append(value)
    return json.dumps(values, ensure_ascii=True)


def selection_phase(config: dict[str, Any], paths: dict[str, Path]) -> None:
    target_provenance = json.loads(paths["target_provenance"].read_text(encoding="utf-8"))
    verify_output_checksum(target_provenance, paths["target_estimates"])
    target_estimates = pd.read_csv(paths["target_estimates"])
    expected_scenes = tuple(config["scenes"])
    if tuple(target_estimates["scene"]) != expected_scenes:
        raise ValueError("Target output does not match the fixed scene order.")
    target_map = dict(
        zip(target_estimates["scene"], target_estimates["hg_estimated_target"])
    )

    membership, metadata = load_phase_metadata(config, "select")
    logger = protocol_io.AccessLogger(paths["access_log"], "select")
    add_control_access_records(logger, config, membership)
    all_trials: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    normalization_by_scene: dict[str, dict[str, float]] = {}
    random_seed = int(config["random_seed"])

    for scene_index, scene in enumerate(config["scenes"]):
        scene_metadata = metadata[metadata["scene_id"] == scene].copy()
        frame = protocol_io.load_scene_features(
            REPO_ROOT, scene_metadata, "select", logger
        )
        values = frame[list(core.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        features, normalization = core.isotropic_normalize(values)
        normalization_by_scene[scene] = normalization
        scene_seed = random_seed + scene_index * 1000
        for method in core.METHODS:
            print(
                f"Selecting candidates: {scene} / {method} / n={len(features)}",
                flush=True,
            )
            trials = core.run_candidate_trials(
                scene,
                method,
                features,
                int(target_map[scene]),
                scene_seed,
                config,
            )
            trials["untargeted_selection_key_json"] = trials.apply(
                lambda row: json_selection_key(core.untargeted_selection_key(row)),
                axis=1,
            )
            trials["hg_expected_selection_key_json"] = trials.apply(
                lambda row: json_selection_key(
                    core.expected_selection_key(row, method)
                ),
                axis=1,
            )
            selected_by_strategy: dict[str, pd.Series] = {}
            for strategy in (
                "untargeted_selection",
                "hg_expected_aware_selection",
            ):
                selected = core.select_candidate(trials, strategy, method)
                selected_by_strategy[strategy] = selected
                selected_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "selection_strategy": strategy,
                        "split": "model_selection",
                        "n_trajectories": int(len(features)),
                        "hg_estimated_target": int(target_map[scene]),
                        "coordinate_representation": "camera_isotropic_shared_scale",
                        "feature_columns_json": json.dumps(core.FEATURE_COLUMNS),
                        "normalization_parameters_json": json.dumps(
                            normalization, sort_keys=True
                        ),
                        "params_json": selected["params_json"],
                        "fit_random_seed": int(selected["fit_random_seed"]),
                        "selection_key_json": selected["selection_key_json"],
                        "trial_index": int(selected["trial_index"]),
                        "n_clusters": int(selected["n_clusters"]),
                        "cluster_count_error": int(
                            selected["cluster_count_error"]
                        ),
                        "pct_outliers": float(selected["pct_outliers"]),
                        "largest_cluster_ratio": float(
                            selected["largest_cluster_ratio"]
                        ),
                        "silhouette_clustered_only": float(
                            selected["silhouette_clustered_only"]
                        ),
                        "davies_bouldin_clustered_only": float(
                            selected["davies_bouldin_clustered_only"]
                        ),
                        "calinski_harabasz_clustered_only": float(
                            selected["calinski_harabasz_clustered_only"]
                        ),
                        "quick_score": float(selected["quick_score"]),
                        "EMAS_HG": float(selected["EMAS_HG"]),
                    }
                )
            trials["selected_untargeted"] = (
                trials["trial_index"]
                == int(selected_by_strategy["untargeted_selection"]["trial_index"])
            )
            trials["selected_hg_expected_aware"] = (
                trials["trial_index"]
                == int(
                    selected_by_strategy["hg_expected_aware_selection"][
                        "trial_index"
                    ]
                )
            )
            all_trials.append(trials)

    candidates = pd.concat(all_trials, ignore_index=True)
    selected_configurations = pd.DataFrame(selected_rows)
    write_csv(candidates, paths["selection_candidates"])
    write_csv(selected_configurations, paths["selected_configurations"])
    logger.flush()

    inputs = input_paths(config)
    provenance = {
        "protocol_version": config["protocol_version"],
        "implementation_version": core.IMPLEMENTATION_VERSION,
        "phase": "select",
        "allowed_split": "model_selection",
        "created_at_utc": protocol_io.utc_timestamp(),
        "target_output": relative(paths["target_estimates"]),
        "target_output_sha256": protocol_io.sha256_file(paths["target_estimates"]),
        "target_provenance_sha256": protocol_io.sha256_file(
            paths["target_provenance"]
        ),
        "sampling_applied": False,
        "sampling_limit": None,
        "normalization_fitted_only_on": "model_selection",
        "normalization_by_scene": normalization_by_scene,
        "optics_max_eps_fitted_only_on": "model_selection",
        "selection_rules": config["selection_rules"],
        "input_checksums": {
            relative(inputs["trajectory_manifest"]): protocol_io.sha256_file(
                inputs["trajectory_manifest"]
            ),
            relative(inputs["evaluation_split"]): protocol_io.sha256_file(
                inputs["evaluation_split"]
            ),
        },
        "output_checksums": {
            relative(path): protocol_io.sha256_file(path)
            for path in (
                paths["selection_candidates"],
                paths["selected_configurations"],
            )
        },
    }
    protocol_io.write_json_atomic(paths["selection_provenance"], provenance)
    print(
        selected_configurations[
            [
                "scene",
                "method",
                "selection_strategy",
                "n_clusters",
                "cluster_count_error",
                "params_json",
            ]
        ].to_string(index=False)
    )


def benchmark_phase(config: dict[str, Any], paths: dict[str, Path]) -> None:
    membership, metadata = load_phase_metadata(config, "select")
    counts = membership.groupby("scene_id").size()
    scene = str(counts.idxmax())
    logger = protocol_io.AccessLogger(paths["access_log"], "select_benchmark")
    # The benchmark is selection-only; use a dedicated local record then append as select.
    logger.phase = "select"
    frame = protocol_io.load_scene_features(
        REPO_ROOT,
        metadata[metadata["scene_id"] == scene].copy(),
        "select",
        logger,
    )
    features, _ = core.isotropic_normalize(
        frame[list(core.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    )
    representative = {
        "kmeans": {"n_clusters": 12, "n_init": 10, "max_iter": 300},
        "hdbscan": {"min_cluster_size": 160, "min_samples": 20},
    }
    eps_values = core.optics_eps_grid(
        features,
        int(config["random_seed"]),
        config["candidate_grids"]["optics"]["max_eps_quantiles"],
        int(config["candidate_grids"]["optics"]["quantile_sample_size"]),
    )
    representative["optics"] = {
        "min_samples": 80,
        "xi": 0.05,
        "max_eps": eps_values[2],
    }
    rows = []
    for method, parameters in representative.items():
        started = time.perf_counter()
        labels = core.fit_predict(
            method, features, parameters, int(config["random_seed"])
        )
        rows.append(
            {
                "scene": scene,
                "method": method,
                "n_trajectories": len(features),
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "fit_time_s": time.perf_counter() - started,
                "n_clusters": len(np.unique(labels[labels >= 0])),
                "pct_outliers": 100.0 * float((labels == -1).mean()),
            }
        )
    write_csv(pd.DataFrame(rows), paths["benchmark"])
    logger.flush()
    print(pd.DataFrame(rows).to_string(index=False))


def git_commit_hash() -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPO_ROOT.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def software_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "hdbscan", "pyarrow", "PyYAML"]
    output = {"python": platform.python_version()}
    for package in packages:
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = "not-installed"
    return output


def freeze_phase(config: dict[str, Any], paths: dict[str, Path], config_path: Path) -> None:
    if paths["frozen_protocol"].exists():
        existing = protocol_io.load_yaml(paths["frozen_protocol"])
        frozen_hash = protocol_io.validate_frozen_payload(existing)
        print(f"Frozen protocol already exists and validates: {frozen_hash}")
        return

    target_provenance = json.loads(paths["target_provenance"].read_text(encoding="utf-8"))
    selection_provenance = json.loads(paths["selection_provenance"].read_text(encoding="utf-8"))
    verify_output_checksum(target_provenance, paths["target_estimates"])
    verify_output_checksum(selection_provenance, paths["selected_configurations"])
    verify_output_checksum(selection_provenance, paths["selection_candidates"])
    selected = pd.read_csv(paths["selected_configurations"])
    targets = pd.read_csv(paths["target_estimates"])
    if len(selected) != len(config["scenes"]) * len(core.METHODS) * 2:
        raise ValueError("Selected configuration table is incomplete.")
    if tuple(targets["scene"]) != tuple(config["scenes"]):
        raise ValueError("Target table is incomplete or out of order.")

    inputs = input_paths(config)
    manifest_source_rows = pd.read_csv(
        inputs["trajectory_manifest"],
        usecols=["scene_id", "source_file", "source_file_checksum_sha256"],
    ).drop_duplicates()
    source_checksums = {
        str(row.scene_id): {
            "path": str(row.source_file),
            "sha256": str(row.source_file_checksum_sha256),
        }
        for row in manifest_source_rows.itertuples(index=False)
    }
    for scene, source in source_checksums.items():
        actual = protocol_io.sha256_file(REPO_ROOT / source["path"])
        if actual != source["sha256"]:
            raise ValueError(f"Source checksum changed for {scene}")

    homography_checksums = {
        scene: {
            "path": relative(homography_path(config, scene)),
            "sha256": protocol_io.sha256_file(homography_path(config, scene)),
        }
        for scene in config["scenes"]
    }
    target_records = targets.where(pd.notna(targets), None).to_dict(orient="records")
    selected_records = selected.where(pd.notna(selected), None).to_dict(orient="records")
    payload: dict[str, Any] = {
        "protocol_version": config["protocol_version"],
        "implementation_version": core.IMPLEMENTATION_VERSION,
        "creation_timestamp_utc": protocol_io.utc_timestamp(),
        "git_commit_hash": git_commit_hash(),
        "independent_test_locked": True,
        "test_protocol": config["test_protocol"],
        "paths": {
            "development_directory": relative(paths["development"]),
            "trajectory_manifest": relative(inputs["trajectory_manifest"]),
            "evaluation_split": relative(inputs["evaluation_split"]),
        },
        "checksums": {
            "runner_config": {
                "path": relative(config_path),
                "sha256": protocol_io.sha256_file(config_path),
            },
            "trajectory_manifest": protocol_io.sha256_file(
                inputs["trajectory_manifest"]
            ),
            "evaluation_split": protocol_io.sha256_file(
                inputs["evaluation_split"]
            ),
            "split_config": protocol_io.sha256_file(inputs["split_config"]),
            "homography_index": protocol_io.sha256_file(
                inputs["homography_index"]
            ),
            "homography_configurations": homography_checksums,
            "target_estimates": protocol_io.sha256_file(
                paths["target_estimates"]
            ),
            "target_candidates": protocol_io.sha256_file(
                paths["target_candidates"]
            ),
            "target_region_candidates": protocol_io.sha256_file(
                paths["target_region_candidates"]
            ),
            "target_provenance": protocol_io.sha256_file(
                paths["target_provenance"]
            ),
            "model_selection_candidates": protocol_io.sha256_file(
                paths["selection_candidates"]
            ),
            "selected_configurations": protocol_io.sha256_file(
                paths["selected_configurations"]
            ),
            "model_selection_provenance": protocol_io.sha256_file(
                paths["selection_provenance"]
            ),
            "source_feature_files": source_checksums,
        },
        "target_outputs": target_records,
        "selected_configurations": selected_records,
        "candidate_grid_definition": config["candidate_grids"],
        "selection_rules": config["selection_rules"],
        "tie_breaking": (
            "stable mergesort semantics with canonical lexical params_json as the "
            "final deterministic tie-breaker; runtime is never a tie-breaker"
        ),
        "random_seeds": {
            "base": int(config["random_seed"]),
            "scene_offset": 1000,
            "candidate_fit_offset": "one-based trial index",
        },
        "sampling_limits": config["sampling"],
        "coordinate_representation": config["coordinate_representation"],
        "software_versions": software_versions(),
    }
    payload["complete_frozen_configuration_sha256"] = protocol_io.canonical_sha256(
        payload
    )
    protocol_io.write_yaml_atomic(paths["frozen_protocol"], payload)
    manifest = {
        "protocol_version": payload["protocol_version"],
        "frozen_protocol_path": relative(paths["frozen_protocol"]),
        "frozen_protocol_file_sha256": protocol_io.sha256_file(
            paths["frozen_protocol"]
        ),
        "complete_frozen_configuration_sha256": payload[
            "complete_frozen_configuration_sha256"
        ],
        "git_commit_hash": payload["git_commit_hash"],
        "independent_test_locked": True,
        "created_at_utc": payload["creation_timestamp_utc"],
    }
    protocol_io.write_json_atomic(paths["frozen_manifest"], manifest)
    report = f"""# Frozen Split-Aware HG-MSA-TC Protocol

- Protocol version: `{payload['protocol_version']}`
- Git implementation commit: `{payload['git_commit_hash']}`
- Frozen configuration hash: `{payload['complete_frozen_configuration_sha256']}`
- Independent test locked: **yes**
- Target rows were read only from `target_estimation`.
- Candidate selection rows were read only from `model_selection`.
- No manual labels were read by either phase.
- The real `independent_test` feature cohort was not loaded.

## Final Test Design

The later final evaluation is transductive. Each frozen method and every selected
hyperparameter will be kept fixed, then the clustering method will be fitted on the
independent-test feature vectors without labels. Cluster assignments must be written
and checksummed before a separate evaluation step may read manual labels. KMeans,
HDBSCAN, and OPTICS therefore follow one consistent test protocol.

## Frozen Artifacts

- `{relative(paths['frozen_protocol'])}`
- `{relative(paths['frozen_manifest'])}`
- `{relative(paths['target_estimates'])}`
- `{relative(paths['selected_configurations'])}`
"""
    protocol_io.write_text_atomic(paths["frozen_report"], report)
    print(payload["complete_frozen_configuration_sha256"])


def validate_frozen_inputs(
    frozen: dict[str, Any], config_path: Path, paths: dict[str, Path]
) -> str:
    frozen_hash = protocol_io.validate_frozen_payload(frozen)
    checksums = frozen["checksums"]
    current = {
        "runner_config": protocol_io.sha256_file(config_path),
        "trajectory_manifest": protocol_io.sha256_file(
            resolve_repo_path(frozen["paths"]["trajectory_manifest"])
        ),
        "evaluation_split": protocol_io.sha256_file(
            resolve_repo_path(frozen["paths"]["evaluation_split"])
        ),
        "target_estimates": protocol_io.sha256_file(paths["target_estimates"]),
        "selected_configurations": protocol_io.sha256_file(
            paths["selected_configurations"]
        ),
    }
    expected = {
        "runner_config": checksums["runner_config"]["sha256"],
        "trajectory_manifest": checksums["trajectory_manifest"],
        "evaluation_split": checksums["evaluation_split"],
        "target_estimates": checksums["target_estimates"],
        "selected_configurations": checksums["selected_configurations"],
    }
    if current != expected:
        raise ValueError(f"Frozen input hash mismatch: {current} != {expected}")
    for scene, item in checksums["homography_configurations"].items():
        if protocol_io.sha256_file(resolve_repo_path(item["path"])) != item["sha256"]:
            raise ValueError(f"Frozen homography changed for {scene}")
    for scene, item in checksums["source_feature_files"].items():
        if protocol_io.sha256_file(resolve_repo_path(item["path"])) != item["sha256"]:
            raise ValueError(f"Frozen source feature changed for {scene}")
    return frozen_hash


def run_synthetic_transductive_smoke(output_dir: Path) -> Path:
    rng = np.random.default_rng(20260702)
    clusters = []
    for center in ((0.1, 0.1, 0.8, 0.8), (0.8, 0.1, 0.1, 0.8), (0.1, 0.8, 0.8, 0.1)):
        clusters.append(rng.normal(center, 0.025, size=(60, 4)))
    features = np.vstack(clusters)
    parameters = {
        "kmeans": {"n_clusters": 3, "n_init": 10, "max_iter": 300},
        "hdbscan": {"min_cluster_size": 15, "min_samples": 5},
        "optics": {"min_samples": 10, "xi": 0.05, "max_eps": 0.20},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame({"synthetic_trajectory_id": range(len(features))})
    for method in core.METHODS:
        assignments[method] = core.fit_predict(
            method, features, parameters[method], 20260702
        )
    path = output_dir / "synthetic_transductive_assignments.csv"
    write_csv(assignments, path)
    protocol_io.write_json_atomic(
        output_dir / "synthetic_transductive_assignments_provenance.json",
        {
            "synthetic_fixture": True,
            "manual_labels_read": False,
            "assignment_sha256": protocol_io.sha256_file(path),
            "methods": list(core.METHODS),
        },
    )
    return path


def test_phase(
    config: dict[str, Any],
    paths: dict[str, Path],
    config_path: Path,
    confirmation: str | None,
    synthetic_fixture: bool,
) -> None:
    if synthetic_fixture:
        output = run_synthetic_transductive_smoke(
            paths["development"] / "synthetic_test_smoke"
        )
        print(output)
        return

    frozen = protocol_io.load_yaml(paths["frozen_protocol"])
    frozen_hash = validate_frozen_inputs(frozen, config_path, paths)
    unlock_path = resolve_repo_path(config["test_protocol"]["unlock_file"])
    protocol_io.require_test_unlock(
        unlock_path,
        frozen_hash,
        confirmation,
        config["test_protocol"]["required_cli_confirmation"],
    )
    membership, metadata = load_phase_metadata(config, "test")
    logger = protocol_io.AccessLogger(paths["access_log"], "test")
    add_control_access_records(logger, config, membership)
    selected = pd.read_csv(paths["selected_configurations"])
    frozen_rows = selected[
        selected["selection_strategy"] == "hg_expected_aware_selection"
    ]
    output_dir = resolve_repo_path(
        config["test_protocol"]["clustering_output_directory"]
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    checksum_records = []
    for scene in config["scenes"]:
        frame = protocol_io.load_scene_features(
            REPO_ROOT,
            metadata[metadata["scene_id"] == scene].copy(),
            "test",
            logger,
        )
        values = frame[list(core.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        scene_rows = frozen_rows[frozen_rows["scene"] == scene]
        normalization = json.loads(
            scene_rows["normalization_parameters_json"].iloc[0]
        )
        features, _ = core.isotropic_normalize(values, normalization)
        for row in scene_rows.itertuples(index=False):
            labels = core.fit_predict(
                row.method,
                features,
                json.loads(row.params_json),
                int(row.fit_random_seed),
            )
            assignment_path = output_dir / f"{scene}_{row.method}_assignments.csv"
            write_csv(
                pd.DataFrame(
                    {
                        "trajectory_id": frame["trajectory_id"],
                        "cluster_id": labels,
                    }
                ),
                assignment_path,
            )
            checksum_records.append(
                {
                    "scene": scene,
                    "method": row.method,
                    "path": relative(assignment_path),
                    "sha256": protocol_io.sha256_file(assignment_path),
                    "manual_labels_read": False,
                }
            )
    logger.flush()
    protocol_io.write_json_atomic(
        output_dir / "clustering_output_manifest.json",
        {
            "frozen_protocol_sha256": frozen_hash,
            "assignments": checksum_records,
            "manual_labels_read": False,
            "evaluation_not_started": True,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["benchmark", "target", "select", "freeze", "test"]
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotation-input", action="append", default=[])
    parser.add_argument("--force-new-protocol-version", action="store_true")
    parser.add_argument("--confirm-independent-test")
    parser.add_argument("--synthetic-fixture", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    paths = output_paths(config)
    protocol_io.reject_annotation_inputs(args.annotation_input, args.phase)
    if args.phase in {"benchmark", "target", "select"}:
        guard_development_mutation(
            config, paths, args.force_new_protocol_version
        )
    if args.phase == "benchmark":
        benchmark_phase(config, paths)
    elif args.phase == "target":
        target_phase(config, paths)
    elif args.phase == "select":
        selection_phase(config, paths)
    elif args.phase == "freeze":
        freeze_phase(config, paths, config_path)
    elif args.phase == "test":
        test_phase(
            config,
            paths,
            config_path,
            args.confirm_independent_test,
            args.synthetic_fixture,
        )


if __name__ == "__main__":
    main()
