"""Task 08 homography reproduction, quality, and sensitivity analyses."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from homography.calibration import (
    HISTORICAL_CONFIDENCE,
    HISTORICAL_MAX_ITERS,
    HISTORICAL_RANSAC_THRESHOLD_PX,
    IMPLEMENTATION_VERSION,
    align_labels_and_change_rate,
    apply_homography,
    classify_homography_quality,
    convex_hull_coverage,
    estimate_historical_homography,
    normalized_dlt_condition,
    normalize_homography,
    reprojection_statistics,
)
from target_estimation.hg_target_estimator import (
    endpoint_features,
    estimate_hg_target_detailed,
    transform_camera_endpoints,
)


SCENES = (
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
)
PERTURBATION_SCALES_PX = (1.0, 2.0, 3.0, 5.0, 10.0)
DEFAULT_PERTURBATION_REPLICATES = 20
PERTURBATION_BASE_SEED = 20260808
NE8TH_EXCLUDED_POINT_IDS = (12, 13, 15, 21, 23)


@dataclass(frozen=True)
class Paths:
    publication: Path
    repo: Path
    results: Path
    figures: Path
    docs: Path
    calibration_inputs: Path
    validation_outputs: Path
    homography_configs: Path
    target_assignments: Path
    runner_config: Path


@dataclass
class SceneInputs:
    scene: str
    points: pd.DataFrame
    point_path: Path
    camera_image_path: Path
    topview_image_path: Path
    camera_size: tuple[int, int]
    topview_size: tuple[int, int]
    frozen_matrix: np.ndarray
    frozen_mask: np.ndarray
    config_path: Path
    config_payload: dict[str, Any]
    camera_endpoints: pd.DataFrame
    frozen_topview_endpoints: pd.DataFrame


def default_paths() -> Paths:
    publication = Path(__file__).resolve().parents[2]
    repo = publication.parents[1]
    workspace = repo / "research_experiments/fov2026_trajectory_clustering"
    return Paths(
        publication=publication,
        repo=repo,
        results=publication / "results/homography",
        figures=publication / "figures/homography",
        docs=publication / "docs",
        calibration_inputs=(
            workspace / "outputs/hg_msa_tc_five_scene/calibration_inputs"
        ),
        validation_outputs=(
            workspace / "outputs/hg_msa_tc_five_scene/homography_validation"
        ),
        homography_configs=workspace / "configs/hg_msa_tc_five_scene",
        target_assignments=(
            publication
            / "results/target_estimation/target_endpoint_region_assignments.parquet"
        ),
        runner_config=publication / "configs/split_aware_runner.yaml",
    )


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def relative(paths: Paths, path: Path) -> str:
    return path.relative_to(paths.repo).as_posix()


def point_path(paths: Paths, scene: str) -> Path:
    if scene == "bellevue_ne8th":
        return (
            paths.validation_outputs
            / scene
            / "point_pair_template_excluding_problem_points_12_13_15_21_23.csv"
        )
    return paths.calibration_inputs / scene / "point_pair_template.csv"


def immutable_input_paths(paths: Paths) -> tuple[Path, ...]:
    fixed = [
        paths.runner_config,
        paths.target_assignments,
        paths.publication / "results/development/target_estimates.csv",
        paths.publication / "results/target_estimation/target_reproduction.csv",
        paths.publication
        / "results/target_estimation/cross_scene_semantic_duplication.csv",
        paths.publication / "configs/frozen_evaluation_protocol.yaml",
    ]
    for scene in SCENES:
        fixed.extend(
            [
                paths.homography_configs / f"homography_{scene}.json",
                point_path(paths, scene),
                paths.calibration_inputs / scene / "representative_camera_frame.png",
                paths.calibration_inputs / scene / "topview_reference_image.png",
            ]
        )
    return tuple(fixed)


def create_input_manifest(paths: Paths) -> dict[str, Any]:
    files = {}
    for path in immutable_input_paths(paths):
        if not path.exists():
            raise FileNotFoundError(f"Required immutable Task 08 input is missing: {path}")
        files[relative(paths, path)] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    payload = {
        "task": "futuretransp-homography-quality-sensitivity",
        "created_at_utc": utc_timestamp(),
        "git_commit_before_analysis": git_head(paths.repo),
        "implementation_version": IMPLEMENTATION_VERSION,
        "target_estimation_split_only_for_propagation": True,
        "independent_test_clustering_executed": False,
        "frozen_homographies_modified": False,
        "frozen_targets_modified": False,
        "quality_gate_tuned_against_reference_metrics": False,
        "files": files,
    }
    write_json(paths.results / "task_08_input_manifest.json", payload)
    return payload


def verify_immutable_inputs(paths: Paths, manifest: dict[str, Any]) -> None:
    changed = []
    for name, metadata in manifest["files"].items():
        path = paths.repo / name
        if not path.exists() or sha256_file(path) != metadata["sha256"]:
            changed.append(name)
    if changed:
        raise RuntimeError(f"Frozen Task 08 inputs changed: {changed}")


def load_scene_inputs(paths: Paths) -> dict[str, SceneInputs]:
    all_endpoints = pd.read_parquet(paths.target_assignments)
    if set(all_endpoints["split"].unique()) != {"target_estimation"}:
        raise ValueError("Homography propagation input must contain target_estimation only.")
    scenes: dict[str, SceneInputs] = {}
    for scene in SCENES:
        csv_path = point_path(paths, scene)
        points = pd.read_csv(csv_path)
        required = ["pair_id", "camera_x", "camera_y", "topview_x", "topview_y"]
        if not set(required).issubset(points.columns):
            raise ValueError(f"Missing calibration columns for {scene}.")
        points = points.dropna(subset=required).copy()
        for column in required:
            points[column] = pd.to_numeric(points[column], errors="raise")
        config_path = paths.homography_configs / f"homography_{scene}.json"
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        frozen_matrix = normalize_homography(
            np.asarray(config_payload["homography_camera_to_topview"], dtype=np.float64)
        )
        frozen_mask = np.asarray(config_payload["inlier_mask"], dtype=np.int8)
        if len(points) != len(frozen_mask):
            raise ValueError(f"Point/mask count mismatch for {scene}.")
        scene_rows = all_endpoints[all_endpoints["scene"] == scene].copy()
        camera_columns = [
            "trajectory_id",
            "source_recording_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]
        camera_image = paths.calibration_inputs / scene / "representative_camera_frame.png"
        topview_image = paths.calibration_inputs / scene / "topview_reference_image.png"
        with Image.open(camera_image) as image:
            camera_size = image.size
        with Image.open(topview_image) as image:
            topview_size = image.size
        scenes[scene] = SceneInputs(
            scene=scene,
            points=points,
            point_path=csv_path,
            camera_image_path=camera_image,
            topview_image_path=topview_image,
            camera_size=camera_size,
            topview_size=topview_size,
            frozen_matrix=frozen_matrix,
            frozen_mask=frozen_mask,
            config_path=config_path,
            config_payload=config_payload,
            camera_endpoints=scene_rows[camera_columns].reset_index(drop=True),
            frozen_topview_endpoints=scene_rows.reset_index(drop=True),
        )
    return scenes


def _point_provenance(row: pd.Series) -> str:
    description = str(row.get("point_description", "")).strip().lower()
    confidence = str(row.get("confidence", "")).strip().lower()
    if "existing_calibration" in description or confidence == "existing":
        return "manual_existing_calibration_point_feature_not_documented"
    return "manual_visual_selection_feature_not_documented"


def run_calibration_reproduction(
    paths: Paths, scenes: dict[str, SceneInputs]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    correspondence_rows = []
    reproduction_rows = []
    quality_rows = []
    point_error_rows = []
    matrix_rows = []
    for scene in SCENES:
        state = scenes[scene]
        source = state.points[["camera_x", "camera_y"]].to_numpy(float)
        destination = state.points[["topview_x", "topview_y"]].to_numpy(float)
        reproduced, mask, method = estimate_historical_homography(source, destination)
        forward = np.linalg.norm(apply_homography(source, state.frozen_matrix) - destination, axis=1)
        statistics = reprojection_statistics(
            source,
            destination,
            state.frozen_matrix,
            state.frozen_mask,
            state.camera_size,
            state.topview_size,
        )
        endpoint_points = state.camera_endpoints[
            ["start_x", "start_y", "end_x", "end_y"]
        ].to_numpy(float).reshape(-1, 2)
        coverage = convex_hull_coverage(source, state.camera_size, endpoint_points)
        metrics = {
            **statistics,
            **{key: value for key, value in coverage.items() if not isinstance(value, np.ndarray)},
            **normalized_dlt_condition(source, destination),
            "homography_matrix_condition_number": float(np.linalg.cond(state.frozen_matrix)),
        }
        gate = classify_homography_quality(metrics)
        source_checksum = sha256_file(state.point_path)
        camera_checksum = sha256_file(state.camera_image_path)
        topview_checksum = sha256_file(state.topview_image_path)
        for index, (_, row) in enumerate(state.points.iterrows()):
            correspondence_rows.append(
                {
                    "scene_id": scene,
                    "point_id": int(row["pair_id"]),
                    "source_x_px": float(row["camera_x"]),
                    "source_y_px": float(row["camera_y"]),
                    "destination_x": float(row["topview_x"]),
                    "destination_y": float(row["topview_y"]),
                    "source_image_width": state.camera_size[0],
                    "source_image_height": state.camera_size[1],
                    "destination_image_width": state.topview_size[0],
                    "destination_image_height": state.topview_size[1],
                    "source_normalized_x": float(row["camera_x"] / state.camera_size[0]),
                    "source_normalized_y": float(row["camera_y"] / state.camera_size[1]),
                    "destination_normalized_x": float(row["topview_x"] / state.topview_size[0]),
                    "destination_normalized_y": float(row["topview_y"] / state.topview_size[1]),
                    "is_inlier_if_applicable": bool(state.frozen_mask[index]),
                    "point_provenance": _point_provenance(row),
                    "calibration_version": "frozen-hg-msa-tc-five-scene-v1",
                    "source_config_path": relative(paths, state.config_path),
                    "source_checksum": source_checksum,
                    "camera_image_checksum": camera_checksum,
                    "topview_image_checksum": topview_checksum,
                }
            )
            point_error_rows.append(
                {
                    "scene_id": scene,
                    "point_id": int(row["pair_id"]),
                    "forward_reprojection_error_px": float(forward[index]),
                    "is_inlier": bool(state.frozen_mask[index]),
                }
            )
        maximum_difference = float(np.max(np.abs(reproduced - state.frozen_matrix)))
        reproduction_rows.append(
            {
                "scene_id": scene,
                "historical_method": (
                    "ransac_10px_excluding_points_12_13_15_21_23"
                    if scene == "bellevue_ne8th"
                    else method
                ),
                "ransac_threshold_px": HISTORICAL_RANSAC_THRESHOLD_PX,
                "max_iterations": HISTORICAL_MAX_ITERS,
                "confidence": HISTORICAL_CONFIDENCE,
                "point_count": len(source),
                "frozen_inlier_count": int(state.frozen_mask.sum()),
                "reproduced_inlier_count": int(mask.sum()),
                "inlier_mask_exact_match": bool(np.array_equal(mask, state.frozen_mask)),
                "matrix_max_abs_difference": maximum_difference,
                "matrix_frobenius_difference": float(
                    np.linalg.norm(reproduced - state.frozen_matrix)
                ),
                "matrix_exact_match": bool(maximum_difference == 0.0),
                "frozen_matrix_replaced": False,
            }
        )
        quality_rows.append(
            {
                "scene_id": scene,
                "point_count": len(source),
                "inlier_count": int(state.frozen_mask.sum()),
                **metrics,
                "quality_class": gate.quality_class,
                "homography_passes_quality_gate": gate.passes,
                "quality_gate_notes": "; ".join(gate.reasons),
                "source_image_width": state.camera_size[0],
                "source_image_height": state.camera_size[1],
                "destination_image_width": state.topview_size[0],
                "destination_image_height": state.topview_size[1],
            }
        )
        matrix_rows.append(
            {
                "scene_id": scene,
                **{
                    f"h{row + 1}{column + 1}": float(state.frozen_matrix[row, column])
                    for row in range(3)
                    for column in range(3)
                },
                "matrix_sha256": sha256_file(state.config_path),
            }
        )
    correspondences = pd.DataFrame(correspondence_rows)
    reproduction = pd.DataFrame(reproduction_rows)
    quality = pd.DataFrame(quality_rows)
    point_errors = pd.DataFrame(point_error_rows)
    write_csv(correspondences, paths.results / "calibration_correspondences.csv")
    write_csv(reproduction, paths.results / "homography_reproduction.csv")
    write_csv(quality, paths.results / "homography_quality_metrics.csv")
    write_csv(point_errors, paths.results / "calibration_point_errors.csv")
    write_csv(pd.DataFrame(matrix_rows), paths.results / "frozen_homography_matrices.csv")
    return correspondences, reproduction, quality, point_errors


def _target_configuration(paths: Paths) -> dict[str, Any]:
    return yaml.safe_load(paths.runner_config.read_text(encoding="utf-8"))


def _wrapped_angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(first - second), np.cos(first - second)))


def _evaluate_transformed_target(
    scene_index: int,
    state: SceneInputs,
    matrix: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any, pd.DataFrame]:
    endpoints = transform_camera_endpoints(state.camera_endpoints, matrix)
    target_config = config["target_estimation"]
    seed = int(config["random_seed"]) + scene_index * 1000
    result = estimate_hg_target_detailed(
        scene=state.scene,
        endpoints=endpoints,
        seed=seed,
        region_counts=[int(value) for value in target_config["region_counts"]],
        support_thresholds=[float(value) for value in target_config["support_thresholds"]],
        region_metric_sample_size=int(target_config["region_metric_sample_size"]),
    )
    return result.summary, result, endpoints


def _propagation_metrics(
    state: SceneInputs,
    result: Any,
    endpoints: pd.DataFrame,
) -> dict[str, float]:
    frozen = state.frozen_topview_endpoints
    frozen_entry = frozen[["start_x_topview", "start_y_topview"]].to_numpy(float)
    frozen_exit = frozen[["end_x_topview", "end_y_topview"]].to_numpy(float)
    candidate_entry = endpoints[["start_x_topview", "start_y_topview"]].to_numpy(float)
    candidate_exit = endpoints[["end_x_topview", "end_y_topview"]].to_numpy(float)
    displacement = np.concatenate(
        [
            np.linalg.norm(candidate_entry - frozen_entry, axis=1),
            np.linalg.norm(candidate_exit - frozen_exit, axis=1),
        ]
    )
    frozen_entry_features = endpoint_features(frozen_entry, np.median(frozen_entry, axis=0))
    frozen_exit_features = endpoint_features(frozen_exit, np.median(frozen_exit, axis=0))
    entry_angles = np.arctan2(result.entry_fit.features[:, 1], result.entry_fit.features[:, 0])
    exit_angles = np.arctan2(result.exit_fit.features[:, 1], result.exit_fit.features[:, 0])
    frozen_entry_angles = np.arctan2(
        frozen_entry_features[:, 1], frozen_entry_features[:, 0]
    )
    frozen_exit_angles = np.arctan2(
        frozen_exit_features[:, 1], frozen_exit_features[:, 0]
    )
    angular = np.concatenate(
        [
            _wrapped_angle_difference(entry_angles, frozen_entry_angles),
            _wrapped_angle_difference(exit_angles, frozen_exit_angles),
        ]
    )
    radial = np.concatenate(
        [
            np.abs(result.entry_fit.features[:, 2] - frozen_entry_features[:, 2]),
            np.abs(result.exit_fit.features[:, 2] - frozen_exit_features[:, 2]),
        ]
    )
    aligned_entry, entry_change = align_labels_and_change_rate(
        frozen["entry_region"].to_numpy(int), result.entry_fit.labels
    )
    aligned_exit, exit_change = align_labels_and_change_rate(
        frozen["exit_region"].to_numpy(int), result.exit_fit.labels
    )
    frozen_od = (
        frozen["entry_region"].astype(int).astype(str)
        + "->"
        + frozen["exit_region"].astype(int).astype(str)
    ).to_numpy()
    candidate_od = np.array(
        [f"{entry}->{exit_}" for entry, exit_ in zip(aligned_entry, aligned_exit, strict=True)]
    )
    return {
        "endpoint_displacement_median_px": float(np.median(displacement)),
        "endpoint_displacement_p95_px": float(np.percentile(displacement, 95)),
        "angular_feature_displacement_median_rad": float(np.median(angular)),
        "angular_feature_displacement_p95_rad": float(np.percentile(angular, 95)),
        "normalized_radial_feature_displacement_median": float(np.median(radial)),
        "normalized_radial_feature_displacement_p95": float(np.percentile(radial, 95)),
        "entry_region_assignment_change_fraction": entry_change,
        "exit_region_assignment_change_fraction": exit_change,
        "od_pair_assignment_change_fraction": float(np.mean(frozen_od != candidate_od)),
    }


def run_jackknife(
    paths: Paths, scenes: dict[str, SceneInputs], config: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for scene_index, scene in enumerate(SCENES):
        state = scenes[scene]
        source = state.points[["camera_x", "camera_y"]].to_numpy(float)
        destination = state.points[["topview_x", "topview_y"]].to_numpy(float)
        frozen_target = int(
            state.frozen_topview_endpoints["supported_at_selected_threshold"].groupby(
                state.frozen_topview_endpoints["od_pair"]
            ).any().sum()
        )
        for position, point_id in enumerate(state.points["pair_id"].astype(int)):
            keep = np.arange(len(source)) != position
            base = {
                "scene_id": scene,
                "omitted_point_id": int(point_id),
                "omitted_point_was_frozen_inlier": bool(state.frozen_mask[position]),
                "remaining_point_count": int(keep.sum()),
                "frozen_target": frozen_target,
            }
            try:
                matrix, mask, method = estimate_historical_homography(
                    source[keep], destination[keep]
                )
                summary, result, endpoints = _evaluate_transformed_target(
                    scene_index, state, matrix, config
                )
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "estimation_method": method,
                        "remaining_inlier_count": int(mask.sum()),
                        "matrix_frobenius_difference": float(
                            np.linalg.norm(matrix - state.frozen_matrix)
                        ),
                        "resulting_entry_regions": int(summary["n_entry_regions"]),
                        "resulting_exit_regions": int(summary["n_exit_regions"]),
                        "resulting_support_threshold": float(summary["support_threshold"]),
                        "resulting_target": int(summary["hg_estimated_target"]),
                        "target_difference_from_frozen": int(summary["hg_estimated_target"])
                        - frozen_target,
                        **_propagation_metrics(state, result, endpoints),
                    }
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                rows.append({**base, "status": "unsupported", "error": str(error)})
    frame = pd.DataFrame(rows)
    write_csv(frame, paths.results / "homography_jackknife_sensitivity.csv")
    return frame


def run_perturbation(
    paths: Paths,
    scenes: dict[str, SceneInputs],
    config: dict[str, Any],
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    frozen_choices = pd.read_csv(
        paths.publication / "results/development/target_estimates.csv"
    ).set_index("scene")
    for scene_index, scene in enumerate(SCENES):
        state = scenes[scene]
        source = state.points[["camera_x", "camera_y"]].to_numpy(float)
        destination = state.points[["topview_x", "topview_y"]].to_numpy(float)
        frozen_target = int(frozen_choices.loc[scene, "hg_estimated_target"])
        for scale_index, scale in enumerate(PERTURBATION_SCALES_PX):
            for replicate in range(replicates):
                seed = (
                    PERTURBATION_BASE_SEED
                    + scene_index * 100000
                    + scale_index * 1000
                    + replicate
                )
                rng = np.random.default_rng(seed)
                perturbed_source = source + rng.uniform(-scale, scale, size=source.shape)
                base = {
                    "scene_id": scene,
                    "noise_scale_px": scale,
                    "replicate": replicate,
                    "seed": seed,
                    "source_noise_distribution": "independent_uniform_bounded",
                    "destination_points_perturbed": False,
                    "frozen_target": frozen_target,
                }
                try:
                    matrix, mask, method = estimate_historical_homography(
                        perturbed_source, destination, rng_seed=seed
                    )
                    summary, result, endpoints = _evaluate_transformed_target(
                        scene_index, state, matrix, config
                    )
                    rows.append(
                        {
                            **base,
                            "status": "ok",
                            "estimation_method": method,
                            "inlier_count": int(mask.sum()),
                            "matrix_frobenius_difference": float(
                                np.linalg.norm(matrix - state.frozen_matrix)
                            ),
                            "resulting_entry_regions": int(summary["n_entry_regions"]),
                            "resulting_exit_regions": int(summary["n_exit_regions"]),
                            "resulting_support_threshold": float(
                                summary["support_threshold"]
                            ),
                            "resulting_target": int(summary["hg_estimated_target"]),
                            "target_difference_from_frozen": int(
                                summary["hg_estimated_target"]
                            )
                            - frozen_target,
                            **_propagation_metrics(state, result, endpoints),
                        }
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    rows.append({**base, "status": "failed", "error": str(error)})
            checkpoint = pd.DataFrame(rows)
            checkpoint.to_parquet(
                paths.results / "homography_perturbation_runs.partial.parquet",
                index=False,
            )
    runs = pd.DataFrame(rows)
    runs.to_parquet(paths.results / "homography_perturbation_runs.parquet", index=False)
    summary, endpoint_stability = summarize_perturbation_runs(paths, runs)
    partial = paths.results / "homography_perturbation_runs.partial.parquet"
    if partial.exists():
        partial.unlink()
    return runs, summary, endpoint_stability


def summarize_perturbation_runs(
    paths: Paths, runs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize persisted perturbations against the frozen scene choices."""
    successful = runs[runs["status"] == "ok"].copy()
    grouped = successful.groupby(["scene_id", "noise_scale_px"], sort=False)
    summary = grouped.agg(
        replicate_count=("replicate", "count"),
        target_preservation_probability=(
            "target_difference_from_frozen",
            lambda values: float(np.mean(np.asarray(values) == 0)),
        ),
        target_mean=("resulting_target", "mean"),
        target_std=("resulting_target", "std"),
        target_min=("resulting_target", "min"),
        target_max=("resulting_target", "max"),
        endpoint_displacement_median_px=("endpoint_displacement_median_px", "median"),
        endpoint_displacement_p95_px=("endpoint_displacement_p95_px", "median"),
        entry_region_change_mean=("entry_region_assignment_change_fraction", "mean"),
        exit_region_change_mean=("exit_region_assignment_change_fraction", "mean"),
        od_pair_change_mean=("od_pair_assignment_change_fraction", "mean"),
    ).reset_index()
    frozen_choices = pd.read_csv(
        paths.publication / "results/development/target_estimates.csv"
    ).set_index("scene")
    summary["frozen_entry_regions"] = [
        int(frozen_choices.loc[scene, "n_entry_regions"])
        for scene in summary["scene_id"]
    ]
    summary["frozen_exit_regions"] = [
        int(frozen_choices.loc[scene, "n_exit_regions"])
        for scene in summary["scene_id"]
    ]
    summary["frozen_support_threshold"] = [
        float(frozen_choices.loc[scene, "support_threshold"])
        for scene in summary["scene_id"]
    ]
    entry_stability = []
    exit_stability = []
    threshold_stability = []
    for row in summary.itertuples():
        group = successful[
            (successful["scene_id"] == row.scene_id)
            & (successful["noise_scale_px"] == row.noise_scale_px)
        ]
        entry_stability.append(
            float(np.mean(group["resulting_entry_regions"] == row.frozen_entry_regions))
        )
        exit_stability.append(
            float(np.mean(group["resulting_exit_regions"] == row.frozen_exit_regions))
        )
        threshold_stability.append(
            float(
                np.mean(
                    np.isclose(
                        group["resulting_support_threshold"],
                        row.frozen_support_threshold,
                        rtol=0.0,
                        atol=1e-15,
                    )
                )
            )
        )
    summary["entry_k_preservation_probability"] = entry_stability
    summary["exit_k_preservation_probability"] = exit_stability
    summary["threshold_preservation_probability"] = threshold_stability
    target_distributions = (
        successful.groupby(["scene_id", "noise_scale_px"])["resulting_target"]
        .value_counts()
        .rename("count")
        .reset_index()
    )
    distribution_lookup = {
        (scene, scale): json.dumps(
            {
                str(int(row.resulting_target)): int(row["count"])
                for _, row in group.iterrows()
            },
            sort_keys=True,
        )
        for (scene, scale), group in target_distributions.groupby(
            ["scene_id", "noise_scale_px"]
        )
    }
    summary["target_distribution_json"] = [
        distribution_lookup[(row.scene_id, row.noise_scale_px)]
        for row in summary.itertuples()
    ]
    write_csv(summary, paths.results / "homography_perturbation_summary.csv")
    endpoint_columns = [
        "scene_id",
        "noise_scale_px",
        "replicate_count",
        "endpoint_displacement_median_px",
        "endpoint_displacement_p95_px",
        "entry_region_change_mean",
        "exit_region_change_mean",
        "od_pair_change_mean",
    ]
    endpoint_stability = summary[endpoint_columns].copy()
    angular_summary = grouped.agg(
        angular_feature_displacement_median_rad=(
            "angular_feature_displacement_median_rad",
            "median",
        ),
        angular_feature_displacement_p95_rad=(
            "angular_feature_displacement_p95_rad",
            "median",
        ),
        normalized_radial_feature_displacement_median=(
            "normalized_radial_feature_displacement_median",
            "median",
        ),
        normalized_radial_feature_displacement_p95=(
            "normalized_radial_feature_displacement_p95",
            "median",
        ),
    ).reset_index()
    endpoint_stability = endpoint_stability.merge(
        angular_summary, on=["scene_id", "noise_scale_px"], validate="one_to_one"
    )
    write_csv(endpoint_stability, paths.results / "endpoint_assignment_stability.csv")
    return summary, endpoint_stability


def build_quality_vs_target_error(paths: Paths, quality: pd.DataFrame) -> pd.DataFrame:
    diagnostic = pd.read_csv(
        paths.publication / "results/target_estimation/cross_scene_semantic_duplication.csv"
    )
    columns = [
        "scene",
        "automatic_od_target",
        "observed_independent_semantic_movement_count",
    ]
    output = quality.merge(
        diagnostic[columns],
        left_on="scene_id",
        right_on="scene",
        validate="one_to_one",
    ).drop(columns="scene")
    output["frozen_target_absolute_error"] = (
        output["automatic_od_target"]
        - output["observed_independent_semantic_movement_count"]
    ).abs()
    output["analysis_role"] = "post_quality-gate_descriptive_only_n_equals_5"
    write_csv(output, paths.results / "homography_quality_vs_target_error.csv")
    return output


def write_result_manifest(
    paths: Paths, manifest: dict[str, Any], replicates: int
) -> dict[str, Any]:
    verify_immutable_inputs(paths, manifest)
    outputs = {}
    for path in sorted(paths.results.iterdir()):
        if path.is_file() and path.name != "task_08_result_manifest.json":
            outputs[relative(paths, path)] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    payload = {
        "task": "futuretransp-homography-quality-sensitivity",
        "created_at_utc": utc_timestamp(),
        "git_commit_at_execution": git_head(paths.repo),
        "perturbation_replicates_per_scene_scale": replicates,
        "perturbation_scales_px": list(PERTURBATION_SCALES_PX),
        "target_estimation_split_only_for_propagation": True,
        "quality_gate_defined_without_reference_metrics": True,
        "frozen_matrices_modified": False,
        "frozen_targets_modified": False,
        "independent_test_clustering_executed": False,
        "outputs": outputs,
    }
    write_json(paths.results / "task_08_result_manifest.json", payload)
    verify_immutable_inputs(paths, manifest)
    return payload


def run_all(
    paths: Paths | None = None, replicates: int = DEFAULT_PERTURBATION_REPLICATES
) -> dict[str, Any]:
    paths = paths or default_paths()
    paths.results.mkdir(parents=True, exist_ok=True)
    if replicates < 1:
        raise ValueError("Perturbation replicates must be positive.")
    input_manifest = create_input_manifest(paths)
    scenes = load_scene_inputs(paths)
    _, reproduction, quality, _ = run_calibration_reproduction(paths, scenes)
    if not reproduction["matrix_exact_match"].all():
        raise RuntimeError("Frozen homography reproduction failed; sensitivity was not run.")
    config = _target_configuration(paths)
    jackknife = run_jackknife(paths, scenes, config)
    runs, perturbation, endpoint = run_perturbation(
        paths, scenes, config, replicates
    )
    build_quality_vs_target_error(paths, quality)
    result_manifest = write_result_manifest(paths, input_manifest, replicates)
    summary = {
        "all_frozen_matrices_reproduced_exactly": True,
        "quality_gate_pass_count": int(quality["homography_passes_quality_gate"].sum()),
        "quality_gate_scene_count": len(quality),
        "jackknife_runs": len(jackknife),
        "perturbation_runs": len(runs),
        "perturbation_replicates_per_scene_scale": replicates,
        "endpoint_stability_rows": len(endpoint),
        "frozen_homographies_modified": False,
        "frozen_targets_modified": False,
        "independent_test_clustering_executed": False,
        "result_manifest_outputs": len(result_manifest["outputs"]),
    }
    write_json(paths.results / "homography_analysis_summary.json", summary)
    verify_immutable_inputs(paths, input_manifest)
    return summary
