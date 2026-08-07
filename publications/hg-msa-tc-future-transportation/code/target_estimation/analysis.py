"""Frozen-target reproduction and post-freeze diagnostic analyses for Task 07."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from pipeline import split_aware_io
from target_estimation.hg_target_estimator import (
    CANONICAL_MODULE_VERSION,
    TargetEstimateResult,
    build_od_support,
    build_threshold_candidates,
    estimate_hg_target_detailed,
    select_threshold_candidate,
    transform_camera_endpoints,
)


SCENES = (
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
)
EXPECTED_FROZEN_TARGETS = {
    "bellevue_116th_ne12th": 10,
    "bellevue_150th_newport": 12,
    "bellevue_150th_eastgate": 9,
    "bellevue_150th_se38th": 18,
    "bellevue_ne8th": 9,
}
THRESHOLD_SENSITIVITY_GRID = (
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0025,
    0.0030,
    0.0040,
    0.0050,
    0.0075,
    0.0100,
)
NUMERIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Paths:
    publication: Path
    repo: Path
    config: Path
    results: Path
    figures: Path
    docs: Path
    homography_directory: Path


@dataclass
class SceneState:
    scene: str
    features: pd.DataFrame
    endpoints: pd.DataFrame
    result: TargetEstimateResult
    homography: np.ndarray


@dataclass
class ReproductionState:
    config: dict[str, Any]
    frozen: dict[str, Any]
    scenes: dict[str, SceneState]
    reproduction_completed_at_utc: str


def default_paths() -> Paths:
    publication = Path(__file__).resolve().parents[2]
    repo = publication.parents[1]
    return Paths(
        publication=publication,
        repo=repo,
        config=publication / "configs" / "split_aware_runner.yaml",
        results=publication / "results" / "target_estimation",
        figures=publication / "figures" / "target_estimation",
        docs=publication / "docs",
        homography_directory=(
            repo
            / "research_experiments"
            / "fov2026_trajectory_clustering"
            / "configs"
            / "hg_msa_tc_five_scene"
        ),
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


def immutable_input_paths(paths: Paths) -> tuple[Path, ...]:
    fixed = [
        paths.publication / "configs/frozen_evaluation_protocol.yaml",
        paths.config,
        paths.publication / "data/manifests/trajectory_manifest.csv",
        paths.publication / "data/splits/evaluation_split.csv",
        paths.publication / "results/development/target_estimates.csv",
        paths.publication / "results/development/target_candidates.csv",
        paths.publication / "results/development/target_region_candidates.csv",
        paths.publication / "results/development/target_od_support_counts.csv",
        paths.publication / "results/development/target_estimation_provenance.json",
        paths.publication / "results/development/selected_configurations.csv",
        paths.publication / "results/independent_test/cluster_assignments.parquet",
        paths.publication / "annotations/protocol/polygon_reference_protocol_v1.yaml",
        paths.publication
        / "annotations/reference_labels/target_estimation_reference_labels.csv",
        paths.publication
        / "annotations/reference_labels/independent_test_reference_labels.csv",
    ]
    fixed.extend(
        paths.homography_directory / f"homography_{scene}.json" for scene in SCENES
    )
    return tuple(fixed)


def relative_to_repo(paths: Paths, path: Path) -> str:
    return path.relative_to(paths.repo).as_posix()


def create_input_manifest(paths: Paths) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in immutable_input_paths(paths):
        if not path.exists():
            raise FileNotFoundError(f"Required immutable Task 07 input is missing: {path}")
        files[relative_to_repo(paths, path)] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "task": "futuretransp-target-estimation-formalization",
        "created_at_utc": utc_timestamp(),
        "git_commit_before_analysis": git_head(paths.repo),
        "canonical_module_version": CANONICAL_MODULE_VERSION,
        "target_recomputation_split": "target_estimation",
        "semantic_reference_labels_read_during_reproduction": False,
        "independent_test_clustering_executed": False,
        "frozen_targets_modified": False,
        "frozen_configurations_modified": False,
        "files": files,
    }
    write_json(paths.results / "task_07_input_manifest.json", manifest)
    return manifest


def verify_immutable_inputs(paths: Paths, manifest: dict[str, Any]) -> None:
    changed = []
    for relative, metadata in manifest["files"].items():
        path = paths.repo / relative
        if not path.exists() or sha256_file(path) != metadata["sha256"]:
            changed.append(relative)
    if changed:
        raise RuntimeError(f"Frozen Task 07 inputs changed: {changed}")


def _load_config(paths: Paths) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
    frozen = yaml.safe_load(
        (paths.publication / "configs/frozen_evaluation_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    if tuple(config["scenes"]) != SCENES:
        raise ValueError("Task 07 requires the exact frozen five-scene order.")
    return config, frozen


def _load_phase_metadata(
    paths: Paths, config: dict[str, Any], phase: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = split_aware_io.load_split_membership(
        paths.repo / config["inputs"]["evaluation_split"], phase, SCENES
    )
    metadata = split_aware_io.load_manifest_metadata(
        paths.repo / config["inputs"]["trajectory_manifest"], membership
    )
    split_aware_io.assert_phase_rows(metadata, phase)
    return membership, metadata


def _load_homography(paths: Paths, scene: str) -> np.ndarray:
    payload = json.loads(
        (paths.homography_directory / f"homography_{scene}.json").read_text(
            encoding="utf-8"
        )
    )
    matrix = np.asarray(payload["homography_camera_to_topview"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid homography for {scene}: {matrix.shape}")
    return matrix


def _max_frame_difference(
    reproduced: pd.DataFrame,
    frozen: pd.DataFrame,
    sort_columns: list[str],
) -> tuple[float, bool]:
    left = reproduced.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    right = frozen.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return float("inf"), False
    maximum = 0.0
    exact = True
    for column in left.columns:
        left_values = left[column]
        right_values = right[column]
        if pd.api.types.is_numeric_dtype(left_values) and pd.api.types.is_numeric_dtype(
            right_values
        ):
            left_numeric = pd.to_numeric(left_values, errors="coerce").to_numpy(float)
            right_numeric = pd.to_numeric(right_values, errors="coerce").to_numpy(float)
            both_nan = np.isnan(left_numeric) & np.isnan(right_numeric)
            differences = np.abs(left_numeric - right_numeric)
            differences[both_nan] = 0.0
            if np.isnan(differences).any():
                exact = False
                maximum = float("inf")
            elif len(differences):
                maximum = max(maximum, float(np.max(differences)))
        else:
            equal = left_values.fillna("__NA__").astype(str).to_numpy() == right_values.fillna(
                "__NA__"
            ).astype(str).to_numpy()
            if not bool(np.all(equal)):
                exact = False
    return maximum, exact and maximum <= NUMERIC_TOLERANCE


def run_reproduction(paths: Paths) -> ReproductionState:
    """Recompute targets from target_estimation only; no reference label is loaded."""
    config, frozen = _load_config(paths)
    _, metadata = _load_phase_metadata(paths, config, "target")
    target_config = config["target_estimation"]
    base_seed = int(config["random_seed"])
    frozen_estimates = pd.read_csv(
        paths.publication / "results/development/target_estimates.csv"
    )
    frozen_thresholds = pd.read_csv(
        paths.publication / "results/development/target_candidates.csv"
    )
    frozen_regions = pd.read_csv(
        paths.publication / "results/development/target_region_candidates.csv"
    )
    frozen_od = pd.read_csv(
        paths.publication / "results/development/target_od_support_counts.csv"
    )

    states: dict[str, SceneState] = {}
    summaries: list[dict[str, Any]] = []
    thresholds: list[pd.DataFrame] = []
    regions: list[pd.DataFrame] = []
    od_tables: list[pd.DataFrame] = []
    assignment_tables: list[pd.DataFrame] = []

    for scene_index, scene in enumerate(SCENES):
        scene_metadata = metadata[metadata["scene_id"] == scene].copy()
        features = split_aware_io.load_scene_features(
            paths.repo, scene_metadata, "target", None
        )
        homography = _load_homography(paths, scene)
        endpoints = transform_camera_endpoints(features, homography)
        seed = base_seed + scene_index * 1000
        result = estimate_hg_target_detailed(
            scene=scene,
            endpoints=endpoints,
            seed=seed,
            region_counts=[int(value) for value in target_config["region_counts"]],
            support_thresholds=[
                float(value) for value in target_config["support_thresholds"]
            ],
            region_metric_sample_size=int(
                target_config["region_metric_sample_size"]
            ),
        )
        states[scene] = SceneState(scene, features, endpoints, result, homography)
        summaries.append(result.summary)
        thresholds.append(result.threshold_candidates)
        regions.append(result.region_candidates)
        od = result.od_counts.copy()
        od.insert(0, "scene", scene)
        od_tables.append(od)
        assignments = result.endpoint_assignments.merge(
            endpoints, on="trajectory_id", how="left", validate="one_to_one"
        ).merge(
            features[
                [
                    "trajectory_id",
                    "source_recording_id",
                    "start_x",
                    "start_y",
                    "end_x",
                    "end_y",
                ]
            ],
            on="trajectory_id",
            how="left",
            validate="one_to_one",
        )
        assignments.insert(0, "scene", scene)
        assignments.insert(2, "split", "target_estimation")
        assignment_tables.append(assignments)

    reproduced_estimates = pd.DataFrame(summaries)
    reproduced_thresholds = pd.concat(thresholds, ignore_index=True)
    reproduced_regions = pd.concat(regions, ignore_index=True)
    reproduced_od = pd.concat(od_tables, ignore_index=True)

    threshold_diff, threshold_exact = _max_frame_difference(
        reproduced_thresholds,
        frozen_thresholds,
        ["scene", "support_threshold"],
    )
    region_diff, region_exact = _max_frame_difference(
        reproduced_regions,
        frozen_regions,
        ["scene", "endpoint_role", "n_regions"],
    )
    od_diff, od_exact = _max_frame_difference(
        reproduced_od, frozen_od, ["scene", "od_pair"]
    )

    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        reproduced = reproduced_estimates[reproduced_estimates["scene"] == scene].iloc[0]
        stored = frozen_estimates[frozen_estimates["scene"] == scene].iloc[0]
        rows.append(
            {
                "scene": scene,
                "split_used": "target_estimation",
                "n_trajectories": int(reproduced["n_trajectories"]),
                "frozen_target": int(stored["hg_estimated_target"]),
                "reproduced_target": int(reproduced["hg_estimated_target"]),
                "target_exact_match": bool(
                    int(stored["hg_estimated_target"])
                    == int(reproduced["hg_estimated_target"])
                    == EXPECTED_FROZEN_TARGETS[scene]
                ),
                "frozen_entry_regions": int(stored["n_entry_regions"]),
                "reproduced_entry_regions": int(reproduced["n_entry_regions"]),
                "frozen_exit_regions": int(stored["n_exit_regions"]),
                "reproduced_exit_regions": int(reproduced["n_exit_regions"]),
                "frozen_support_threshold": float(stored["support_threshold"]),
                "reproduced_support_threshold": float(reproduced["support_threshold"]),
                "threshold_candidates_max_abs_difference": threshold_diff,
                "threshold_candidates_exact": threshold_exact,
                "region_candidates_max_abs_difference": region_diff,
                "region_candidates_exact": region_exact,
                "od_support_max_abs_difference": od_diff,
                "od_support_exact": od_exact,
                "canonical_module_version": CANONICAL_MODULE_VERSION,
            }
        )
    reproduction = pd.DataFrame(rows)
    if not reproduction["target_exact_match"].all() or not all(
        (threshold_exact, region_exact, od_exact)
    ):
        raise RuntimeError("Frozen target-estimator reproduction failed.")

    write_csv(reproduction, paths.results / "target_reproduction.csv")
    assignments = pd.concat(assignment_tables, ignore_index=True)
    write_csv(assignments, paths.results / "target_endpoint_region_assignments.csv")
    assignments.to_parquet(
        paths.results / "target_endpoint_region_assignments.parquet", index=False
    )
    completed = utc_timestamp()
    write_json(
        paths.results / "target_reproduction_lock.json",
        {
            "completed_at_utc": completed,
            "split_used": "target_estimation",
            "all_targets_exact": True,
            "all_candidate_tables_exact": True,
            "semantic_reference_loaded": False,
            "independent_test_clustering_executed": False,
        },
    )
    return ReproductionState(config, frozen, states, completed)


def _circular_metrics(angles: np.ndarray) -> tuple[float, float, float]:
    mean = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    unwrapped = np.angle(np.exp(1j * (angles - mean)))
    return mean, float(np.std(unwrapped)), float(np.ptp(unwrapped))


def build_endpoint_region_diagnostics(
    paths: Paths, state: ReproductionState
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scene, scene_state in state.scenes.items():
        for role, fit, points in (
            (
                "entry",
                scene_state.result.entry_fit,
                scene_state.endpoints[
                    ["start_x_topview", "start_y_topview"]
                ].to_numpy(float),
            ),
            (
                "exit",
                scene_state.result.exit_fit,
                scene_state.endpoints[
                    ["end_x_topview", "end_y_topview"]
                ].to_numpy(float),
            ),
        ):
            centroids: dict[int, np.ndarray] = {}
            mean_angles: dict[int, float] = {}
            for region in sorted(np.unique(fit.labels)):
                selected = points[fit.labels == region]
                centroid = np.mean(selected, axis=0)
                centroids[int(region)] = centroid
                delta = selected - fit.center
                angles = np.arctan2(delta[:, 1], delta[:, 0])
                radii = np.linalg.norm(delta, axis=1)
                mean_angle, angular_std, angular_span = _circular_metrics(angles)
                mean_angles[int(region)] = mean_angle
                spread = np.linalg.norm(selected - centroid, axis=1)
                rows.append(
                    {
                        "scene": scene,
                        "endpoint_role": role,
                        "record_type": "region",
                        "region_a": int(region),
                        "region_b": np.nan,
                        "region_size": int(len(selected)),
                        "region_share": float(len(selected) / len(points)),
                        "reference_center_x": float(fit.center[0]),
                        "reference_center_y": float(fit.center[1]),
                        "centroid_x": float(centroid[0]),
                        "centroid_y": float(centroid[1]),
                        "mean_angle_rad": mean_angle,
                        "angular_std_rad": angular_std,
                        "angular_span_rad": angular_span,
                        "mean_radius": float(np.mean(radii)),
                        "median_radius": float(np.median(radii)),
                        "within_region_mean_spread": float(np.mean(spread)),
                        "within_region_max_spread": float(np.max(spread)),
                        "pairwise_centroid_distance": np.nan,
                        "pairwise_angular_separation_rad": np.nan,
                    }
                )
            identifiers = sorted(centroids)
            for index, first in enumerate(identifiers):
                for second in identifiers[index + 1 :]:
                    angular_difference = abs(mean_angles[first] - mean_angles[second])
                    angular_difference = min(
                        angular_difference, 2.0 * np.pi - angular_difference
                    )
                    rows.append(
                        {
                            "scene": scene,
                            "endpoint_role": role,
                            "record_type": "pairwise",
                            "region_a": first,
                            "region_b": second,
                            "region_size": np.nan,
                            "region_share": np.nan,
                            "reference_center_x": float(fit.center[0]),
                            "reference_center_y": float(fit.center[1]),
                            "centroid_x": np.nan,
                            "centroid_y": np.nan,
                            "mean_angle_rad": np.nan,
                            "angular_std_rad": np.nan,
                            "angular_span_rad": np.nan,
                            "mean_radius": np.nan,
                            "median_radius": np.nan,
                            "within_region_mean_spread": np.nan,
                            "within_region_max_spread": np.nan,
                            "pairwise_centroid_distance": float(
                                np.linalg.norm(centroids[first] - centroids[second])
                            ),
                            "pairwise_angular_separation_rad": angular_difference,
                        }
                    )
    output = pd.DataFrame(rows)
    write_csv(output, paths.results / "endpoint_region_diagnostics.csv")
    return output


def build_threshold_sensitivity(
    paths: Paths, state: ReproductionState
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scene, scene_state in state.scenes.items():
        selected_threshold = float(scene_state.result.summary["support_threshold"])
        counts = scene_state.result.od_counts
        frozen_pairs = set(counts.loc[counts["share"] >= selected_threshold, "od_pair"])
        n = int(scene_state.result.summary["n_trajectories"])
        for threshold in THRESHOLD_SENSITIVITY_GRID:
            supported = counts[counts["share"] >= threshold]
            pairs = set(supported["od_pair"])
            union = pairs | frozen_pairs
            rows.append(
                {
                    "scene": scene,
                    "support_threshold": threshold,
                    "support_threshold_percent": 100.0 * threshold,
                    "minimum_required_support_count": int(np.ceil(threshold * n)),
                    "resulting_target_count": int(len(supported)),
                    "od_coverage": float(supported["count"].sum() / n),
                    "supported_pair_jaccard_vs_frozen": (
                        float(len(pairs & frozen_pairs) / len(union)) if union else 1.0
                    ),
                    "retained_frozen_pairs": int(len(pairs & frozen_pairs)),
                    "added_pairs_vs_frozen": int(len(pairs - frozen_pairs)),
                    "removed_pairs_vs_frozen": int(len(frozen_pairs - pairs)),
                    "is_frozen_selected_threshold": bool(
                        math.isclose(threshold, selected_threshold, abs_tol=1e-15)
                    ),
                    "frozen_target": int(scene_state.result.summary["hg_estimated_target"]),
                }
            )
    output = pd.DataFrame(rows)
    write_csv(output, paths.results / "support_threshold_sensitivity.csv")
    return output


def build_region_count_sensitivity(
    paths: Paths, state: ReproductionState
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    thresholds = [float(value) for value in state.config["target_estimation"]["support_thresholds"]]
    for scene, scene_state in state.scenes.items():
        entry_fit = scene_state.result.entry_fit
        exit_fit = scene_state.result.exit_fit
        frozen_threshold = float(scene_state.result.summary["support_threshold"])
        n = len(scene_state.endpoints)
        for entry_count, entry_labels in sorted(entry_fit.labels_by_count.items()):
            for exit_count, exit_labels in sorted(exit_fit.labels_by_count.items()):
                _, counts = build_od_support(
                    scene_state.endpoints["trajectory_id"], entry_labels, exit_labels
                )
                frozen_supported = counts[counts["share"] >= frozen_threshold]
                candidates = build_threshold_candidates(
                    scene,
                    counts,
                    n,
                    entry_count,
                    exit_count,
                    thresholds,
                )
                selected = select_threshold_candidate(candidates)
                rows.append(
                    {
                        "scene": scene,
                        "entry_region_count": entry_count,
                        "exit_region_count": exit_count,
                        "frozen_support_threshold": frozen_threshold,
                        "target_at_frozen_threshold": int(len(frozen_supported)),
                        "coverage_at_frozen_threshold": float(
                            frozen_supported["count"].sum() / n
                        ),
                        "heuristic_selected_threshold": float(
                            selected["support_threshold"]
                        ),
                        "target_with_threshold_heuristic": int(
                            selected["hg_estimated_target"]
                        ),
                        "coverage_with_threshold_heuristic": float(
                            selected["od_coverage"]
                        ),
                        "is_frozen_region_pair": bool(
                            entry_count == entry_fit.selected_count
                            and exit_count == exit_fit.selected_count
                        ),
                        "frozen_target": int(
                            scene_state.result.summary["hg_estimated_target"]
                        ),
                    }
                )
    output = pd.DataFrame(rows)
    write_csv(output, paths.results / "region_count_sensitivity.csv")
    return output


def build_frozen_scene_parameters(
    paths: Paths, state: ReproductionState
) -> pd.DataFrame:
    index = yaml.safe_load(
        (paths.homography_directory / "homography_five_scene_index.yaml").read_text(
            encoding="utf-8"
        )
    )
    index_rows = {row["scene"]: row for row in index["scenes"]}
    rows: list[dict[str, Any]] = []
    for scene, scene_state in state.scenes.items():
        summary = scene_state.result.summary
        topview_path = paths.repo / index_rows[scene]["topview_image_path"]
        with Image.open(topview_path) as image:
            image_width, image_height = image.size
        rows.append(
            {
                "scene": scene,
                "target_estimation_trajectory_count": int(summary["n_trajectories"]),
                "homography_id": f"homography_{scene}",
                "homography_sha256": sha256_file(
                    paths.homography_directory / f"homography_{scene}.json"
                ),
                "topview_image_width_px": image_width,
                "topview_image_height_px": image_height,
                "entry_reference_center_x": float(
                    scene_state.result.entry_fit.center[0]
                ),
                "entry_reference_center_y": float(
                    scene_state.result.entry_fit.center[1]
                ),
                "exit_reference_center_x": float(
                    scene_state.result.exit_fit.center[0]
                ),
                "exit_reference_center_y": float(
                    scene_state.result.exit_fit.center[1]
                ),
                "candidate_region_counts": json.dumps(
                    state.config["target_estimation"]["region_counts"]
                ),
                "chosen_entry_regions": int(summary["n_entry_regions"]),
                "chosen_exit_regions": int(summary["n_exit_regions"]),
                "support_threshold": float(summary["support_threshold"]),
                "support_threshold_percent": float(
                    summary["support_threshold_percent"]
                ),
                "minimum_required_support_count": int(
                    summary["support_threshold_absolute_count"]
                ),
                "frozen_target": int(summary["hg_estimated_target"]),
                "entry_random_seed": int(summary["random_seed"]),
                "exit_random_seed": int(summary["random_seed"]) + 17,
                "implementation_version": summary["implementation_version"],
                "threshold_origin": (
                    "data-dependent frozen heuristic over one global pre-specified grid; "
                    "not a manually assigned scene target"
                ),
            }
        )
    output = pd.DataFrame(rows)
    write_csv(output, paths.results / "frozen_scene_parameters.csv")
    return output


def _load_reference_protocol(paths: Paths) -> dict[str, Any]:
    return yaml.safe_load(
        (
            paths.publication
            / "annotations/protocol/polygon_reference_protocol_v1.yaml"
        ).read_text(encoding="utf-8")
    )


def _dominant_mapping(
    values: pd.Series, total_region_size: int
) -> tuple[str, float, int, str]:
    valid = values.astype(str)
    valid = valid[valid.str.len() > 0]
    if valid.empty:
        return "", 0.0, 0, "{}"
    counts = valid.value_counts().sort_index().sort_values(
        ascending=False, kind="mergesort"
    )
    dominant = str(counts.index[0])
    mapped_count = int(counts.sum())
    return (
        dominant,
        float(counts.iloc[0] / mapped_count),
        mapped_count,
        json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True),
    )


def _predict_independent_endpoint_regions(
    paths: Paths, state: ReproductionState
) -> pd.DataFrame:
    _, metadata = _load_phase_metadata(paths, state.config, "test")
    tables: list[pd.DataFrame] = []
    for scene in SCENES:
        scene_metadata = metadata[metadata["scene_id"] == scene].copy()
        features = split_aware_io.load_scene_features(
            paths.repo, scene_metadata, "test", None
        )
        target_state = state.scenes[scene]
        endpoints = transform_camera_endpoints(features, target_state.homography)
        start = endpoints[["start_x_topview", "start_y_topview"]].to_numpy(float)
        end = endpoints[["end_x_topview", "end_y_topview"]].to_numpy(float)
        table = endpoints.copy()
        table["entry_region"] = target_state.result.entry_fit.predict(start)
        table["exit_region"] = target_state.result.exit_fit.predict(end)
        table["od_pair"] = (
            table["entry_region"].astype(str)
            + "->"
            + table["exit_region"].astype(str)
        )
        table.insert(0, "scene", scene)
        table.insert(2, "split", "independent_test")
        tables.append(table)
    output = pd.concat(tables, ignore_index=True)
    output.to_parquet(
        paths.results / "independent_endpoint_region_diagnostic_assignments.parquet",
        index=False,
    )
    return output


def build_semantic_diagnostics(
    paths: Paths, state: ReproductionState, endpoint_diagnostics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load frozen human reference only after exact target reproduction is locked."""
    target_reference = pd.read_csv(
        paths.publication
        / "annotations/reference_labels/target_estimation_reference_labels.csv",
        keep_default_na=False,
    )
    independent_reference = pd.read_csv(
        paths.publication
        / "annotations/reference_labels/independent_test_reference_labels.csv",
        keep_default_na=False,
    )
    assignments = pd.read_parquet(
        paths.results / "target_endpoint_region_assignments.parquet"
    )
    target_joined = assignments.merge(
        target_reference[
            [
                "scene_id",
                "trajectory_id",
                "entry_polygon_id",
                "exit_polygon_id",
                "reference_movement_id",
                "reference_status",
                "exclusion_reason",
                "finite_canonical_point_count",
            ]
        ],
        left_on=["scene", "trajectory_id"],
        right_on=["scene_id", "trajectory_id"],
        how="left",
        validate="one_to_one",
    )
    independent_assignments = _predict_independent_endpoint_regions(paths, state)
    independent_joined = independent_assignments.merge(
        independent_reference[
            [
                "scene_id",
                "trajectory_id",
                "entry_polygon_id",
                "exit_polygon_id",
                "reference_movement_id",
                "reference_status",
            ]
        ],
        left_on=["scene", "trajectory_id"],
        right_on=["scene_id", "trajectory_id"],
        how="left",
        validate="one_to_one",
    )

    region_rows: list[dict[str, Any]] = []
    mapping_lookup: dict[tuple[str, str, int], str] = {}
    for scene in SCENES:
        scene_rows = target_joined[target_joined["scene"] == scene]
        for role, region_column, manual_column, x_column, y_column in (
            (
                "entry",
                "entry_region",
                "entry_polygon_id",
                "start_x_topview",
                "start_y_topview",
            ),
            (
                "exit",
                "exit_region",
                "exit_polygon_id",
                "end_x_topview",
                "end_y_topview",
            ),
        ):
            for region, group in scene_rows.groupby(region_column, sort=True):
                dominant, purity, mapped_count, counts_json = _dominant_mapping(
                    group[manual_column], len(group)
                )
                mapping_lookup[(scene, role, int(region))] = dominant
                diagnostic = endpoint_diagnostics[
                    (endpoint_diagnostics["scene"] == scene)
                    & (endpoint_diagnostics["endpoint_role"] == role)
                    & (endpoint_diagnostics["record_type"] == "region")
                    & (endpoint_diagnostics["region_a"] == int(region))
                ].iloc[0]
                region_rows.append(
                    {
                        "scene": scene,
                        "endpoint_role": role,
                        "automatic_region_id": int(region),
                        "dominant_manual_approach": dominant,
                        "mapping_purity": purity,
                        "mapped_endpoint_count": mapped_count,
                        "automatic_region_size": int(len(group)),
                        "unmapped_endpoint_count": int(len(group) - mapped_count),
                        "counts_by_manual_approach_json": counts_json,
                        "centroid_x_topview": float(group[x_column].mean()),
                        "centroid_y_topview": float(group[y_column].mean()),
                        "angular_span_rad": float(diagnostic["angular_span_rad"]),
                    }
                )
    region_mapping = pd.DataFrame(region_rows)
    write_csv(
        region_mapping,
        paths.results / "automatic_region_to_manual_approach_mapping.csv",
    )

    protocol = _load_reference_protocol(paths)
    legal_pairs: dict[str, set[tuple[str, str]]] = {}
    for guide in protocol["scene_guides"]:
        legal_pairs[guide["scene_id"]] = {
            (row["entry"], row["exit"])
            for row in guide["maneuver_type_mapping"]
        }

    od_rows: list[dict[str, Any]] = []
    for scene in SCENES:
        scene_state = state.scenes[scene]
        threshold = float(scene_state.result.summary["support_threshold"])
        supported = scene_state.result.od_counts[
            scene_state.result.od_counts["share"] >= threshold
        ]
        target_scene = target_joined[target_joined["scene"] == scene]
        independent_scene = independent_joined[
            independent_joined["scene"] == scene
        ]
        for row in supported.itertuples(index=False):
            entry_region, exit_region = (int(value) for value in row.od_pair.split("->"))
            mapped_entry = mapping_lookup.get((scene, "entry", entry_region), "")
            mapped_exit = mapping_lookup.get((scene, "exit", exit_region), "")
            mapped_movement = (
                f"{scene}:{mapped_entry}>{mapped_exit}"
                if mapped_entry and mapped_exit
                else ""
            )
            target_pair = target_scene[target_scene["od_pair"] == row.od_pair]
            valid_target = target_pair[target_pair["reference_status"] == "valid"]
            empirical, empirical_purity, empirical_count, empirical_json = _dominant_mapping(
                valid_target["reference_movement_id"], len(target_pair)
            )
            all_target_movement_counts = (
                valid_target["reference_movement_id"].value_counts().sort_index()
            )
            all_target_movements_json = json.dumps(
                {
                    str(key): int(value)
                    for key, value in all_target_movement_counts.items()
                },
                sort_keys=True,
            )
            independent_pair = independent_scene[
                independent_scene["od_pair"] == row.od_pair
            ]
            valid_independent = independent_pair[
                independent_pair["reference_status"] == "valid"
            ]
            _, _, independent_valid_count, independent_json = _dominant_mapping(
                valid_independent["reference_movement_id"], len(independent_pair)
            )
            od_rows.append(
                {
                    "scene": scene,
                    "automatic_entry_region": entry_region,
                    "automatic_exit_region": exit_region,
                    "automatic_od_pair": row.od_pair,
                    "target_estimation_support_count": int(row.count),
                    "target_estimation_support_percentage": 100.0 * float(row.share),
                    "support_threshold": threshold,
                    "low_support_mode": bool(float(row.share) < 2.0 * threshold),
                    "mapped_manual_entry": mapped_entry,
                    "mapped_manual_exit": mapped_exit,
                    "mapped_manual_movement": mapped_movement,
                    "mapped_pair_is_legal": bool(
                        (mapped_entry, mapped_exit) in legal_pairs[scene]
                    ),
                    "empirical_target_reference_movement": empirical,
                    "empirical_target_reference_purity": empirical_purity,
                    "empirical_target_valid_count": empirical_count,
                    "empirical_target_movements_json": empirical_json,
                    "all_target_reference_movements_json": all_target_movements_json,
                    "unique_target_reference_movement_count": int(
                        valid_target["reference_movement_id"].nunique()
                    ),
                    "target_reference_invalid_count": int(
                        len(target_pair) - len(valid_target)
                    ),
                    "median_canonical_point_count": float(
                        pd.to_numeric(
                            target_pair["finite_canonical_point_count"], errors="coerce"
                        ).median()
                    ),
                    "short_track_fraction_le_30_points": float(
                        (
                            pd.to_numeric(
                                target_pair["finite_canonical_point_count"],
                                errors="coerce",
                            )
                            <= 30
                        ).mean()
                    ),
                    "independent_prediction_count": int(len(independent_pair)),
                    "independent_valid_reference_count": independent_valid_count,
                    "independent_reference_movements_json": independent_json,
                    "observed_in_independent_valid_reference": bool(
                        independent_valid_count > 0
                    ),
                }
            )
    od_mapping = pd.DataFrame(od_rows)
    duplicate_sizes = od_mapping.groupby(
        ["scene", "mapped_manual_movement"], dropna=False
    )["automatic_od_pair"].transform("size")
    od_mapping["semantic_duplicate_group_size"] = np.where(
        od_mapping["mapped_manual_movement"].astype(str).str.len() > 0,
        duplicate_sizes,
        0,
    ).astype(int)
    od_mapping["duplicate_semantic_movement_flag"] = (
        od_mapping["semantic_duplicate_group_size"] > 1
    )
    empirical_duplicate_sizes = od_mapping.groupby(
        ["scene", "empirical_target_reference_movement"], dropna=False
    )["automatic_od_pair"].transform("size")
    od_mapping["empirical_dominant_duplicate_group_size"] = np.where(
        od_mapping["empirical_target_reference_movement"]
        .astype(str)
        .str.len()
        .gt(0),
        empirical_duplicate_sizes,
        0,
    ).astype(int)
    od_mapping["empirical_dominant_duplicate_flag"] = (
        od_mapping["empirical_dominant_duplicate_group_size"] > 1
    )
    write_csv(
        od_mapping,
        paths.results / "automatic_od_to_manual_movement_mapping.csv",
    )

    se38 = od_mapping[od_mapping["scene"] == "bellevue_150th_se38th"].copy()
    se38 = se38[
        [
            "automatic_entry_region",
            "automatic_exit_region",
            "automatic_od_pair",
            "target_estimation_support_count",
            "target_estimation_support_percentage",
            "mapped_manual_entry",
            "mapped_manual_exit",
            "mapped_manual_movement",
            "duplicate_semantic_movement_flag",
            "semantic_duplicate_group_size",
            "low_support_mode",
            "empirical_target_reference_movement",
            "empirical_target_reference_purity",
            "empirical_dominant_duplicate_group_size",
            "empirical_dominant_duplicate_flag",
            "all_target_reference_movements_json",
            "unique_target_reference_movement_count",
            "target_reference_invalid_count",
            "median_canonical_point_count",
            "short_track_fraction_le_30_points",
            "independent_prediction_count",
            "independent_valid_reference_count",
            "observed_in_independent_valid_reference",
        ]
    ]
    write_csv(se38, paths.results / "se38th_od_collapse_table.csv")

    cross_rows: list[dict[str, Any]] = []
    for scene in SCENES:
        region_scene = region_mapping[region_mapping["scene"] == scene]
        od_scene = od_mapping[od_mapping["scene"] == scene]
        valid_test = independent_reference[
            (independent_reference["scene_id"] == scene)
            & (independent_reference["reference_status"] == "valid")
        ]
        valid_mapped = od_scene[
            od_scene["mapped_pair_is_legal"]
            & od_scene["mapped_manual_movement"].astype(str).str.len().gt(0)
        ]
        unique_semantic = int(valid_mapped["mapped_manual_movement"].nunique())
        mapped_pair_count = int(len(valid_mapped))
        supported_pair_ids = set(od_scene["automatic_od_pair"])
        target_supported = target_joined[
            (target_joined["scene"] == scene)
            & (target_joined["od_pair"].isin(supported_pair_ids))
            & (target_joined["reference_status"] == "valid")
        ]
        empirical_dominant_unique = int(
            od_scene["empirical_target_reference_movement"]
            .replace("", np.nan)
            .nunique()
        )
        entry_splits = int(
            (
                region_scene[region_scene["endpoint_role"] == "entry"]
                .groupby("dominant_manual_approach")["automatic_region_id"]
                .nunique()
                > 1
            ).sum()
        )
        exit_splits = int(
            (
                region_scene[region_scene["endpoint_role"] == "exit"]
                .groupby("dominant_manual_approach")["automatic_region_id"]
                .nunique()
                > 1
            ).sum()
        )
        cross_rows.append(
            {
                "scene": scene,
                "automatic_entry_region_count": int(
                    state.scenes[scene].result.summary["n_entry_regions"]
                ),
                "manual_entry_approach_count": 4,
                "automatic_exit_region_count": int(
                    state.scenes[scene].result.summary["n_exit_regions"]
                ),
                "manual_exit_approach_count": 4,
                "automatic_od_target": int(
                    state.scenes[scene].result.summary["hg_estimated_target"]
                ),
                "observed_independent_semantic_movement_count": int(
                    valid_test["reference_movement_id"].nunique()
                ),
                "supported_automatic_pairs_mapped_to_legal_manual_movements": mapped_pair_count,
                "unique_mapped_semantic_movements": unique_semantic,
                "unique_empirical_dominant_semantic_movements": empirical_dominant_unique,
                "all_semantic_movements_represented_in_supported_pairs": int(
                    target_supported["reference_movement_id"].nunique()
                ),
                "empirical_dominant_duplicate_pair_count": int(
                    len(od_scene) - empirical_dominant_unique
                ),
                "empirical_dominant_duplication_ratio": float(
                    (len(od_scene) - empirical_dominant_unique) / len(od_scene)
                ),
                "semantically_mixed_automatic_pair_count": int(
                    (od_scene["unique_target_reference_movement_count"] > 1).sum()
                ),
                "semantic_duplicate_pair_count": max(
                    0, mapped_pair_count - unique_semantic
                ),
                "semantic_duplication_ratio": (
                    float((mapped_pair_count - unique_semantic) / mapped_pair_count)
                    if mapped_pair_count
                    else np.nan
                ),
                "unmapped_or_illegal_automatic_pair_count": int(
                    len(od_scene) - mapped_pair_count
                ),
                "manual_entry_approaches_split_across_multiple_auto_regions": entry_splits,
                "manual_exit_approaches_split_across_multiple_auto_regions": exit_splits,
            }
        )
    cross_scene = pd.DataFrame(cross_rows)
    write_csv(
        cross_scene, paths.results / "cross_scene_semantic_duplication.csv"
    )
    return region_mapping, od_mapping, se38, cross_scene


def build_se38th_fragmentation(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_parquet(
        paths.publication / "results/independent_test/cluster_assignments.parquet"
    )
    reference = pd.read_csv(
        paths.publication
        / "annotations/reference_labels/independent_test_reference_labels.csv",
        keep_default_na=False,
    )
    scene = "bellevue_150th_se38th"
    valid = reference[
        (reference["scene_id"] == scene)
        & (reference["reference_status"] == "valid")
    ][["scene_id", "trajectory_id", "reference_movement_id"]]
    joined = assignments[assignments["scene_id"] == scene].merge(
        valid, on=["scene_id", "trajectory_id"], how="inner", validate="many_to_one"
    )
    rows: list[dict[str, Any]] = []
    for (method, strategy), result in joined.groupby(
        ["method", "selection_strategy"], sort=True
    ):
        nonnoise_result = result[~result["is_noise"].astype(bool)]
        cluster_totals = nonnoise_result.groupby("cluster_label").size()
        movement_cluster = (
            nonnoise_result.groupby(["reference_movement_id", "cluster_label"])
            .size()
            .rename("count")
            .reset_index()
        )
        for movement, movement_all in result.groupby("reference_movement_id", sort=True):
            support = len(movement_all)
            movement_nonnoise = movement_all[~movement_all["is_noise"].astype(bool)]
            counts = movement_nonnoise["cluster_label"].value_counts().sort_index()
            nonnoise = int(counts.sum())
            if nonnoise:
                probabilities = counts.to_numpy(float) / nonnoise
                entropy = float(-np.sum(probabilities * np.log(probabilities)))
                effective = float(np.exp(entropy))
                normalized_entropy = (
                    float(entropy / np.log(len(counts))) if len(counts) > 1 else 0.0
                )
                dominant_share = float(counts.max() / nonnoise)
                weighted_purity_numerator = 0.0
                for cluster, count in counts.items():
                    cluster_rows = movement_cluster[
                        movement_cluster["cluster_label"] == cluster
                    ]
                    cluster_max = int(cluster_rows["count"].max())
                    weighted_purity_numerator += count * (
                        cluster_max / int(cluster_totals.loc[cluster])
                    )
                weighted_purity = float(weighted_purity_numerator / nonnoise)
            else:
                entropy = np.nan
                effective = 0.0
                normalized_entropy = np.nan
                dominant_share = np.nan
                weighted_purity = np.nan
            rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "selection_strategy": strategy,
                    "reference_movement_id": movement,
                    "movement_support": support,
                    "non_noise_support": nonnoise,
                    "noise_count": int(support - nonnoise),
                    "noise_ratio": float((support - nonnoise) / support),
                    "unique_non_noise_clusters": int(len(counts)),
                    "clusters_with_at_least_1pct_of_movement": int(
                        sum((counts / support) >= 0.01)
                    ),
                    "effective_number_of_clusters": effective,
                    "dominant_cluster_share_non_noise": dominant_share,
                    "cluster_allocation_entropy": entropy,
                    "normalized_fragmentation_entropy": normalized_entropy,
                    "movement_completeness_proxy": (
                        1.0 - normalized_entropy
                        if np.isfinite(normalized_entropy)
                        else np.nan
                    ),
                    "weighted_cluster_purity": weighted_purity,
                }
            )
    output = pd.DataFrame(rows)
    write_csv(output, paths.results / "se38th_cluster_fragmentation.csv")

    candidates = output[
        (output["method"] == "kmeans")
        & (output["selection_strategy"] == "hg_expected_aware_selection")
        & (output["movement_support"] >= 50)
    ].sort_values(
        ["effective_number_of_clusters", "movement_support"],
        ascending=[False, False],
        kind="mergesort",
    )
    movement = str(candidates.iloc[0]["reference_movement_id"])
    figure_rows = joined[
        (joined["method"] == "kmeans")
        & (joined["selection_strategy"] == "hg_expected_aware_selection")
        & (joined["reference_movement_id"] == movement)
        & (~joined["is_noise"].astype(bool))
    ].sort_values(["cluster_label", "trajectory_id"], kind="mergesort")
    figure_rows = figure_rows.groupby("cluster_label", sort=True).head(20)
    manifest = pd.read_csv(
        paths.publication / "data/manifests/trajectory_manifest.csv",
        usecols=[
            "scene_id",
            "trajectory_id",
            "source_recording_id",
            "source_recording_track_id",
            "start_frame",
            "end_frame",
        ],
        keep_default_na=False,
    )
    queue = figure_rows.merge(
        manifest, on=["scene_id", "trajectory_id"], how="left", validate="one_to_one"
    )
    queue["trajectory_source_path"] = queue.apply(
        lambda row: (
            f"data/interim/{row['scene_id']}/tracks_{row['source_recording_id']}.parquet"
        ),
        axis=1,
    )
    write_csv(queue, paths.results / "se38th_fragmentation_figure_queue.csv")
    return output, queue


def _write_access_log(paths: Paths, state: ReproductionState) -> None:
    records: list[dict[str, Any]] = []
    for scene, scene_state in state.scenes.items():
        source = paths.repo / str(scene_state.features["source_file"].iloc[0])
        records.append(
            {
                "stage": "frozen_target_reproduction",
                "scene": scene,
                "split": "target_estimation",
                "input_kind": "feature_endpoints",
                "path": relative_to_repo(paths, source),
                "row_count": len(scene_state.features),
                "sha256": sha256_file(source),
                "reference_labels_loaded": False,
            }
        )
    for relative, kind in (
        (
            "annotations/reference_labels/target_estimation_reference_labels.csv",
            "diagnostic_target_reference",
        ),
        (
            "annotations/reference_labels/independent_test_reference_labels.csv",
            "diagnostic_independent_reference",
        ),
        (
            "results/independent_test/cluster_assignments.parquet",
            "persisted_independent_cluster_assignments",
        ),
    ):
        path = paths.publication / relative
        records.append(
            {
                "stage": "post_reproduction_diagnostic",
                "scene": "ALL",
                "split": "diagnostic_only",
                "input_kind": kind,
                "path": relative_to_repo(paths, path),
                "row_count": None,
                "sha256": sha256_file(path),
                "reference_labels_loaded": "reference" in kind,
                "reproduction_completed_at_utc": state.reproduction_completed_at_utc,
            }
        )
    path = paths.results / "target_estimation_data_access_log.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )


def _result_manifest(paths: Paths, state: ReproductionState) -> dict[str, Any]:
    result_files = sorted(
        path
        for path in paths.results.iterdir()
        if path.is_file() and path.name != "task_07_result_manifest.json"
    )
    payload = {
        "task": "futuretransp-target-estimation-formalization",
        "created_at_utc": utc_timestamp(),
        "git_commit_at_execution": git_head(paths.repo),
        "reproduction_completed_at_utc": state.reproduction_completed_at_utc,
        "reference_access_stage": "post_reproduction_diagnostic_only",
        "all_frozen_targets_reproduced": True,
        "frozen_targets_modified": False,
        "frozen_selected_configurations_modified": False,
        "independent_test_clustering_executed": False,
        "outputs": {
            relative_to_repo(paths, path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in result_files
        },
    }
    write_json(paths.results / "task_07_result_manifest.json", payload)
    return payload


def run_all(paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or default_paths()
    paths.results.mkdir(parents=True, exist_ok=True)
    input_manifest = create_input_manifest(paths)
    state = run_reproduction(paths)
    endpoint = build_endpoint_region_diagnostics(paths, state)
    threshold = build_threshold_sensitivity(paths, state)
    region = build_region_count_sensitivity(paths, state)
    parameters = build_frozen_scene_parameters(paths, state)

    # The human reference and persisted test assignments are accessed only below,
    # after target reproduction has been written and locked.
    region_mapping, od_mapping, se38, cross = build_semantic_diagnostics(
        paths, state, endpoint
    )
    fragmentation, queue = build_se38th_fragmentation(paths)
    _write_access_log(paths, state)
    verify_immutable_inputs(paths, input_manifest)
    result_manifest = _result_manifest(paths, state)
    summary = {
        "all_five_targets_reproduced": True,
        "frozen_targets": EXPECTED_FROZEN_TARGETS,
        "endpoint_diagnostic_rows": len(endpoint),
        "threshold_sensitivity_rows": len(threshold),
        "region_count_sensitivity_rows": len(region),
        "frozen_parameter_rows": len(parameters),
        "automatic_region_mapping_rows": len(region_mapping),
        "supported_automatic_od_rows": len(od_mapping),
        "se38th_supported_od_rows": len(se38),
        "cross_scene_rows": len(cross),
        "fragmentation_rows": len(fragmentation),
        "fragmentation_figure_queue_rows": len(queue),
        "reference_labels_used_only_after_reproduction": True,
        "independent_test_clustering_executed": False,
        "result_manifest_outputs": len(result_manifest["outputs"]),
    }
    write_json(paths.results / "target_estimation_analysis_summary.json", summary)
    verify_immutable_inputs(paths, input_manifest)
    return summary
