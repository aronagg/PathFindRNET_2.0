"""Development-only data loading with strict split and path guards."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pipeline import split_aware_io
from target_estimation.hg_target_estimator import apply_homography, transform_camera_endpoints

from .descriptors import directed_heading
from .provenance import reject_forbidden_paths, sha256_file


PUBLICATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PUBLICATION_ROOT.parents[1]
PROTOCOL_PATH = PUBLICATION_ROOT / "configs/hg_smg_protocol_v1.yaml"
ABLATION_PATH = PUBLICATION_ROOT / "configs/hg_smg_ablation_protocol.yaml"
RUNNER_CONFIG_PATH = PUBLICATION_ROOT / "configs/split_aware_runner.yaml"
RESULTS_ROOT = PUBLICATION_ROOT / "results/hg_smg/development"
FIGURES_ROOT = PUBLICATION_ROOT / "figures/hg_smg/development"


@dataclass(frozen=True)
class DevelopmentInputs:
    protocol: dict[str, Any]
    ablations: dict[str, Any]
    runner_config: dict[str, Any]
    membership: pd.DataFrame
    metadata: pd.DataFrame


def load_yaml(path: Path) -> dict[str, Any]:
    reject_forbidden_paths([path])
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


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


def load_development_inputs(phase: str) -> DevelopmentInputs:
    if phase not in {"target", "select"}:
        raise PermissionError("HG-SMG development accepts target/select phases only")
    protocol = load_yaml(PROTOCOL_PATH)
    ablations = load_yaml(ABLATION_PATH)
    runner = load_yaml(RUNNER_CONFIG_PATH)
    input_paths = {
        key: REPOSITORY_ROOT / value for key, value in runner["inputs"].items()
    }
    membership = split_aware_io.load_split_membership(
        input_paths["evaluation_split"], phase, tuple(protocol["scenes"]["fixed"])
    )
    metadata = split_aware_io.load_manifest_metadata(
        input_paths["trajectory_manifest"], membership
    )
    split_aware_io.assert_phase_rows(metadata, phase)
    return DevelopmentInputs(protocol, ablations, runner, membership, metadata)


def homography_path(runner_config: dict[str, Any], scene: str) -> Path:
    directory = REPOSITORY_ROOT / runner_config["inputs"]["homography_config_directory"]
    return directory / f"homography_{scene}.json"


def load_homography(runner_config: dict[str, Any], scene: str) -> np.ndarray:
    path = homography_path(runner_config, scene)
    reject_forbidden_paths([path])
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["homography_camera_to_topview"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid homography for {scene}: {matrix.shape}")
    return matrix


def _trajectory_path(scene: str) -> Path:
    return REPOSITORY_ROOT / "data/processed" / scene / "trajectories.parquet"


def _load_heading_table(
    scene: str,
    authorized_features: pd.DataFrame,
    homography: np.ndarray,
    windows: tuple[int, ...] = (3, 5, 7),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _trajectory_path(scene)
    reject_forbidden_paths([path])
    track_ids = authorized_features["original_track_id"].astype(int).tolist()
    trajectories = pd.read_parquet(
        path,
        columns=["track_id", "frame", "x", "y"],
        filters=[("track_id", "in", track_ids)],
    )
    observed_ids = set(trajectories["track_id"].astype(int))
    if not observed_ids.issubset(set(track_ids)) or observed_ids != set(track_ids):
        raise PermissionError("Trajectory parquet filter returned an unauthorized cohort")
    feature_source_values = authorized_features["source_file"].drop_duplicates()
    if len(feature_source_values) != 1:
        raise ValueError(f"{scene}: expected one authorized feature source")
    feature_source = REPOSITORY_ROOT / str(feature_source_values.iloc[0])
    interval_values = pd.read_parquet(
        feature_source,
        columns=["track_id", "frame_start", "frame_end"],
        filters=[("track_id", "in", track_ids)],
    )
    intervals = authorized_features[["trajectory_id", "original_track_id"]].merge(
        interval_values,
        left_on="original_track_id",
        right_on="track_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"frame_start": "start_frame", "frame_end": "end_frame"})
    intervals = intervals.drop(columns=["track_id"])
    intervals["original_track_id"] = intervals["original_track_id"].astype(int)
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
    rows: list[dict[str, Any]] = []
    endpoint_differences: list[float] = []
    feature_lookup = authorized_features.set_index("original_track_id")
    for track_id, group in joined.groupby("track_id", sort=True):
        points = group[["x", "y"]].to_numpy(dtype=np.float64)
        if not len(points):
            raise ValueError(f"{scene}:{track_id} has no canonical points")
        topview = apply_homography(points, homography)
        feature = feature_lookup.loc[int(track_id)]
        expected = np.asarray(
            [feature.start_x, feature.start_y, feature.end_x, feature.end_y],
            dtype=np.float64,
        )
        actual = np.asarray([*points[0], *points[-1]], dtype=np.float64)
        endpoint_differences.append(float(np.max(np.abs(actual - expected))))
        row: dict[str, Any] = {
            "trajectory_id": str(feature.trajectory_id),
            "track_id": int(track_id),
            "n_canonical_points": int(len(points)),
        }
        for window in windows:
            for role in ("entry", "exit"):
                row[f"heading_camera_w{window}_{role}"] = directed_heading(
                    points, role, window
                )
                row[f"heading_topview_w{window}_{role}"] = directed_heading(
                    topview, role, window
                )
        rows.append(row)
    output = pd.DataFrame(rows)
    if output["trajectory_id"].duplicated().any() or len(output) != len(authorized_features):
        raise ValueError(f"{scene}: heading extraction changed cohort size")
    maximum_difference = max(endpoint_differences, default=float("nan"))
    if not np.isfinite(maximum_difference) or maximum_difference > 1e-9:
        raise ValueError(
            f"{scene}: canonical polyline endpoints differ from frozen features by "
            f"{maximum_difference}"
        )
    provenance = {
        "trajectory_source": relative(path),
        "trajectory_source_sha256": sha256_file(path),
        "returned_track_count": int(output["track_id"].nunique()),
        "returned_row_count": int(len(joined)),
        "maximum_endpoint_difference_px": maximum_difference,
        "authorized_split": "target_estimation",
    }
    return output, provenance


def load_target_scene(
    inputs: DevelopmentInputs,
    scene: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    metadata = inputs.metadata[inputs.metadata["scene_id"] == scene].copy()
    split_aware_io.assert_phase_rows(metadata, "target")
    features = split_aware_io.load_scene_features(
        REPOSITORY_ROOT, metadata, "target", access_logger=None
    )
    homography = load_homography(inputs.runner_config, scene)
    endpoints = transform_camera_endpoints(features, homography)
    headings, heading_provenance = _load_heading_table(
        scene, features, homography
    )
    frame = features[
        [
            "trajectory_id",
            "source_recording_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]
    ].copy()
    frame = frame.rename(columns={"source_recording_id": "recording_id"})
    frame = frame.merge(endpoints, on=["trajectory_id", "recording_id"], validate="one_to_one")
    frame = frame.merge(headings, on="trajectory_id", validate="one_to_one")
    if set(frame["split"] if "split" in frame else ["target_estimation"]) != {
        "target_estimation"
    }:
        raise PermissionError("Non-target row reached HG-SMG target scene")
    return frame.sort_values("trajectory_id", kind="mergesort").reset_index(drop=True), homography, heading_provenance


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
