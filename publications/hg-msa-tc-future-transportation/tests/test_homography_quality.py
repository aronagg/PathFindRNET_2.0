from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PUBLICATION_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from homography import analysis  # noqa: E402
from homography.calibration import (  # noqa: E402
    apply_homography,
    classify_homography_quality,
    estimate_historical_homography,
    reprojection_statistics,
)


def test_identity_homography_and_invalid_denominator() -> None:
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(apply_homography(points, np.eye(3)), points)
    invalid = np.eye(3)
    invalid[2] = 0.0
    with pytest.raises(ValueError, match="normalized|denominator"):
        apply_homography(points, invalid)


def test_synthetic_reprojection_metrics_are_exact() -> None:
    source = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    destination = source + np.array([5.0, -3.0])
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    metrics = reprojection_statistics(
        source, destination, matrix, np.ones(4), (10, 10), (10, 10)
    )
    assert metrics["forward_rmse_error_px"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["inverse_rmse_error_px"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["inlier_fraction"] == 1.0


def test_quality_gate_is_deterministic_and_target_independent() -> None:
    metrics = {
        "normalized_rmse_fraction_destination_diagonal": 0.005,
        "normalized_p95_fraction_destination_diagonal": 0.010,
        "inlier_fraction": 0.8,
        "source_calibration_hull_area_fraction": 0.3,
        "endpoint_extrapolation_fraction": 0.5,
    }
    first = classify_homography_quality(metrics)
    second = classify_homography_quality(metrics)
    assert first == second
    assert first.quality_class == "good"
    assert first.passes is True
    source = inspect.getsource(classify_homography_quality)
    for forbidden in ("reference", "observed", "target_error"):
        assert forbidden not in source


def test_all_frozen_homographies_reproduce_exactly() -> None:
    paths = analysis.default_paths()
    scenes = analysis.load_scene_inputs(paths)
    for scene in analysis.SCENES:
        state = scenes[scene]
        source = state.points[["camera_x", "camera_y"]].to_numpy(float)
        destination = state.points[["topview_x", "topview_y"]].to_numpy(float)
        reproduced, mask, _ = estimate_historical_homography(source, destination)
        np.testing.assert_array_equal(reproduced, state.frozen_matrix)
        np.testing.assert_array_equal(mask, state.frozen_mask)


def test_correspondence_order_and_ne8th_exclusions_are_frozen() -> None:
    paths = analysis.default_paths()
    scenes = analysis.load_scene_inputs(paths)
    for scene, state in scenes.items():
        assert state.points["pair_id"].is_unique
        assert state.points["pair_id"].tolist() == sorted(state.points["pair_id"])
        assert len(state.points) == len(state.frozen_mask)
        assert state.camera_size == (1280, 720)
        assert state.topview_size[0] > 0 and state.topview_size[1] > 0
        if scene == "bellevue_ne8th":
            assert set(analysis.NE8TH_EXCLUDED_POINT_IDS).isdisjoint(
                set(state.points["pair_id"].astype(int))
            )
            assert len(state.points) == 18


def test_propagation_reads_target_estimation_only() -> None:
    paths = analysis.default_paths()
    frame = pd.read_parquet(paths.target_assignments, columns=["split", "scene"])
    assert set(frame["split"]) == {"target_estimation"}
    assert tuple(frame["scene"].drop_duplicates()) == analysis.SCENES
    source = inspect.getsource(analysis.run_perturbation)
    assert "independent_test" not in source
    assert "reference_labels" not in source


def test_completed_outputs_preserve_frozen_inputs_and_record_no_retuning() -> None:
    paths = analysis.default_paths()
    input_manifest_path = paths.results / "task_08_input_manifest.json"
    result_manifest_path = paths.results / "task_08_result_manifest.json"
    if not input_manifest_path.exists() or not result_manifest_path.exists():
        pytest.skip("Task 08 analysis has not been executed.")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    result_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    analysis.verify_immutable_inputs(paths, input_manifest)
    assert result_manifest["frozen_matrices_modified"] is False
    assert result_manifest["frozen_targets_modified"] is False
    assert result_manifest["independent_test_clustering_executed"] is False
    assert result_manifest["quality_gate_defined_without_reference_metrics"] is True


def test_completed_sensitivity_is_deterministic_and_complete() -> None:
    paths = analysis.default_paths()
    perturbation_path = paths.results / "homography_perturbation_runs.parquet"
    jackknife_path = paths.results / "homography_jackknife_sensitivity.csv"
    if not perturbation_path.exists() or not jackknife_path.exists():
        pytest.skip("Task 08 sensitivity analysis has not been executed.")
    perturbation = pd.read_parquet(perturbation_path)
    result_manifest = json.loads(
        (paths.results / "task_08_result_manifest.json").read_text(encoding="utf-8")
    )
    replicates = int(result_manifest["perturbation_replicates_per_scene_scale"])
    assert len(perturbation) == (
        len(analysis.SCENES) * len(analysis.PERTURBATION_SCALES_PX) * replicates
    )
    assert not perturbation.duplicated(["scene_id", "noise_scale_px", "replicate"]).any()
    assert perturbation["seed"].is_unique
    jackknife = pd.read_csv(jackknife_path)
    expected_points = sum(
        len(state.points) for state in analysis.load_scene_inputs(paths).values()
    )
    assert len(jackknife) == expected_points
    assert not jackknife.duplicated(["scene_id", "omitted_point_id"]).any()


def test_quality_output_contains_objective_gate_fields() -> None:
    path = analysis.default_paths().results / "homography_quality_metrics.csv"
    if not path.exists():
        pytest.skip("Task 08 quality analysis has not been executed.")
    frame = pd.read_csv(path)
    assert tuple(frame["scene_id"]) == analysis.SCENES
    required = {
        "forward_rmse_error_px",
        "forward_p95_error_px",
        "normalized_rmse_fraction_destination_diagonal",
        "source_calibration_hull_area_fraction",
        "endpoint_extrapolation_fraction",
        "homography_passes_quality_gate",
    }
    assert required.issubset(frame.columns)
