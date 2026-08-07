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

from pipeline import hg_msa_tc_core as core  # noqa: E402
from target_estimation import analysis  # noqa: E402
from target_estimation.hg_target_estimator import (  # noqa: E402
    apply_homography,
    build_od_support,
    build_threshold_candidates,
    endpoint_features,
    estimate_endpoint_region_fit,
    select_threshold_candidate,
)


def test_homography_and_endpoint_feature_definition() -> None:
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    transformed = apply_homography(points, np.eye(3))
    np.testing.assert_allclose(transformed, points, atol=0.0)
    features = endpoint_features(points, np.array([2.0, 3.0]))
    assert features.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(features[:, :2], axis=1), 1.0)
    assert np.all((features[:, 2] >= 0.0) & (features[:, 2] <= 0.25))


def test_near_zero_homogeneous_scale_is_rejected() -> None:
    matrix = np.eye(3)
    matrix[2] = 0.0
    with pytest.raises(ValueError, match="near-zero"):
        apply_homography(np.array([[1.0, 2.0]]), matrix)


def test_endpoint_region_fit_is_deterministic() -> None:
    angles = np.linspace(-np.pi, np.pi, 120, endpoint=False)
    radius = 10.0 + 0.3 * np.sin(4 * angles)
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    first = estimate_endpoint_region_fit(points, "entry", "synthetic", 42, [3, 4], 80)
    second = estimate_endpoint_region_fit(points, "entry", "synthetic", 42, [3, 4], 80)
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.metric_indices, second.metric_indices)
    pd.testing.assert_frame_equal(first.candidates, second.candidates)


def test_od_threshold_is_inclusive_and_uses_total_denominator() -> None:
    trajectory_ids = pd.Series(["a", "b", "c", "d"])
    _, counts = build_od_support(
        trajectory_ids,
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 0]),
    )
    candidates = build_threshold_candidates(
        "synthetic", counts, 4, 2, 2, [0.25, 0.50]
    )
    assert candidates.loc[0, "hg_estimated_target"] == 3
    assert candidates.loc[1, "hg_estimated_target"] == 1
    assert candidates.loc[0, "support_threshold_absolute_count"] == 1


def test_threshold_selection_matches_frozen_lexicographic_rule() -> None:
    candidates = pd.DataFrame(
        {
            "support_threshold": [0.001, 0.005, 0.01],
            "od_coverage": [0.95, 0.95, 0.95],
            "hg_estimated_target": [8, 8, 7],
            "target_local_instability": [0.0, 0.0, 1.0],
        }
    )
    selected = select_threshold_candidate(candidates)
    assert selected["support_threshold"] == pytest.approx(0.005)


def test_pipeline_core_delegates_to_canonical_estimator() -> None:
    source = inspect.getsource(core.estimate_hg_target)
    assert "target_estimator.estimate_hg_target" in source
    assert "KMeans(" not in source
    assert "support_threshold" in source


def test_exact_five_scene_reproduction_outputs() -> None:
    path = PUBLICATION_ROOT / "results/target_estimation/target_reproduction.csv"
    if not path.exists():
        pytest.skip("Task 07 analysis has not been executed.")
    frame = pd.read_csv(path)
    assert tuple(frame["scene"]) == analysis.SCENES
    assert dict(zip(frame["scene"], frame["reproduced_target"], strict=True)) == (
        analysis.EXPECTED_FROZEN_TARGETS
    )
    assert frame["target_exact_match"].all()
    assert frame["threshold_candidates_exact"].all()
    assert frame["region_candidates_exact"].all()
    assert frame["od_support_exact"].all()
    assert frame[
        [
            "threshold_candidates_max_abs_difference",
            "region_candidates_max_abs_difference",
            "od_support_max_abs_difference",
        ]
    ].to_numpy(float).max() <= analysis.NUMERIC_TOLERANCE


def test_reproduction_function_cannot_read_reference_or_test_assignments() -> None:
    source = inspect.getsource(analysis.run_reproduction)
    assert "reference_labels" not in source
    assert "cluster_assignments" not in source
    assert '"test"' not in source
    assert '"target"' in source


def test_run_order_locks_reproduction_before_diagnostics() -> None:
    source = inspect.getsource(analysis.run_all)
    reproduction_position = source.index("run_reproduction")
    semantic_position = source.index("build_semantic_diagnostics")
    fragmentation_position = source.index("build_se38th_fragmentation")
    assert reproduction_position < semantic_position < fragmentation_position


def test_task_manifest_preserves_all_immutable_inputs() -> None:
    paths = analysis.default_paths()
    manifest_path = paths.results / "task_07_input_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Task 07 analysis has not been executed.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis.verify_immutable_inputs(paths, manifest)
    assert manifest["target_recomputation_split"] == "target_estimation"
    assert manifest["semantic_reference_labels_read_during_reproduction"] is False
    assert manifest["independent_test_clustering_executed"] is False
    assert manifest["frozen_targets_modified"] is False
    assert manifest["frozen_configurations_modified"] is False


def test_semantic_mapping_and_se38th_collapse_are_diagnostic_and_deterministic() -> None:
    root = PUBLICATION_ROOT / "results/target_estimation"
    mapping = pd.read_csv(root / "automatic_region_to_manual_approach_mapping.csv")
    collapse = pd.read_csv(root / "se38th_od_collapse_table.csv")
    assert not mapping.duplicated(
        ["scene", "endpoint_role", "automatic_region_id"]
    ).any()
    assert len(collapse) == 18
    assert collapse["automatic_od_pair"].is_unique
    assert collapse["empirical_dominant_duplicate_flag"].any()
    assert (collapse["unique_target_reference_movement_count"] > 1).any()


def test_sensitivity_outputs_do_not_overwrite_frozen_choices() -> None:
    root = PUBLICATION_ROOT / "results/target_estimation"
    threshold = pd.read_csv(root / "support_threshold_sensitivity.csv")
    region = pd.read_csv(root / "region_count_sensitivity.csv")
    assert len(threshold) == len(analysis.SCENES) * len(
        analysis.THRESHOLD_SENSITIVITY_GRID
    )
    assert threshold.groupby("scene")["is_frozen_selected_threshold"].sum().eq(1).all()
    assert region.groupby("scene")["is_frozen_region_pair"].sum().eq(1).all()
    frozen = pd.read_csv(PUBLICATION_ROOT / "results/development/target_estimates.csv")
    selected = threshold[threshold["is_frozen_selected_threshold"]]
    assert dict(zip(frozen["scene"], frozen["hg_estimated_target"], strict=True)) == dict(
        zip(selected["scene"], selected["resulting_target_count"], strict=True)
    )


def test_fragmentation_reads_persisted_assignments_without_clustering() -> None:
    source = inspect.getsource(analysis.build_se38th_fragmentation)
    for forbidden in ("KMeans(", "HDBSCAN(", "OPTICS(", "fit_predict(", ".fit("):
        assert forbidden not in source
    frame = pd.read_csv(
        PUBLICATION_ROOT / "results/target_estimation/se38th_cluster_fragmentation.csv"
    )
    assert set(frame["method"]) == {"kmeans", "hdbscan", "optics"}
    assert set(frame["selection_strategy"]) == {
        "untargeted_selection",
        "hg_expected_aware_selection",
    }


def test_result_manifest_confirms_no_retuning_or_test_rerun() -> None:
    manifest = json.loads(
        (
            PUBLICATION_ROOT
            / "results/target_estimation/task_07_result_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["all_frozen_targets_reproduced"] is True
    assert manifest["frozen_targets_modified"] is False
    assert manifest["frozen_selected_configurations_modified"] is False
    assert manifest["independent_test_clustering_executed"] is False
    assert manifest["reference_access_stage"] == "post_reproduction_diagnostic_only"
