from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PUBLICATION_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline import hg_msa_tc_core as core  # noqa: E402
from pipeline import run_split_aware_hg_msa_tc as runner  # noqa: E402
from pipeline import split_aware_io as protocol_io  # noqa: E402


SPLIT_PATH = PUBLICATION_ROOT / "data" / "splits" / "evaluation_split.csv"
CONFIG_PATH = PUBLICATION_ROOT / "configs" / "split_aware_runner.yaml"
FROZEN_PATH = PUBLICATION_ROOT / "configs" / "frozen_evaluation_protocol.yaml"
FROZEN_MANIFEST_PATH = (
    PUBLICATION_ROOT / "results" / "development" / "frozen_selection_manifest.json"
)
NEWPORT_CONFIG = REPO_ROOT / "configs" / "dataset" / "bellevue_150th_newport.yaml"


@pytest.mark.parametrize(
    ("phase", "allowed"),
    [
        ("target", "target_estimation"),
        ("select", "model_selection"),
        ("test", "independent_test"),
    ],
)
def test_phase_accepts_only_its_split(phase: str, allowed: str) -> None:
    protocol_io.assert_phase_rows(pd.DataFrame({"split": [allowed, allowed]}), phase)
    with pytest.raises(PermissionError):
        protocol_io.assert_phase_rows(pd.DataFrame({"split": [allowed, "forbidden"]}), phase)


@pytest.mark.parametrize("phase", ["target", "select"])
def test_development_phases_reject_annotation_inputs(phase: str) -> None:
    with pytest.raises(PermissionError):
        protocol_io.reject_annotation_inputs(
            [PUBLICATION_ROOT / "annotations" / "manual_labels.csv"], phase
        )


def test_test_phase_refuses_without_unlock_file(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        protocol_io.require_test_unlock(tmp_path / "missing.json", "abc", None, "CONFIRM")


def synthetic_endpoints() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    starts = []
    ends = []
    for start_angle, end_angle in (
        (0.0, np.pi / 2),
        (np.pi / 2, np.pi),
        (np.pi, -np.pi / 2),
        (-np.pi / 2, 0.0),
    ):
        starts.append(
            np.column_stack(
                [
                    np.cos(start_angle) + rng.normal(0, 0.02, 40),
                    np.sin(start_angle) + rng.normal(0, 0.02, 40),
                ]
            )
        )
        ends.append(
            np.column_stack(
                [
                    np.cos(end_angle) + rng.normal(0, 0.02, 40),
                    np.sin(end_angle) + rng.normal(0, 0.02, 40),
                ]
            )
        )
    start = np.vstack(starts)
    end = np.vstack(ends)
    return pd.DataFrame(
        {
            "trajectory_id": [f"synthetic:{index}" for index in range(len(start))],
            "recording_id": "synthetic-recording",
            "start_x_topview": start[:, 0],
            "start_y_topview": start[:, 1],
            "end_x_topview": end[:, 0],
            "end_y_topview": end[:, 1],
        }
    )


def test_target_output_is_deterministic() -> None:
    arguments = dict(
        scene="synthetic",
        endpoints=synthetic_endpoints(),
        seed=123,
        region_counts=[3, 4, 5],
        support_thresholds=[0.001, 0.005, 0.02],
        region_metric_sample_size=100,
    )
    first = core.estimate_hg_target(**arguments)
    second = core.estimate_hg_target(**arguments)
    assert first[0] == second[0]
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[2], second[2])


def test_selected_configuration_is_deterministic() -> None:
    trials = pd.DataFrame(
        {
            "error": ["", "", ""],
            "cluster_count_error": [1, 0, 0],
            "pct_outliers": [0.0, 5.0, 5.0],
            "silhouette_clustered_only": [0.7, 0.5, 0.5],
            "davies_bouldin_clustered_only": [0.4, 0.6, 0.6],
            "calinski_harabasz_clustered_only": [10.0, 8.0, 8.0],
            "largest_cluster_ratio": [0.6, 0.5, 0.5],
            "params_json": ['{"x": 1}', '{"x": 3}', '{"x": 2}'],
            "trial_index": [1, 2, 3],
        }
    )
    first = core.select_candidate(trials, "hg_expected_aware_selection", "hdbscan")
    second = core.select_candidate(
        trials.sample(frac=1, random_state=9),
        "hg_expected_aware_selection",
        "hdbscan",
    )
    assert first["params_json"] == second["params_json"] == '{"x": 2}'


def test_optics_data_dependent_grid_uses_supplied_model_selection_features() -> None:
    config = runner.load_config(CONFIG_PATH)
    rng = np.random.default_rng(4)
    first = rng.normal(0, 0.01, size=(300, 4))
    second = first * 20.0
    first_grid = core.candidate_grid("optics", first, 7, config)
    second_grid = core.candidate_grid("optics", second, 7, config)
    first_eps = sorted({row["max_eps"] for row in first_grid})
    second_eps = sorted({row["max_eps"] for row in second_grid})
    assert first_eps != second_eps
    np.testing.assert_allclose(np.asarray(second_eps), np.asarray(first_eps) * 20.0)


def test_frozen_manifest_hash_validates_and_detects_mutation() -> None:
    payload = {"protocol_version": "test", "selected": [{"method": "kmeans"}]}
    payload["complete_frozen_configuration_sha256"] = protocol_io.canonical_sha256(payload)
    assert (
        protocol_io.validate_frozen_payload(payload)
        == payload["complete_frozen_configuration_sha256"]
    )
    payload["selected"][0]["method"] = "optics"
    with pytest.raises(ValueError):
        protocol_io.validate_frozen_payload(payload)


def test_generated_frozen_protocol_and_manifest_validate() -> None:
    frozen = protocol_io.load_yaml(FROZEN_PATH)
    frozen_hash = protocol_io.validate_frozen_payload(frozen)
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["complete_frozen_configuration_sha256"] == frozen_hash
    assert manifest["frozen_protocol_file_sha256"] == protocol_io.sha256_file(FROZEN_PATH)
    assert manifest["independent_test_locked"] is True
    assert frozen["independent_test_locked"] is True


def test_frozen_protocol_cannot_be_overwritten_silently(tmp_path: Path) -> None:
    config = runner.load_config(CONFIG_PATH)
    frozen = {"protocol_version": config["protocol_version"]}
    frozen["complete_frozen_configuration_sha256"] = protocol_io.canonical_sha256(frozen)
    frozen_path = tmp_path / "frozen.yaml"
    protocol_io.write_yaml_atomic(frozen_path, frozen)
    paths = runner.output_paths(config)
    paths["frozen_protocol"] = frozen_path
    with pytest.raises(PermissionError):
        runner.guard_development_mutation(config, paths, False)
    with pytest.raises(PermissionError):
        runner.guard_development_mutation(config, paths, True)


def test_no_exact_trajectory_crosses_phases() -> None:
    split = pd.read_csv(SPLIT_PATH, usecols=["trajectory_id", "data_fingerprint", "split"])
    assert split.groupby("trajectory_id")["split"].nunique().max() == 1
    assert split.groupby("data_fingerprint")["split"].nunique().max() == 1


def test_feature_order_is_stable() -> None:
    config = runner.load_config(CONFIG_PATH)
    assert tuple(config["coordinate_representation"]["feature_columns"]) == (
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    )
    assert core.FEATURE_COLUMNS == tuple(config["coordinate_representation"]["feature_columns"])


def test_synthetic_transductive_test_phase_smoke(tmp_path: Path) -> None:
    output = runner.run_synthetic_transductive_smoke(tmp_path)
    assignments = pd.read_csv(output)
    assert tuple(assignments.columns) == (
        "synthetic_trajectory_id",
        "kmeans",
        "hdbscan",
        "optics",
    )
    provenance = json.loads(
        (tmp_path / "synthetic_transductive_assignments_provenance.json").read_text()
    )
    assert provenance["manual_labels_read"] is False
    assert provenance["assignment_sha256"] == protocol_io.sha256_file(output)


def test_real_independent_test_outputs_follow_authorized_contract() -> None:
    config = runner.load_config(CONFIG_PATH)
    output_path = runner.resolve_repo_path(config["test_protocol"]["clustering_output_directory"])
    manifest = json.loads(
        (output_path / "clustering_run_manifest.json").read_text(encoding="utf-8")
    )
    assignments = pd.read_parquet(output_path / "cluster_assignments.parquet")
    assert manifest["reference_labels_read"] is False
    assert manifest["target_recomputed"] is False
    assert manifest["hyperparameters_recomputed"] is False
    assert manifest["normalization_refitted"] is False
    assert len(assignments) == 27393 * 3 * 2
    assert not assignments.duplicated(
        ["scene_id", "trajectory_id", "method", "selection_strategy"]
    ).any()


def test_newport_cross_scene_video_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="another scene"):
        protocol_io.validate_scene_video_field(NEWPORT_CONFIG, "bellevue_150th_newport")
