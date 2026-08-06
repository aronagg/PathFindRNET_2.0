from __future__ import annotations

import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PUBLICATION_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from independent_test import evaluator, protocol, runner  # noqa: E402
from pipeline import hg_msa_tc_core as core  # noqa: E402


@pytest.fixture(scope="module")
def paths() -> protocol.Paths:
    return protocol.default_paths()


def test_preflight_checks_all_frozen_artifacts_without_loading_test_data(paths) -> None:
    result = protocol.run_preflight(paths, write_report=False)
    assert result["status"] == "PASS"
    assert result["real_test_features_loaded"] is False
    assert result["reference_label_rows_loaded"] is False
    assert result["selected_configuration_rows"] == 30
    assert all(item["status"] == "PASS" for item in result["checks"])


def test_runner_refuses_before_feature_loading_when_preflight_fails(
    monkeypatch, paths, tmp_path
) -> None:
    isolated = replace(paths, results=tmp_path / "results")
    called = False

    def fail_preflight(*args, **kwargs):
        raise ValueError("hash mismatch")

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("features loaded")

    monkeypatch.setattr(runner, "run_preflight", fail_preflight)
    monkeypatch.setattr(runner, "_load_test_metadata", forbidden_loader)
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run_frozen_clustering(isolated, explicit_confirmation=True)
    assert called is False


def test_unlock_requires_file_and_explicit_confirmation(paths, tmp_path) -> None:
    preflight = protocol.run_preflight(paths, write_report=False)
    missing = replace(paths, unlock=tmp_path / "missing.json")
    with pytest.raises(PermissionError, match="locked"):
        protocol.verify_unlock(missing, preflight, explicit_confirmation=True)
    if paths.unlock.exists():
        with pytest.raises(PermissionError, match="confirmation"):
            protocol.verify_unlock(paths, preflight, explicit_confirmation=False)


def test_clustering_module_cannot_read_reference_labels() -> None:
    source = inspect.getsource(runner).lower()
    assert "from reference_labels" not in source
    assert "import reference_labels" not in source
    assert "read_csv(paths.reference" not in source
    assert "read_parquet(paths.reference" not in source
    assert "candidate_grid(" not in source
    assert "estimate_hg_target(" not in source


def _synthetic_partition() -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.DataFrame(
        {
            "scene_id": ["scene"] * 7,
            "trajectory_id": [f"t{i}" for i in range(7)],
            "cluster_label": [8, 8, 3, 3, 3, -1, -1],
            "is_noise": [False, False, False, False, False, True, True],
        }
    )
    reference = pd.DataFrame(
        {
            "scene_id": ["scene"] * 7,
            "trajectory_id": [f"t{i}" for i in range(7)],
            "reference_movement_id": ["A>E", "A>E", "A>F", "A>F", "A>F", "A>G", "A>G"],
        }
    )
    return assignments, reference


def test_synthetic_ari_nmi_purity_mapping_and_noise_handling() -> None:
    assignments, reference = _synthetic_partition()
    metrics, per_movement, mapping = evaluator.evaluate_partition(
        assignments,
        reference,
        observed_movement_count=3,
        frozen_hg_target=3,
    )
    expected_labels = np.array([0, 0, 1, 1, 1, 2, 2])
    predicted = np.array([8, 8, 3, 3, 3, -1, -1])
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    assert metrics["ari"] == pytest.approx(adjusted_rand_score(expected_labels, predicted))
    assert metrics["nmi"] == pytest.approx(normalized_mutual_info_score(expected_labels, predicted))
    assert metrics["purity"] == pytest.approx(1.0)
    assert metrics["mapped_accuracy"] == pytest.approx(5 / 7)
    assert metrics["noise_count_valid_reference"] == 2
    assert metrics["unmatched_reference_movement_count"] == 1
    assert len(mapping) == 2
    assert per_movement.set_index("reference_movement_id").loc["A>G", "recall"] == 0


def test_observed_and_legal_target_errors_are_separate() -> None:
    assignments, reference = _synthetic_partition()
    metrics, _, _ = evaluator.evaluate_partition(
        assignments,
        reference,
        observed_movement_count=3,
        frozen_hg_target=2,
    )
    assert metrics["n_non_noise_clusters"] == 2
    assert metrics["observed_target_abs_error"] == 1
    assert metrics["legal_target_abs_error"] == 10
    assert metrics["hg_target_abs_error"] == 0


def test_frozen_algorithms_are_deterministic_on_synthetic_fixture() -> None:
    rng = np.random.default_rng(42)
    features = np.vstack([rng.normal(0.2, 0.02, (40, 4)), rng.normal(0.8, 0.02, (40, 4))])
    parameters = {
        "kmeans": {"n_clusters": 2, "n_init": 10, "max_iter": 300},
        "hdbscan": {"min_cluster_size": 10, "min_samples": 5},
        "optics": {"min_samples": 5, "xi": 0.05, "max_eps": 0.3},
    }
    for method, params in parameters.items():
        first = core.fit_predict(method, features, params, 123)
        second = core.fit_predict(method, features, params, 123)
        assert np.array_equal(first, second)


def test_persisted_assignment_contract_when_outputs_exist(paths) -> None:
    parquet = paths.results / "cluster_assignments.parquet"
    if not parquet.exists():
        pytest.skip("Real independent-test clustering has not yet been executed.")
    assignments = pd.read_parquet(parquet)
    assert len(assignments) == 27_393 * 6
    assert not assignments.duplicated(
        ["scene_id", "trajectory_id", "method", "selection_strategy"]
    ).any()
    assert set(assignments["method"]) == set(protocol.METHODS)
    assert set(assignments["selection_strategy"]) == set(protocol.STRATEGIES)
    counts = assignments.groupby(["scene_id", "method", "selection_strategy"]).size()
    assert counts.groupby(level="scene_id").nunique().eq(1).all()


def test_access_log_contains_no_development_split_when_outputs_exist(paths) -> None:
    log = paths.results / "data_access_log.jsonl"
    if not log.exists():
        pytest.skip("Real independent-test clustering has not yet been executed.")
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    clustering_records = [record for record in records if record.get("phase") == "test"]
    assert clustering_records
    for record in clustering_records:
        assert record["observed_split_values"] == ["independent_test"]


def test_evaluation_reports_excluded_rows_and_joins_on_both_ids(paths) -> None:
    source = inspect.getsource(evaluator)
    assert 'on=["scene_id", "trajectory_id"]' in source
    metrics_path = paths.results / "independent_test_metrics.csv"
    if not metrics_path.exists():
        pytest.skip("Real independent-test evaluation has not yet been executed.")
    metrics = pd.read_csv(metrics_path)
    assert (metrics["excluded_reference_trajectories"] > 0).all()
    assert (metrics["cluster_assignment_coverage_pct"] == 100).all()


def test_target_validation_preserves_se38th_failure(paths) -> None:
    target_path = paths.results / "target_estimation_validation.csv"
    if not target_path.exists():
        pytest.skip("Real independent-test evaluation has not yet been executed.")
    targets = pd.read_csv(target_path).set_index("scene_id")
    row = targets.loc["bellevue_150th_se38th"]
    assert row["frozen_hg_target"] == 18
    assert row["observed_independent_test_movement_count"] == 9
    assert row["hg_target_abs_error_vs_observed"] == 9


def test_sensitivity_is_separate_from_primary_metrics(paths) -> None:
    primary = paths.results / "independent_test_metrics.csv"
    sensitivity = paths.results / "reference_sensitivity_evaluation.csv"
    if not sensitivity.exists():
        pytest.skip("Reference sensitivity evaluation has not yet been executed.")
    manifest = json.loads(
        (paths.results / "evaluation_run_manifest.json").read_text(encoding="utf-8")
    )
    assert protocol_io_sha(primary) == manifest["output_checksums"][primary.name]
    frame = pd.read_csv(sensitivity)
    assert set(frame["reference_variant"]) == {
        "primary",
        "median_first_last_3",
        "median_first_last_5",
        "polygon_inward_3px",
        "polygon_outward_3px",
    }


def protocol_io_sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
