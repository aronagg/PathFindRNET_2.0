from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PUBLICATION_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from emas_sensitivity import analysis  # noqa: E402
from metrics.emas_hg import (  # noqa: E402
    ORIGINAL_WEIGHTS,
    EmasWeights,
    cluster_balance_component,
    compute_emas_hg,
    davies_bouldin_component,
    non_outlier_component,
    silhouette_component,
    target_agreement_component,
)
from pipeline import hg_msa_tc_core as core  # noqa: E402


def test_original_weights_and_invalid_weights() -> None:
    assert ORIGINAL_WEIGHTS.as_tuple() == (0.5, 0.2, 0.1, 0.1, 0.1)
    ORIGINAL_WEIGHTS.validate()
    with pytest.raises(ValueError, match="sum to one"):
        EmasWeights(0.5, 0.2, 0.1, 0.1, 0.2).validate()
    with pytest.raises(ValueError, match="non-negative"):
        EmasWeights(0.7, 0.2, 0.1, 0.1, -0.1).validate()
    with pytest.raises(ValueError, match="finite"):
        EmasWeights(np.nan, 0.2, 0.2, 0.2, 0.2).validate()


@pytest.mark.parametrize(
    ("function", "inputs"),
    [
        (target_agreement_component, (12, 3)),
        (non_outlier_component, (17.5,)),
        (cluster_balance_component, (0.42,)),
        (silhouette_component, (0.61,)),
        (davies_bouldin_component, (0.33,)),
    ],
)
def test_every_component_is_in_unit_interval(function, inputs) -> None:
    value = function(*inputs)
    assert 0.0 <= value <= 1.0


def test_combined_score_is_in_unit_interval_and_matches_legacy_wrapper() -> None:
    inputs = {
        "expected_target": 12,
        "cluster_count_error": 2,
        "pct_outliers": 9.5,
        "largest_cluster_ratio": 0.31,
        "silhouette_clustered_only": 0.52,
        "davies_bouldin_clustered_only": 0.44,
    }
    score = compute_emas_hg(**inputs)
    legacy = core.emas_hg(
        {
            "hg_estimated_target": inputs["expected_target"],
            "cluster_count_error": inputs["cluster_count_error"],
            "pct_outliers": inputs["pct_outliers"],
            "largest_cluster_ratio": inputs["largest_cluster_ratio"],
            "silhouette_clustered_only": inputs["silhouette_clustered_only"],
            "davies_bouldin_clustered_only": inputs["davies_bouldin_clustered_only"],
        }
    )
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(legacy, abs=0.0)


def test_edge_cases_are_explicit() -> None:
    assert target_agreement_component(0, 0) == 1.0
    assert target_agreement_component(0, 1) == 0.0
    with pytest.raises(ValueError, match="expected_target"):
        compute_emas_hg(
            expected_target=np.nan,
            cluster_count_error=0,
            pct_outliers=0,
        )
    with pytest.raises(ValueError, match="pct_outliers"):
        compute_emas_hg(
            expected_target=10,
            cluster_count_error=10,
            pct_outliers=np.nan,
        )
    assert non_outlier_component(0.0) == 1.0  # KMeans no-noise case
    assert non_outlier_component(100.0) == 0.0  # all-noise case
    assert cluster_balance_component(1.0) == 0.0  # one clustered group
    assert cluster_balance_component(np.nan) == 0.5
    assert silhouette_component(np.nan) == 0.5
    assert davies_bouldin_component(np.nan) == 0.5
    assert davies_bouldin_component(np.inf) == 0.0
    assert davies_bouldin_component(-np.inf) == 0.5


def test_all_noise_score_uses_frozen_neutral_internal_fallbacks() -> None:
    score = compute_emas_hg(
        expected_target=10,
        cluster_count_error=10,
        pct_outliers=100,
        largest_cluster_ratio=np.nan,
        silhouette_clustered_only=np.nan,
        davies_bouldin_clustered_only=np.nan,
    )
    assert score == pytest.approx(0.15)


def test_every_stored_emas_value_is_reproduced() -> None:
    paths = analysis.default_paths()
    candidates = pd.read_csv(
        PUBLICATION_ROOT / "results/development/model_selection_candidates.csv"
    )
    differences = []
    for _, row in candidates.iterrows():
        inputs = analysis._normalized_inputs(row, "development_candidates")
        differences.append(abs(compute_emas_hg(**inputs) - float(row["EMAS_HG"])))
    independent = pd.read_csv(
        PUBLICATION_ROOT / "results/independent_test/independent_test_metrics.csv"
    )
    for _, row in independent.iterrows():
        inputs = analysis._normalized_inputs(row, "independent_test_metrics")
        differences.append(abs(compute_emas_hg(**inputs) - float(row["EMAS_HG"])))
    assert max(differences) <= analysis.REPRODUCTION_TOLERANCE
    assert (
        analysis.sha256_file(
            paths.publication / "results/development/model_selection_candidates.csv"
        )
        == "c9243c0adb3135d327b4bbaaa3944feefe4b891289a51490fc5ad38734418469"
    )


def test_weight_samples_are_deterministic_and_valid(tmp_path: Path) -> None:
    paths = analysis.default_paths()
    first_paths = replace(paths, results=tmp_path / "first")
    second_paths = replace(paths, results=tmp_path / "second")
    first_local, first_global = analysis.create_weight_samples(first_paths)
    second_local, second_global = analysis.create_weight_samples(second_paths)
    pd.testing.assert_frame_equal(first_local, second_local)
    pd.testing.assert_frame_equal(first_global, second_global)
    assert len(first_global) == 1000
    assert np.allclose(first_local[list(analysis.COMPONENTS)].sum(axis=1), 1.0)
    assert np.allclose(first_global[list(analysis.COMPONENTS)].sum(axis=1), 1.0)


def test_all_named_weight_scenarios_are_valid() -> None:
    scenarios = analysis.load_named_weights(analysis.default_paths())
    assert set(scenarios) == {
        "original",
        "reviewer_example",
        "moderate_target",
        "balanced_task_internal",
        "equal_weights",
        "target_heavy",
        "outlier_heavy",
    }
    for weights in scenarios.values():
        weights.validate()


def test_weight_analysis_is_development_only_and_does_not_cluster() -> None:
    source = inspect.getsource(analysis.run_grid_analysis)
    candidate_source = inspect.getsource(analysis.load_candidate_components)
    module_source = inspect.getsource(analysis)
    assert "independent_test" not in source
    assert "reference_labels" not in source
    assert "independent_test" not in candidate_source
    for forbidden in ("fit_predict(", "KMeans(", "HDBSCAN(", "OPTICS("):
        assert forbidden not in module_source


def test_task_manifest_confirms_frozen_inputs_if_analysis_exists() -> None:
    paths = analysis.default_paths()
    manifest_path = paths.results / "task_06_input_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Task 06 analysis has not been executed yet.")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis.verify_immutable_inputs(paths, manifest)
    assert manifest["semantic_reference_labels_read"] is False
    assert manifest["independent_test_clustering_executed"] is False
    reproduction = pd.read_csv(paths.results / "emas_reproduction_check.csv")
    assert len(reproduction) == 345
    assert set(reproduction["source"]) == {
        "development_candidates",
        "development_selected",
        "frozen_protocol_selected",
        "independent_test_metrics",
    }
    assert reproduction["within_tolerance"].all()
    assert reproduction["absolute_difference"].max() <= analysis.REPRODUCTION_TOLERANCE
