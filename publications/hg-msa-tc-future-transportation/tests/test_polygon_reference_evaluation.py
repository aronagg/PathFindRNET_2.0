from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLICATION_ROOT / "code" / "evaluation"))

import polygon_reference_evaluation as evaluation  # noqa: E402


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scene_id": ["scene"] * 7,
            "trajectory_id": [f"t{index}" for index in range(7)],
            "split": ["independent_test"] * 6 + ["model_selection"],
            "reference_status": ["valid"] * 5 + ["unassigned", "valid"],
            "reference_movement_id": ["A>E", "A>E", "A>F", "A>F", "A>F", "", "A>E"],
        }
    )


def test_perfect_permuted_clusters_have_perfect_scores() -> None:
    clusters = pd.DataFrame(
        {
            "scene_id": ["scene"] * 6,
            "trajectory_id": [f"t{index}" for index in range(6)],
            "cluster_id": [9, 9, 4, 4, 4, -1],
        }
    )
    scene, movement, coverage = evaluation.evaluate_assignments(clusters, _reference())
    assert scene.iloc[0][["ari", "nmi", "purity", "mapped_accuracy"]].tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )
    assert (movement[["precision", "recall", "f1"]] == 1.0).all().all()
    assert coverage.iloc[0]["valid_reference_rows"] == 5
    assert coverage.iloc[0]["excluded_reference_rows"] == 1


def test_missing_cluster_rows_are_reported_not_fabricated() -> None:
    clusters = pd.DataFrame(
        {
            "scene_id": ["scene"] * 4,
            "trajectory_id": ["t0", "t1", "t2", "t3"],
            "cluster_id": [0, 0, 1, 1],
        }
    )
    _, _, coverage = evaluation.evaluate_assignments(clusters, _reference())
    assert coverage.iloc[0]["missing_cluster_rows"] == 1
    assert coverage.iloc[0]["evaluated_rows"] == 4


def test_duplicate_cluster_rows_are_rejected() -> None:
    clusters = pd.DataFrame(
        {
            "scene_id": ["scene", "scene"],
            "trajectory_id": ["t0", "t0"],
            "cluster_id": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="not unique"):
        evaluation.evaluate_assignments(clusters, _reference())
