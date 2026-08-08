from __future__ import annotations

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

from baselines import independent_test_baselines as baselines  # noqa: E402
from independent_test.evaluator import evaluate_partition  # noqa: E402


def test_baseline_grids_are_deterministic() -> None:
    features = np.arange(200, dtype=float).reshape(50, 4)
    for method in baselines.METHODS:
        first = baselines.deterministic_grid(method, features, 123)
        second = baselines.deterministic_grid(method, features, 123)
        assert first == second
        assert len(first) > 0


def test_reference_free_metric_fixture() -> None:
    assignments = pd.DataFrame(
        {
            "scene_id": ["s"] * 4,
            "trajectory_id": ["a", "b", "c", "d"],
            "cluster_label": [0, 0, 1, 1],
            "is_noise": [False, False, False, False],
        }
    )
    reference = pd.DataFrame(
        {
            "scene_id": ["s"] * 4,
            "trajectory_id": ["a", "b", "c", "d"],
            "reference_movement_id": ["x", "x", "y", "y"],
        }
    )
    metrics, per_movement, mapping = evaluate_partition(assignments, reference, 2, 12)
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["nmi"] == pytest.approx(1.0)
    assert metrics["purity"] == pytest.approx(1.0)
    assert set(per_movement["f1"]) == {1.0}
    assert len(mapping) == 2


def test_dtw_frechet_skip_is_non_silent() -> None:
    assert baselines.DTW_STATUS == "skipped_full_pairwise_infeasible"


def test_baseline_outputs_if_present() -> None:
    root = PUBLICATION_ROOT / "results/baselines/independent_test"
    if not root.exists():
        pytest.skip("Task 11 baseline outputs are not present")
    required = [
        "baseline_assignments.parquet",
        "baseline_metrics.csv",
        "baseline_per_movement_metrics.csv",
        "baseline_cluster_mapping.csv",
        "baseline_runtime_summary.csv",
        "baseline_comparison_to_hg_smg.csv",
        "baseline_assignment_manifest.json",
        "baseline_evaluation_manifest.json",
    ]
    for name in required:
        assert (root / name).exists()
    assignment_manifest = json.loads(
        (root / "baseline_assignment_manifest.json").read_text(encoding="utf-8")
    )
    evaluation_manifest = json.loads(
        (root / "baseline_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert assignment_manifest["reference_labels_read"] is False
    assert evaluation_manifest["assignments_loaded_before_reference"] is True
    assignments = pd.read_parquet(root / "baseline_assignments.parquet")
    assert len(assignments) == 27_393 * len(baselines.BASELINES) * len(baselines.METHODS)
    assert not assignments.duplicated(
        ["scene_id", "trajectory_id", "baseline_id", "method"]
    ).any()
