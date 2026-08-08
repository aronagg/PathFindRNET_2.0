from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PUBLICATION_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from hg_smg import locked_test  # noqa: E402


def test_locked_test_refuses_without_confirmation() -> None:
    with pytest.raises(PermissionError):
        locked_test.run_assignments(None)
    with pytest.raises(PermissionError):
        locked_test.run_assignments("WRONG")


def test_locked_test_ablation_set_is_frozen() -> None:
    assert locked_test.ALL_ABLATIONS == tuple(f"A{index}" for index in range(11))
    assert locked_test.EXECUTABLE_ABLATIONS == ("A2", "A3", "A5", "A6", "A7", "A9", "A10")
    assert locked_test.COPIED_ORIGINAL_ABLATIONS == ("A0", "A1")
    assert locked_test.NON_EXECUTABLE_ABLATIONS == ("A4", "A8")


def test_ablation_comparison_synthetic_direction() -> None:
    frame = pd.DataFrame(
        [
            {
                "scene_id": "s",
                "method": "kmeans",
                "ablation_id": "A1",
                "observed_target_abs_error": 4,
                "legal_target_abs_error": 4,
                "nmi": 0.4,
                "ari": 0.3,
                "purity": 0.9,
                "macro_f1": 0.4,
                "weighted_f1": 0.5,
                "completeness": 0.5,
                "noise_pct_all_test": 10.0,
            },
            {
                "scene_id": "s",
                "method": "kmeans",
                "ablation_id": "A5",
                "observed_target_abs_error": 1,
                "legal_target_abs_error": 2,
                "nmi": 0.6,
                "ari": 0.5,
                "purity": 0.88,
                "macro_f1": 0.7,
                "weighted_f1": 0.8,
                "completeness": 0.7,
                "noise_pct_all_test": 5.0,
            },
        ]
    )
    comparison = locked_test._ablation_comparison(frame)
    paired = comparison[comparison["ablation_id"] == "A5_vs_A1_paired"].iloc[0]
    assert paired["mean_delta_observed_target_abs_error_a5_minus_a1"] == -3
    assert paired["a5_better_observed_target_abs_error_count"] == 1
    assert paired["a5_better_nmi_count"] == 1


def test_locked_test_manifest_records_assignment_before_reference_if_present() -> None:
    result_root = PUBLICATION_ROOT / "results/hg_smg/independent_test"
    assignment_manifest = result_root / "clustering_run_manifest.json"
    evaluation_manifest = result_root / "evaluation_run_manifest.json"
    if not assignment_manifest.exists() or not evaluation_manifest.exists():
        pytest.skip("Task 10 locked-test outputs are not present")
    assignments = json.loads(assignment_manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    assert assignments["reference_labels_read"] is False
    assert assignments["evaluation_started"] is False
    assert evaluation["assignments_loaded_before_reference"] is True
    assert evaluation["assignments_persisted_at_utc"] == assignments["assignments_persisted_at_utc"]


def test_locked_test_outputs_cover_all_executed_runs_if_present() -> None:
    result_root = PUBLICATION_ROOT / "results/hg_smg/independent_test"
    assignments_path = result_root / "cluster_assignments.parquet"
    if not assignments_path.exists():
        pytest.skip("Task 10 locked-test assignments are not present")
    assignments = pd.read_parquet(assignments_path)
    expected_runs = (
        len(locked_test.SCENES)
        * len(locked_test.METHODS)
        * (len(locked_test.COPIED_ORIGINAL_ABLATIONS) + len(locked_test.EXECUTABLE_ABLATIONS))
    )
    assert len(assignments.groupby(["scene_id", "method", "ablation_id"])) == expected_runs
    assert not assignments.duplicated(
        ["scene_id", "trajectory_id", "method", "ablation_id", "selection_strategy"]
    ).any()
