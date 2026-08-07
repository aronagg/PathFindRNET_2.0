from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PUBLICATION_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from hg_smg.ablations import executable_sac_variants  # noqa: E402
from hg_smg.bootstrap import (  # noqa: E402
    hierarchical_recording_indices,
    percentile_interval,
    shannon_entropy_bits,
    smaller_tie_mode,
)
from hg_smg.descriptors import (  # noqa: E402
    circular_distance,
    circular_l1_median,
    directed_heading,
)
from hg_smg.pcms import interval_distance, select_pcms_candidates  # noqa: E402
from hg_smg.provenance import derive_seed, reject_forbidden_paths  # noqa: E402
from hg_smg.runner import future_test_gate  # noqa: E402
from hg_smg.sac import SACVariant, run_sac  # noqa: E402
from hg_smg.smg import run_smg  # noqa: E402


def test_seed_derivation_matches_frozen_formula() -> None:
    payload = b"20260901|scene|SAC|entry|3"
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647
    assert derive_seed("scene", "SAC", "entry", 3) == expected


def test_circular_l1_median_matches_observed_brute_force() -> None:
    rng = np.random.default_rng(42)
    for _ in range(30):
        values = rng.uniform(-np.pi, np.pi, size=17)
        result = circular_l1_median(values)
        wrapped = np.sort((values + np.pi) % (2 * np.pi) - np.pi)
        objectives = np.asarray([circular_distance(values, item).sum() for item in wrapped])
        expected = wrapped[np.flatnonzero(np.isclose(objectives, objectives.min()))[0]]
        assert result == pytest.approx(expected, abs=1e-12)


def test_directed_heading_uses_exact_endpoint_window() -> None:
    points = np.column_stack([np.arange(7.0), np.zeros(7)])
    assert directed_heading(points, "entry", 5) == pytest.approx(0.0)
    assert directed_heading(points, "exit", 5) == pytest.approx(0.0)
    assert np.isnan(directed_heading(points[:4], "entry", 5))


def test_hierarchical_bootstrap_is_deterministic_and_preserves_size() -> None:
    frame = pd.DataFrame({"recording_id": ["a"] * 3 + ["b"] * 2})
    first = hierarchical_recording_indices(frame, 7)
    second = hierarchical_recording_indices(frame, 7)
    assert np.array_equal(first, second)
    assert len(first) in {4, 5, 6}


def test_uatp_summary_primitives() -> None:
    values = np.asarray([2, 2, 3, 4])
    assert smaller_tie_mode(values) == 2
    assert percentile_interval(values, 0.90) == (2, 4)
    assert 0.0 < shannon_entropy_bits(values) < 2.0


def _synthetic_sac_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(8)],
            "start_x_topview": [-2.0, -2.1, -1.9, -2.0, 2.0, 2.1, 1.9, 2.0],
            "start_y_topview": [0.0, 0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.0],
            "end_x_topview": [0.0] * 8,
            "end_y_topview": [2.0, 2.1, 1.9, 2.0, -2.0, -2.1, -1.9, -2.0],
            "heading_camera_w5_entry": [0.0] * 4 + [np.pi] * 4,
            "heading_camera_w5_exit": [np.pi / 2] * 4 + [-np.pi / 2] * 4,
        }
    )
    return frame, np.asarray([0] * 4 + [1] * 4), np.asarray([0] * 4 + [1] * 4)


def test_sac_never_merges_across_roles_and_smg_aggregates() -> None:
    frame, entry, exit_labels = _synthetic_sac_frame()
    result = run_sac(
        "scene",
        frame,
        entry,
        exit_labels,
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 0.0]),
        SACVariant("test", "heading_camera_w5_{role}"),
        bootstrap_replicates=20,
    )
    assert set(result.assignments["role"]) == {"entry", "exit"}
    assert result.assignments.groupby("supernode_id")["role"].nunique().max() == 1
    smg = run_smg("scene", frame["trajectory_id"], entry, exit_labels, result, [0.001, 0.005])
    assert smg.summary["valid_assignment_denominator"] == 8
    assert smg.edges["count"].sum() == 8


def test_pcms_interval_distance_and_selection_order() -> None:
    assert interval_distance(3, 4, 6) == 1
    assert interval_distance(5, 4, 6) == 0
    assert interval_distance(8, 4, 6) == 2
    candidates = pd.DataFrame(
        [
            {"scene": "s", "method": "kmeans", "params_json": "a", "n_clusters": 5, "silhouette_clustered_only": 0.4, "davies_bouldin_clustered_only": 1.0, "calinski_harabasz_clustered_only": 2.0, "largest_cluster_ratio": 0.5, "pct_outliers": 0.0},
            {"scene": "s", "method": "kmeans", "params_json": "b", "n_clusters": 5, "silhouette_clustered_only": 0.6, "davies_bouldin_clustered_only": 1.2, "calinski_harabasz_clustered_only": 2.0, "largest_cluster_ratio": 0.5, "pct_outliers": 0.0},
        ]
    )
    _, selected = select_pcms_candidates(
        candidates,
        pd.DataFrame([{"scene": "s", "interval_lower": 4, "interval_upper": 6}]),
    )
    assert selected.iloc[0]["params_json"] == "b"


def test_forbidden_paths_and_future_gate_refuse() -> None:
    with pytest.raises(PermissionError):
        reject_forbidden_paths(["annotations/reference_labels/x.csv"])
    with pytest.raises(PermissionError):
        future_test_gate(None)


def test_preregistered_variants_are_exact_and_a8_is_not_silently_defined() -> None:
    ids = {variant.variant_id for variant in executable_sac_variants()}
    assert ids == {
        "A5",
        "A6",
        "A7",
        "A9",
        "sensitivity_heading_w3",
        "sensitivity_heading_w7",
        "sensitivity_self_consistency_q090",
        "sensitivity_self_consistency_q0975",
    }
    ablations = yaml.safe_load(
        (PUBLICATION_ROOT / "configs/hg_smg_ablation_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert list(ablations["ablations"]) == [f"A{index}" for index in range(11)]


def test_hg_smg_source_has_no_reference_or_test_data_imports() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in sorted((CODE_ROOT / "hg_smg").glob("*.py"))
    )
    forbidden = (
        "independent_test_reference_labels",
        "independent_test_metrics.csv",
        "cluster_movement_mapping.csv",
        "scene_guides/",
    )
    assert all(value not in source for value in forbidden)
