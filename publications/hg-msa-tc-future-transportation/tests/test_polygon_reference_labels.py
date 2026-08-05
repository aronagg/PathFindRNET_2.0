from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from shapely.geometry import Polygon


PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PUBLICATION_ROOT.parents[1]
CODE_ROOT = PUBLICATION_ROOT / "code"
REFERENCE_CODE = CODE_ROOT / "reference_labels"
sys.path.insert(0, str(CODE_ROOT))

from reference_labels import generator, protocol  # noqa: E402


EXPECTED_SCENE_SPLIT_COUNTS = {
    ("bellevue_116th_ne12th", "target_estimation"): 723,
    ("bellevue_116th_ne12th", "model_selection"): 634,
    ("bellevue_116th_ne12th", "independent_test"): 964,
    ("bellevue_150th_newport", "target_estimation"): 2423,
    ("bellevue_150th_newport", "model_selection"): 3101,
    ("bellevue_150th_newport", "independent_test"): 3924,
    ("bellevue_150th_eastgate", "target_estimation"): 8720,
    ("bellevue_150th_eastgate", "model_selection"): 6791,
    ("bellevue_150th_eastgate", "independent_test"): 10755,
    ("bellevue_150th_se38th", "target_estimation"): 2482,
    ("bellevue_150th_se38th", "model_selection"): 2865,
    ("bellevue_150th_se38th", "independent_test"): 3713,
    ("bellevue_ne8th", "target_estimation"): 5576,
    ("bellevue_ne8th", "model_selection"): 6321,
    ("bellevue_ne8th", "independent_test"): 8037,
}


@pytest.fixture(scope="module")
def frozen_protocol() -> dict:
    return protocol.load_frozen_protocol(
        PUBLICATION_ROOT / "annotations/protocol/polygon_reference_protocol_v1.yaml"
    )


@pytest.fixture(scope="module")
def labels() -> pd.DataFrame:
    return pd.read_parquet(
        PUBLICATION_ROOT / "annotations/reference_labels/polygon_rule_reference_labels_all.parquet"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_protocol_matches_all_five_guides(frozen_protocol: dict) -> None:
    assert tuple(frozen_protocol["scene_order"]) == protocol.SCENES
    guide_dir = PUBLICATION_ROOT / "annotations/protocol/scene_guides"
    for scene in protocol.SCENES:
        checksums = frozen_protocol["scene_guide_checksums"][scene]
        assert checksums["yaml_sha256"] == _sha256(guide_dir / f"{scene}.yaml")
        assert checksums["rendered_guide_sha256"] == _sha256(guide_dir / f"{scene}_guide.png")
        guide = next(
            value for value in frozen_protocol["scene_guides"] if value["scene_id"] == scene
        )
        assert len(guide["valid_entry_approaches"]) == 4
        assert len(guide["valid_exit_approaches"]) == 4
        assert len(guide["maneuver_type_mapping"]) == 12


def test_all_trajectories_appear_once_and_split_totals_match(labels: pd.DataFrame) -> None:
    assert len(labels) == generator.EXPECTED_TOTAL
    assert labels["trajectory_id"].is_unique
    counts = labels.groupby(["scene_id", "split"]).size().to_dict()
    assert counts == EXPECTED_SCENE_SPLIT_COUNTS


def test_valid_and_invalid_assignment_invariants(labels: pd.DataFrame) -> None:
    valid = labels[labels["reference_status"] == "valid"]
    invalid = labels[labels["reference_status"] != "valid"]
    assert (valid["entry_match_count"] == 1).all()
    assert (valid["exit_match_count"] == 1).all()
    assert valid["reference_movement_id"].str.contains(r":[A-D]>[E-H]$", regex=True).all()
    assert (invalid["reference_movement_id"].fillna("") == "").all()
    assert (invalid["reference_maneuver_type"].fillna("") == "").all()


def test_legal_inventory_is_exactly_the_frozen_mapping() -> None:
    inventory = pd.read_csv(
        PUBLICATION_ROOT / "annotations/reference_labels/legal_movement_inventory.csv"
    )
    assert len(inventory) == 60
    assert inventory.groupby("scene_id").size().to_dict() == {
        scene: 12 for scene in protocol.SCENES
    }
    assert inventory[["scene_id", "movement_id"]].duplicated().sum() == 0


def test_overlaps_are_ambiguous_and_nearest_is_diagnostic_only() -> None:
    polygons = {
        "A": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        "B": Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]),
    }
    overlap = generator.assign_endpoint(1.5, 1.0, polygons, 1e-7)
    outside = generator.assign_endpoint(10.0, 10.0, polygons, 1e-7)
    assert overlap.matched_ids == ("A", "B")
    assert outside.matched_ids == ()
    assert outside.nearest_id in {"A", "B"}


def test_protocol_rejects_cluster_or_hg_derived_fields() -> None:
    guide_path = PUBLICATION_ROOT / "annotations/protocol/scene_guides/bellevue_116th_ne12th.yaml"
    guide = yaml.safe_load(guide_path.read_text(encoding="utf-8"))
    guide["hg_target"] = 12
    with pytest.raises(ValueError, match="Forbidden"):
        protocol.validate_scene_guide(guide, "bellevue_116th_ne12th")


def test_output_manifest_uses_only_canonical_inputs_and_preserves_lock() -> None:
    output_manifest = json.loads(
        (
            PUBLICATION_ROOT / "annotations/reference_labels/reference_output_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert set(output_manifest["input_files"]) == {
        "trajectory_manifest",
        "evaluation_split",
        "source_catalog",
        "frozen_polygon_protocol",
        "frozen_model_selection_manifest",
    }
    frozen_selection = PUBLICATION_ROOT / "results/development/frozen_selection_manifest.json"
    assert output_manifest["input_files"]["frozen_model_selection_manifest"]["sha256"] == _sha256(
        frozen_selection
    )
    assert (
        json.loads(frozen_selection.read_text(encoding="utf-8"))["independent_test_locked"] is True
    )


def test_sensitivity_contains_every_trajectory_for_each_variant(labels: pd.DataFrame) -> None:
    sensitivity = pd.read_csv(
        PUBLICATION_ROOT / "annotations/reference_labels/polygon_assignment_sensitivity.csv"
    )
    assert sensitivity["variant"].nunique() == 4
    assert len(sensitivity) == 4 * len(labels)
    assert not sensitivity.duplicated(["trajectory_id", "variant"]).any()


def test_reference_code_has_no_clustering_execution_surface() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(REFERENCE_CODE.glob("*.py"))
    ).lower()
    for forbidden in ("fit_predict(", "kmeans(", "hdbscan(", "optics(", "emas_hg"):
        assert forbidden not in source
