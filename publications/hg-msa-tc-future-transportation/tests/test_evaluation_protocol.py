from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PUBLICATION_ROOT / "code/data"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import build_evaluation_split as split_builder  # noqa: E402
import build_trajectory_manifest as manifest_builder  # noqa: E402


MANIFEST_PATH = PUBLICATION_ROOT / "data/manifests/trajectory_manifest.csv"
SPLIT_PATH = PUBLICATION_ROOT / "data/splits/evaluation_split.csv"
CONFIG_PATH = PUBLICATION_ROOT / "configs/evaluation_split.yaml"
ANNOTATION_PATH = PUBLICATION_ROOT / "annotations/annotation_template.csv"


@pytest.fixture(scope="session")
def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH, keep_default_na=False)


@pytest.fixture(scope="session")
def split() -> pd.DataFrame:
    return pd.read_csv(SPLIT_PATH, keep_default_na=False)


@pytest.fixture(scope="session")
def rebuilt_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    checksums_before = {
        row.source_file: manifest_builder.sha256_file(REPO_ROOT / row.source_file)
        for row in manifest[["source_file"]].drop_duplicates().itertuples(index=False)
    }
    rebuilt = manifest_builder.build_manifest(REPO_ROOT)
    checksums_after = {
        path: manifest_builder.sha256_file(REPO_ROOT / path) for path in checksums_before
    }
    assert checksums_after == checksums_before
    return rebuilt


def test_required_manifest_columns(manifest: pd.DataFrame) -> None:
    assert tuple(manifest.columns) == manifest_builder.MANIFEST_COLUMNS


def test_required_split_columns(split: pd.DataFrame) -> None:
    assert tuple(split.columns) == split_builder.SPLIT_COLUMNS


def test_all_five_required_scenes_present(manifest: pd.DataFrame, split: pd.DataFrame) -> None:
    assert tuple(manifest["scene_id"].drop_duplicates()) == manifest_builder.SCENES
    assert tuple(split["scene_id"].drop_duplicates()) == manifest_builder.SCENES
    assert (split.groupby("scene_id")["split"].nunique() == 3).all()


def test_deterministic_trajectory_ids(
    manifest: pd.DataFrame, rebuilt_manifest: pd.DataFrame
) -> None:
    expected = manifest["scene_id"] + ":" + manifest["original_track_id"].astype(str)
    assert manifest["trajectory_id"].equals(expected)
    assert rebuilt_manifest["trajectory_id"].tolist() == manifest["trajectory_id"].tolist()
    assert rebuilt_manifest["data_fingerprint"].tolist() == manifest["data_fingerprint"].tolist()


def test_source_files_match_recorded_checksums(manifest: pd.DataFrame) -> None:
    sources = manifest[
        ["source_file", "source_file_checksum_sha256"]
    ].drop_duplicates()
    assert len(sources) == len(manifest_builder.SCENES)
    for row in sources.itertuples(index=False):
        assert manifest_builder.sha256_file(REPO_ROOT / row.source_file) == row.source_file_checksum_sha256


def test_deterministic_split_assignment(manifest: pd.DataFrame, split: pd.DataFrame) -> None:
    first, _ = split_builder.build_split(manifest)
    second, _ = split_builder.build_split(manifest)
    columns = ["scene_id", "trajectory_id", "split", "split_assignment_basis"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    pd.testing.assert_frame_equal(first[columns], split[columns])


def test_no_trajectory_id_in_multiple_splits(split: pd.DataFrame) -> None:
    assert split["trajectory_id"].is_unique
    assert split.groupby("trajectory_id")["split"].nunique().max() == 1


def test_no_exact_fingerprint_in_multiple_splits(split: pd.DataFrame) -> None:
    assert split.groupby("data_fingerprint")["split"].nunique().max() == 1


def test_split_proportions_within_recording_block_tolerance(split: pd.DataFrame) -> None:
    expected = split_builder.SPLIT_PROPORTIONS
    for _, scene in split.groupby("scene_id", sort=False):
        observed = scene["split"].value_counts(normalize=True)
        for split_name, target in expected.items():
            assert abs(float(observed[split_name]) - target) <= 0.06


def test_stable_output_ordering(manifest: pd.DataFrame, split: pd.DataFrame) -> None:
    scene_rank = {scene: index for index, scene in enumerate(manifest_builder.SCENES)}
    order = pd.DataFrame(
        {
            "scene_rank": manifest["scene_id"].map(scene_rank),
            "recording": manifest["source_recording_start_time"],
            "start": manifest["start_frame"],
            "end": manifest["end_frame"],
            "track": manifest["original_track_id"],
        }
    )
    expected_index = order.sort_values(
        ["scene_rank", "recording", "start", "end", "track"], kind="mergesort"
    ).index.tolist()
    assert expected_index == list(range(len(manifest)))
    assert split["trajectory_id"].tolist() == manifest["trajectory_id"].tolist()


def test_annotation_template_has_no_manual_labels(split: pd.DataFrame) -> None:
    annotation = pd.read_csv(ANNOTATION_PATH, keep_default_na=False)
    assert tuple(annotation.columns) == split_builder.ANNOTATION_COLUMNS
    assert annotation[["scene_id", "trajectory_id", "split"]].equals(
        split[["scene_id", "trajectory_id", "split"]]
    )
    assert not (annotation.iloc[:, 3:] != "").any().any()


def test_config_records_leakage_controls(manifest: pd.DataFrame) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["protocol_version"] == manifest_builder.PROTOCOL_VERSION
    assert tuple(config["scene_list"]) == manifest_builder.SCENES
    assert config["random_seed"] == split_builder.RANDOM_SEED
    assert "not used" in config["random_seed_usage"]
    assert "manual labels are forbidden" in config["manual_label_policy"]
    assert set(config["source_data_checksums"]) == set(manifest_builder.SCENES)
