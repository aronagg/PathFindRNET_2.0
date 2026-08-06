from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml


PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PUBLICATION_ROOT.parents[1]
APP_DIR = PUBLICATION_ROOT / "code" / "annotation_app"
sys.path.insert(0, str(APP_DIR))

import agreement  # noqa: E402
import models  # noqa: E402
import protocol  # noqa: E402
import queues  # noqa: E402
import rendering  # noqa: E402
import source_data  # noqa: E402
import storage  # noqa: E402


EXPECTED_COUNTS = {
    "bellevue_116th_ne12th": 964,
    "bellevue_150th_newport": 3924,
    "bellevue_150th_eastgate": 10755,
    "bellevue_150th_se38th": 3713,
    "bellevue_ne8th": 8037,
}


@pytest.fixture(scope="module")
def primary_queues() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = PUBLICATION_ROOT / "annotations" / "queues"
    return (
        pd.read_csv(root / "independent_test_annotator_A.csv", keep_default_na=False),
        pd.read_csv(root / "independent_test_annotator_B.csv", keep_default_na=False),
    )


def test_pilot_uses_only_target_estimation_and_all_scenes() -> None:
    pilot = pd.read_csv(PUBLICATION_ROOT / "annotations" / "queues" / "protocol_pilot_queue.csv")
    assert set(pilot["split"]) == {"target_estimation"}
    assert pilot.groupby("scene_id").size().to_dict() == {scene: 100 for scene in models.SCENES}
    assert len(pilot) == 500


def test_primary_queues_are_complete_independent_test(primary_queues) -> None:
    a, b = primary_queues
    queues.validate_primary_queues(a, b, EXPECTED_COUNTS)
    for frame in (a, b):
        assert len(frame) == 27393
        assert set(frame["split"]) == {"independent_test"}
        assert set(frame["queue_status"]) == {"locked_pending_protocol_freeze"}


def test_queue_order_is_deterministic_and_annotator_specific(primary_queues) -> None:
    a, b = primary_queues
    assert a["trajectory_id"].tolist() != b["trajectory_id"].tolist()
    source = source_data.attach_sources(
        source_data.load_manifest_and_split(
            PUBLICATION_ROOT / "data" / "manifests" / "trajectory_manifest.csv",
            PUBLICATION_ROOT / "data" / "splits" / "evaluation_split.csv",
        ),
        pd.read_csv(
            PUBLICATION_ROOT / "annotations" / "provenance" / "trajectory_source_catalog.csv"
        ),
    )
    rebuilt = queues.primary_annotator_queue(source, "annotator_A", protocol_frozen=False)
    assert rebuilt["trajectory_id"].tolist() == a["trajectory_id"].tolist()


def test_blind_ui_contract_rejects_forbidden_and_unexpected_columns(primary_queues) -> None:
    a, _ = primary_queues
    models.validate_blind_queue_columns(tuple(a.columns))
    for forbidden in models.FORBIDDEN_BLIND_COLUMNS:
        with pytest.raises(PermissionError):
            models.validate_blind_queue_columns(tuple(a.columns) + (forbidden,))
    app_text = (APP_DIR / "app.py").read_text(encoding="utf-8")
    assert "suggested_label" not in app_text
    assert "rare_movement" not in app_text


def test_annotation_app_has_no_scientific_method_imports() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in APP_DIR.glob("*.py"))
    forbidden_imports = (
        "hg_msa_tc_core",
        "run_split_aware_hg_msa_tc",
        "sklearn.cluster",
        "hdbscan",
    )
    assert not any(value in text for value in forbidden_imports)


def _synthetic_protocol_tree(tmp_path: Path) -> Path:
    annotations = tmp_path / "annotations"
    protocol_dir = annotations / "protocol"
    guide_dir = protocol_dir / "scene_guides"
    guide_dir.mkdir(parents=True)
    (protocol_dir / "annotation_protocol_draft.yaml").write_text(
        yaml.safe_dump(
            {
                "scenes": list(models.SCENES),
                "blinding_policy": {"forbidden": ["cluster_id"]},
                "allowed_values": {"maneuver_type": list(models.MANEUVER_TYPES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    guideline = protocol_dir / "annotation_guideline.md"
    guideline.write_text("Synthetic guideline.\n", encoding="utf-8")
    for scene in models.SCENES:
        frame = guide_dir / f"{scene}_representative_frame.png"
        cv2.imwrite(str(frame), np.zeros((80, 120, 3), dtype=np.uint8))
        guide = {
            "scene_id": scene,
            "status": "ready_for_freeze",
            "representative_frame": f"protocol/scene_guides/{frame.name}",
            "approaches": [
                {
                    "id": "A",
                    "label_position_normalized": {"x": 0.1, "y": 0.5},
                    "arrow_end_normalized": {"x": 0.2, "y": 0.5},
                },
                {
                    "id": "B",
                    "label_position_normalized": {"x": 0.9, "y": 0.5},
                    "arrow_end_normalized": {"x": 0.8, "y": 0.5},
                },
            ],
            "valid_entry_approaches": ["A", "B"],
            "valid_exit_approaches": ["A", "B"],
            "maneuver_type_mapping": [{"entry": "A", "exit": "B", "maneuver_type": "straight"}],
            "ambiguity_notes": "synthetic",
        }
        guide_path = guide_dir / f"{scene}.yaml"
        guide_path.write_text(yaml.safe_dump(guide, sort_keys=False), encoding="utf-8")
        cv2.imwrite(str(guide_dir / f"{scene}_guide.png"), np.zeros((80, 120, 3), dtype=np.uint8))
    manifest = tmp_path / "frozen_selection_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_version": "future-transportation-split-aware-v1",
                "complete_frozen_configuration_sha256": "a" * 64,
                "independent_test_locked": True,
            }
        ),
        encoding="utf-8",
    )
    protocol.freeze_protocol(protocol_dir, manifest, guideline)
    return protocol_dir


def test_protocol_freeze_is_required_and_immutable(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    with pytest.raises(PermissionError):
        protocol.require_frozen_protocol(empty)
    protocol_dir = _synthetic_protocol_tree(tmp_path)
    frozen = protocol.require_frozen_protocol(protocol_dir)
    assert frozen["protocol_version"] == protocol.ANNOTATION_PROTOCOL_VERSION
    with pytest.raises(FileExistsError):
        protocol.freeze_protocol(
            protocol_dir,
            tmp_path / "frozen_selection_manifest.json",
            protocol_dir / "annotation_guideline.md",
        )


def test_scene_guide_editor_skips_blank_rows_and_normalizes_values() -> None:
    rows = protocol.normalize_approach_rows(
        [
            {
                "id": "A",
                "human_readable_name": None,
                "label_x": "0.1",
                "label_y": 0.2,
                "arrow_x": 0.3,
                "arrow_y": 0.4,
            },
            {
                "id": None,
                "human_readable_name": None,
                "label_x": None,
                "label_y": None,
                "arrow_x": None,
                "arrow_y": None,
            },
        ]
    )
    assert rows == [
        {
            "id": "A",
            "human_readable_name": "",
            "label_position_normalized": {"x": 0.1, "y": 0.2},
            "arrow_end_normalized": {"x": 0.3, "y": 0.4},
        }
    ]


def test_scene_guide_editor_normalizes_manual_region() -> None:
    rows = protocol.normalize_approach_rows(
        [
            {
                "id": "A",
                "human_readable_name": "North entry",
                "label_x": 0.2,
                "label_y": 0.3,
                "arrow_x": 0.2,
                "arrow_y": 0.3,
                "region_x_min": 0.1,
                "region_y_min": 0.2,
                "region_x_max": 0.3,
                "region_y_max": 0.4,
                "region_role": "entry",
            }
        ]
    )
    assert rows[0]["region_normalized"] == {
        "x_min": 0.1,
        "y_min": 0.2,
        "x_max": 0.3,
        "y_max": 0.4,
    }
    assert rows[0]["region_role"] == "entry"


def test_scene_guide_editor_normalizes_manual_polygon() -> None:
    polygon = [
        {"x": 0.1, "y": 0.2},
        {"x": 0.4, "y": 0.2},
        {"x": 0.3, "y": 0.5},
    ]
    rows = protocol.normalize_approach_rows(
        [
            {
                "id": "A",
                "human_readable_name": "North entry",
                "label_x": 0.25,
                "label_y": 0.3,
                "arrow_x": 0.25,
                "arrow_y": 0.3,
                "polygon_points": json.dumps(polygon),
                "region_role": "entry",
            }
        ]
    )
    assert rows[0]["polygon_normalized"] == polygon
    assert rows[0]["region_role"] == "entry"


@pytest.mark.parametrize(
    ("polygon", "message"),
    [
        ([{"x": 0.1, "y": 0.2}, {"x": 0.4, "y": 0.2}], "at least three"),
        (
            [{"x": 0.1, "y": 0.2}, {"x": 0.2, "y": 0.2}, {"x": 0.3, "y": 0.2}],
            "positive area",
        ),
    ],
)
def test_scene_guide_editor_rejects_invalid_polygon(
    polygon: list[dict[str, float]], message: str
) -> None:
    row = {
        "id": "A",
        "human_readable_name": "",
        "label_x": 0.2,
        "label_y": 0.3,
        "arrow_x": 0.2,
        "arrow_y": 0.3,
        "polygon_points": json.dumps(polygon),
        "region_role": "entry",
    }
    with pytest.raises(ValueError, match=message):
        protocol.normalize_approach_rows([row])


def test_scene_guide_editor_rejects_incomplete_or_inverted_region() -> None:
    base = {
        "id": "A",
        "human_readable_name": "",
        "label_x": 0.2,
        "label_y": 0.3,
        "arrow_x": 0.2,
        "arrow_y": 0.3,
        "region_x_min": 0.4,
        "region_y_min": 0.2,
        "region_x_max": 0.3,
        "region_y_max": 0.5,
        "region_role": "entry",
    }
    with pytest.raises(ValueError, match="positive area"):
        protocol.normalize_approach_rows([base])
    incomplete = dict(base, region_x_min=0.1, region_x_max=None)
    with pytest.raises(ValueError, match="all four region coordinates"):
        protocol.normalize_approach_rows([incomplete])


def test_scene_guide_renderer_draws_manual_point_and_region(tmp_path: Path) -> None:
    background = np.full((240, 320, 3), 230, dtype=np.uint8)
    guide = {
        "status": "draft_manual_configuration_required",
        "approaches": [
            {
                "id": "A",
                "label_position_normalized": {"x": 0.25, "y": 0.3},
                "arrow_end_normalized": {"x": 0.25, "y": 0.3},
                "region_normalized": {
                    "x_min": 0.1,
                    "y_min": 0.2,
                    "x_max": 0.4,
                    "y_max": 0.5,
                },
                "region_role": "entry",
            },
            {
                "id": "B",
                "label_position_normalized": {"x": 0.7, "y": 0.35},
                "arrow_end_normalized": {"x": 0.7, "y": 0.35},
                "polygon_normalized": [
                    {"x": 0.55, "y": 0.2},
                    {"x": 0.85, "y": 0.25},
                    {"x": 0.8, "y": 0.55},
                    {"x": 0.6, "y": 0.5},
                ],
                "region_role": "exit",
            },
        ],
    }
    output = rendering.render_scene_guide(background, guide, tmp_path / "guide.png")
    rendered = cv2.imread(str(output))
    assert output.stat().st_size > 0
    assert rendered is not None
    assert rendered.std() > 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label_x", None, "label_x is required"),
        ("label_y", "not-a-number", "label_y must be numeric"),
        ("arrow_x", 1.1, "arrow_x must be between 0 and 1"),
    ],
)
def test_scene_guide_editor_reports_partial_row_errors(
    field: str, value: object, message: str
) -> None:
    row = {
        "id": "A",
        "human_readable_name": "Approach A",
        "label_x": 0.1,
        "label_y": 0.2,
        "arrow_x": 0.3,
        "arrow_y": 0.4,
    }
    row[field] = value
    with pytest.raises(ValueError, match=message):
        protocol.normalize_approach_rows([row])


def test_ready_scene_guide_rejects_unknown_entry_and_exit_ids() -> None:
    guide = {
        "scene_id": models.SCENES[0],
        "status": "ready_for_freeze",
        "approaches": [
            {
                "id": "A",
                "label_position_normalized": {"x": 0.1, "y": 0.2},
                "arrow_end_normalized": {"x": 0.3, "y": 0.4},
            },
            {
                "id": "B",
                "label_position_normalized": {"x": 0.5, "y": 0.6},
                "arrow_end_normalized": {"x": 0.7, "y": 0.8},
            },
        ],
        "valid_entry_approaches": ["A", "C"],
        "valid_exit_approaches": ["B"],
        "maneuver_type_mapping": [],
    }
    with pytest.raises(ValueError, match="Invalid entry/exit approach set"):
        protocol.validate_scene_guide_definition(guide, models.SCENES[0])


def _queue_record(source_path: Path, annotator: str = "annotator_A") -> dict:
    return {
        "annotator_id": annotator,
        "scene_id": models.SCENES[0],
        "trajectory_id": f"{models.SCENES[0]}:1",
        "split": "independent_test",
        "recording_id": "recording",
        "trajectory_source_checksum": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }


def _values(entry: str = "A", exit_value: str = "B") -> dict:
    return {
        "entry_approach": entry,
        "exit_approach": exit_value,
        "maneuver_type": "straight",
        "validity": "valid",
        "confidence": "high",
        "notes": "",
    }


def test_autosave_revision_history_and_annotator_isolation(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    database = tmp_path / "A.sqlite"
    storage.initialize_database(database, "annotator_A", "v1")
    queue = _queue_record(source)
    first = storage.save_annotation(database, queue, _values(), "annotator_A", "v1", "renderer")
    second = storage.save_annotation(
        database, queue, _values("B", "A"), "annotator_A", "v1", "renderer"
    )
    history = storage.annotation_history(database, "annotator_A", queue["trajectory_id"])
    assert history["annotation_id"].tolist() == [first, second]
    assert history["is_active"].tolist() == [0, 1]
    assert history["revision_number"].tolist() == [1, 2]
    with pytest.raises(PermissionError):
        storage.assert_database_identity(database, "annotator_B", "v1")
    with pytest.raises(PermissionError):
        storage.assert_database_identity(database, "annotator_A", "v2")
    assert storage.validate_storage_integrity(database)["duplicate_active"] == 0
    backup = storage.backup_database(database, tmp_path / "backups")
    assert storage.validate_storage_integrity(backup)["annotations"] == 2


def test_rare_movement_is_not_human_input() -> None:
    record = {
        "scene_id": models.SCENES[0],
        "entry_approach": "A",
        "exit_approach": "B",
        "manual_maneuver_id": models.manual_maneuver_id(models.SCENES[0], "A", "B"),
        "maneuver_type": "straight",
        "validity": "valid",
        "confidence": "high",
        "rare_movement": False,
    }
    with pytest.raises(ValueError):
        models.validate_annotation_values(record)


def test_manual_maneuver_id_is_deterministic() -> None:
    assert models.manual_maneuver_id("bellevue_ne8th", "A", "B") == "bellevue_ne8th:A>B"


def _synthetic_exports() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_a, rows_b = [], []
    labels_a = [
        ("A", "B", "valid", "straight", "high"),
        ("A", "C", "valid", "left", "medium"),
        ("B", "A", "ambiguous", "unknown", "low"),
        ("B", "C", "valid", "right", "high"),
    ]
    labels_b = [
        ("A", "B", "valid", "straight", "high"),
        ("A", "C", "valid", "left", "high"),
        ("B", "C", "ambiguous", "unknown", "low"),
        ("B", "C", "valid", "right", "high"),
    ]
    for index, (left, right) in enumerate(zip(labels_a, labels_b)):
        for label, rows, annotator in (
            (left, rows_a, "annotator_A"),
            (right, rows_b, "annotator_B"),
        ):
            entry, exit_value, validity, maneuver, confidence = label
            rows.append(
                {
                    "annotation_id": f"{annotator}-{index}",
                    "scene_id": models.SCENES[0],
                    "trajectory_id": f"t{index}",
                    "entry_approach": entry,
                    "exit_approach": exit_value,
                    "manual_maneuver_id": models.manual_maneuver_id(
                        models.SCENES[0], entry, exit_value
                    ),
                    "validity": validity,
                    "maneuver_type": maneuver,
                    "confidence": confidence,
                }
            )
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def test_agreement_metrics_match_synthetic_fixture() -> None:
    a, b = _synthetic_exports()
    result = agreement.agreement_analysis(a, b)
    exact = result["summary"].set_index("field").loc["manual_maneuver_id"]
    assert exact["raw_agreement_pct"] == 75.0
    assert len(result["disagreements"]) == 1


def test_adjudication_preserves_raw_annotations(tmp_path: Path) -> None:
    a, b = _synthetic_exports()
    raw_a, raw_b = a.iloc[2].to_dict(), b.iloc[2].to_dict()
    consensus = {
        **raw_a,
        "entry_approach": "B",
        "exit_approach": "C",
        "maneuver_type": "right",
        "validity": "valid",
        "confidence": "medium",
    }
    consensus["manual_maneuver_id"] = models.manual_maneuver_id(consensus["scene_id"], "B", "C")
    database = tmp_path / "adjudication.sqlite"
    agreement.initialize_adjudication_database(database, "v1")
    agreement.save_adjudication(
        database, raw_a, raw_b, consensus, "reviewer", "v1", "resolved from video"
    )
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT raw_label_A_json, raw_label_B_json FROM adjudications"
        ).fetchone()
    assert json.loads(row[0])["manual_maneuver_id"] == raw_a["manual_maneuver_id"]
    assert json.loads(row[1])["manual_maneuver_id"] == raw_b["manual_maneuver_id"]


def test_source_checksum_mismatch_is_rejected_and_source_unmodified(tmp_path: Path) -> None:
    path = tmp_path / "tracks.parquet"
    pd.DataFrame(
        {"track_id": [1, 1], "frame": [1, 2], "cx": [1.0, 2.0], "cy": [3.0, 4.0]}
    ).to_parquet(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        source_data.load_full_polyline(tmp_path, path.name, 1, "0" * 64)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_camera_aspect_ratio_is_preserved() -> None:
    x_limits, y_limits = rendering.compute_viewport(
        1920, 1080, np.array([500, 800]), np.array([300, 700])
    )
    width = x_limits[1] - x_limits[0]
    height = y_limits[0] - y_limits[1]
    assert width / height == pytest.approx(1920 / 1080)


def test_real_scientific_test_was_explicitly_unlocked_after_freeze() -> None:
    frozen = json.loads(
        (PUBLICATION_ROOT / "results" / "development" / "frozen_selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert frozen["independent_test_locked"] is True
    unlock = json.loads(
        (PUBLICATION_ROOT / "configs" / "INDEPENDENT_TEST_UNLOCK.json").read_text(encoding="utf-8")
    )
    assert unlock["frozen_protocol_sha256"] == frozen["complete_frozen_configuration_sha256"]
    output = PUBLICATION_ROOT / "results" / "independent_test"
    clustering = json.loads((output / "clustering_run_manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output / "evaluation_run_manifest.json").read_text(encoding="utf-8"))
    assert clustering["reference_labels_read"] is False
    assert clustering["evaluation_started"] is False
    assert clustering["assignment_rows"] == 27393 * 6
    assert evaluation["assignments_loaded_before_reference"] is True


def test_real_preflight_maps_every_primary_trajectory() -> None:
    summary = json.loads(
        (
            PUBLICATION_ROOT
            / "annotations"
            / "preflight"
            / "independent_test_renderability_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["total"] == 27393
    assert summary["full_polyline_missing"] == 0
    assert summary["video_missing"] == 0
    assert summary["nonfinite"] == 0
    assert summary["frame_range_not_covered"] == 0
    assert summary["all_renderable"] is True


def test_no_real_human_labels_or_databases_were_generated() -> None:
    database_dir = PUBLICATION_ROOT / "annotations" / "databases"
    export_dir = PUBLICATION_ROOT / "annotations" / "exports"
    assert not database_dir.exists() or not any(database_dir.glob("*.sqlite*"))
    assert not export_dir.exists() or not any(export_dir.glob("independent_test_*.csv"))
    assert not (
        PUBLICATION_ROOT / "annotations" / "protocol" / "annotation_protocol_v1.yaml"
    ).exists()
