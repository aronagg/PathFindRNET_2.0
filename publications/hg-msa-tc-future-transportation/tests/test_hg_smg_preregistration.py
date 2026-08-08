from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PUBLICATION_ROOT.parents[1]
CODE_ROOT = PUBLICATION_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from reproduction import cli  # noqa: E402


BASE_COMMIT = "8bfb4826725ea5c4c04042c037b521edcf216ec4"
SCENES = (
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
)


def load_protocol() -> dict:
    return yaml.safe_load(
        (PUBLICATION_ROOT / "configs" / "hg_smg_protocol_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_exact_task08_base_is_recorded_and_is_branch_ancestor() -> None:
    protocol = load_protocol()
    assert protocol["base_commit"] == BASE_COMMIT
    merge_base = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY_ROOT.as_posix()}",
            "merge-base",
            "HEAD",
            BASE_COMMIT,
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()
    assert merge_base == BASE_COMMIT


def test_protocol_hash_is_deterministic() -> None:
    protocol_path = PUBLICATION_ROOT / "configs" / "hg_smg_protocol_v1.yaml"
    sidecar = (
        PUBLICATION_ROOT / "configs" / "hg_smg_protocol_v1.sha256"
    ).read_text(encoding="ascii")
    expected = sidecar.split()[0]
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == expected
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == expected


def test_preregistration_path_has_no_independent_data_reads() -> None:
    source = (PUBLICATION_ROOT / "code" / "reproduction" / "cli.py").read_text(
        encoding="utf-8"
    ).lower()
    forbidden_path_fragments = (
        "independent_test_reference_labels",
        "cluster_assignments.parquet",
        "independent_test_metrics",
        "cluster_movement_mapping",
        "per_movement_metrics",
    )
    assert all(fragment not in source for fragment in forbidden_path_fragments)
    protocol = load_protocol()
    assert all(
        "independent_test" not in item["path"]
        and "reference_labels" not in item["path"]
        for item in protocol["frozen_inputs"].values()
    )


def test_all_required_literature_entries_exist() -> None:
    text = (PUBLICATION_ROOT / "docs" / "literature_register_hg_smg.md").read_text(
        encoding="utf-8"
    )
    identifiers = (
        "2607.10949",
        "10.1080/13658816.2024.2433086",
        "10.1080/13658816.2023.2279977",
        "10.1016/j.compenvurbsys.2016.12.006",
        "10.1109/ITSC.2013.6728262",
        "2112.01570",
        "10.1007/s00500-020-04967-9",
        "10.1016/j.procs.2025.03.032",
        "10.3390/su151914299",
    )
    assert all(identifier in text for identifier in identifiers)
    assert text.count("Status: preprint") == 2


def test_task08_facts_are_inherited_without_scene_removal() -> None:
    protocol = load_protocol()
    facts = protocol["inherited_task_08_facts"]
    assert facts["frozen_matrices_reproduce_exactly"] is True
    assert facts["opencv_version"] == "4.12"
    assert facts["ransac_reprojection_threshold_px"] == 10.0
    assert facts["max_iterations"] == 2000
    assert facts["confidence"] == 0.995
    assert facts["endpoint_extrapolation_percent_range"] == [90.4, 96.9]
    assert facts["se38th_target_preserved_for_perturbation_px"] == [1, 2, 3, 5]
    assert tuple(protocol["scenes"]["fixed"]) == SCENES
    assert protocol["scenes"]["removal_permitted"] is False


def test_primary_sac_and_sensitivity_choices_are_frozen() -> None:
    protocol = load_protocol()
    sac = protocol["modules"]["SAC"]
    assert sac["primary_heading_domain"] == "camera_isotropic_shared_scale"
    assert sac["heading_window_points"] == 5
    assert sac["self_consistency"]["bootstrap_replicates"] == 500
    assert sac["self_consistency"]["quantile"] == 0.95
    assert sac["consolidation"]["linkage"] == "complete"
    assert sac["od_profile"]["primary_merge_condition"] is False
    sensitivity = protocol["sensitivities"]
    assert sensitivity["sac_heading_window_points"] == [3, 5, 7]
    assert sensitivity["sac_self_consistency_quantiles"] == [0.90, 0.95, 0.975]
    assert sensitivity["sac_heading_domains"] == [
        "camera_isotropic_shared_scale",
        "homography_topview",
    ]


def test_uatp_and_pcms_primary_rules_are_frozen() -> None:
    protocol = load_protocol()
    uatp = protocol["modules"]["UATP"]
    pcms = protocol["modules"]["PCMS"]
    assert uatp["bootstrap_replicates"] == 500
    assert uatp["primary_interval_level"] == 0.90
    assert uatp["primary_interval_quantiles"] == [0.05, 0.95]
    assert uatp["sensitivity_interval_levels"] == [0.80, 0.90, 0.95]
    assert pcms["kmeans_forced_to_target"] is False
    assert pcms["emas_hg_role"] == "diagnostic_only"


def test_ablation_list_is_exact_and_immutable() -> None:
    path = PUBLICATION_ROOT / "configs" / "hg_smg_ablation_protocol.yaml"
    ablations = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = [f"A{index}" for index in range(11)]
    assert ablations["ablation_order"] == expected
    assert list(ablations["ablations"]) == expected
    assert ablations["immutable_after_freeze"] is True
    assert ablations["primary_ablation"] == "A5"


def test_preregistered_protocol_remains_immutable_and_no_test_result_exists() -> None:
    assert load_protocol()["status"] == "preregistered_not_implemented"
    assert not (PUBLICATION_ROOT / "results" / "hg_smg_tc").exists()
    assert not (
        PUBLICATION_ROOT / "results" / "hg_smg" / "independent_test"
    ).exists()
    registry = yaml.safe_load(
        (PUBLICATION_ROOT / "reproducibility" / "experiment_registry.yaml").read_text(
            encoding="utf-8"
        )
    )
    planned = [
        row for row in registry["entries"] if row["experiment_id"].startswith("hg_smg_A")
    ]
    assert len(planned) == 11
    assert all(row["reference_label_access"] is False for row in planned)
    assert all(row.get("independent_test_access", False) is False for row in planned)
    development = PUBLICATION_ROOT / "results" / "hg_smg" / "development"
    if development.exists():
        assert all("not_run" not in row["status"] for row in planned)
        assert (
            PUBLICATION_ROOT / "configs" / "hg_smg_development_freeze_v1.yaml"
        ).exists()
    else:
        assert all("not_run" in row["status"] for row in planned)


def test_frozen_inputs_remain_unchanged_and_cli_validates() -> None:
    protocol = load_protocol()
    for item in protocol["frozen_inputs"].values():
        path = REPOSITORY_ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    development_root = PUBLICATION_ROOT / "results" / "hg_smg"
    if development_root.exists():
        assert not (development_root / "independent_test").exists()
        assert (development_root / "preflight.json").exists()
    messages = cli.validate_preregistration()
    assert "independent_test_outputs=0" in messages
