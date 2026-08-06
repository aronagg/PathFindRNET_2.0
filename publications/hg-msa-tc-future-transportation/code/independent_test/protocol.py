"""Scientific preflight and one-time unlock guards for independent testing."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline import split_aware_io as protocol_io
from reference_labels.protocol import load_frozen_protocol


TASK_NAME = "future-transportation-independent-test-evaluation-v1"
SCENES = (
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
)
METHODS = ("kmeans", "hdbscan", "optics")
STRATEGIES = ("untargeted_selection", "hg_expected_aware_selection")


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    publication_root: Path
    runner_config: Path
    frozen_protocol: Path
    frozen_manifest: Path
    selected_configurations: Path
    target_estimates: Path
    trajectory_manifest: Path
    evaluation_split: Path
    polygon_protocol: Path
    reference_manifest: Path
    reference_export: Path
    unlock: Path
    results: Path
    figures: Path
    docs: Path


def default_paths() -> Paths:
    publication_root = Path(__file__).resolve().parents[2]
    repo_root = publication_root.parents[1]
    return Paths(
        repo_root=repo_root,
        publication_root=publication_root,
        runner_config=publication_root / "configs/split_aware_runner.yaml",
        frozen_protocol=publication_root / "configs/frozen_evaluation_protocol.yaml",
        frozen_manifest=publication_root / "results/development/frozen_selection_manifest.json",
        selected_configurations=publication_root
        / "results/development/selected_configurations.csv",
        target_estimates=publication_root / "results/development/target_estimates.csv",
        trajectory_manifest=publication_root / "data/manifests/trajectory_manifest.csv",
        evaluation_split=publication_root / "data/splits/evaluation_split.csv",
        polygon_protocol=publication_root
        / "annotations/protocol/polygon_reference_protocol_v1.yaml",
        reference_manifest=publication_root
        / "annotations/reference_labels/reference_output_manifest.json",
        reference_export=publication_root
        / "annotations/reference_labels/independent_test_reference_labels.csv",
        unlock=publication_root / "configs/INDEPENDENT_TEST_UNLOCK.json",
        results=publication_root / "results/independent_test",
        figures=publication_root / "figures/independent_test",
        docs=publication_root / "docs",
    )


def _repo_path(paths: Paths, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else paths.repo_root / path


def _git_commit(paths: Paths) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={paths.repo_root.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=paths.repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _check(
    checks: list[dict[str, Any]],
    name: str,
    path: Path,
    expected: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = protocol_io.sha256_file(path)
    checks.append(
        {
            "artifact": name,
            "path": path.as_posix(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if actual == expected else "FAIL",
        }
    )
    if actual != expected:
        raise ValueError(f"Scientific hash mismatch for {name}: {actual} != {expected}")


def run_preflight(paths: Paths, write_report: bool = True) -> dict[str, Any]:
    """Validate every frozen artifact without loading independent-test features or labels."""
    required = (
        paths.runner_config,
        paths.frozen_protocol,
        paths.frozen_manifest,
        paths.selected_configurations,
        paths.target_estimates,
        paths.trajectory_manifest,
        paths.evaluation_split,
        paths.polygon_protocol,
        paths.reference_manifest,
        paths.reference_export,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preflight artifacts: {missing}")

    frozen = _load_yaml(paths.frozen_protocol)
    frozen_hash = protocol_io.validate_frozen_payload(frozen)
    manifest = json.loads(paths.frozen_manifest.read_text(encoding="utf-8"))
    if manifest.get("complete_frozen_configuration_sha256") != frozen_hash:
        raise ValueError("Frozen manifest and canonical protocol hash differ.")
    if manifest.get("independent_test_locked") is not True:
        raise ValueError("Frozen manifest does not preserve the independent-test lock.")

    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "frozen_protocol_file",
        paths.frozen_protocol,
        manifest["frozen_protocol_file_sha256"],
    )
    frozen_checks = frozen["checksums"]
    direct = {
        "runner_config": paths.runner_config,
        "trajectory_manifest": paths.trajectory_manifest,
        "evaluation_split": paths.evaluation_split,
        "split_config": _repo_path(
            paths, _load_yaml(paths.runner_config)["inputs"]["split_config"]
        ),
        "homography_index": _repo_path(
            paths, _load_yaml(paths.runner_config)["inputs"]["homography_index"]
        ),
        "target_estimates": paths.target_estimates,
        "target_candidates": paths.publication_root / "results/development/target_candidates.csv",
        "target_region_candidates": paths.publication_root
        / "results/development/target_region_candidates.csv",
        "target_provenance": paths.publication_root
        / "results/development/target_estimation_provenance.json",
        "model_selection_candidates": paths.publication_root
        / "results/development/model_selection_candidates.csv",
        "selected_configurations": paths.selected_configurations,
        "model_selection_provenance": paths.publication_root
        / "results/development/model_selection_provenance.json",
    }
    for name, path in direct.items():
        expected = frozen_checks[name]
        if isinstance(expected, dict):
            expected = expected["sha256"]
        _check(checks, name, path, str(expected))
    for scene, item in frozen_checks["homography_configurations"].items():
        _check(checks, f"homography:{scene}", _repo_path(paths, item["path"]), item["sha256"])
    for scene, item in frozen_checks["source_feature_files"].items():
        _check(checks, f"source_feature:{scene}", _repo_path(paths, item["path"]), item["sha256"])

    selected = pd.read_csv(paths.selected_configurations)
    expected_combinations = {
        (scene, method, strategy)
        for scene in SCENES
        for method in METHODS
        for strategy in STRATEGIES
    }
    actual_combinations = set(
        selected[["scene", "method", "selection_strategy"]].itertuples(index=False, name=None)
    )
    if len(selected) != 30 or actual_combinations != expected_combinations:
        raise ValueError("Frozen selected configurations are incomplete or duplicated.")
    if set(selected["split"]) != {"model_selection"}:
        raise ValueError("Frozen configurations were not selected solely on model_selection.")

    polygon = load_frozen_protocol(paths.polygon_protocol)
    polygon_file_hash = protocol_io.sha256_file(paths.polygon_protocol)
    polygon_sidecar = paths.polygon_protocol.with_suffix(".sha256")
    expected_sidecar = polygon_sidecar.read_text(encoding="ascii").split()[0]
    _check(checks, "polygon_reference_protocol", paths.polygon_protocol, expected_sidecar)

    reference_manifest = json.loads(paths.reference_manifest.read_text(encoding="utf-8"))
    reference_record = reference_manifest["output_files"][paths.reference_export.name]
    _check(
        checks,
        "independent_test_reference_export",
        paths.reference_export,
        reference_record["sha256"],
    )
    if int(reference_record["rows"]) != 27_393:
        raise ValueError("Reference manifest independent-test row count changed.")

    split_counts = (
        pd.read_csv(paths.evaluation_split, usecols=["scene_id", "split"])
        .query("split == 'independent_test'")
        .groupby("scene_id")
        .size()
        .to_dict()
    )
    expected_counts = {
        "bellevue_116th_ne12th": 964,
        "bellevue_150th_newport": 3924,
        "bellevue_150th_eastgate": 10755,
        "bellevue_150th_se38th": 3713,
        "bellevue_ne8th": 8037,
    }
    if split_counts != expected_counts:
        raise ValueError(f"Independent-test split counts changed: {split_counts}")

    result = {
        "status": "PASS",
        "task_name": TASK_NAME,
        "preflight_timestamp_utc": protocol_io.utc_timestamp(),
        "git_commit": _git_commit(paths),
        "frozen_protocol_hash": frozen_hash,
        "frozen_protocol_file_sha256": protocol_io.sha256_file(paths.frozen_protocol),
        "polygon_reference_protocol_hash": polygon["protocol_hash"],
        "polygon_reference_protocol_file_sha256": polygon_file_hash,
        "reference_export_sha256": reference_record["sha256"],
        "reference_export_rows_from_manifest": int(reference_record["rows"]),
        "test_counts_by_scene": expected_counts,
        "selected_configuration_rows": len(selected),
        "checks": checks,
        "real_test_features_loaded": False,
        "reference_label_rows_loaded": False,
        "development_configuration_changed": False,
    }
    if write_report:
        paths.docs.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(checks)[["artifact", "expected_sha256", "actual_sha256", "status"]]
        text = (
            "# Independent-Test Preflight Report\n\n"
            f"- Status: **PASS**\n"
            f"- Timestamp: `{result['preflight_timestamp_utc']}`\n"
            f"- Git commit: `{result['git_commit']}`\n"
            f"- Frozen configuration hash: `{frozen_hash}`\n"
            f"- Polygon-reference protocol hash: `{polygon['protocol_hash']}`\n"
            f"- Independent-test rows: **27,393**\n"
            f"- Frozen selected configurations: **30**\n"
            "- No independent-test feature vector or reference-label row was loaded.\n"
            "- Reference validation at this stage used file hashes and manifest metadata only.\n"
            "- No development configuration changed.\n\n"
            "## Hash Checks\n\n" + table.to_markdown(index=False) + "\n"
        )
        (paths.docs / "independent_test_preflight_report.md").write_text(text, encoding="utf-8")
        protocol_io.write_json_atomic(
            paths.docs / "independent_test_preflight_manifest.json", result
        )
    return result


def create_unlock(paths: Paths, preflight: dict[str, Any]) -> dict[str, Any]:
    if paths.unlock.exists():
        raise FileExistsError(f"One-time unlock already exists: {paths.unlock}")
    payload = {
        "unlock_version": "independent-test-unlock-v1",
        "protocol_version": "future-transportation-split-aware-v1",
        "frozen_protocol_sha256": preflight["frozen_protocol_hash"],
        "frozen_protocol_file_sha256": preflight["frozen_protocol_file_sha256"],
        "polygon_reference_protocol_hash": preflight["polygon_reference_protocol_hash"],
        "reference_export_sha256": preflight["reference_export_sha256"],
        "author_authorization_statement": (
            "Authorized by the repository owner through Codex Task 05 for the first "
            "frozen independent-test evaluation."
        ),
        "authorization_timestamp_utc": protocol_io.utc_timestamp(),
        "git_commit": _git_commit(paths),
        "intended_task_name": TASK_NAME,
        "no_further_tuning_confirmation": (
            "No target, parameter, normalization, preprocessing, EMAS weight, or "
            "selection rule will be changed after independent-test results are seen."
        ),
        "required_cli_confirmation": "--confirm-independent-test-evaluation",
        "single_use_output_directory_must_not_exist": True,
    }
    protocol_io.write_json_atomic(paths.unlock, payload)
    return payload


def verify_unlock(
    paths: Paths,
    preflight: dict[str, Any],
    explicit_confirmation: bool,
) -> dict[str, Any]:
    if not paths.unlock.exists():
        raise PermissionError(f"Independent test remains locked: {paths.unlock}")
    unlock = json.loads(paths.unlock.read_text(encoding="utf-8"))
    protocol_io.require_test_unlock(
        paths.unlock,
        preflight["frozen_protocol_hash"],
        "I_CONFIRM_FROZEN_TRANSDUCTIVE_TEST" if explicit_confirmation else None,
        "I_CONFIRM_FROZEN_TRANSDUCTIVE_TEST",
    )
    required = {
        "unlock_version": "independent-test-unlock-v1",
        "polygon_reference_protocol_hash": preflight["polygon_reference_protocol_hash"],
        "reference_export_sha256": preflight["reference_export_sha256"],
        "intended_task_name": TASK_NAME,
    }
    for key, expected in required.items():
        if unlock.get(key) != expected:
            raise PermissionError(f"Unlock field mismatch: {key}")
    if not explicit_confirmation:
        raise PermissionError("Explicit --confirm-independent-test-evaluation flag is required.")
    return unlock
