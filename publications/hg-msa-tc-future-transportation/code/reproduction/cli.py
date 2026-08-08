"""Validate the HG-SMG protocol and run explicitly development-only stages."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PUBLICATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PUBLICATION_ROOT.parents[1]
PROTOCOL_PATH = PUBLICATION_ROOT / "configs" / "hg_smg_protocol_v1.yaml"
HASH_PATH = PUBLICATION_ROOT / "configs" / "hg_smg_protocol_v1.sha256"
REGISTRY_PATH = PUBLICATION_ROOT / "reproducibility" / "experiment_registry.yaml"
DEVELOPMENT_FREEZE_PATH = PUBLICATION_ROOT / "configs/hg_smg_development_freeze_v1.yaml"
DEVELOPMENT_FREEZE_HASH_PATH = PUBLICATION_ROOT / "configs/hg_smg_development_freeze_v1.sha256"
EXPECTED_BASE_COMMIT = "8bfb4826725ea5c4c04042c037b521edcf216ec4"


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={REPOSITORY_ROOT.as_posix()}", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def validate_preregistration() -> list[str]:
    """Validate immutable preregistration and either planned or frozen development state."""
    protocol = _load_yaml(PROTOCOL_PATH)
    recorded_hash = HASH_PATH.read_text(encoding="ascii").strip().split()[0]
    actual_hash = _sha256(PROTOCOL_PATH)
    if recorded_hash != actual_hash:
        raise RuntimeError("HG-SMG-TC protocol hash mismatch.")
    if protocol["base_commit"] != EXPECTED_BASE_COMMIT:
        raise RuntimeError("Unexpected preregistration base commit.")
    merge_base = _git_output("merge-base", "HEAD", EXPECTED_BASE_COMMIT)
    if merge_base != EXPECTED_BASE_COMMIT:
        raise RuntimeError("Current branch is not based on the frozen Task-08 commit.")
    if protocol["status"] != "preregistered_not_implemented":
        raise RuntimeError("Protocol no longer records the required no-run state.")

    for item in protocol["frozen_inputs"].values():
        path = REPOSITORY_ROOT / item["path"]
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen input changed: {item['path']}")

    registry = _load_yaml(REGISTRY_PATH)
    planned = [
        entry
        for entry in registry["entries"]
        if str(entry["experiment_id"]).startswith("hg_smg_A")
    ]
    if len(planned) != 11:
        raise RuntimeError("HG-SMG ablation registry is incomplete.")
    if any(entry["reference_label_access"] is not False for entry in planned):
        raise RuntimeError("A development ablation records reference-label access.")
    if any(entry.get("independent_test_access", False) is not False for entry in planned):
        raise RuntimeError("A development ablation records independent-test access.")

    development_root = PUBLICATION_ROOT / "results/hg_smg/development"
    test_result_roots = (
        PUBLICATION_ROOT / "results/hg_smg/independent_test",
        PUBLICATION_ROOT / "results/hg_smg_tc",
    )
    if any(path.exists() for path in test_result_roots):
        raise RuntimeError("Unexpected HG-SMG-TC independent-test output exists.")
    if development_root.exists():
        if not DEVELOPMENT_FREEZE_PATH.exists() or not DEVELOPMENT_FREEZE_HASH_PATH.exists():
            raise RuntimeError("Development outputs exist without a complete freeze.")
        recorded_freeze_hash = (
            DEVELOPMENT_FREEZE_HASH_PATH.read_text(encoding="ascii").strip().split()[0]
        )
        if _sha256(DEVELOPMENT_FREEZE_PATH) != recorded_freeze_hash:
            raise RuntimeError("HG-SMG development-freeze hash mismatch.")
        if any("not_run" in str(entry["status"]) for entry in planned):
            raise RuntimeError("Development outputs exist but registry still records not-run state.")
        development_state = "frozen"
        output_count = len(list(development_root.glob("*")))
    else:
        if any("not_run" not in str(entry["status"]) for entry in planned):
            raise RuntimeError("Ablation registry is marked as run without development outputs.")
        development_state = "planned"
        output_count = 0
    return [
        f"protocol_sha256={actual_hash}",
        f"base_commit={EXPECTED_BASE_COMMIT}",
        "planned_ablations=11",
        f"development_state={development_state}",
        f"development_output_files={output_count}",
        "independent_test_outputs=0",
    ]


def list_entries() -> list[str]:
    """Return concise registry rows without executing their commands."""
    registry = _load_yaml(REGISTRY_PATH)
    return [
        f"{entry['experiment_id']}: {entry['status']}"
        for entry in registry["entries"]
    ]


def plan_experiment(experiment_id: str) -> str:
    """Return the frozen future command; never execute it."""
    registry = _load_yaml(REGISTRY_PATH)
    for entry in registry["entries"]:
        if entry["experiment_id"] == experiment_id:
            return f"{entry['status']}\n{entry['command']}"
    raise KeyError(f"Unknown experiment: {experiment_id}")


def run_hg_smg_development(stage: str, confirmation: str | None) -> None:
    """Run one explicitly confirmed development-only HG-SMG stage."""
    if confirmation != "I_CONFIRM_DEVELOPMENT_SPLITS_ONLY":
        raise PermissionError("Exact development-only confirmation phrase is required.")
    allowed = {"preflight", "full-split", "uatp", "pcms"}
    if stage not in allowed:
        raise ValueError(f"Unsupported HG-SMG development stage: {stage}")
    subprocess.run(
        [sys.executable, "-m", "hg_smg.cli", stage],
        cwd=PUBLICATION_ROOT,
        env={**os.environ, "PYTHONPATH": str(PUBLICATION_ROOT / "code")},
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-preregistration")
    subparsers.add_parser("list")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--experiment", required=True)
    development = subparsers.add_parser("hg-smg-development")
    development.add_argument(
        "--stage", choices=["preflight", "full-split", "uatp", "pcms"], required=True
    )
    development.add_argument("--confirm-development-only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate-preregistration":
        print("\n".join(validate_preregistration()))
    elif args.command == "list":
        print("\n".join(list_entries()))
    elif args.command == "plan":
        print(plan_experiment(args.experiment))
    else:
        run_hg_smg_development(args.stage, args.confirm_development_only)


if __name__ == "__main__":
    main()
