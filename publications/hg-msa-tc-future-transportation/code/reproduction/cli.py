"""Validate and inspect the frozen HG-SMG-TC preregistration.

This module is intentionally unable to execute scientific experiments. A later,
versioned implementation must consume the frozen protocol and add its own execution
gate.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml


PUBLICATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PUBLICATION_ROOT.parents[1]
PROTOCOL_PATH = PUBLICATION_ROOT / "configs" / "hg_smg_protocol_v1.yaml"
HASH_PATH = PUBLICATION_ROOT / "configs" / "hg_smg_protocol_v1.sha256"
REGISTRY_PATH = PUBLICATION_ROOT / "reproducibility" / "experiment_registry.yaml"
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
    """Validate hashes, ancestry, frozen inputs, and the no-result state."""
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
    if len(planned) != 11 or any("not_run" not in entry["status"] for entry in planned):
        raise RuntimeError("Planned ablation registry is incomplete or marked as run.")
    result_candidates = (
        PUBLICATION_ROOT / "results" / "hg_smg",
        PUBLICATION_ROOT / "results" / "hg_smg_tc",
    )
    if any(path.exists() for path in result_candidates):
        raise RuntimeError("Unexpected HG-SMG-TC scientific output exists.")
    return [
        f"protocol_sha256={actual_hash}",
        f"base_commit={EXPECTED_BASE_COMMIT}",
        "planned_ablations=11",
        "scientific_outputs=0",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-preregistration")
    subparsers.add_parser("list")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--experiment", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate-preregistration":
        print("\n".join(validate_preregistration()))
    elif args.command == "list":
        print("\n".join(list_entries()))
    else:
        print(plan_experiment(args.experiment))


if __name__ == "__main__":
    main()
