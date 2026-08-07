"""Deterministic identifiers, hashing, and leakage guards for HG-SMG-TC."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


IMPLEMENTATION_VERSION = "hg-smg-tc-development-v1"
MASTER_SEED = 20260901
SEED_MODULUS = 2_147_483_647
FORBIDDEN_PATH_MARKERS = (
    "independent_test",
    "reference_labels",
    "scene_guide",
    "polygon_reference",
    "cluster_movement_mapping",
    "per_movement_metrics",
)


def derive_seed(
    scene: str,
    module: str,
    role: str,
    region_or_replicate: str | int,
    master_seed: int = MASTER_SEED,
) -> int:
    """Apply the frozen SHA-256 seed derivation rule."""
    payload = (
        f"{int(master_seed)}|{scene}|{module}|{role}|{region_or_replicate}"
    ).encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % SEED_MODULUS


def reject_forbidden_paths(paths: Iterable[str | Path]) -> None:
    """Reject any semantic-reference or independent-test input path."""
    rejected = []
    for path in paths:
        normalized = str(path).replace("\\", "/").lower()
        if any(marker in normalized for marker in FORBIDDEN_PATH_MARKERS):
            rejected.append(str(path))
    if rejected:
        raise PermissionError(f"HG-SMG development forbids these paths: {rejected}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
