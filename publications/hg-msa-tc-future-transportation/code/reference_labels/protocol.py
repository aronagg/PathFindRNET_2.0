"""Validation and immutable freezing of manual polygon scene guides."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from shapely.geometry import Polygon


PROTOCOL_VERSION = "future-transportation-polygon-reference-v1"
SCENES = (
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
)
MANEUVER_TYPES = {"left", "right", "straight", "u_turn", "other", "unknown"}
FORBIDDEN_GUIDE_KEY_FRAGMENTS = (
    "cluster",
    "emas",
    "hg_target",
    "homography_target",
    "pseudo",
    "automatic_od",
    "model_selection",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _atomic_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding=encoding, newline="\n", delete=False, dir=path.parent
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_yaml(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False, width=100),
    )


def current_git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key).lower())
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _normalized_polygon(approach: dict[str, Any], scene: str) -> list[dict[str, float]]:
    polygon = approach.get("polygon_normalized")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError(f"{scene}/{approach.get('id')}: polygon needs at least 3 vertices")
    points: list[dict[str, float]] = []
    for point in polygon:
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{scene}/{approach.get('id')}: invalid polygon vertex") from exc
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"{scene}/{approach.get('id')}: polygon vertex outside image")
        points.append({"x": x, "y": y})
    shape = Polygon([(point["x"], point["y"]) for point in points])
    if not shape.is_valid or shape.area <= 0:
        raise ValueError(f"{scene}/{approach.get('id')}: polygon is not a valid positive area")
    return points


def validate_scene_guide(guide: dict[str, Any], expected_scene: str) -> dict[str, Any]:
    """Return a normalized validated guide without changing the source file."""
    if guide.get("scene_id") != expected_scene:
        raise ValueError(f"Scene mismatch for {expected_scene}")
    if guide.get("status") != "ready_for_freeze":
        raise ValueError(f"Scene is not ready for freeze: {expected_scene}")
    forbidden = sorted(
        {
            key
            for key in _walk_keys(guide)
            if any(fragment in key for fragment in FORBIDDEN_GUIDE_KEY_FRAGMENTS)
        }
    )
    if forbidden:
        raise ValueError(
            f"Forbidden cluster/HG-derived guide keys in {expected_scene}: {forbidden}"
        )

    approaches = guide.get("approaches")
    if not isinstance(approaches, list):
        raise ValueError(f"Approaches must be a list: {expected_scene}")
    normalized_approaches: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for approach in approaches:
        identifier = str(approach.get("id", "")).strip()
        role = str(approach.get("region_role", "")).strip()
        if not identifier:
            raise ValueError(f"Empty polygon ID: {expected_scene}")
        if role not in {"entry", "exit"}:
            raise ValueError(
                f"Each polygon needs exactly one entry/exit role: {expected_scene}/{identifier}"
            )
        identifiers.append(identifier)
        normalized_approaches.append(
            {
                "id": identifier,
                "human_readable_name": str(approach.get("human_readable_name", "")),
                "region_role": role,
                "polygon_normalized": _normalized_polygon(approach, expected_scene),
            }
        )
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate polygon IDs: {expected_scene}")

    entries = [str(value) for value in guide.get("valid_entry_approaches", [])]
    exits = [str(value) for value in guide.get("valid_exit_approaches", [])]
    role_entries = [row["id"] for row in normalized_approaches if row["region_role"] == "entry"]
    role_exits = [row["id"] for row in normalized_approaches if row["region_role"] == "exit"]
    if len(entries) != 4 or len(set(entries)) != 4 or set(entries) != set(role_entries):
        raise ValueError(f"Exactly four valid entry polygons are required: {expected_scene}")
    if len(exits) != 4 or len(set(exits)) != 4 or set(exits) != set(role_exits):
        raise ValueError(f"Exactly four valid exit polygons are required: {expected_scene}")

    mappings = guide.get("maneuver_type_mapping")
    if not isinstance(mappings, list) or len(mappings) != 12:
        raise ValueError(f"Exactly 12 legal mappings are required: {expected_scene}")
    normalized_mappings: list[dict[str, str]] = []
    pairs: list[tuple[str, str]] = []
    for mapping in mappings:
        entry = str(mapping.get("entry", ""))
        exit_id = str(mapping.get("exit", ""))
        maneuver = str(mapping.get("maneuver_type", ""))
        if entry not in entries or exit_id not in exits:
            raise ValueError(f"Invalid mapping reference in {expected_scene}: {entry}>{exit_id}")
        if maneuver not in MANEUVER_TYPES:
            raise ValueError(
                f"Missing or invalid maneuver type in {expected_scene}: {entry}>{exit_id}"
            )
        pairs.append((entry, exit_id))
        normalized_mappings.append({"entry": entry, "exit": exit_id, "maneuver_type": maneuver})
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"Duplicate legal movement mappings: {expected_scene}")

    notes = str(guide.get("ambiguity_notes", "")).strip()
    if not notes or "MANUAL INPUT REQUIRED" in notes:
        raise ValueError(f"Scene-specific ambiguity notes are incomplete: {expected_scene}")
    return {
        "scene_id": expected_scene,
        "representative_frame": str(guide["representative_frame"]),
        "representative_recording_id": str(guide["representative_recording_id"]),
        "representative_frame_number": int(guide["representative_frame_number"]),
        "approaches": normalized_approaches,
        "valid_entry_approaches": entries,
        "valid_exit_approaches": exits,
        "maneuver_type_mapping": normalized_mappings,
        "ambiguity_notes": notes,
    }


def freeze_polygon_protocol(
    repo_root: Path,
    publication_root: Path,
    output_path: Path,
) -> tuple[dict[str, Any], str]:
    """Validate and freeze the five guides. Existing v1 artifacts are immutable."""
    sidecar = output_path.with_suffix(".sha256")
    report_path = output_path.with_name("polygon_reference_protocol_freeze_report.md")
    for path in (output_path, sidecar, report_path):
        if path.exists():
            raise FileExistsError(f"Frozen polygon protocol cannot be overwritten: {path}")

    guide_dir = publication_root / "annotations/protocol/scene_guides"
    guides: list[dict[str, Any]] = []
    guide_checksums: dict[str, dict[str, str]] = {}
    for scene in SCENES:
        guide_path = guide_dir / f"{scene}.yaml"
        image_path = guide_dir / f"{scene}_guide.png"
        if not guide_path.exists() or not image_path.exists():
            raise FileNotFoundError(f"Scene guide artifact missing: {scene}")
        guides.append(validate_scene_guide(load_yaml(guide_path), scene))
        guide_checksums[scene] = {
            "yaml_path": guide_path.relative_to(repo_root).as_posix(),
            "yaml_sha256": sha256_file(guide_path),
            "rendered_guide_path": image_path.relative_to(repo_root).as_posix(),
            "rendered_guide_sha256": sha256_file(image_path),
        }

    created_at = utc_timestamp()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": created_at,
        "label_generation_timestamp_utc": created_at,
        "git_commit": current_git_commit(repo_root),
        "scene_order": list(SCENES),
        "coordinate_space": "normalized_camera_image_coordinates",
        "endpoint_rule": "first_and_last_finite_points_inside_canonical_manifest_interval",
        "containment_rule": "shapely_polygon_covers_without_nearest_polygon_fallback",
        "boundary_tolerance_px": 1e-7,
        "sensitivity_polygon_buffer_px": 3.0,
        "scene_guide_checksums": guide_checksums,
        "scene_guides": guides,
        "scientific_guards": {
            "manual_cherry_picked_labels_used": False,
            "cluster_outputs_used": False,
            "hg_targets_used": False,
            "emas_scores_used": False,
            "automatic_od_assignments_used": False,
            "independent_test_clustering_executed": False,
        },
    }
    payload["protocol_hash"] = sha256_payload(payload)
    write_yaml(output_path, payload)
    file_hash = sha256_file(output_path)
    _atomic_text(sidecar, f"{file_hash}  {output_path.name}\n", encoding="ascii")
    report = (
        "# Polygon Reference Protocol Freeze Report\n\n"
        f"- Protocol: `{PROTOCOL_VERSION}`\n"
        f"- Created: `{created_at}`\n"
        f"- Git commit containing the manual guides: `{payload['git_commit']}`\n"
        f"- Canonical protocol hash: `{payload['protocol_hash']}`\n"
        f"- Frozen YAML file SHA-256: `{file_hash}`\n"
        "- Five fixed Bellevue scenes validated.\n"
        "- Each scene contains four entry polygons, four exit polygons, and twelve unique legal mappings.\n"
        "- No cluster, HG-target, EMAS, pseudo-label, or automatic-OD field was accepted.\n"
        "- The protocol is immutable; a later change requires a new version.\n"
    )
    _atomic_text(report_path, report)
    return payload, file_hash


def load_frozen_protocol(path: Path) -> dict[str, Any]:
    payload = load_yaml(path)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Unexpected polygon protocol version: {path}")
    expected = payload.get("protocol_hash")
    unhashed = dict(payload)
    unhashed.pop("protocol_hash", None)
    if expected != sha256_payload(unhashed):
        raise ValueError(f"Frozen protocol canonical hash mismatch: {path}")
    return payload
