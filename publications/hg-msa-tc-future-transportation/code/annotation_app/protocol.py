"""Manual scene-guide protocol editing, validation, and immutable freezing."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from .models import MANEUVER_TYPES, SCENES
    from .source_data import sha256_file
except ImportError:  # Direct script execution.
    from models import MANEUVER_TYPES, SCENES
    from source_data import sha256_file


ANNOTATION_PROTOCOL_VERSION = "future-transportation-manual-annotation-v1"
APPROACH_COORDINATE_FIELDS = ("label_x", "label_y", "arrow_x", "arrow_y")
REGION_COORDINATE_FIELDS = ("region_x_min", "region_y_min", "region_x_max", "region_y_max")
REGION_ROLES = {"entry", "exit", "both"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_yaml(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False, width=100),
    )


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def normalize_approach_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate editable table rows and discard only completely empty new rows."""
    normalized = []
    approach_ids = set()
    for row_number, item in enumerate(records, start=1):
        relevant = (
            item.get("id"),
            item.get("human_readable_name"),
            *(item.get(field) for field in APPROACH_COORDINATE_FIELDS),
            *(item.get(field) for field in REGION_COORDINATE_FIELDS),
        )
        if all(_blank(value) for value in relevant):
            continue
        approach_id = "" if _blank(item.get("id")) else str(item["id"]).strip()
        if not approach_id:
            raise ValueError(f"Approach row {row_number}: id is required.")
        if approach_id in approach_ids:
            raise ValueError(f"Approach row {row_number}: duplicate id '{approach_id}'.")
        coordinates = {}
        for field in APPROACH_COORDINATE_FIELDS:
            value = item.get(field)
            if _blank(value):
                raise ValueError(f"Approach row {row_number}: {field} is required.")
            try:
                coordinate = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Approach row {row_number}: {field} must be numeric.") from exc
            if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
                raise ValueError(f"Approach row {row_number}: {field} must be between 0 and 1.")
            coordinates[field] = coordinate
        name = item.get("human_readable_name")
        normalized_row = {
            "id": approach_id,
            "human_readable_name": "" if _blank(name) else str(name).strip(),
            "label_position_normalized": {
                "x": coordinates["label_x"],
                "y": coordinates["label_y"],
            },
            "arrow_end_normalized": {
                "x": coordinates["arrow_x"],
                "y": coordinates["arrow_y"],
            },
        }
        region_values = [item.get(field) for field in REGION_COORDINATE_FIELDS]
        if any(not _blank(value) for value in region_values):
            if any(_blank(value) for value in region_values):
                raise ValueError(
                    f"Approach row {row_number}: all four region coordinates are required."
                )
            region = {}
            for field, value in zip(REGION_COORDINATE_FIELDS, region_values, strict=True):
                try:
                    coordinate = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Approach row {row_number}: {field} must be numeric."
                    ) from exc
                if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
                    raise ValueError(f"Approach row {row_number}: {field} must be between 0 and 1.")
                region[field.removeprefix("region_")] = coordinate
            if region["x_min"] >= region["x_max"] or region["y_min"] >= region["y_max"]:
                raise ValueError(f"Approach row {row_number}: region must have positive area.")
            role = "both" if _blank(item.get("region_role")) else str(item["region_role"]).strip()
            if role not in REGION_ROLES:
                raise ValueError(
                    f"Approach row {row_number}: region_role must be entry, exit, or both."
                )
            normalized_row["region_normalized"] = region
            normalized_row["region_role"] = role
        normalized.append(normalized_row)
        approach_ids.add(approach_id)
    return normalized


def verify_scientific_protocol_frozen(frozen_manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "complete_frozen_configuration_sha256",
        "independent_test_locked",
    }
    if not required.issubset(payload):
        raise ValueError("Scientific freeze manifest is incomplete.")
    if payload["independent_test_locked"] is not True:
        raise ValueError("Scientific protocol must remain frozen and test-locked.")
    return {
        "protocol_version": payload["protocol_version"],
        "complete_frozen_configuration_sha256": payload["complete_frozen_configuration_sha256"],
        "independent_test_locked": True,
    }


def validate_scene_guide_definition(guide: dict[str, Any], scene: str) -> None:
    if guide.get("scene_id") != scene:
        raise ValueError(f"Guide scene mismatch: {scene}")
    if guide.get("status") != "ready_for_freeze":
        raise ValueError(f"Scene guide is not ready for freeze: {scene}")
    approaches = guide.get("approaches", [])
    if len(approaches) < 2:
        raise ValueError(f"At least two manually defined approaches required: {scene}")
    approach_ids = [str(row.get("id", "")).strip() for row in approaches]
    if any(not value for value in approach_ids) or len(set(approach_ids)) != len(approach_ids):
        raise ValueError(f"Approach IDs must be non-empty and unique: {scene}")
    for approach in approaches:
        position = approach.get("label_position_normalized", {})
        arrow = approach.get("arrow_end_normalized", {})
        for point in (position, arrow):
            if not (0 <= float(point.get("x", -1)) <= 1 and 0 <= float(point.get("y", -1)) <= 1):
                raise ValueError(f"Approach label/arrow position outside image: {scene}")
        region = approach.get("region_normalized")
        if region is not None:
            try:
                x_min = float(region["x_min"])
                y_min = float(region["y_min"])
                x_max = float(region["x_max"])
                y_max = float(region["y_max"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid approach region: {scene}") from exc
            if not (
                0 <= x_min < x_max <= 1
                and 0 <= y_min < y_max <= 1
                and approach.get("region_role", "both") in REGION_ROLES
            ):
                raise ValueError(f"Invalid approach region: {scene}")
    entries = set(guide.get("valid_entry_approaches", []))
    exits = set(guide.get("valid_exit_approaches", []))
    if (
        not entries
        or not exits
        or not entries.issubset(approach_ids)
        or not exits.issubset(approach_ids)
    ):
        raise ValueError(f"Invalid entry/exit approach set: {scene}")
    mappings = guide.get("maneuver_type_mapping", [])
    for mapping in mappings:
        if mapping.get("entry") not in entries or mapping.get("exit") not in exits:
            raise ValueError(f"Maneuver mapping uses an unknown approach: {scene}")
        if mapping.get("maneuver_type") not in MANEUVER_TYPES:
            raise ValueError(f"Invalid coarse maneuver type in guide: {scene}")


def validate_scene_guide(guide: dict[str, Any], scene: str, guide_path: Path) -> None:
    validate_scene_guide_definition(guide, scene)
    frame_path = guide_path.parent.parent.parent / str(guide["representative_frame"])
    image_path = guide_path.with_name(f"{scene}_guide.png")
    if not frame_path.exists() or not image_path.exists():
        raise FileNotFoundError(f"Scene guide imagery missing for {scene}")


def protocol_is_frozen(protocol_dir: Path) -> bool:
    protocol_path = protocol_dir / "annotation_protocol_v1.yaml"
    checksum_path = protocol_dir / "annotation_protocol_v1.sha256"
    if not protocol_path.exists() or not checksum_path.exists():
        return False
    recorded = checksum_path.read_text(encoding="ascii").strip().split()[0]
    return sha256_file(protocol_path) == recorded


def require_frozen_protocol(protocol_dir: Path) -> dict[str, Any]:
    if not protocol_is_frozen(protocol_dir):
        raise PermissionError("Primary annotation requires a valid frozen protocol.")
    return load_yaml(protocol_dir / "annotation_protocol_v1.yaml")


def freeze_protocol(
    protocol_dir: Path,
    frozen_scientific_manifest: Path,
    guideline_path: Path,
) -> tuple[Path, str]:
    frozen_path = protocol_dir / "annotation_protocol_v1.yaml"
    checksum_path = protocol_dir / "annotation_protocol_v1.sha256"
    report_path = protocol_dir / "protocol_freeze_report.md"
    if frozen_path.exists() or checksum_path.exists():
        raise FileExistsError("Frozen annotation protocol cannot be overwritten.")
    draft = load_yaml(protocol_dir / "annotation_protocol_draft.yaml")
    if tuple(draft.get("scenes", [])) != SCENES:
        raise ValueError("Draft protocol does not contain exactly five fixed scenes.")
    scientific_reference = verify_scientific_protocol_frozen(frozen_scientific_manifest)
    guide_checksums = {}
    guides = []
    for scene in SCENES:
        guide_path = protocol_dir / "scene_guides" / f"{scene}.yaml"
        guide = load_yaml(guide_path)
        validate_scene_guide(guide, scene, guide_path)
        image_path = guide_path.with_name(f"{scene}_guide.png")
        guide_checksums[scene] = {
            "yaml_sha256": sha256_file(guide_path),
            "image_sha256": sha256_file(image_path),
        }
        guides.append(guide)
    payload = {
        "protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "frozen_at_utc": utc_timestamp(),
        "scientific_protocol_reference": scientific_reference,
        "blinding_policy": draft["blinding_policy"],
        "allowed_values": draft["allowed_values"],
        "manual_maneuver_id_rule": "<scene_id>:<entry_approach>><exit_approach>",
        "rare_movement_policy": "derived_after_consensus_only",
        "guideline_path": guideline_path.as_posix(),
        "guideline_sha256": sha256_file(guideline_path),
        "scene_guide_checksums": guide_checksums,
        "scene_guides": guides,
    }
    write_yaml(frozen_path, payload)
    checksum = sha256_file(frozen_path)
    _atomic_text(checksum_path, f"{checksum}  annotation_protocol_v1.yaml\n")
    _atomic_text(
        report_path,
        "# Annotation Protocol Freeze Report\n\n"
        f"- Protocol: `{ANNOTATION_PROTOCOL_VERSION}`\n"
        f"- Frozen at: `{payload['frozen_at_utc']}`\n"
        f"- SHA-256: `{checksum}`\n"
        "- All five scene guides passed manual-configuration validation.\n"
        "- The scientific model-selection protocol was already frozen and remains locked.\n"
        "- HG targets, automatic regions, clusters, and pseudo-labels are not annotation inputs.\n",
    )
    return frozen_path, checksum
