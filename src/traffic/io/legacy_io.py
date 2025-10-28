from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

# This module loads legacy JSON files which contain a list of "track" entities.
# Each entity has the shape shown below (fields not strictly required except detections):
# {
#   "id": 3,
#   "history_X": [...],
#   "history_Y": [...],
#   "history_VX_calculated": [...],
#   "history_VY_calculated": [...],
#   "history_AX_calculated": [...],
#   "history_AY_calculated": [...],
#   "detections": [
#       {"label": "car", "confidence": 0.62, "X": 0.4875, "Y": 0.2597,
#        "Width": 0.0861, "Height": 0.1389, "frameID": 9.0, "VX": 0.0, ...},
#       ...
#   ]
# }
#
# We normalize this into a DataFrame with columns:
#   frame:int, track_id:int, cls:int, conf:float, cx:float, cy:float, w:float, h:float
#
# Notes:
# - label is mapped to integer class via an optional class_map; if not provided, cls=-1.
# - X/Y are considered center coordinates; Width/Height are box sizes, all typically normalized.
# - If an entity lacks "detections" but has history_X/Y, we fall back to those with frames 0..N-1.


def _label_to_cls(label: Any, class_map: Mapping[str, int] | None) -> int:
    # If numeric-like, coerce to int
    if isinstance(label, (int,)):
        return int(label)
    if isinstance(label, str):
        # direct numeric string
        try:
            return int(label)
        except ValueError:
            pass
        if class_map and label in class_map:
            return int(class_map[label])
    return -1


def _entity_to_rows(
    entity: Mapping[str, Any], class_map: Mapping[str, int] | None
) -> list[dict[str, Any]]:
    tid = int(entity.get("id", -1))
    rows: list[dict[str, Any]] = []

    dets = entity.get("detections")
    if isinstance(dets, list) and dets:
        for d in dets:
            if not isinstance(d, Mapping):
                continue
            label = d.get("label")
            cls = _label_to_cls(label, class_map)
            conf = float(d.get("confidence", 0.0) or 0.0)
            cx = float(d.get("X", 0.0) or 0.0)
            cy = float(d.get("Y", 0.0) or 0.0)
            w = float(d.get("Width", 0.0) or 0.0)
            h = float(d.get("Height", 0.0) or 0.0)
            frame = d.get("frameID", d.get("frame", 0))
            try:
                frame = int(round(float(frame)))
            except Exception:
                frame = 0
            rows.append(
                {
                    "frame": frame,
                    "track_id": tid,
                    "cls": cls,
                    "conf": conf,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                }
            )
        return rows

    # Fallback: build from histories if available
    hx = entity.get("history_X") or []
    hy = entity.get("history_Y") or []
    n = min(len(hx), len(hy))
    for i in range(n):
        rows.append(
            {
                "frame": i,
                "track_id": tid,
                "cls": -1,
                "conf": 0.0,
                "cx": float(hx[i]),
                "cy": float(hy[i]),
                "w": 0.0,
                "h": 0.0,
            }
        )
    return rows


def load_legacy_json(
    path: str | Path,
    *,
    class_map: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Load a legacy JSON file with a list of entities and return normalized detections.

    Parameters
    ----------
    path : str | Path
        Path to JSON file. The file should contain a JSON array of entities.
        It may also be NDJSON/JSONL with one entity per line.
    class_map : Mapping[str, int] | None
        Optional mapping from string labels (e.g., "car") to integer class ids.

    Returns
    -------
    pd.DataFrame
        Columns: frame, track_id, cls, conf, cx, cy, w, h
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    entities: list[Mapping[str, Any]]
    # Try to parse as a single JSON payload first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            entities = [e for e in obj if isinstance(e, Mapping)]
        elif isinstance(obj, Mapping):
            # Some files may wrap payload under a key like {"tracks": [...]}
            found = None
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], Mapping):
                    found = v
                    break
            entities = [e for e in (found or []) if isinstance(e, Mapping)]
        else:
            entities = []
    except Exception:
        # Fallback: NDJSON/JSONL
        entities = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if isinstance(e, Mapping):
                    entities.append(e)
            except Exception:
                continue

    # Normalize
    rows: list[dict[str, Any]] = []
    for ent in entities:
        rows.extend(_entity_to_rows(ent, class_map))

    df = pd.DataFrame(rows, columns=["frame", "track_id", "cls", "conf", "cx", "cy", "w", "h"])
    if len(df) == 0:
        return df
    # Enforce dtypes for downstream compatibility
    return df.astype(
        {
            "frame": int,
            "track_id": int,
            "cls": int,
            "conf": float,
            "cx": float,
            "cy": float,
            "w": float,
            "h": float,
        }
    )
