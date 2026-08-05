"""Dependency-free Streamlit component for manual point and region placement."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "manual_approach_picker",
    path=str(Path(__file__).resolve().parent / "components" / "approach_picker"),
)


def image_data_url(path: Path) -> str:
    """Encode a local guide frame for the isolated browser component."""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def approach_picker(
    image_path: Path,
    mode: str,
    approach_id: str,
    region_role: str,
    existing_shapes: list[dict[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    """Return a manually clicked point or dragged rectangle in normalized coordinates."""
    return _COMPONENT(
        image_data_url=image_data_url(image_path),
        mode=mode,
        approach_id=approach_id,
        region_role=region_role,
        existing_shapes=existing_shapes,
        key=key,
        default=None,
    )
