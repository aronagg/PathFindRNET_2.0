"""Aspect-ratio-preserving trajectory and video rendering utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle


RENDERING_VERSION = "camera-polyline-renderer-v1"


def compute_viewport(
    image_width: int,
    image_height: int,
    x: np.ndarray,
    y: np.ndarray,
    margin_fraction: float = 0.08,
    zoom: float = 1.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    zoom = max(float(zoom), 1.0)
    center_x = float((np.nanmin(x) + np.nanmax(x)) / 2.0)
    center_y = float((np.nanmin(y) + np.nanmax(y)) / 2.0)
    path_width = max(float(np.ptp(x)), image_width / zoom * 0.2)
    path_height = max(float(np.ptp(y)), image_height / zoom * 0.2)
    margin = max(float(margin_fraction), 0.0)
    needed_width = path_width * (1.0 + 2.0 * margin)
    needed_height = path_height * (1.0 + 2.0 * margin)
    aspect = image_width / image_height
    if needed_width / needed_height < aspect:
        needed_width = needed_height * aspect
    else:
        needed_height = needed_width / aspect
    max_width = image_width / zoom if zoom > 1 else image_width
    max_height = image_height / zoom if zoom > 1 else image_height
    width = min(max(needed_width, image_width * 0.1), max_width)
    height = width / aspect
    if height > max_height:
        height = max_height
        width = height * aspect
    center_x = min(max(center_x, width / 2), image_width - width / 2)
    center_y = min(max(center_y, height / 2), image_height - height / 2)
    return (
        (center_x - width / 2, center_x + width / 2),
        (center_y + height / 2, center_y - height / 2),
    )


def render_trajectory(
    polyline: pd.DataFrame,
    background_rgb: np.ndarray | None,
    output_path: Path | None = None,
    line_width: float = 2.5,
    margin_fraction: float = 0.08,
    zoom: float = 1.0,
    show_background: bool = True,
    title: str | None = None,
) -> plt.Figure:
    x = polyline["cx"].to_numpy(dtype=float)
    y = polyline["cy"].to_numpy(dtype=float)
    if background_rgb is not None:
        height, width = background_rgb.shape[:2]
    else:
        width = int(max(np.nanmax(x) + 1, 1920))
        height = int(max(np.nanmax(y) + 1, 1080))
    x_limits, y_limits = compute_viewport(width, height, x, y, margin_fraction, zoom)
    figure_width = 12.0
    figure_height = figure_width * height / width
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=120)
    if background_rgb is not None and show_background:
        ax.imshow(background_rgb, extent=(0, width, height, 0))
    else:
        ax.set_facecolor("#f4f4f4")
    ax.plot(x, y, color="#0b57d0", linewidth=line_width, alpha=0.92)
    ax.scatter([x[0]], [y[0]], color="#16833b", s=75, zorder=4)
    ax.scatter([x[-1]], [y[-1]], color="#c5221f", marker="X", s=85, zorder=4)
    ax.annotate(
        "START",
        (x[0], y[0]),
        xytext=(6, 6),
        textcoords="offset points",
        color="#0b5428",
        weight="bold",
    )
    ax.annotate(
        "END",
        (x[-1], y[-1]),
        xytext=(6, 6),
        textcoords="offset points",
        color="#8f1512",
        weight="bold",
    )
    arrow_indices = np.unique(
        np.linspace(0, max(len(x) - 2, 0), min(4, max(len(x) - 1, 1))).astype(int)
    )
    for index in arrow_indices:
        next_index = min(index + max(1, len(x) // 15), len(x) - 1)
        ax.annotate(
            "",
            xy=(x[next_index], y[next_index]),
            xytext=(x[index], y[index]),
            arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.5},
        )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout(pad=0.2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    return fig


def extract_clip(
    video_path: Path,
    frame_start: int,
    frame_end: int,
    output_path: Path,
    padding_frames: int = 15,
) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open source video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = max(0, int(frame_start) - padding_frames)
    last = max(first, int(frame_end) + padding_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for _ in range(first, last + 1):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise IOError(f"Clip extraction failed: {output_path}")
    return output_path


def render_scene_guide(
    background_rgb: np.ndarray,
    guide: dict,
    output_path: Path,
) -> Path:
    """Render only manually entered scene-guide approach markers."""
    height, width = background_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 12 * height / width), dpi=120)
    ax.imshow(background_rgb)
    for approach in guide.get("approaches", []):
        start = approach["label_position_normalized"]
        end = approach["arrow_end_normalized"]
        start_xy = (float(start["x"]) * width, float(start["y"]) * height)
        end_xy = (float(end["x"]) * width, float(end["y"]) * height)
        region = approach.get("region_normalized")
        role = approach.get("region_role", "both")
        color = {"entry": "#00a66a", "exit": "#d1495b", "both": "#f2b134"}.get(role, "#f2b134")
        if region:
            region_x = float(region["x_min"]) * width
            region_y = float(region["y_min"]) * height
            region_width = (float(region["x_max"]) - float(region["x_min"])) * width
            region_height = (float(region["y_max"]) - float(region["y_min"])) * height
            ax.add_patch(
                Rectangle(
                    (region_x, region_y),
                    region_width,
                    region_height,
                    linewidth=3,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.2,
                )
            )
        if np.hypot(start_xy[0] - end_xy[0], start_xy[1] - end_xy[1]) < 2:
            ax.scatter(*end_xy, s=130, color=color, edgecolor="white", linewidth=1.5)
            arrow_properties = None
        else:
            arrow_properties = {"arrowstyle": "->", "color": color, "lw": 2.5}
        ax.annotate(
            str(approach["id"]),
            xy=end_xy,
            xytext=start_xy,
            color="white",
            fontsize=15,
            weight="bold",
            bbox={"boxstyle": "square,pad=0.25", "facecolor": "#111111", "alpha": 0.85},
            arrowprops=arrow_properties,
        )
    if guide.get("status") != "ready_for_freeze":
        ax.text(
            0.01,
            0.02,
            "DRAFT - APPROACHES REQUIRE MANUAL CONFIGURATION",
            transform=ax.transAxes,
            color="white",
            fontsize=11,
            weight="bold",
            bbox={"facecolor": "#9c1c1c", "alpha": 0.9, "pad": 5},
        )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def render_scene_guide_from_yaml(guide_path: Path, annotations_root: Path) -> Path:
    guide = yaml.safe_load(guide_path.read_text(encoding="utf-8"))
    frame_path = annotations_root / guide["representative_frame"]
    background = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
    if background is None:
        raise ValueError(f"Cannot read representative frame: {frame_path}")
    return render_scene_guide(
        background, guide, guide_path.with_name(f"{guide['scene_id']}_guide.png")
    )
