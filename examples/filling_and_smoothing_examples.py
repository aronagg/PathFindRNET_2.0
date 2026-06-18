from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "sample_data" / "sample_trajectory_with_gaps.csv"
OUT = ROOT / "sample_data" / "filling_smoothing_output.csv"


def complete_frame_grid(g: pd.DataFrame, frame_step: int = 10) -> pd.DataFrame:
    g = g.sort_values("frame_id").set_index("frame_id")
    full_index = range(int(g.index.min()), int(g.index.max()) + frame_step, frame_step)
    out = g.reindex(full_index)
    out.index.name = "frame_id"
    out["track_id"] = out["track_id"].ffill().bfill().astype(int)
    out["class_name"] = out["class_name"].ffill().bfill()
    out["was_missing_detection"] = out["x_image"].isna().astype(int)
    return out.reset_index()


def fill_missing_detections(g: pd.DataFrame) -> pd.DataFrame:
    # Linear interpolation handles short missed detections.
    cols = ["timestamp", "x_image", "y_image", "bbox_w", "bbox_h", "confidence"]
    g[cols] = g[cols].interpolate(method="linear", limit_direction="both")
    return g


def mark_hidden_object_segments(g: pd.DataFrame, max_hidden_gap: int = 3) -> pd.DataFrame:
    # Consecutive missing detections are treated as a temporary occlusion if the gap is short.
    missing = g["was_missing_detection"].to_numpy()
    hidden = np.zeros_like(missing)
    start = None
    for i, value in enumerate(missing):
        if value == 1 and start is None:
            start = i
        if (value == 0 or i == len(missing) - 1) and start is not None:
            end = i if value == 0 else i + 1
            if end - start <= max_hidden_gap:
                hidden[start:end] = 1
            start = None
    g["filled_hidden_object"] = hidden
    return g


def smooth_motion(g: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    # Rolling smoothing stabilizes position and bounding rectangle dimensions.
    for col in ["x_image", "y_image", "bbox_w", "bbox_h"]:
        g[f"{col}_smooth"] = g[col].rolling(window=window, min_periods=1, center=True).mean()

    dt = g["timestamp"].diff().replace(0, np.nan)
    g["vx"] = g["x_image_smooth"].diff() / dt
    g["vy"] = g["y_image_smooth"].diff() / dt
    g["speed_px_s"] = np.hypot(g["vx"], g["vy"])
    g["speed_px_s_smooth"] = g["speed_px_s"].rolling(window=window, min_periods=1, center=True).mean()
    g["acceleration_px_s2"] = g["speed_px_s_smooth"].diff() / dt
    g["acceleration_px_s2_smooth"] = g["acceleration_px_s2"].rolling(window=window, min_periods=1, center=True).mean()
    return g


def application_based_postprocess(g: pd.DataFrame) -> pd.DataFrame:
    # Example rule: clamp unrealistic confidence values for interpolated detections.
    g.loc[g["was_missing_detection"] == 1, "confidence"] = g.loc[g["was_missing_detection"] == 1, "confidence"].clip(upper=0.75)
    return g


def process_track(g: pd.DataFrame) -> pd.DataFrame:
    g = complete_frame_grid(g)
    g = fill_missing_detections(g)
    g = mark_hidden_object_segments(g)
    g = smooth_motion(g)
    g = application_based_postprocess(g)
    return g


def main() -> None:
    df = pd.read_csv(INPUT)
    result = pd.concat([process_track(g) for _, g in df.groupby("track_id")], ignore_index=True)
    result.to_csv(OUT, index=False)
    print(result[["track_id", "frame_id", "was_missing_detection", "filled_hidden_object", "x_image", "x_image_smooth", "speed_px_s_smooth", "acceleration_px_s2_smooth", "bbox_w", "bbox_w_smooth"]])
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
