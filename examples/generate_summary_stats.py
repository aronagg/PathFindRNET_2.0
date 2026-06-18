from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
trajectory_path = ROOT / "sample_data" / "sample_trajectories.csv"
out_path = ROOT / "sample_data" / "sample_summary_stats.csv"

df = pd.read_csv(trajectory_path)

summary = (
    df.groupby(["track_id", "class_name"])
    .agg(
        first_frame=("frame_id", "min"),
        last_frame=("frame_id", "max"),
        point_count=("frame_id", "count"),
        mean_confidence=("confidence", "mean"),
        start_x=("x_image", "first"),
        start_y=("y_image", "first"),
        end_x=("x_image", "last"),
        end_y=("y_image", "last"),
    )
    .reset_index()
)

summary["duration_frames"] = summary["last_frame"] - summary["first_frame"]
summary.to_csv(out_path, index=False)
print(summary)
print(f"Saved: {out_path}")
