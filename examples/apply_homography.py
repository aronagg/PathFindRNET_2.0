from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
trajectory_path = ROOT / "sample_data" / "sample_trajectories.csv"
homography_path = ROOT / "sample_data" / "sample_homography.json"
out_path = ROOT / "sample_data" / "sample_trajectories_homography_output.csv"


def apply_homography(x: float, y: float, H: np.ndarray) -> tuple[float, float]:
    point = np.array([x, y, 1.0], dtype=float)
    transformed = H @ point
    transformed = transformed / transformed[2]
    return float(transformed[0]), float(transformed[1])


df = pd.read_csv(trajectory_path)
with open(homography_path, "r", encoding="utf-8") as f:
    H = np.array(json.load(f)["matrix"], dtype=float)

xy = df.apply(lambda row: apply_homography(row["x_image"], row["y_image"], H), axis=1)
df[["x_homography", "y_homography"]] = pd.DataFrame(xy.tolist(), index=df.index)
df.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(df.head())
