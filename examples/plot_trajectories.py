from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
trajectory_path = ROOT / "sample_data" / "sample_trajectories.csv"
out_path = ROOT / "sample_data" / "sample_trajectory_plot.png"

df = pd.read_csv(trajectory_path)

fig, ax = plt.subplots(figsize=(7, 5))
for track_id, group in df.groupby("track_id"):
    ax.plot(group["x_image"], group["y_image"], marker="o", label=f"track {track_id}")

ax.invert_yaxis()
ax.set_xlabel("x image [px]")
ax.set_ylabel("y image [px]")
ax.set_title("Sample image-plane trajectories")
ax.legend()
fig.tight_layout()
fig.savefig(out_path, dpi=160)
print(f"Saved: {out_path}")
