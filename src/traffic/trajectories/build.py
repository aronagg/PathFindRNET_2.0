from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def build_trajectories(df: pd.DataFrame, fps: float, win: int = 9, poly: int = 2):
    rows = []
    grouped = df.sort_values("frame").groupby("track_id")
    for tid, g in grouped:
        cx = g["cx"].to_numpy()
        cy = g["cy"].to_numpy()
        if len(cx) >= win:
            sx = savgol_filter(cx, win, poly, mode="interp")
            sy = savgol_filter(cy, win, poly, mode="interp")
        else:
            sx, sy = cx, cy
        vx = np.gradient(sx) * fps
        vy = np.gradient(sy) * fps
        ax = np.gradient(vx) * fps
        ay = np.gradient(vy) * fps
        rows.append(
            dict(track_id=tid, frame=g["frame"].to_numpy(), x=sx, y=sy, vx=vx, vy=vy, ax=ax, ay=ay)
        )
    return rows
