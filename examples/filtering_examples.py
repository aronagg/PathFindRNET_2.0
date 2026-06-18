from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "sample_data" / "sample_trajectories_filtering.csv"
OUT_DIR = ROOT / "sample_data"


def add_track_level_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track_id, g in df.sort_values("frame_id").groupby("track_id"):
        first = g.iloc[0]
        last = g.iloc[-1]
        dx = float(last["x_image"] - first["x_image"])
        dy = float(last["y_image"] - first["y_image"])
        displacement = float(np.hypot(dx, dy))
        duration = float(last["timestamp"] - first["timestamp"])
        rows.append(
            {
                "track_id": track_id,
                "point_count": len(g),
                "class_name": g["class_name"].mode().iat[0],
                "mean_speed": float(g["speed"].mean()),
                "max_speed": float(g["speed"].max()),
                "displacement_px": displacement,
                "duration_sec": duration,
                "inside_application_roi_ratio": float(g["inside_application_roi"].mean()),
                "start_y": float(first["y_image"]),
                "end_y": float(last["y_image"]),
            }
        )
    return pd.DataFrame(rows)


def filter_too_short_tracks(features: pd.DataFrame, min_points: int = 4) -> pd.DataFrame:
    return features[features["point_count"] >= min_points]


def filter_only_cars(features: pd.DataFrame) -> pd.DataFrame:
    return features[features["class_name"] == "car"]


def filter_parking_or_stationary(features: pd.DataFrame, min_displacement_px: float = 25.0, min_mean_speed: float = 0.5) -> pd.DataFrame:
    return features[(features["displacement_px"] >= min_displacement_px) & (features["mean_speed"] >= min_mean_speed)]


def filter_application_based(features: pd.DataFrame, min_roi_ratio: float = 0.75, require_forward_motion: bool = True) -> pd.DataFrame:
    filtered = features[features["inside_application_roi_ratio"] >= min_roi_ratio]
    if require_forward_motion:
        # Example application rule: keep vehicles moving from lower image area toward upper image area.
        filtered = filtered[filtered["end_y"] < filtered["start_y"]]
    return filtered


def main() -> None:
    df = pd.read_csv(INPUT)
    features = add_track_level_features(df)
    features.to_csv(OUT_DIR / "filtering_track_features.csv", index=False)

    step_1 = filter_too_short_tracks(features)
    step_2 = filter_only_cars(step_1)
    step_3 = filter_parking_or_stationary(step_2)
    step_4 = filter_application_based(step_3)

    step_1.to_csv(OUT_DIR / "filtering_step_01_length.csv", index=False)
    step_2.to_csv(OUT_DIR / "filtering_step_02_only_cars.csv", index=False)
    step_3.to_csv(OUT_DIR / "filtering_step_03_no_parking.csv", index=False)
    step_4.to_csv(OUT_DIR / "filtering_step_04_application.csv", index=False)

    final_tracks = df[df["track_id"].isin(step_4["track_id"])]
    final_tracks.to_csv(OUT_DIR / "filtering_output_final_tracks.csv", index=False)

    print("Track-level features")
    print(features)
    print("\nFinal kept track IDs:", list(step_4["track_id"]))
    print("Saved filtering outputs into sample_data/")


if __name__ == "__main__":
    main()
