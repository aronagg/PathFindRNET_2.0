from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "sample_data" / "sample_trajectories.csv"
OUT = ROOT / "sample_data" / "isotropic_normalized_trajectories.csv"
PARAMS = ROOT / "sample_data" / "isotropic_normalization_params.json"


def estimate_isotropic_transform(points: np.ndarray, target_rms_radius: float = np.sqrt(2.0)) -> tuple[np.ndarray, dict]:
    """
    Estimate a 2D isotropic normalization transform.

    The transform translates points to zero mean and applies one shared scale factor
    for both axes. This keeps the x/y aspect ratio unchanged, unlike separate
    min-max normalization per axis.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    distances = np.linalg.norm(centered, axis=1)
    mean_distance = float(distances.mean())
    scale = 1.0 if mean_distance == 0 else float(target_rms_radius / mean_distance)
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    params = {"centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]), "scale": scale, "target_rms_radius": float(target_rms_radius)}
    return T, params


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def inverse_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return apply_transform(points, np.linalg.inv(T))


def normalize_per_scene(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    points = df[["x_image", "y_image"]].to_numpy(float)
    T, params = estimate_isotropic_transform(points)
    normalized = apply_transform(points, T)
    restored = inverse_transform(normalized, T)

    out = df.copy()
    out["x_iso"] = normalized[:, 0]
    out["y_iso"] = normalized[:, 1]
    out["x_restored"] = restored[:, 0]
    out["y_restored"] = restored[:, 1]
    params["transform"] = T.tolist()
    params["max_restoration_error"] = float(np.max(np.linalg.norm(points - restored, axis=1)))
    return out, params


def normalize_per_track(df: pd.DataFrame) -> pd.DataFrame:
    # Optional variant: each trajectory is normalized independently.
    chunks = []
    for track_id, g in df.groupby("track_id"):
        points = g[["x_image", "y_image"]].to_numpy(float)
        T, params = estimate_isotropic_transform(points)
        norm = apply_transform(points, T)
        gg = g.copy()
        gg["x_iso_track"] = norm[:, 0]
        gg["y_iso_track"] = norm[:, 1]
        gg["track_iso_scale"] = params["scale"]
        chunks.append(gg)
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    df = pd.read_csv(INPUT)
    scene_normalized, params = normalize_per_scene(df)
    track_normalized = normalize_per_track(df)

    merged = scene_normalized.merge(
        track_normalized[["track_id", "frame_id", "x_iso_track", "y_iso_track", "track_iso_scale"]],
        on=["track_id", "frame_id"],
        how="left",
    )
    merged.to_csv(OUT, index=False)
    with open(PARAMS, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print("Scene-level isotropic normalization parameters:")
    print(params)
    print("\nOutput preview:")
    print(merged.head())
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
