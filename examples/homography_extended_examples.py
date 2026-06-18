from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTROL_POINTS = ROOT / "sample_data" / "sample_homography_control_points.csv"
TRAJECTORIES = ROOT / "sample_data" / "sample_trajectories.csv"
OUT_DIR = ROOT / "sample_data"


def estimate_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate H with a minimal Direct Linear Transform implementation."""
    if len(src) < 4:
        raise ValueError("At least four point pairs are required.")
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, dtype=float)
    _, _, vh = np.linalg.svd(A)
    H = vh[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack([points, ones])
    transformed = (H @ homogeneous.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    return transformed


def reprojection_error(src: np.ndarray, dst: np.ndarray, H: np.ndarray) -> pd.DataFrame:
    pred = apply_homography(src, H)
    err = np.linalg.norm(pred - dst, axis=1)
    return pd.DataFrame({"pred_x": pred[:, 0], "pred_y": pred[:, 1], "target_x": dst[:, 0], "target_y": dst[:, 1], "error": err})


def main() -> None:
    cp = pd.read_csv(CONTROL_POINTS)
    src = cp[["image_x", "image_y"]].to_numpy(float)
    dst = cp[["map_x", "map_y"]].to_numpy(float)

    H_image_to_map = estimate_homography_dlt(src, dst)
    H_map_to_image = np.linalg.inv(H_image_to_map)
    H_map_to_image = H_map_to_image / H_map_to_image[2, 2]

    err = reprojection_error(src, dst, H_image_to_map)
    err.to_csv(OUT_DIR / "homography_reprojection_error.csv", index=False)

    traj = pd.read_csv(TRAJECTORIES)
    image_points = traj[["x_image", "y_image"]].to_numpy(float)
    map_points = apply_homography(image_points, H_image_to_map)
    traj["x_map_from_H"] = map_points[:, 0]
    traj["y_map_from_H"] = map_points[:, 1]
    traj.to_csv(OUT_DIR / "homography_transformed_trajectories.csv", index=False)

    example_map_points = np.array([[0, 0], [20, 10], [40, 35]], dtype=float)
    back_projected = apply_homography(example_map_points, H_map_to_image)
    back_projected_df = pd.DataFrame(back_projected, columns=["image_x_backprojected", "image_y_backprojected"])
    back_projected_df[["map_x", "map_y"]] = example_map_points
    back_projected_df.to_csv(OUT_DIR / "homography_backprojected_points.csv", index=False)

    with open(OUT_DIR / "homography_matrix_estimated.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "DLT homography from manually selected image/map control points",
                "image_to_map": H_image_to_map.tolist(),
                "map_to_image": H_map_to_image.tolist(),
                "mean_reprojection_error": float(err["error"].mean()),
                "max_reprojection_error": float(err["error"].max()),
            },
            f,
            indent=2,
        )

    print("Estimated image-to-map homography:")
    print(H_image_to_map)
    print("\nReprojection error:")
    print(err)
    print("\nSaved homography examples into sample_data/")


if __name__ == "__main__":
    main()
