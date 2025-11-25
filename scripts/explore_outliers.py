"""Explore and visualize OPTICS outliers from clustering results.

This script helps analyze trajectories that OPTICS couldn't cluster (label -1),
which often represent unusual or anomalous traffic patterns.
"""

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from traffic.io.dataset_loader import get_paths
from traffic.io.serialization import read_parquet, write_parquet


@hydra.main(config_path="../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    _, _, processed = get_paths(cfg.dataset)

    # Load data
    outliers = read_parquet(processed / "outliers.parquet")
    clusters = read_parquet(processed / "clusters.parquet")
    trajs = read_parquet(processed / "trajectories.parquet")

    print(f"=== Outlier Analysis for {cfg.dataset.scene} ===\n")

    # Basic statistics
    n_outliers = len(outliers)
    n_total = len(clusters)
    n_clustered = (clusters["cluster"] >= 0).sum()

    print(f"Total tracks: {n_total}")
    print(f"Clustered: {n_clustered} ({100.0*n_clustered/n_total:.1f}%)")
    print(f"Outliers: {n_outliers} ({100.0*n_outliers/n_total:.1f}%)")

    if n_outliers == 0:
        print("\nNo outliers found!")
        return

    # Reachability distribution
    print("\nReachability Statistics:")
    print(f"  Min: {outliers['reachability'].min():.3f}")
    print(f"  Max: {outliers['reachability'].max():.3f}")
    print(f"  Mean: {outliers['reachability'].mean():.3f}")
    print(f"  Median: {outliers['reachability'].median():.3f}")

    # Spatial distribution
    print("\nSpatial Distribution:")
    print(f"  Entry X range: [{outliers['x_entry'].min():.3f}, {outliers['x_entry'].max():.3f}]")
    print(f"  Entry Y range: [{outliers['y_entry'].min():.3f}, {outliers['y_entry'].max():.3f}]")
    print(f"  Exit X range: [{outliers['x_exit'].min():.3f}, {outliers['x_exit'].max():.3f}]")
    print(f"  Exit Y range: [{outliers['y_exit'].min():.3f}, {outliers['y_exit'].max():.3f}]")

    # Top outliers
    print("\nTop 10 Most Anomalous Tracks (highest reachability):")
    for idx, row in outliers.head(10).iterrows():
        track_traj = trajs[trajs["track_id"] == row["track_id"]]
        duration = len(track_traj)
        print(
            f"  Track {row['track_id']:4d}: reach={row['reachability']:6.3f}, "
            f"duration={duration:3d} frames, "
            f"entry=({row['x_entry']:.3f},{row['y_entry']:.3f}), "
            f"exit=({row['x_exit']:.3f},{row['y_exit']:.3f})"
        )

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Entry points colored by reachability
    ax = axes[0, 0]
    scatter = ax.scatter(
        outliers["x_entry"],
        outliers["y_entry"],
        c=outliers["reachability"],
        cmap="hot",
        alpha=0.6,
        s=50,
    )
    ax.set_xlabel("Entry X")
    ax.set_ylabel("Entry Y")
    ax.set_title("Outlier Entry Points (colored by reachability)")
    plt.colorbar(scatter, ax=ax, label="Reachability")
    ax.grid(True, alpha=0.3)

    # 2. Exit points colored by reachability
    ax = axes[0, 1]
    scatter = ax.scatter(
        outliers["x_exit"],
        outliers["y_exit"],
        c=outliers["reachability"],
        cmap="hot",
        alpha=0.6,
        s=50,
    )
    ax.set_xlabel("Exit X")
    ax.set_ylabel("Exit Y")
    ax.set_title("Outlier Exit Points (colored by reachability)")
    plt.colorbar(scatter, ax=ax, label="Reachability")
    ax.grid(True, alpha=0.3)

    # 3. Entry-exit flow (arrows)
    ax = axes[1, 0]
    # Sample top 50 to avoid overcrowding
    sample = outliers.head(50)
    for _, row in sample.iterrows():
        ax.arrow(
            row["x_entry"],
            row["y_entry"],
            row["x_exit"] - row["x_entry"],
            row["y_exit"] - row["y_entry"],
            alpha=0.3,
            head_width=0.02,
            head_length=0.02,
            fc="red",
            ec="red",
            linewidth=0.5,
        )
    ax.scatter(sample["x_entry"], sample["y_entry"], c="blue", s=30, alpha=0.6, label="Entry")
    ax.scatter(sample["x_exit"], sample["y_exit"], c="green", s=30, alpha=0.6, label="Exit")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Top {len(sample)} Outlier Trajectories (Entry→Exit)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Reachability distribution
    ax = axes[1, 1]
    ax.hist(outliers["reachability"], bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(
        outliers["reachability"].median(),
        color="red",
        linestyle="--",
        label=f'Median={outliers["reachability"].median():.2f}',
    )
    ax.set_xlabel("Reachability Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Outlier Reachability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = processed / "outlier_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {plot_path}")

    # Optional: Save detailed trajectory data for top outliers
    top_n = 20
    top_outlier_ids = outliers.head(top_n)["track_id"].values
    top_trajs = trajs[trajs["track_id"].isin(top_outlier_ids)]

    detail_path = processed / f"top_{top_n}_outlier_trajectories.parquet"
    write_parquet(top_trajs, detail_path)
    print(f"Saved top {top_n} outlier trajectories to {detail_path.name}")


if __name__ == "__main__":
    main()
