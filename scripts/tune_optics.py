"""
OPTICS Parameter Tuning Guide for Traffic Intersections

For a 4-way intersection, we expect:
- 12 primary movements: 4 straight, 4 left turns, 4 right turns
- Plus U-turns, lane changes = ~12-20 clusters total

Key Parameters:
--------------
1. min_samples: Controls density threshold
   - Too high → Under-clustering (everything merges)
   - Too low → Over-clustering (noise becomes clusters)
   - Recommended: 5-15% of smallest expected cluster size
   - For 1000 tracks: Start with 15-30

2. max_eps: Maximum distance for neighborhood
   - Controls spatial reach in 4D space [x_entry, y_entry, x_exit, y_exit]
   - Too large → Merges dissimilar patterns
   - Too small → Fragments valid clusters
   - Recommended: Depends on normalized coordinates (0-1 scale)
   - Start with 0.05-0.15 for normalized data

3. xi: Steepness threshold for cluster extraction
   - Controls sensitivity to density changes
   - Too high → Misses small clusters, merges similar ones
   - Too low → Over-fragments, every bump becomes a cluster
   - Recommended: 0.01-0.05 for intersection analysis
   - Start with 0.03

Tuning Strategy:
---------------
1. Normalize your coordinates to [0,1] scale first
2. Start conservative (find obvious clusters):
   - min_samples = 20-30
   - xi = 0.05
   - max_eps = 0.1
   
3. If you get too few clusters (< 8), try:
   - Decrease min_samples (e.g., 15)
   - Decrease xi (e.g., 0.03)
   
4. If you get too many clusters (> 25), try:
   - Increase min_samples (e.g., 40)
   - Increase xi (e.g., 0.07)

5. Use reachability plot to visualize:
   - Valleys = cluster boundaries
   - Deep valleys = well-separated clusters
   - Shallow valleys = marginal separations

Quick Diagnostic:
----------------
- 1 cluster → Parameters WAY too conservative (decrease all)
- 2-5 clusters → Still too conservative (decrease xi, min_samples)
- 8-20 clusters → Good range for intersection
- 50+ clusters → Too aggressive (increase xi, min_samples)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler


def normalize_coordinates(entry_exit_points):
    """Normalize entry-exit coordinates to [0,1] scale."""
    scaler = StandardScaler()
    normalized = scaler.fit_transform(entry_exit_points)
    # Scale to [0,1] range
    min_vals = normalized.min(axis=0)
    max_vals = normalized.max(axis=0)
    normalized = (normalized - min_vals) / (max_vals - min_vals + 1e-10)
    return normalized, scaler


def plot_reachability(model, labels):
    """Plot reachability plot to visualize cluster structure."""
    reachability = model.reachability_[model.ordering_]
    labels_ordered = labels[model.ordering_]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot reachability distances
    colors = ["g" if label >= 0 else "r" for label in labels_ordered]
    ax.bar(range(len(reachability)), reachability, color=colors, alpha=0.6)

    ax.set_xlabel("Sample Order")
    ax.set_ylabel("Reachability Distance")
    ax.set_title("OPTICS Reachability Plot (Green=Clustered, Red=Outliers)")
    ax.grid(True, alpha=0.3, axis="y")

    return fig


def tune_optics_grid_search(entry_exit_points, param_grid):
    """
    Try multiple parameter combinations and report results.

    param_grid = {
        'min_samples': [10, 20, 30],
        'xi': [0.01, 0.03, 0.05],
        'max_eps': [0.1, 0.15, 0.2]
    }
    """
    results = []

    for min_samp in param_grid["min_samples"]:
        for xi_val in param_grid["xi"]:
            for max_eps_val in param_grid["max_eps"]:
                model = OPTICS(min_samples=min_samp, xi=xi_val, max_eps=max_eps_val)
                labels = model.fit_predict(entry_exit_points)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = (labels == -1).sum()
                pct_outliers = 100.0 * n_outliers / len(labels)

                results.append(
                    {
                        "min_samples": min_samp,
                        "xi": xi_val,
                        "max_eps": max_eps_val,
                        "n_clusters": n_clusters,
                        "n_outliers": n_outliers,
                        "pct_outliers": pct_outliers,
                    }
                )

    return pd.DataFrame(results)


# Example usage:
if __name__ == "__main__":
    # Load your data
    processed_dir = Path("data/processed/bellevue_116th_ne12th")
    trajs = pd.read_parquet(processed_dir / "trajectories_cleaned.parquet")

    # Build entry-exit points
    g = trajs.sort_values("frame").groupby("track_id")
    entry = g.first()[["x", "y"]].to_numpy()
    exit_ = g.last()[["x", "y"]].to_numpy()
    exy = np.hstack([entry, exit_])

    print(f"Data shape: {exy.shape}")
    print("Coordinate ranges:")
    print(f"  X: [{exy[:, 0].min():.3f}, {exy[:, 0].max():.3f}]")
    print(f"  Y: [{exy[:, 1].min():.3f}, {exy[:, 1].max():.3f}]")

    # Normalize coordinates
    exy_norm, scaler = normalize_coordinates(exy)

    # Grid search for best parameters
    param_grid = {
        "min_samples": [15, 25, 40],
        "xi": [0.02, 0.04, 0.06],
        "max_eps": [0.1, 0.15, 0.2],
    }

    print("\nTuning OPTICS parameters...")
    results = tune_optics_grid_search(exy_norm, param_grid)

    # Show results sorted by number of clusters
    print("\nParameter tuning results:")
    print(results.sort_values("n_clusters", ascending=False).to_string(index=False))

    # Find parameters that give reasonable cluster count (8-20)
    good_params = results[(results["n_clusters"] >= 8) & (results["n_clusters"] <= 20)]
    if len(good_params) > 0:
        print("\nRecommended parameters (8-20 clusters):")
        print(good_params.sort_values("n_clusters").to_string(index=False))
