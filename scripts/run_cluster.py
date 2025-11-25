from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from traffic.cluster.consolidate import consolidate_by_exit
from traffic.cluster.optics import optics_cluster
from traffic.io.dataset_loader import get_paths
from traffic.io.serialization import read_parquet, write_parquet


@hydra.main(config_path="../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    # Get paths relative to original working directory
    orig_cwd = Path(get_original_cwd())
    _, _, processed = get_paths(cfg.dataset)
    processed = orig_cwd / processed

    trajs = read_parquet(processed / "trajectories_cleaned.parquet")
    # entry/exit per track
    g = trajs.sort_values("frame").groupby("track_id")
    entry = g.first()[["x", "y"]].to_numpy()
    exit_ = g.last()[["x", "y"]].to_numpy()
    exy = np.hstack([entry, exit_])
    track_ids = g.size().index.to_numpy()

    # Run OPTICS clustering with config parameters
    print("\nClustering configuration:")
    print(f"  min_samples: {cfg.dataset.cluster.min_samples}")
    print(f"  xi: {cfg.dataset.cluster.xi}")
    print(f"  max_eps: {cfg.dataset.cluster.get('max_eps', np.inf)}")
    labels, model = optics_cluster(
        exy,
        min_samples=cfg.dataset.cluster.min_samples,
        xi=cfg.dataset.cluster.xi,
        max_eps=cfg.dataset.cluster.get("max_eps", np.inf),
    )

    # Save cluster assignments
    df = pd.DataFrame(dict(track_id=track_ids, cluster=labels))
    write_parquet(df, processed / "clusters.parquet")

    # Analyze and save outliers
    from traffic.cluster.optics import analyze_outliers, get_outlier_stats

    outlier_stats = get_outlier_stats(labels, track_ids)
    print("\nOPTICS Clustering Results:")
    print(f"  Total samples: {outlier_stats['n_total']}")
    print(f"  Clusters found: {outlier_stats['n_clusters']}")
    print(f"  Outliers: {outlier_stats['n_outliers']} ({outlier_stats['pct_outliers']:.1f}%)")

    outlier_df = analyze_outliers(exy, labels, model, track_ids)
    if len(outlier_df) > 0:
        write_parquet(outlier_df, processed / "outliers.parquet")
        n_unreachable = outlier_df["is_unreachable"].sum()
        print(f"  Wrote {len(outlier_df)} outliers to outliers.parquet")
        if n_unreachable > 0:
            print(
                f"  Completely unreachable: {n_unreachable} ({100.0*n_unreachable/len(outlier_df):.1f}%)"
            )
        print("  Top 5 outliers by reachability:")
        for idx, row in outlier_df.head(5).iterrows():
            unreachable_flag = " [UNREACHABLE]" if row["is_unreachable"] else ""
            print(
                f"    Track {row['track_id']}: reachability={row['reachability']:.3f}{unreachable_flag}, "
                f"entry=({row['x_entry']:.3f}, {row['y_entry']:.3f}), "
                f"exit=({row['x_exit']:.3f}, {row['y_exit']:.3f})"
            )

    # Consolidate by exit points
    exit_labels, _ = consolidate_by_exit(exit_, k=8)
    df2 = pd.DataFrame(dict(track_id=track_ids, exit_group=exit_labels))
    write_parquet(df2, processed / "exit_groups.parquet")
    print("\nWrote cluster and exit-group tables.")


if __name__ == "__main__":
    main()
