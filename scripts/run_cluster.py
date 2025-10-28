import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from traffic.cluster.consolidate import consolidate_by_exit
from traffic.cluster.optics import optics_cluster
from traffic.io.dataset_loader import get_paths
from traffic.io.serialization import read_parquet, write_parquet


@hydra.main(config_path="../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    _, _, processed = get_paths(cfg.dataset)
    trajs = read_parquet(processed / "trajectories.parquet")
    # entry/exit per track
    g = trajs.sort_values("frame").groupby("track_id")
    entry = g.first()[["x", "y"]].to_numpy()
    exit_ = g.last()[["x", "y"]].to_numpy()
    exy = np.hstack([entry, exit_])

    labels, _ = optics_cluster(exy, min_samples=30, xi=0.05)
    df = pd.DataFrame(dict(track_id=g.size().index.to_numpy(), cluster=labels))
    write_parquet(df, processed / "clusters.parquet")

    exit_labels, _ = consolidate_by_exit(exit_, k=8)
    df2 = pd.DataFrame(dict(track_id=g.size().index.to_numpy(), exit_group=exit_labels))
    write_parquet(df2, processed / "exit_groups.parquet")
    print("Wrote cluster and exit-group tables.")


if __name__ == "__main__":
    main()
