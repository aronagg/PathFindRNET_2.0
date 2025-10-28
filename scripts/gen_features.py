import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from traffic.features.vector_specs import FVS
from traffic.features.vectorize import vectorize
from traffic.io.dataset_loader import get_paths
from traffic.io.serialization import read_parquet, write_parquet


@hydra.main(config_path="../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    _, _, processed = get_paths(cfg.dataset)
    trajs = read_parquet(processed / "trajectories.parquet")
    preset_name = getattr(cfg.features, "preset", "ReVeRs")
    spec = FVS.get(preset_name, FVS["ReVeRs"])

    feats, labels = [], []
    for tid, g in trajs.sort_values("frame").groupby("track_id"):
        traj = {
            k: g[k].to_numpy()
            for k in ["x", "y", "vx", "vy", "ax", "ay", "frame"]
            if k in g.columns
        }
        if len(traj["x"]) == 0:
            continue
        fv = vectorize(traj, spec)
        feats.append(fv)
        labels.append(tid)

    X = np.vstack(feats) if feats else np.zeros((0, 6), dtype=np.float32)
    df = pd.DataFrame(X)
    df.insert(0, "track_id", labels)
    write_parquet(df, processed / "features.parquet")
    print("Wrote features.parquet")


if __name__ == "__main__":
    main()
