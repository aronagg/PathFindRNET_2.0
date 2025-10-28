import hydra
import pandas as pd
from omegaconf import DictConfig

from traffic.io.dataset_loader import get_paths
from traffic.io.serialization import read_parquet, write_parquet
from traffic.trajectories.build import build_trajectories


@hydra.main(config_path="../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    _, interim, processed = get_paths(cfg.dataset)
    df = read_parquet(interim / "tracks.parquet")
    fps = cfg.dataset.fps
    trajs = build_trajectories(df, fps=fps)
    # flatten to long table
    rows = []
    for t in trajs:
        for f, x, y, vx, vy, ax, ay in zip(
            t["frame"], t["x"], t["y"], t["vx"], t["vy"], t["ax"], t["ay"]
        ):
            rows.append(
                dict(
                    track_id=t["track_id"],
                    frame=int(f),
                    x=float(x),
                    y=float(y),
                    vx=float(vx),
                    vy=float(vy),
                    ax=float(ax),
                    ay=float(ay),
                )
            )
    out = processed / "trajectories.parquet"
    write_parquet(pd.DataFrame(rows), out)
    print(f"Wrote trajectories -> {out}")


if __name__ == "__main__":
    main()
