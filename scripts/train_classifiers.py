import hydra
import numpy as np
from omegaconf import DictConfig

from traffic.classify.evaluate import crossval_scores
from traffic.classify.models import make_model
from traffic.io.dataset_loader import get_paths
from traffic.io.serialization import read_parquet


@hydra.main(config_path="../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    _, _, processed = get_paths(cfg.dataset)
    Xdf = read_parquet(processed / "features.parquet")

    # Labels: prefer exit_groups if available; else cluster
    ymap = None
    try:
        ydf = read_parquet(processed / "exit_groups.parquet")
        ymap = dict(zip(ydf["track_id"], ydf["exit_group"]))
    except Exception:
        try:
            ydf = read_parquet(processed / "clusters.parquet")
            ymap = dict(zip(ydf["track_id"], ydf["cluster"]))
        except Exception:
            print("No label tables found. Run run_cluster.py first.")
            return

    y = np.array([ymap.get(tid, -1) for tid in Xdf["track_id"]], dtype=int)
    X = Xdf.drop(columns=["track_id"]).to_numpy(dtype=np.float32)
    if len(X) == 0:
        print("No features found. Run gen_features.py first.")
        return

    clf = make_model(cfg.clf.name)
    scores = crossval_scores(clf, X, y, k=5, repeats=2, seed=42)
    print(f"{cfg.clf.name} balanced-accuracy: mean={scores.mean():.3f} +- {scores.std():.3f}")


if __name__ == "__main__":
    main()
