import os

import hydra
import pandas as pd
from omegaconf import DictConfig

from traffic.detect.ultralytics_runner import UltralyticsDetector
from traffic.io.dataset_loader import get_paths
from traffic.io.serialization import write_parquet
from traffic.track.tracker_api import UltralyticsTracker


@hydra.main(config_path="../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    raw, interim, _ = get_paths(cfg.dataset)
    # Prefer an explicit CLI override 'source'; else use configured video path
    source = getattr(cfg, "source", None)
    if source is None:
        source = os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.video)

    weights = cfg.detect.weights
    conf = cfg.detect.conf
    device = cfg.detect.device
    classes = cfg.detect.classes

    rows = []
    if cfg.tracker.name == "none":
        det = UltralyticsDetector(weights, device=device, conf=conf, classes=classes)
        for i, res in enumerate(det.detect(source=source)):
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            for b in res.boxes:
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cls = int(b.cls[0].item()) if b.cls is not None else -1
                c = float(b.conf[0].item()) if b.conf is not None else 0.0
                rows.append(
                    dict(frame=i, track_id=-1, cls=cls, conf=c, cx=cx, cy=cy, w=x2 - x1, h=y2 - y1)
                )
    else:
        tracker_yaml = cfg.tracker.yaml_path
        tr = UltralyticsTracker(weights, tracker_yaml, device=device, conf=conf, classes=classes)
        for i, res in enumerate(tr.track(source=source)):
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            ids = res.boxes.id
            for j, b in enumerate(res.boxes):
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cls = int(b.cls[0].item()) if b.cls is not None else -1
                c = float(b.conf[0].item()) if b.conf is not None else 0.0
                tid = int(ids[j].item()) if ids is not None else -1
                rows.append(
                    dict(frame=i, track_id=tid, cls=cls, conf=c, cx=cx, cy=cy, w=x2 - x1, h=y2 - y1)
                )

    df = pd.DataFrame(rows)
    out = interim / "tracks.parquet"
    write_parquet(df, out)
    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
