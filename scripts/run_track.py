import os
import cv2
import numpy as np
from visualize import draw_annotations, show_frame

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
    imgsz = cfg.detect.get("imgsz", cfg.detect.get("size", None))
    # visualize flag now comes from the dataset config (per-scene)
    visualize = bool(cfg.dataset.get("visualize", False))

    # colors must be provided per-scene in dataset config as a map label->RGB
    ds_colors = getattr(cfg.dataset, "colors", None)
    class_map = getattr(cfg.dataset, "class_map", None)
    if ds_colors is None or class_map is None:
        raise RuntimeError("cfg.dataset.colors and cfg.dataset.class_map are required for visualization.")
    # ensure same label set
    if set(ds_colors.keys()) != set(class_map.keys()):
        raise RuntimeError("cfg.dataset.colors keys must exactly match cfg.dataset.class_map keys.")
    # build COLORS list indexed by class_id (max id + 1), unused slots remain None
    max_id = max(class_map.values())
    COLORS = [None] * (max_id + 1)
    for label_name, cid in class_map.items():
        rgb = ds_colors.get(label_name)
        if rgb is None:
            raise RuntimeError(f"Missing color for label '{label_name}' in cfg.dataset.colors")
        COLORS[int(cid)] = tuple(map(int, rgb))

    rows = []
    if cfg.tracker.name == "none":
        det = UltralyticsDetector(weights, device=device, conf=conf, classes=classes, imgsz=imgsz)
        stop = False
        for i, res in enumerate(det.detect(source=source)):
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            img = getattr(res, "orig_img", None)
            annos = []
            for b in res.boxes:
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cls = int(b.cls[0].item()) if b.cls is not None else -1
                c = float(b.conf[0].item()) if b.conf is not None else 0.0
                rows.append(dict(frame=i, track_id=-1, cls=cls, conf=c, cx=cx, cy=cy, w=x2 - x1, h=y2 - y1))
                annos.append((x1, y1, x2, y2, cls, c, None))
            if visualize and img is not None:
                draw_annotations(img, annos, COLORS, class_names=getattr(cfg.detect, "class_names", None))
                if show_frame("detections", img):
                    stop = True
            if stop:
                break
    else:
        tracker_yaml = cfg.tracker.yaml_path
        tr = UltralyticsTracker(weights, tracker_yaml, device=device, conf=conf, classes=classes, imgsz=imgsz)
        stop = False
        for i, res in enumerate(tr.track(source=source)):
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            img = getattr(res, "orig_img", None)
            ids = res.boxes.id
            annos = []
            for j, b in enumerate(res.boxes):
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cls = int(b.cls[0].item()) if b.cls is not None else -1
                c = float(b.conf[0].item()) if b.conf is not None else 0.0
                tid = int(ids[j].item()) if ids is not None else -1
                rows.append(dict(frame=i, track_id=tid, cls=cls, conf=c, cx=cx, cy=cy, w=x2 - x1, h=y2 - y1))
                annos.append((x1, y1, x2, y2, cls, c, tid))
            if visualize and img is not None:
                draw_annotations(img, annos, COLORS, class_names=getattr(cfg.detect, "class_names", None))
                if show_frame("tracking", img):
                    stop = True
            if stop:
                break

    df = pd.DataFrame(rows)
    out = interim / "tracks.parquet"
    write_parquet(df, out)
    print(f"Wrote {len(df)} rows -> {out}")
    if visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
