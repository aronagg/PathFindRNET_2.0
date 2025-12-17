from typing import Iterable

from ultralytics import YOLO


class UltralyticsTracker:
    def __init__(
        self,
        weights: str,
        tracker_yaml: str,
        device: str = "auto",
        conf: float = 0.3,
        classes=None,
        imgsz=None,
    ):
        self.model = YOLO(weights)
        self.kw = dict(device=device, conf=conf, classes=classes, tracker=tracker_yaml)
        if imgsz is not None:
            self.kw["imgsz"] = imgsz

    def track(self, source: str | int, stream: bool = True) -> Iterable:
        for res in self.model.track(source=source, stream=stream, **self.kw):
            yield res
