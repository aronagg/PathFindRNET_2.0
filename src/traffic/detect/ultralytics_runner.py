from typing import Iterable
from ultralytics import YOLO


class UltralyticsDetector:
    def __init__(self, weights: str, device: str = "auto", conf: float = 0.25, classes=None, imgsz=None):
        self.model = YOLO(weights)
        self.kw = dict(device=device, conf=conf, classes=classes)
        if imgsz is not None:
            self.kw["imgsz"] = imgsz

    def detect(self, source: str | int, stream: bool = True) -> Iterable:
        return self.model.predict(source=source, stream=stream, **self.kw)
