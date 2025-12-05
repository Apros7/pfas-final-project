from typing import List

import numpy as np
from ultralytics import YOLO

from constants import YOLO_MODEL_PATH
from detection import Detection


class ObjectDetector:
    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)
        self.conf = conf
        self.iou = iou

    def detect(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            image, conf=self.conf, iou=self.iou, verbose=False
        )
        boxes = results[0].boxes if results else None
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        detections: List[Detection] = []
        for i in range(len(cls_ids)):
            bbox = xyxy[i]
            cx = float((bbox[0] + bbox[2]) / 2.0)
            cy = float((bbox[1] + bbox[3]) / 2.0)
            detections.append(
                Detection(
                    bbox=bbox,
                    cls_id=int(cls_ids[i]),
                    score=float(scores[i]),
                    center=(cx, cy),
                )
            )
        return detections

