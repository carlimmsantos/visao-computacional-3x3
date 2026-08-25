from pathlib import Path
from typing import Any, List

import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector


class YOLO26Detector(BaseDetector):
    """Detector de jogadores com modelo YOLO fine-tuned para basquete."""

    def __init__(
        self,
        weights_path: str = "models/detector/yolo26m-basketball.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.7,
        image_size: int = 640,
        tracker_config: str = "botsort.yaml",
        track_persist: bool = True,
    ):
        resolved_path = Path(weights_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Arquivo de pesos nao encontrado: {resolved_path}")

        self.model = YOLO(str(resolved_path))
        self.class_names = self.model.names
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.tracker_config = tracker_config
        self.track_persist = track_persist

    @staticmethod
    def _parse_results(results: List[Any], include_track_ids: bool) -> List[Any]:
        if not results:
            return []

        detections: List[Any] = []
        boxes = results[0].boxes

        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        track_ids = None
        if include_track_ids and getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.cpu().numpy().astype(np.int32)

        for index, (box, conf, cls_id) in enumerate(zip(xyxy, confs, classes)):
            x1, y1, x2, y2 = box.tolist()
            det = [x1, y1, x2, y2, float(conf), int(cls_id)]

            if include_track_ids:
                track_id = int(track_ids[index]) if track_ids is not None and index < len(track_ids) else -1
                det.append(track_id)

            detections.append(det)

        return detections

    def detect(self, image: np.ndarray) -> List[Any]:
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            verbose=False,
        )


        detections = self._parse_results(results=results, include_track_ids=False)

        return detections

    def track(self, image: np.ndarray) -> List[Any]:
        results = self.model.track(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            tracker=self.tracker_config,
            persist=self.track_persist,
            verbose=False,
        )

        detections = self._parse_results(results=results, include_track_ids=True)

        print(len(detections))
        return detections
