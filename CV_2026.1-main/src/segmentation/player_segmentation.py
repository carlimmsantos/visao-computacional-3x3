import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from ultralytics import YOLO

class PlayerSegmenter:
    """Responsável exclusivo por isolar os pixels dos jogadores do fundo."""
    def __init__(
        self,
        weights_path: str = "models/seg/yolov8m-seg.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.7,
        image_size: int = 640,
        crop_scale: float = 1.0,
    ):
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.crop_scale = crop_scale
        
        resolved_path = Path(weights_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Pesos de segmentação ausentes: {resolved_path}")
        self.model = YOLO(str(resolved_path))

    def _scale_box(self, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(np.float32)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = (x2 - x1) * self.crop_scale, (y2 - y1) * self.crop_scale
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

    def _clamp_box(self, box: np.ndarray, width: int, height: int) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.array([max(0, x1), max(0, y1), min(x2, width - 1), min(y2, height - 1)], dtype=np.int32)

    def _box_intersection_area(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        return inter_w * inter_h

    def get_segments(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(source=frame, conf=self.confidence, iou=self.iou_threshold, imgsz=self.image_size, verbose=False)
        if not results or results[0].boxes is None or results[0].masks is None: 
            return []
        
        segments = []
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(np.int32)
        masks_xy = results[0].masks.xy

        for box, class_id, polygon in zip(xyxy, classes, masks_xy):
            if polygon is None or len(polygon) < 3: continue
            frame_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(frame_mask, [np.round(polygon).astype(np.int32)], 1)
            segments.append({"bbox": box.astype(np.float32), "class_id": class_id, "mask": frame_mask.astype(bool)})
        return segments

    def find_best_mask(self, box: np.ndarray, segments: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        best_mask, best_score = None, 0.0
        box_area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
        for seg in segments:
            inter = self._box_intersection_area(box, seg["bbox"])
            if inter / box_area > best_score:
                best_score = inter / box_area
                best_mask = seg["mask"]
        return best_mask

    def crop_masked_image(self, frame: np.ndarray, box: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._clamp_box(box, width=w, height=h)
        if x2 <= x1 or y2 <= y1: return None
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        if mask is None: return crop
        
        mask_crop = mask[y1:y2, x1:x2]
        if mask_crop.shape[:2] != crop.shape[:2] or not np.any(mask_crop): return crop
        
        masked_crop = np.zeros_like(crop)
        masked_crop[mask_crop] = crop[mask_crop]
        return masked_crop