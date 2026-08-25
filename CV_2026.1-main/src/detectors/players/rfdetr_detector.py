from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
from rfdetr import RFDETRMedium
import supervision as sv
from trackers import BoTSORTTracker

from .base_detector import BaseDetector


class RFDETRDetector(BaseDetector):
    """Detector de jogadores com modelo RF-DETR fine-tuned para basquete."""

    def __init__(
        self,
        weights_path: str = "models/rfdetr-m-basketball.pth",
        confidence: float = 0.4,
        resolution: int = 576,
        optimize_inference: bool = True,
        track_activation_threshold: float = 0.5,   
        lost_track_buffer: int = 50,               
        minimum_consecutive_frames: int = 2,        
        high_conf_det_threshold: float = 0.5,      
        minimum_iou_threshold_first_assoc: float = 0.15,
        minimum_iou_threshold_second_assoc: float = 0.10,  
        minimum_iou_threshold_unconfirmed_assoc: float = 0.10,  
        enable_cmc: bool = True,
        cmc_method: str = "sparseOptFlow", 
        frame_rate: int = 30,
    ):
        resolved_path = Path(weights_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Arquivo de pesos nao encontrado: {resolved_path}")

        self.model = RFDETRMedium(
            pretrain_weights=str(resolved_path),
            resolution=resolution,
        )
        self.class_names = getattr(self.model, "class_names", {0: "Player"})
        self.confidence = confidence

        if optimize_inference:
            self.model.optimize_for_inference()

        self.tracker = BoTSORTTracker(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_consecutive_frames=minimum_consecutive_frames,
            high_conf_det_threshold=high_conf_det_threshold,
            minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
            minimum_iou_threshold_second_assoc=minimum_iou_threshold_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=minimum_iou_threshold_unconfirmed_assoc,
            enable_cmc=enable_cmc,
            cmc_method=cmc_method,
            frame_rate=frame_rate,
        )

    @staticmethod
    def _to_rgb_image(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Formato de imagem nao suportado: {image.shape}")

    def _sv_detections_to_list(self, detections: sv.Detections, include_track_ids: bool = False) -> List[Any]:
        """Converte sv.Detections para [[x1,y1,x2,y2,conf,cls,(track_id)], ...]."""
        if detections is None or len(detections) == 0:
            return []

        result = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].tolist()
            conf = float(detections.confidence[i])
            cls_id = int(detections.class_id[i])
            det = [x1, y1, x2, y2, conf, cls_id]

            if include_track_ids:
                track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                det.append(track_id)

            result.append(det)

        return result

    def detect(self, image: np.ndarray) -> List[Any]:
        """Detecção simples sem tracking. Retorna [[x1,y1,x2,y2,conf,cls], ...]."""
        rgb_image = self._to_rgb_image(image)
        detections = self.model.predict(rgb_image, threshold=self.confidence)

        if detections is None or len(detections) == 0:
            return []

        return self._sv_detections_to_list(detections, include_track_ids=False)

    def track(self, image: np.ndarray) -> List[Any]:
        """Detecção + tracking com BoTSORT. Retorna [[x1,y1,x2,y2,conf,cls,track_id], ...]."""
        rgb_image = self._to_rgb_image(image)
        detections = self.model.predict(rgb_image, threshold=self.confidence)

        if detections is None or len(detections) == 0:
            return []

        tracked_detections = self.tracker.update(detections, frame=image)

        return self._sv_detections_to_list(tracked_detections, include_track_ids=True)

    def reset_tracker(self) -> None:
        """Reinicia o estado interno do tracker (chamar entre vídeos distintos)."""
        self.tracker.reset()