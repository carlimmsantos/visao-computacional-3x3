import numpy as np

from pathlib import Path
from typing import List, Optional, Tuple
from ultralytics import YOLO

DEFAULT_POSE_WEIGHTS = "models/pose/yolo11n-pose.pt"

class PoseEstimator:
    """Responsável exclusivo por estimar keypoints COCO-17 em crops de jogadores.

    image_size=256 mediu o melhor recall por evento de arremesso no dataset
    combinado (crops de jogador têm 60-580px de altura; 640 amplia demais os
    crops pequenos e degrada os keypoints). Também é o mais rápido (~40ms por
    frame com 10 crops numa RTX 3050).
    """
    def __init__(
        self,
        weights_path: str = DEFAULT_POSE_WEIGHTS,
        confidence: float = 0.25,
        keypoint_conf_threshold: float = 0.3,
        image_size: int = 256,
        device: Optional[str] = None,
    ):
        self.confidence = confidence
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.image_size = image_size
        self.device = device

        resolved_path = Path(weights_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Pesos de pose ausentes: {resolved_path}")
        self.model = YOLO(str(resolved_path))

    def estimate_on_crops(
        self,
        crops: List[np.ndarray],
        origins: List[Tuple[int, int]],
        inner_boxes: Optional[List[np.ndarray]] = None,
    ) -> List[Optional[np.ndarray]]:
        """Estima a pose de uma pessoa por crop, em batch.

        Args:
            crops: lista de crops BGR (um jogador esperado por crop).
            origins: canto superior-esquerdo (x1, y1) de cada crop no frame.
            inner_boxes: box original (sem padding) de cada jogador em coords
                do FRAME [x1, y1, x2, y2]. Quando fornecido, a pessoa é
                selecionada por IoU com esse box — essencial quando há
                jogadores sobrepostos no crop.

        Returns:
            Por crop, um array (17, 3) [x, y, conf] em coordenadas do FRAME
            (keypoints com conf < keypoint_conf_threshold viram NaN), ou None
            se nenhuma pessoa foi detectada.
        """
        if not crops:
            return []

        results = self.model.predict(
            source=crops,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        if inner_boxes is None:
            inner_boxes = [None] * len(crops)

        poses: List[Optional[np.ndarray]] = []
        for result, crop, (origin_x, origin_y), inner_box in zip(
            results, crops, origins, inner_boxes
        ):
            if inner_box is not None:
                # Converte o box do frame para coords do crop
                inner_box = np.asarray(inner_box, dtype=np.float32).copy()
                inner_box[[0, 2]] -= float(origin_x)
                inner_box[[1, 3]] -= float(origin_y)

            person_idx = self._select_best_person(result, crop.shape, inner_box)
            if person_idx is None:
                poses.append(None)
                continue

            kpts = result.keypoints.data[person_idx].cpu().numpy().astype(np.float32)
            low_conf = kpts[:, 2] < self.keypoint_conf_threshold
            kpts[low_conf, 0:2] = np.nan

            # Keypoints vêm em pixels do crop — translada para o frame.
            kpts[:, 0] += float(origin_x)
            kpts[:, 1] += float(origin_y)
            poses.append(kpts)

        return poses

    @staticmethod
    def _select_best_person(
        result, crop_shape, inner_box: Optional[np.ndarray] = None
    ) -> Optional[int]:
        """Escolhe a pessoa cujo bbox melhor casa com o box do jogador alvo.

        Um crop de jogador pode conter defensores sobrepostos; pontuamos cada
        esqueleto por IoU com o box original (sem padding) do jogador. Sem
        inner_box, usa o crop inteiro como referência, modulado pela
        proximidade ao centro.
        """
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            return None
        if keypoints.data is None or len(keypoints.data) == 0:
            return None

        crop_h, crop_w = crop_shape[:2]
        if inner_box is not None:
            ref_x1, ref_y1, ref_x2, ref_y2 = (float(v) for v in inner_box)
        else:
            ref_x1, ref_y1, ref_x2, ref_y2 = 0.0, 0.0, float(crop_w), float(crop_h)
        ref_area = max(1.0, (ref_x2 - ref_x1) * (ref_y2 - ref_y1))
        ref_cx, ref_cy = (ref_x1 + ref_x2) / 2.0, (ref_y1 + ref_y2) / 2.0
        crop_diag = float(np.hypot(crop_w, crop_h))

        xyxy = boxes.xyxy.cpu().numpy()
        best_idx, best_score = None, 0.0
        for idx, (x1, y1, x2, y2) in enumerate(xyxy):
            inter_w = max(0.0, min(x2, ref_x2) - max(x1, ref_x1))
            inter_h = max(0.0, min(y2, ref_y2) - max(y1, ref_y1))
            inter = inter_w * inter_h
            box_area = max(1.0, (x2 - x1) * (y2 - y1))
            iou = inter / (box_area + ref_area - inter)

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            center_proximity = 1.0 - np.hypot(cx - ref_cx, cy - ref_cy) / crop_diag

            score = iou * max(0.0, center_proximity)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx
