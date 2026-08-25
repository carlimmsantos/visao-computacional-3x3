import numpy as np

from typing import List, Optional, Tuple

from ..core.frame_data import FrameData
from ..core.pipeline_step import PipelineStep
from ..detectors.players.classes import PLAYER_CLASS_IDS, SHOOTING_CLASS_IDS
from ..pose import PoseEstimator
from .pose_features import PoseFeatureExtractor
from .shooting_state_manager import ShootingStateManager

class ShootingDetectionStep(PipelineStep):
    """Orquestrador da detecção de movimento de arremesso por jogador trackeado."""

    # Classes do detector que já indicam arremesso (prior fraco)
    SHOOTING_PRIOR_CLASS_IDS = SHOOTING_CLASS_IDS

    def __init__(
        self,
        pose_estimator: PoseEstimator,
        state_manager: ShootingStateManager,
        feature_extractor: Optional[PoseFeatureExtractor] = None,
        player_class_ids: Optional[List[int]] = None,
        crop_padding: float = 0.15,
        min_crop_height_px: int = 64,
    ):
        self.pose_estimator = pose_estimator
        self.state_manager = state_manager
        self.feature_extractor = feature_extractor or PoseFeatureExtractor()

        self.player_class_ids = set(player_class_ids) if player_class_ids else set(PLAYER_CLASS_IDS)
        self.crop_padding = crop_padding
        self.min_crop_height_px = min_crop_height_px

    def process(self, data: FrameData) -> FrameData:
        # 1. Prepara o FrameData (indexado por detecção, como team_ids)
        data.poses = [None] * len(data.detections)
        data.shooting_flags = [False] * len(data.detections)
        data.shooting_confidences = [0.0] * len(data.detections)

        if not data.detections:
            return data

        height, width = data.image.shape[:2]
        valid_indices, valid_crops, valid_origins, valid_boxes = [], [], [], []
        valid_track_ids, valid_priors = [], []

        for idx, det in enumerate(data.detections):
            if int(det[5]) not in self.player_class_ids:
                continue

            track_id = int(det[6]) if len(det) > 6 else -1
            prior = int(det[5]) in self.SHOOTING_PRIOR_CLASS_IDS

            box = self._pad_box(np.array(det[:4], dtype=np.float32))
            x1, y1, x2, y2 = self._clamp_box(box, width, height)
            if x2 <= x1 or y2 <= y1:
                continue

            # Jogador longe demais para pose confiável: carrega o último estado
            if (y2 - y1) < self.min_crop_height_px:
                flag, score = self.state_manager.get_last_state(track_id)
                data.shooting_flags[idx] = flag
                data.shooting_confidences[idx] = score
                continue

            valid_indices.append(idx)
            valid_crops.append(data.image[y1:y2, x1:x2])
            valid_origins.append((x1, y1))
            valid_boxes.append(np.array(det[:4], dtype=np.float32))
            valid_track_ids.append(track_id)
            valid_priors.append(prior)

        # 2. Pose em lote (uma chamada por frame); o box original desambigua
        # qual pessoa do crop é o jogador alvo
        if valid_crops:
            poses = self.pose_estimator.estimate_on_crops(
                valid_crops, valid_origins, inner_boxes=valid_boxes
            )

            for idx, kpts, track_id, prior in zip(
                valid_indices, poses, valid_track_ids, valid_priors
            ):
                data.poses[idx] = kpts
                features = self.feature_extractor.extract(kpts)
                flag, score = self.state_manager.register_and_classify(
                    track_id=track_id,
                    features=features,
                    detector_prior=prior,
                    frame_id=data.frame_id,
                )
                data.shooting_flags[idx] = flag
                data.shooting_confidences[idx] = score

        self.state_manager.prune_stale(data.frame_id)
        return data

    def _pad_box(self, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w = (x2 - x1) * (1.0 + self.crop_padding)
        h = (y2 - y1) * (1.0 + self.crop_padding)
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

    @staticmethod
    def _clamp_box(box: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            int(max(0, x1)),
            int(max(0, y1)),
            int(min(x2, width - 1)),
            int(min(y2, height - 1)),
        )
