from typing import Dict, List, Optional
import numpy as np

from ..core.frame_data import FrameData
from ..core.pipeline_step import PipelineStep
from ..segmentation.player_segmentation import PlayerSegmenter
from .team_classifier import TeamClassifier
from .team_vote_manager import TeamVoteManager

class TeamAssignmentStep(PipelineStep):
    """Orquestrador do fluxo de atribuição de times."""
    def __init__(
        self,
        classifier: TeamClassifier,
        segmenter: PlayerSegmenter,
        vote_manager: TeamVoteManager,
        player_class_ids: Optional[List[int]] = None,
        team_names: Optional[Dict[int, str]] = None,
    ):
        self.classifier = classifier
        self.segmenter = segmenter
        self.vote_manager = vote_manager
        
        self.player_class_ids = set(player_class_ids) if player_class_ids else {0}
        self.team_names = team_names or {0: "Time 1", 1: "Time 2"}

    def process(self, data: FrameData) -> FrameData:
        # 1. Prepara o FrameData
        data.team_names = self.team_names
        data.team_ids = [-1] * len(data.detections)
        data.segmentation_masks = [None] * len(data.detections)

        if not data.detections:
            return data

        # 2. Processamento de Visão (Isolado no Segmenter)
        segments = self.segmenter.get_segments(data.image)
        
        valid_indices, valid_crops, valid_track_ids = [], [], []

        for idx, det in enumerate(data.detections):
            if int(det[5]) not in self.player_class_ids:
                continue

            track_id = int(det[6]) if len(det) > 6 else -1
            box = np.array(det[:4], dtype=np.float32)
            scaled_box = self.segmenter._scale_box(box)
            
            mask = self.segmenter.find_best_mask(scaled_box, segments)
            data.segmentation_masks[idx] = mask  # Salva para o DrawStep
            
            crop = self.segmenter.crop_masked_image(data.image, scaled_box, mask)
            
            # Se a oclusão destruiu o crop, resgata do histórico
            if crop is None:
                last_team = self.vote_manager.get_last_known_team(track_id)
                if last_team >= 0:
                    data.team_ids[idx] = last_team
                continue

            valid_indices.append(idx)
            valid_crops.append(crop)
            valid_track_ids.append(track_id)

        # 3. Classificação e Votação Temporal
        if valid_crops:  # Removido self.calibrator.is_calibrated
            predictions, confidences = self.classifier.predict_with_confidence(valid_crops)
            
            for idx, pred_team, conf, track_id in zip(valid_indices, predictions, confidences, valid_track_ids):
                final_team_id = self.vote_manager.register_vote_and_get_winner(
                    track_id=track_id, 
                    predicted_team_id=int(pred_team), 
                    confidence=conf
                )
                data.team_ids[idx] = final_team_id

        return data