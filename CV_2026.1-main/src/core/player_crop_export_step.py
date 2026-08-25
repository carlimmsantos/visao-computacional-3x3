from pathlib import Path
from typing import Iterable, Optional, Set

import cv2
import numpy as np

from .frame_data import FrameData
from .pipeline_step import PipelineStep


class PlayerCropExportStep(PipelineStep):
    """Exporta crops dos jogadores detectados para um dataset em disco."""

    def __init__(
        self,
        output_dir: str = "players_dataset",
        player_class_ids: Optional[Iterable[int]] = None,
        min_confidence: float = 0.0,
        min_box_width: int = 15,
        min_box_height: int = 30,
        video_source: Optional[str] = None,  # Required parameter - no fallback
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.player_class_ids: Optional[Set[int]] = set(player_class_ids) if player_class_ids is not None else None
        self.min_confidence = float(min_confidence)
        self.min_box_width = int(min_box_width)
        self.min_box_height = int(min_box_height)

        # video_source is now REQUIRED - no fallback
        if video_source is None:
            raise ValueError("video_source parameter is required for PlayerCropExportStep to avoid fallback naming")

        self.video_source = self._sanitize_filename(video_source)
        self.exported_count = 0

    @staticmethod
    def _clamp_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.array(
            [
                max(0, int(np.floor(x1))),
                max(0, int(np.floor(y1))),
                min(width, int(np.ceil(x2))),
                min(height, int(np.ceil(y2))),
            ],
            dtype=np.int32,
        )

    def _sanitize_filename(self, name: str) -> str:
        """Remove problematic characters from filename."""
        # Replace common problematic characters
        name = name.replace(" ", "_")
        name = name.replace("/", "_")
        name = name.replace("\\", "_")
        name = name.replace(":", "_")
        name = name.replace("*", "_")
        name = name.replace("?", "_")
        name = name.replace("\"", "_")
        name = name.replace("<", "_")
        name = name.replace(">", "_")
        name = name.replace("|", "_")
        # Remove any other non-alphanumeric characters except dot and underscore
        name = "".join(c for c in name if c.isalnum() or c in "._-")
        return name

    def process(self, data: FrameData) -> FrameData:
        if not data.detections:
            return data

        height, width = data.image.shape[:2]

        for index, det in enumerate(data.detections):
            if len(det) < 6:
                continue

            x1, y1, x2, y2, confidence = det[:5]
            class_id = int(det[5])

            if self.player_class_ids is not None and class_id not in self.player_class_ids:
                continue
            if float(confidence) < self.min_confidence:
                continue

            box = self._clamp_box(np.array([x1, y1, x2, y2], dtype=np.float32), width=width, height=height)
            box_width = int(box[2] - box[0])
            box_height = int(box[3] - box[1])
            if box_width < self.min_box_width or box_height < self.min_box_height:
                continue

            crop = data.image[box[1] : box[3], box[0] : box[2]]
            if crop.size == 0:
                continue

            track_id = int(det[6]) if len(det) > 6 else -1
            self.exported_count += 1

            # Always use video_source naming (actual video name)
            filename = (
                f"{self.video_source}_frame_{data.frame_id:06d}"
                f"_det_{index:02d}_cls_{class_id}_conf_{float(confidence):.2f}"
            )

            # Add track ID if available
            if track_id >= 0:
                filename += f"_track_{track_id}"
            filename += ".jpg"

            cv2.imwrite(str(self.output_dir / filename), crop)

        return data
