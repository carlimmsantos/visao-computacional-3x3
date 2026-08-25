from typing import Any, List

import numpy as np
import supervision as sv

from .base_tracker import BaseTracker


class ByteTrackTracker(BaseTracker):
    """Tracker baseado em supervision.ByteTrack.

    Aceita detecoes no formato `[x1, y1, x2, y2, conf, class_id]` e retorna
    `[x1, y1, x2, y2, conf, class_id, track_id]` (mesmo formato que o
    YOLO26Detector.track() produz hoje).
    """

    def __init__(
        self,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def update(self, detections: List[Any], image: np.ndarray) -> List[Any]:
        if not detections:
            return []

        arr = np.asarray(detections, dtype=np.float32)
        sv_det = sv.Detections(
            xyxy=arr[:, :4],
            confidence=arr[:, 4],
            class_id=arr[:, 5].astype(int),
        )

        tracked = self.tracker.update_with_detections(sv_det)

        out: List[List[float]] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i].tolist()
            tid = (
                int(tracked.tracker_id[i])
                if tracked.tracker_id is not None
                else -1
            )
            out.append(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(tracked.confidence[i]),
                    int(tracked.class_id[i]),
                    tid,
                ]
            )
        return out
