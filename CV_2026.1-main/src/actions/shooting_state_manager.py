from collections import deque
from typing import Deque, Dict, Tuple

from .base_shooting_classifier import BaseShootingClassifier
from .pose_features import PoseFeatures

class ShootingStateManager:
    """Mantém janelas deslizantes de features por track_id e estabiliza o estado com histerese."""
    def __init__(
        self,
        classifier: BaseShootingClassifier,
        window_size: int = 24,          # ~0.8 s @ 30 fps
        enter_score: float = 0.6,
        exit_score: float = 0.35,
        stale_after_frames: int = 60,
    ):
        self.classifier = classifier
        self.window_size = window_size
        self.enter_score = enter_score
        self.exit_score = exit_score
        self.stale_after_frames = stale_after_frames

        self._windows: Dict[int, Deque[Tuple[PoseFeatures, bool]]] = {}
        self._state: Dict[int, bool] = {}
        self._last_score: Dict[int, float] = {}
        self._last_seen_frame: Dict[int, int] = {}

    def register_and_classify(
        self, track_id: int, features: PoseFeatures, detector_prior: bool, frame_id: int
    ) -> Tuple[bool, float]:
        """Acumula as features do frame e retorna (is_shooting, score) estabilizados."""
        # Sem ID (perda temporária do tracker): não polui janelas, score instantâneo
        if track_id < 0:
            return False, self.classifier.score_frame(features)

        window = self._windows.setdefault(track_id, deque(maxlen=self.window_size))
        window.append((features, detector_prior))
        self._last_seen_frame[track_id] = frame_id

        score = self.classifier.score_window(
            [f for f, _ in window], [p for _, p in window]
        )
        self._last_score[track_id] = score

        # Histerese: liga em enter_score, só desliga abaixo de exit_score
        currently_on = self._state.get(track_id, False)
        if currently_on:
            is_shooting = score > self.exit_score
        else:
            is_shooting = score >= self.enter_score
        self._state[track_id] = is_shooting

        return is_shooting, score

    def get_last_state(self, track_id: int) -> Tuple[bool, float]:
        """Último estado conhecido do track (False/0.0 se nunca visto)."""
        return self._state.get(track_id, False), self._last_score.get(track_id, 0.0)

    def prune_stale(self, current_frame_id: int) -> None:
        """Descarta tracks não vistos há stale_after_frames (limita memória)."""
        stale = [
            track_id
            for track_id, last_seen in self._last_seen_frame.items()
            if current_frame_id - last_seen > self.stale_after_frames
        ]
        for track_id in stale:
            self._windows.pop(track_id, None)
            self._state.pop(track_id, None)
            self._last_score.pop(track_id, None)
            self._last_seen_frame.pop(track_id, None)
