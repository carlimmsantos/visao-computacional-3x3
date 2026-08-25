from abc import ABC, abstractmethod
from typing import Sequence

from .pose_features import PoseFeatures

class BaseShootingClassifier(ABC):
    """Interface comum para classificadores de movimento de arremesso.

    Permite trocar o classificador por regras por um modelo aprendido
    (logistic/GBM sobre features da janela, ST-GCN sobre keypoints, etc.)
    sem alterar o ShootingDetectionStep.
    """

    @abstractmethod
    def score_frame(self, features: PoseFeatures) -> float:
        """Score [0,1] de pose-de-arremesso num único frame (sem contexto temporal)."""

    @abstractmethod
    def score_window(
        self, window: Sequence[PoseFeatures], detector_prior: Sequence[bool]
    ) -> float:
        """Score [0,1] de movimento de arremesso na janela temporal.

        Args:
            window: features dos últimos N frames do track (ordem cronológica).
            detector_prior: por frame, se o detector classificou a detecção
                como classe de arremesso (player-jump-shot / player-layup-dunk).
        """
