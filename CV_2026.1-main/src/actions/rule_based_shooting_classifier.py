import numpy as np

from dataclasses import dataclass
from typing import Sequence, Tuple

from .base_shooting_classifier import BaseShootingClassifier
from .pose_features import PoseFeatures

@dataclass(frozen=True)
class ShootingRuleConfig:
    """Thresholds e pesos das heurísticas de arremesso (calibráveis)."""
    wrist_above_shoulder_ratio: float = 0.5  # punho >= 0.5 alturas-de-torso acima dos ombros
    elbow_angle_flexed_max: float = 110.0    # "set point" (graus)
    elbow_angle_extended_min: float = 150.0  # extensão no release (graus)
    knee_flex_min_delta: float = 15.0        # flexiona-e-estende dos joelhos (graus)
    wrist_upward_velocity_min: float = 0.05  # alturas-de-torso por frame, para cima
    min_valid_frames_in_window: int = 5
    # Pesos dos sub-sinais (somam 1.0)
    w_wrist_high: float = 0.35
    w_elbow_extension: float = 0.25
    w_wrist_velocity: float = 0.15
    w_knee_pattern: float = 0.10
    w_detector_prior: float = 0.15

class RuleBasedShootingClassifier(BaseShootingClassifier):
    """Heurísticas geométricas sobre a janela temporal de features de pose."""

    def __init__(self, config: ShootingRuleConfig = ShootingRuleConfig()):
        self.config = config

    def score_frame(self, features: PoseFeatures) -> float:
        """Score instantâneo: punho alto + cotovelo estendido (sem temporal)."""
        if not features.valid:
            return 0.0

        wrist_rel, elbow_angle = self._best_arm(features)

        s_wrist = self._wrist_high_signal(wrist_rel, features.wrist_above_nose)
        s_elbow = self._elbow_extension_signal(elbow_angle)

        # Punho elevado é condição necessária (braço esticado para baixo não é
        # arremesso); a extensão do cotovelo apenas modula o score.
        w_total = self.config.w_wrist_high + self.config.w_elbow_extension
        w_wrist = self.config.w_wrist_high / w_total
        w_elbow = self.config.w_elbow_extension / w_total
        return s_wrist * (w_wrist + w_elbow * s_elbow)

    def score_window(
        self, window: Sequence[PoseFeatures], detector_prior: Sequence[bool]
    ) -> float:
        cfg = self.config
        valid_mask = np.array([f.valid for f in window], dtype=bool)
        if valid_mask.sum() < cfg.min_valid_frames_in_window:
            return 0.0

        # Arrays por frame (NaN onde inválido) do braço de arremesso
        wrist_rel = np.full(len(window), np.nan, dtype=np.float32)
        elbow = np.full(len(window), np.nan, dtype=np.float32)
        knee = np.full(len(window), np.nan, dtype=np.float32)
        above_nose = np.zeros(len(window), dtype=bool)
        for i, f in enumerate(window):
            if not f.valid:
                continue
            wrist_rel[i], elbow[i] = self._best_arm(f)
            knees = [f.knee_angle_left, f.knee_angle_right]
            knee[i] = np.nan if np.isnan(knees).all() else float(np.nanmin(knees))
            above_nose[i] = f.wrist_above_nose

        s_wrist_high = self._window_wrist_high(wrist_rel, above_nose, valid_mask)
        s_elbow = self._window_elbow_extension(elbow, wrist_rel)
        s_velocity = self._window_wrist_velocity(wrist_rel)
        s_knee = self._window_knee_pattern(knee)
        s_prior = float(np.mean([bool(p) for p in detector_prior])) if len(detector_prior) else 0.0

        return float(
            cfg.w_wrist_high * s_wrist_high
            + cfg.w_elbow_extension * s_elbow
            + cfg.w_wrist_velocity * s_velocity
            + cfg.w_knee_pattern * s_knee
            + cfg.w_detector_prior * s_prior
        )

    # ----- Sinais auxiliares -----

    @staticmethod
    def _best_arm(features: PoseFeatures) -> Tuple[float, float]:
        """Retorna (wrist_rel_height, elbow_angle) do braço mais elevado."""
        rel_l, rel_r = features.wrist_rel_height_left, features.wrist_rel_height_right
        if np.isnan(rel_l) and np.isnan(rel_r):
            return np.nan, np.nan
        if np.isnan(rel_r) or (not np.isnan(rel_l) and rel_l >= rel_r):
            return rel_l, features.elbow_angle_left
        return rel_r, features.elbow_angle_right

    def _wrist_high_signal(self, wrist_rel: float, above_nose: bool) -> float:
        if above_nose:
            return 1.0
        if np.isnan(wrist_rel):
            return 0.0
        return float(np.clip(wrist_rel / self.config.wrist_above_shoulder_ratio, 0.0, 1.0))

    def _elbow_extension_signal(self, elbow_angle: float) -> float:
        cfg = self.config
        if np.isnan(elbow_angle):
            return 0.0
        return float(np.clip(
            (elbow_angle - cfg.elbow_angle_flexed_max)
            / (cfg.elbow_angle_extended_min - cfg.elbow_angle_flexed_max),
            0.0, 1.0,
        ))

    def _window_wrist_high(
        self, wrist_rel: np.ndarray, above_nose: np.ndarray, valid_mask: np.ndarray
    ) -> float:
        """Fração dos frames válidos com punho acima do limiar (ou do nariz)."""
        high = (wrist_rel >= self.config.wrist_above_shoulder_ratio) | above_nose
        return float(np.sum(high & valid_mask) / max(1, valid_mask.sum()))

    def _window_elbow_extension(self, elbow: np.ndarray, wrist_rel: np.ndarray) -> float:
        """Sequência flexionado (1ª metade) -> estendido com punho alto (2ª metade)."""
        cfg = self.config
        half = len(elbow) // 2
        first_half = elbow[:half]
        # Extensão só conta como release se o punho está na linha do ombro ou
        # acima — braço esticado para baixo também tem cotovelo "estendido".
        second_half = np.where(wrist_rel[half:] >= 0.0, elbow[half:], np.nan)
        if np.isnan(second_half).all():
            return 0.0

        extended = float(np.nanmax(second_half)) >= cfg.elbow_angle_extended_min
        if not extended:
            return 0.0
        flexed_before = (
            not np.isnan(first_half).all()
            and float(np.nanmin(first_half)) <= cfg.elbow_angle_flexed_max
        )
        # Crédito parcial para extensão sem o set point visível
        return 1.0 if flexed_before else 0.5

    def _window_wrist_velocity(self, wrist_rel: np.ndarray) -> float:
        """Velocidade média do punho para cima (alturas-de-torso por frame)."""
        diffs = np.diff(wrist_rel)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            return 0.0
        mean_velocity = float(np.mean(diffs))
        return float(np.clip(mean_velocity / self.config.wrist_upward_velocity_min, 0.0, 1.0))

    def _window_knee_pattern(self, knee: np.ndarray) -> float:
        """Joelho flexiona >= delta e recupera dentro da janela."""
        cfg = self.config
        if np.isnan(knee).all():
            return 0.0
        idx_min = int(np.nanargmin(knee))
        knee_min = float(knee[idx_min])
        before = knee[: idx_min + 1]
        after = knee[idx_min:]
        if np.isnan(before).all() or np.isnan(after).all():
            return 0.0
        flexed = float(np.nanmax(before)) - knee_min >= cfg.knee_flex_min_delta
        recovered = float(np.nanmax(after)) - knee_min >= cfg.knee_flex_min_delta
        return 1.0 if (flexed and recovered) else 0.0
