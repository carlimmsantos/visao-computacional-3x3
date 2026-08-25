import numpy as np

from dataclasses import dataclass
from typing import Optional

from ..core.coco_skeleton import (
    NOSE,
    L_SHOULDER, R_SHOULDER,
    L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST,
    L_HIP, R_HIP,
    L_KNEE, R_KNEE,
    L_ANKLE, R_ANKLE,
)

@dataclass
class PoseFeatures:
    """Features por frame derivadas dos keypoints (NaN quando indisponível)."""
    elbow_angle_left: float        # graus, ombro-cotovelo-punho
    elbow_angle_right: float
    knee_angle_left: float         # graus, quadril-joelho-tornozelo
    knee_angle_right: float
    wrist_rel_height_left: float   # (y_ombros - y_punho) / altura_torso (>0 = acima)
    wrist_rel_height_right: float
    wrist_above_nose: bool         # melhor punho acima do nariz
    torso_height: float            # px, escala para normalizações
    valid: bool                    # torso detectável (2 ombros + >=1 quadril)

    @classmethod
    def invalid(cls) -> "PoseFeatures":
        return cls(
            elbow_angle_left=np.nan,
            elbow_angle_right=np.nan,
            knee_angle_left=np.nan,
            knee_angle_right=np.nan,
            wrist_rel_height_left=np.nan,
            wrist_rel_height_right=np.nan,
            wrist_above_nose=False,
            torso_height=np.nan,
            valid=False,
        )

class PoseFeatureExtractor:
    """Converte keypoints COCO-17 em features geométricas para classificação."""

    def extract(self, keypoints: Optional[np.ndarray]) -> PoseFeatures:
        """Extrai features de um array (17, 3) [x, y, conf]; NaN-safe."""
        if keypoints is None:
            return PoseFeatures.invalid()

        pts = keypoints[:, :2]

        shoulders_y = pts[[L_SHOULDER, R_SHOULDER], 1]
        hips_y = pts[[L_HIP, R_HIP], 1]
        if np.isnan(shoulders_y).any() or np.isnan(hips_y).all():
            return PoseFeatures.invalid()

        shoulder_line_y = float(np.mean(shoulders_y))
        hip_line_y = float(np.nanmean(hips_y))
        torso_height = abs(hip_line_y - shoulder_line_y)
        if torso_height < 1.0:
            return PoseFeatures.invalid()

        def wrist_rel_height(wrist_idx: int) -> float:
            wrist_y = pts[wrist_idx, 1]
            if np.isnan(wrist_y):
                return np.nan
            # y cresce para baixo na imagem: punho acima dos ombros => positivo
            return (shoulder_line_y - float(wrist_y)) / torso_height

        rel_left = wrist_rel_height(L_WRIST)
        rel_right = wrist_rel_height(R_WRIST)

        nose_y = pts[NOSE, 1]
        wrist_ys = pts[[L_WRIST, R_WRIST], 1]
        best_wrist_y = np.nan if np.isnan(wrist_ys).all() else float(np.nanmin(wrist_ys))
        wrist_above_nose = bool(
            not np.isnan(nose_y)
            and not np.isnan(best_wrist_y)
            and best_wrist_y < nose_y
        )

        return PoseFeatures(
            elbow_angle_left=self._angle(pts[L_SHOULDER], pts[L_ELBOW], pts[L_WRIST]),
            elbow_angle_right=self._angle(pts[R_SHOULDER], pts[R_ELBOW], pts[R_WRIST]),
            knee_angle_left=self._angle(pts[L_HIP], pts[L_KNEE], pts[L_ANKLE]),
            knee_angle_right=self._angle(pts[R_HIP], pts[R_KNEE], pts[R_ANKLE]),
            wrist_rel_height_left=rel_left,
            wrist_rel_height_right=rel_right,
            wrist_above_nose=wrist_above_nose,
            torso_height=torso_height,
            valid=True,
        )

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Ângulo em graus no vértice b, formado pelos segmentos b->a e b->c."""
        if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
            return np.nan
        ba, bc = a - b, c - b
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm < 1e-6:
            return np.nan
        cos_angle = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
