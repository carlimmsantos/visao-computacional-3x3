import json
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..core.frame_data import FrameData
from ..core.pipeline_step import PipelineStep

MAX_KEYPOINTS = 99

def fix_homography_flip(H: np.ndarray) -> np.ndarray:
    """
    Detecta e corrige reflexões (flips) indesejadas na matriz de homografia.
    Se o determinante da parte afim for negativo, usa Decomposição em Valores Singulares (SVD)
    para forçar a rotação válida mais próxima, removendo a reflexão fisicamente impossível.
    """
    R = H[:2, :2]
    if np.linalg.det(R) < 0:
        U, S, Vt = np.linalg.svd(R)
        # Inverte o sinal da componente de menor variância para remover o espelhamento
        S[-1] *= -1
        H[:2, :2] = U @ np.diag(S) @ Vt
    return H

def reprojection_error(H: np.ndarray, img_pts: np.ndarray, court_pts: np.ndarray) -> float:
    if len(img_pts) == 0:
        return float('inf')
    img_pts = img_pts.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(img_pts, H).reshape(-1, 2)
    return np.mean(np.linalg.norm(projected - court_pts, axis=1))

def blend_homographies(H1: np.ndarray, H2: np.ndarray, alpha: float) -> np.ndarray:
    H_blend = alpha * H1 + (1 - alpha) * H2
    return H_blend / H_blend[-1, -1]

class HomographyStep(PipelineStep):
    def __init__(
        self,
        template_path: str = "assets/dst_homo.json",
        tactical_width: int = 300,
        tactical_height: int = 161,
        min_keypoints: int = 4,
        max_keypoints: int = MAX_KEYPOINTS,
        alpha: float = 0.85,
        reprojection_thresh: float = 40.0,
        ransac_reproj_thresh: float = 5.0,
        ransac_max_iters: int = 2000
    ):
        self.min_keypoints = min_keypoints
        self.max_keypoints = max_keypoints
        self.alpha = alpha
        self.reprojection_thresh = reprojection_thresh
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.ransac_max_iters = ransac_max_iters
        
        self._H_prev: Optional[np.ndarray] = None
        self._template: Dict[int, Tuple[float, float]] = self._carregar_template(
            template_path, tactical_width, tactical_height
        )

    @staticmethod
    def _carregar_template(path: str, target_width: int, target_height: int) -> Dict[int, Tuple[float, float]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        points = {p["id"]: (p["x"], p["y"]) for p in data["dst_homo"]}
        if not points: return points
            
        min_x = min(x for x, y in points.values())
        max_x = max(x for x, y in points.values())
        min_y = min(y for x, y in points.values())
        max_y = max(y for x, y in points.values())
        
        court_w = max_x - min_x
        court_h = max_y - min_y
        
        scaled_points = {}
        for kp_id, (x, y) in points.items():
            nx = ((x - min_x) / court_w) * target_width
            ny = ((y - min_y) / court_h) * target_height
            scaled_points[kp_id] = (nx, ny)
            
        return scaled_points

    def _select_best_keypoints(self, keypoints: List[tuple]) -> List[tuple]:
        valid = [kp for kp in keypoints if int(kp[2]) in self._template]
        valid.sort(key=lambda kp: kp[3] if len(kp) >= 4 else 0.0, reverse=True)
        return valid[: self.max_keypoints]

    def _build_point_pairs(self, keypoints: List[tuple]) -> Tuple[np.ndarray, np.ndarray]:
        src, dst = [], []
        for kp in keypoints:
            class_id = int(kp[2])
            if class_id in self._template:
                src.append([kp[0], kp[1]])
                dst.append(list(self._template[class_id]))
        return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)

    def process(self, data: FrameData) -> FrameData:
        keypoints = data.court_keypoints

        if keypoints:
            best_kps = self._select_best_keypoints(keypoints)
            src_pts, dst_pts = self._build_point_pairs(best_kps)

            if len(src_pts) >= self.min_keypoints and len(src_pts) == len(dst_pts):
                H, mask = cv2.findHomography(
                    src_pts, 
                    dst_pts, 
                    method=cv2.RANSAC, 
                    ransacReprojThreshold=self.ransac_reproj_thresh,
                    maxIters=self.ransac_max_iters
                )

                if H is not None and mask is not None:
                    inliers_count = np.sum(mask)
                    if inliers_count >= self.min_keypoints:
                        H = fix_homography_flip(H)
                        
                        inlier_img_pts = src_pts[mask.ravel() == 1]
                        inlier_court_pts = dst_pts[mask.ravel() == 1]
                        err = reprojection_error(H, inlier_img_pts, inlier_court_pts)

                        if err <= self.reprojection_thresh:
                            if self._H_prev is not None:
                                H = blend_homographies(self._H_prev, H, self.alpha)
                            
                            self._H_prev = H

        data.homography_matrix = self._H_prev
        data.template_points = self._template
        return data
