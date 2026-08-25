import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import os
import json

from .coco_skeleton import COCO_SKELETON_EDGES
from .frame_data import FrameData
from .pipeline_step import PipelineStep

class DrawStep(PipelineStep):
    def __init__(
        self,
        overlay_size: Tuple[int, int] = (300, 161),
        overlay_pos: Tuple[int, int] = (20, 20),
        minimap_size: Tuple[int, int] = (161, 300),
        draw_segmentation: bool = False,
        segmentation_alpha: float = 0.35,
        draw_minimap_keypoints: bool = True,
        draw_template_keypoints: bool = True,
        draw_frame_keypoints: bool = True,
        template_path: str = "assets/dst_homo.json",
        minimap_image_path: Optional[str] = None,
        draw_poses: bool = False,
        highlight_shooting: bool = True,
    ):
        self.overlay_size = overlay_size
        self.overlay_pos = overlay_pos
        self.minimap_size = minimap_size
        self.draw_minimap_keypoints = draw_minimap_keypoints
        self.draw_template_keypoints = draw_template_keypoints
        self.draw_frame_keypoints = draw_frame_keypoints
        self.draw_poses = draw_poses
        self.highlight_shooting = highlight_shooting

        self.template_points = self._carregar_template(template_path, self.minimap_size[1], self.minimap_size[0])

        self.draw_segmentation = draw_segmentation
        self.segmentation_alpha = float(np.clip(segmentation_alpha, 0.0, 1.0))
        
        if minimap_image_path and os.path.exists(minimap_image_path):
            self.court_template = cv2.imread(minimap_image_path)
            self.minimap_size = self.court_template.shape[:2] 
        else:
            self.court_template = self._draw_procedural_court()

    def _carregar_template(self, path: str, target_width: int, target_height: int) -> Dict[int, Tuple[float, float]]:
        if not path or not os.path.exists(path):
            return {}
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

    TEAM_COLOR_MAP = {
        0: (40, 210, 80),   # Time 1 (Verde)
        1: (40, 120, 255),  # Time 2 (Laranja/Vermelho)
    }

    SHOOTING_COLOR = (0, 255, 255)  # Amarelo

    def _draw_procedural_court(self) -> np.ndarray:
        minimap = np.zeros((self.minimap_size[0], self.minimap_size[1], 3), dtype=np.uint8)
        
        bg_color = (30, 30, 30)
        court_color = (200, 150, 100)
        paint_color = (150, 90, 40)
        line_color = (255, 255, 255)
        
        minimap[:] = bg_color
        
        x_min, y_min = 0, 0
        x_max, y_max = 300, 161
        x_center, y_center = 150, 80
        
        cv2.rectangle(minimap, (x_min, y_min), (x_max, y_max), court_color, -1)
        cv2.rectangle(minimap, (x_min, y_min), (x_max, y_max), line_color, 2)
        
        cv2.line(minimap, (x_center, y_min), (x_center, y_max), line_color, 2)
        cv2.circle(minimap, (x_center, y_center), 19, line_color, 2)
        
        cv2.rectangle(minimap, (x_min, 54), (62, 106), paint_color, -1)
        cv2.rectangle(minimap, (x_min, 54), (62, 106), line_color, 2)
        cv2.ellipse(minimap, (62, y_center), (19, 19), 0, -90, 90, line_color, 2)
        
        cv2.rectangle(minimap, (238, 54), (x_max, 106), paint_color, -1)
        cv2.rectangle(minimap, (238, 54), (x_max, 106), line_color, 2)
        cv2.ellipse(minimap, (238, y_center), (19, 19), 0, 90, 270, line_color, 2)
        
        cv2.ellipse(minimap, (x_min, y_center), (90, 80), 0, -70, 70, line_color, 2)
        cv2.ellipse(minimap, (x_max, y_center), (90, 80), 0, 110, 250, line_color, 2)
        
        return minimap

    @staticmethod
    def _blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float) -> None:
        if alpha <= 0.0 or mask is None or mask.shape[:2] != image.shape[:2] or not np.any(mask):
            return
        color_arr = np.array(color, dtype=np.float32)
        image_pixels = image[mask].astype(np.float32)
        image[mask] = np.clip(image_pixels * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)

    def _draw_skeleton(self, image: np.ndarray, kpts: np.ndarray) -> None:
        """Desenha o esqueleto COCO-17; ignora keypoints NaN."""
        for i, j in COCO_SKELETON_EDGES:
            pt_a, pt_b = kpts[i, :2], kpts[j, :2]
            if not (np.isfinite(pt_a).all() and np.isfinite(pt_b).all()):
                continue
            cv2.line(image, (int(pt_a[0]), int(pt_a[1])), (int(pt_b[0]), int(pt_b[1])),
                     (255, 200, 0), 2)
        for x, y, _ in kpts:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(image, (int(x), int(y)), 3, (0, 200, 255), -1)

    @staticmethod
    def _color_from_class_id(class_id: int) -> tuple:
        palette = [(0, 255, 0), (0, 165, 255), (255, 255, 0), (255, 0, 0), (255, 0, 255)]
        return palette[0] if class_id < 0 else palette[class_id % len(palette)]

    @staticmethod
    def _resolve_class_name(data: FrameData, class_id: int) -> str:
        class_names = getattr(data, 'class_names', {})
        if isinstance(class_names, dict):
            return class_names.get(class_id, f"class_{class_id}")
        if isinstance(class_names, list) and 0 <= class_id < len(class_names):
            return class_names[class_id]
        return "Player"

    @staticmethod
    def _resolve_team_name(data: FrameData, team_id: int) -> str:
        if team_id < 0: return "Sem time"
        if isinstance(data.team_names, dict):
            return data.team_names.get(team_id, f"Time {team_id + 1}")
        return f"Time {team_id + 1}"

    def _desenhar_minimapa(self, data: FrameData):
        H = getattr(data, 'homography_matrix', None)
        if H is None: return

        minimap = self.court_template.copy()
        player_feet, player_colors = [], []

        for index, det in enumerate(data.detections):
            x1, y1, x2, y2 = det[:4]
            player_feet.append([(x1 + x2) / 2.0, y2])
            
            team_id = int(data.team_ids[index]) if hasattr(data, "team_ids") and index < len(data.team_ids) else -1
            player_colors.append(self.TEAM_COLOR_MAP.get(team_id, (200, 200, 200)))

        if player_feet:
            pts_f = np.array(player_feet, dtype=np.float32).reshape(-1, 1, 2)
            pts_trans = cv2.perspectiveTransform(pts_f, H).reshape(-1, 2)

            for (x, y), color in zip(pts_trans, player_colors):
                if not np.isfinite(x) or not np.isfinite(y): continue
                xi, yi = int(x), int(y)
                
                if 0 <= xi <= self.minimap_size[1] and 0 <= yi <= self.minimap_size[0]:
                    cv2.circle(minimap, (xi, yi), 6, color, -1)
                    cv2.circle(minimap, (xi, yi), 6, (0, 0, 0), 1)

        # Desenha os keypoints da quadra projetados no minimapa
        if self.draw_minimap_keypoints and data.court_keypoints:
            kps_f = np.array([[kp[0], kp[1]] for kp in data.court_keypoints], dtype=np.float32).reshape(-1, 1, 2)
            if len(kps_f) > 0:
                kps_trans = cv2.perspectiveTransform(kps_f, H).reshape(-1, 2)
                
                for index, (x, y) in enumerate(kps_trans):
                    if not np.isfinite(x) or not np.isfinite(y): continue
                    xi, yi = int(x), int(y)
                    
                    if 0 <= xi <= self.minimap_size[1] and 0 <= yi <= self.minimap_size[0]:
                        cv2.circle(minimap, (xi, yi), 3, (0, 255, 255), -1)
                        cv2.circle(minimap, (xi, yi), 3, (0, 0, 0), 1)
                        
                        kp_data = data.court_keypoints[index]
                        if len(kp_data) >= 3:
                            kp_id = str(int(kp_data[2]))
                            cv2.putText(minimap, kp_id, (xi + 4, yi - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)
                            cv2.putText(minimap, kp_id, (xi + 4, yi - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Desenha os keypoints do template (vermelho) no minimapa
        if self.draw_template_keypoints and hasattr(self, 'template_points') and self.template_points:
            for kp_id, (x, y) in self.template_points.items():
                xi, yi = int(x), int(y)
                if 0 <= xi <= self.minimap_size[1] and 0 <= yi <= self.minimap_size[0]:
                    cv2.circle(minimap, (xi, yi), 4, (0, 0, 255), -1) # Ponto vermelho
                    cv2.circle(minimap, (xi, yi), 4, (0, 0, 0), 1)      # Borda preta
                    kp_str = str(kp_id)
                    cv2.putText(minimap, kp_str, (xi - 6, yi + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)
                    cv2.putText(minimap, kp_str, (xi - 6, yi + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Desenha os arremessos registrados no minimapa (como um 'X' vermelho com borda preta)
        if hasattr(data, 'recorded_shots') and data.recorded_shots:
            for _, _, x_world, y_world in data.recorded_shots:
                xi, yi = int(x_world), int(y_world)
                if 0 <= xi <= self.minimap_size[1] and 0 <= yi <= self.minimap_size[0]:
                    length = 4
                    # Desenha borda preta (grossa)
                    cv2.line(minimap, (xi - length, yi - length), (xi + length, yi + length), (0, 0, 0), 3)
                    cv2.line(minimap, (xi - length, yi + length), (xi + length, yi - length), (0, 0, 0), 3)
                    # Desenha cruz vermelha (fina)
                    cv2.line(minimap, (xi - length, yi - length), (xi + length, yi + length), (0, 0, 255), 1)
                    cv2.line(minimap, (xi - length, yi + length), (xi + length, yi - length), (0, 0, 255), 1)

        # Desenha a sobreposição na imagem principal
        mini = cv2.resize(minimap, self.overlay_size)
        x, y = self.overlay_pos
        w, h = self.overlay_size

        roi = data.image[y : y + h, x : x + w]
        blended = cv2.addWeighted(mini, 0.9, roi, 0.1, 0)
        data.image[y : y + h, x : x + w] = blended
        cv2.rectangle(data.image, (x, y), (x + w, y + h), (255, 255, 255), 2)

    def process(self, data: FrameData) -> FrameData:
        cv2.putText(data.image, f"Frame: {data.frame_id}", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for index, det in enumerate(data.detections):
            x1, y1, x2, y2, conf = det[:5]
            class_id = int(det[5]) if len(det) > 5 else 0
            track_id = int(det[6]) if len(det) > 6 else -1

            team_id = int(data.team_ids[index]) if hasattr(data, "team_ids") and index < len(data.team_ids) else -1
            color = self.TEAM_COLOR_MAP.get(team_id, self._color_from_class_id(class_id))

            is_shooting = bool(data.shooting_flags[index]) if hasattr(data, "shooting_flags") and index < len(data.shooting_flags) else False
            box_thickness = 2

            is_shooting = bool(data.shooting_flags[index]) if hasattr(data, "shooting_flags") and index < len(data.shooting_flags) else False
            if self.highlight_shooting and is_shooting:
                color = self.SHOOTING_COLOR
                box_thickness = 4

            if self.draw_segmentation and hasattr(data, "segmentation_masks") and index < len(data.segmentation_masks):
                seg_mask = data.segmentation_masks[index]
                self._blend_mask(data.image, seg_mask, color, self.segmentation_alpha)

            class_name = self._resolve_class_name(data, class_id)
            cv2.rectangle(data.image, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thickness)

            if self.draw_poses and hasattr(data, "poses") and index < len(data.poses) and data.poses[index] is not None:
                self._draw_skeleton(data.image, data.poses[index])

            track_prefix = f"ID {track_id} | " if track_id >= 0 else ""
            label = f"{track_prefix}{self._resolve_team_name(data, team_id)} | {class_name} {conf:.2f}" if team_id >= 0 else f"{track_prefix}{class_name} {conf:.2f}"
            if self.highlight_shooting and is_shooting:
                shooting_conf = float(data.shooting_confidences[index]) if hasattr(data, "shooting_confidences") and index < len(data.shooting_confidences) else 0.0
                label += f" | SHOOTING {shooting_conf:.2f}"

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(data.image, (int(x1), int(y1) - 20), (int(x1) + text_w, int(y1)), color, -1)
            cv2.putText(data.image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.draw_frame_keypoints:
            for kp in data.court_keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(data.image, (x, y), 5, (0, 255, 255), -1)
                if len(kp) >= 3:
                    kp_id = int(kp[2])
                    cv2.putText(data.image, str(kp_id), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                    cv2.putText(data.image, str(kp_id), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        self._desenhar_minimapa(data)
        return data