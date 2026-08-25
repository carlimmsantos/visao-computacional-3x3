from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import pandas as pd
 
 
class PlayerHeatmapTracker:
 
    def __init__(
            self,
            fps: float = 30,
            heatmap_resolution: float = 1.0,  # fixed: was 0.1 (metres), now 1.0 (pixels/units)
    ):
        self.fps = fps
        self.heatmap_resolution = heatmap_resolution
 
        self.player_positions: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        self.player_distances: Dict[int, float] = defaultdict(float)
        self.player_speeds: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.player_keypoints: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        self.player_teams: Dict[int, int] = {}
        self.player_shooting_active: Dict[int, bool] = {}
        self.recorded_shots: List[Tuple[int, int, float, float]] = []
 
        self.heatmap: Optional[np.ndarray] = None
        self.court_width: Optional[float] = None
        self.court_length: Optional[float] = None
        self._world_to_metres: float = 1.0
 
        self._last_position: Dict[int, np.ndarray] = {}
        self._last_frame: Dict[int, int] = {}
 
    def set_court_dimensions(self, width: float, length: float):
        """
        Configura as dimensões da quadra
        """
        self.court_width = width
        self.court_length = length
        nx = int(width  / self.heatmap_resolution) + 1
        ny = int(length / self.heatmap_resolution) + 1 
        self.heatmap = np.zeros((ny, nx), dtype=np.float32)
 
    def set_court_dimensions_from_template(self, template: Dict[int, Tuple[float, float]]):
        """
        Infere o tamanho da quadra a partir da Homografia. Recomendado utilizar quando não se conhece
        os tamanhos da quadra.
        """
        if not template:
            raise ValueError("Template is empty — cannot infer court dimensions.")
 
        xs = [x for x, _ in template.values()]
        ys = [y for _, y in template.values()]
        width  = float(max(xs))
        length = float(max(ys))
        print(f"[PlayerHeatmapTracker] Auto-detected court bounds: {width:.1f} x {length:.1f} (template units)")
        self.set_court_dimensions(width, length)
 
    def set_world_scale(self, pixels_per_metre: float):
        """
        Converte de pixels para distância real
        """
        if pixels_per_metre <= 0:
            raise ValueError("pixels_per_metre must be > 0")
        self._world_to_metres = 1.0 / pixels_per_metre
 
    def reset(self):
        """Reseta todas as estatísticas de um vídeo"""
        self.player_positions.clear()
        self.player_distances.clear()
        self.player_speeds.clear()
        self.player_keypoints.clear()
        self.player_teams.clear()
        self.player_shooting_active.clear()
        self.recorded_shots.clear()
        self._last_position.clear()
        self._last_frame.clear()
        if self.heatmap is not None:
            self.heatmap[:] = 0
 
    # ------------------------------------------------------------------
    # Homography validation
    # ------------------------------------------------------------------
 
    @staticmethod
    def _is_valid_homography(H: np.ndarray) -> bool:
        if H is None:
            return False
        if H.shape != (3, 3):
            return False
        if not np.all(np.isfinite(H)):
            return False
        if abs(np.linalg.det(H)) < 1e-5:
            return False
        return True
 
    def _is_within_court(self, world_pt: np.ndarray) -> bool:
        """Verifica se um ponto esta dentro da quadra"""
        if self.court_width is None or self.court_length is None:
            return True
        x, y = world_pt
        margin = 0.05 * max(self.court_width, self.court_length)
        return (-margin <= x <= self.court_width  + margin and
                -margin <= y <= self.court_length + margin)
 

    def update_from_frame(
            self,
            frame_id: int,
            detections: List[Any],
            homography_matrix: Optional[np.ndarray],
            player_keypoints_dict: Optional[Dict[int, np.ndarray]] = None,
            team_ids: Optional[List[int]] = None,
            shooting_flags: Optional[List[bool]] = None,
    ):
        if not self._is_valid_homography(homography_matrix):
            return
 
        for index, det in enumerate(detections):
            if len(det) < 7:
                continue
 
            x1, y1, x2, y2, _, _, track_id = det[:7]
            track_id = int(track_id)
 
            if track_id < 0:
                continue
 
            center_px = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            world_pt  = self._pixel_to_world(center_px, homography_matrix)
 
            if not np.all(np.isfinite(world_pt)):
                continue
 
            if not self._is_within_court(world_pt):
                continue
 
            self.player_positions[track_id].append((frame_id, float(world_pt[0]), float(world_pt[1])))
            self._update_distance_and_speed(track_id, frame_id, world_pt)
 
            if self.heatmap is not None:
                self._update_heatmap(world_pt)
 
            if player_keypoints_dict and track_id in player_keypoints_dict:
                self.player_keypoints[track_id].append((frame_id, player_keypoints_dict[track_id]))
 
            if team_ids is not None and index < len(team_ids):
                self.player_teams[track_id] = int(team_ids[index])
 
            is_shooting = bool(shooting_flags[index]) if shooting_flags is not None and index < len(shooting_flags) else False
            was_shooting = self.player_shooting_active.get(track_id, False)
            if is_shooting and not was_shooting:
                self.recorded_shots.append((frame_id, track_id, float(world_pt[0]), float(world_pt[1])))
            self.player_shooting_active[track_id] = is_shooting
 

    def _pixel_to_world(self, pixel_pt: np.ndarray, H: np.ndarray) -> np.ndarray:
        pt    = np.array([pixel_pt[0], pixel_pt[1], 1.0])
        world = H @ pt
        world = world / world[2]
        return world[:2]
 
    def _update_distance_and_speed(self, track_id: int, frame_id: int, world_pt: np.ndarray):
        if track_id in self._last_position:
            prev_pt    = self._last_position[track_id]
            prev_frame = self._last_frame[track_id]
 
            if frame_id > prev_frame:
                # distance in template units → convert to metres
                dist_units = float(np.linalg.norm(world_pt - prev_pt))
                dist_m     = dist_units * self._world_to_metres
                self.player_distances[track_id] += dist_m
 
                dt = (frame_id - prev_frame) / self.fps
                if dt > 0:
                    speed_ms = dist_m / dt          # m/s
                    self.player_speeds[track_id].append((frame_id, speed_ms))
 
        self._last_position[track_id] = world_pt
        self._last_frame[track_id]    = frame_id   # fixed: was overwriting the whole dict
 
    def _update_heatmap(self, world_pt: np.ndarray):
        x, y = world_pt
        xi = int(x / self.heatmap_resolution)
        yi = int(y / self.heatmap_resolution)
        if 0 <= xi < self.heatmap.shape[1] and 0 <= yi < self.heatmap.shape[0]:
            self.heatmap[yi, xi] += 1
 
    def get_player_heatmap(self, track_id: Optional[int] = None, blur_sigma: float = 1.5) -> np.ndarray:
        if self.heatmap is None:
            raise ValueError("Court dimensions not set. Call set_court_dimensions() or set_court_dimensions_from_template() first.")
 
        if track_id is None:
            hm = self.heatmap.copy()
        else:
            hm = np.zeros_like(self.heatmap)
            for _, x, y in self.player_positions.get(track_id, []):
                xi = int(x / self.heatmap_resolution)
                yi = int(y / self.heatmap_resolution)
                if 0 <= xi < hm.shape[1] and 0 <= yi < hm.shape[0]:
                    hm[yi, xi] += 1
 
        if blur_sigma > 0:
            k  = int(blur_sigma * 3) * 2 + 1
            hm = cv2.GaussianBlur(hm, (k, k), blur_sigma)
 
        return hm
 
    def get_team_heatmap(self, team_id: int, blur_sigma: float = 1.5) -> np.ndarray:
        if self.heatmap is None:
            raise ValueError("Court dimensions not set. Call set_court_dimensions() or set_court_dimensions_from_template() first.")
 
        hm = np.zeros_like(self.heatmap)
        for tid, positions in self.player_positions.items():
            if self.player_teams.get(tid, -1) == team_id:
                for _, x, y in positions:
                    xi = int(x / self.heatmap_resolution)
                    yi = int(y / self.heatmap_resolution)
                    if 0 <= xi < hm.shape[1] and 0 <= yi < hm.shape[0]:
                        hm[yi, xi] += 1
 
        if blur_sigma > 0:
            k  = int(blur_sigma * 3) * 2 + 1
            hm = cv2.GaussianBlur(hm, (k, k), blur_sigma)
 
        return hm
 
    def get_stats_dataframe(self) -> pd.DataFrame:
        rows = []
 
        for tid in self.player_positions.keys():
            positions = self.player_positions[tid]
            if not positions:
                continue
 
            # fixed: speeds are (frame_id, speed_m/s) tuples
            speeds    = [s for _, s in self.player_speeds.get(tid, [])]
            avg_speed = float(np.mean(speeds)) * 3.6 if speeds else 0.0   # km/h
            max_speed = float(np.max(speeds))  * 3.6 if speeds else 0.0   # km/h
 
            pts  = np.array([[x, y] for _, x, y in positions])
            area = 0.0
            if len(pts) >= 3:
                hull = cv2.convexHull(pts.astype(np.float32))
                area = float(cv2.contourArea(hull)) * (self._world_to_metres ** 2)
 
            total_time = (positions[-1][0] - positions[0][0]) / self.fps if len(positions) > 1 else 0.0
 
            rows.append({
                "track_id":             tid,
                "total_distance (m)":   self.player_distances[tid],
                "avg_speed (km/h)":     avg_speed,
                "max_speed (km/h)":     max_speed,
                "movement_area (m^2)":  area,
                "active_time (s)":      total_time,
                "num_frames":           len(positions),
            })
 
        return pd.DataFrame(rows)
 
    def export_positions_csv(self, output_path: str):
        rows = []
        for tid, pos_list in self.player_positions.items():
            for frame, x, y in pos_list:
                rows.append({"track_id": tid, "frame": frame, "x_world": x, "y_world": y})
        pd.DataFrame(rows).to_csv(output_path, index=False)
 
    def get_keypoints_stats(self) -> Dict:
        stats = {}
        for tid, kpt_list in self.player_keypoints.items():
            if not kpt_list:
                continue
            kpt_arrays = [kpts for _, kpts in kpt_list]
            stacked    = np.stack(kpt_arrays, axis=0)
            stats[tid] = {
                "keypoints_mean": np.mean(stacked, axis=0).tolist(),
                "std_keypoints":  np.std(stacked,  axis=0).tolist(),
                "ranges":         (np.max(stacked, axis=0) - np.min(stacked, axis=0)).tolist(),
                "num_frames":     len(kpt_list),
            }
            
        return stats