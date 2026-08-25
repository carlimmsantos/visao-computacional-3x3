import cv2
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, Union, Tuple

@dataclass
class FrameData:
    """Objeto que guarda a imagem e os metadados gerados a cada frame."""
    frame_id: int
    image: np.ndarray
    detections: List[Any] = field(default_factory=list)
    tracks: List[Any] = field(default_factory=list)
    court_keypoints: List[tuple] = field(default_factory=list)
    homography_matrix: Optional[np.ndarray] = None
    class_names: Optional[Union[Dict[int, str], List[str]]] = None
    team_ids: List[int] = field(default_factory=list)
    team_names: Optional[Dict[int, str]] = None
    segmentation_masks: List[Optional[np.ndarray]] = field(default_factory=list)
    poses: List[Optional[np.ndarray]] = field(default_factory=list)
    shooting_flags: List[bool] = field(default_factory=list)
    shooting_confidences: List[float] = field(default_factory=list)
    template_points: Optional[Dict[int, Tuple[float, float]]] = None
    fps: float = 30.0
    recorded_shots: List[Any] = field(default_factory=list)