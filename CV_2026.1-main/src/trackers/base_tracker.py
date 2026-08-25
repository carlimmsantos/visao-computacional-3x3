import numpy as np

from abc import ABC, abstractmethod
from typing import List, Any

class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: List[Any], image: np.ndarray) -> List[Any]:
        """Recebe detecções e atualiza os IDs de rastreamento."""
        pass