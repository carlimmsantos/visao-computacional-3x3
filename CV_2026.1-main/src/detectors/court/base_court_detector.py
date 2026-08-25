import numpy as np

from abc import ABC, abstractmethod
from typing import List

class BaseCourtDetector(ABC):
    @abstractmethod
    def extract_keypoints(self, image: np.ndarray) -> List[tuple]:
        """Encontra os pontos-chave da quadra (ex: cantos do garrafão)."""
        pass