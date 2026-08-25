import numpy as np

from abc import ABC, abstractmethod
from typing import List, Any


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Any]:
        """Recebe uma imagem e retorna uma lista de detecções (ex: bounding boxes)."""
        pass

    def track(self, image: np.ndarray) -> List[Any]:
        """Rastreia objetos no frame.

        Implementacao padrao: fallback para detect quando o detector nao
        implementa tracking nativo.
        """
        return self.detect(image)