import numpy as np
from typing import List, Any

from .base_detector import BaseDetector

class ToyDetector(BaseDetector):
    def __init__(self):
        self.x = 50
        self.y = 50

    def detect(self, image: np.ndarray) -> List[Any]:
        self.x = (self.x + 5) % image.shape[1]  # Volta pro começo se bater na borda da largura
        self.y = (self.y + 2) % image.shape[0]  # Volta pro começo se bater na borda da altura
        
        x1, y1 = self.x, self.y
        x2, y2 = self.x + 60, self.y + 120 # Largura de 60 e altura de 120 (proporção de uma pessoa)
        conf = 0.95
        class_id = 0 # 0 para jogador
        
        return [[x1, y1, x2, y2, conf, class_id]]