from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaseTeamClassifier(ABC):
    """Interface comum para todos os classificadores de times."""
    
    # A interface exige que a classe defina quantos clusters ela usa
    n_clusters: int

    @abstractmethod
    def fit(self, crops: List[np.ndarray]) -> None:
        """Treina o classificador com uma lista de crops."""
        pass

    @abstractmethod
    def predict_with_confidence(self, crops: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna uma tupla contendo (predicoes, confiancas)."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Salva o estado treinado do modelo no disco."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Carrega o estado treinado do modelo do disco."""
        pass