from __future__ import annotations

import pickle
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .base_team_classifier import BaseTeamClassifier

class TeamClassifierHSV(BaseTeamClassifier):
    """Classifica jogadores agrupando histogramas HSV 2D (Matiz e Saturação) das camisas."""

    def __init__(
        self,
        n_clusters: int = 2,
        h_bins: int = 32,
        s_bins: int = 16,
        random_state: int = 42,
    ):
        if n_clusters != 2:
            raise ValueError("Este classificador foi configurado para exatamente 2 times.")

        self.n_clusters = n_clusters
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.random_state = random_state

        self.kmeans = None

    def _extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 0), dtype=np.float32)

        features = []

        for crop in crops:
            # 1. Recuperar a máscara usando o fundo preto deixado pela segmentação
            mask = np.any(crop > 0, axis=-1).astype(np.uint8) * 255

            if cv2.countNonZero(mask) == 0:
                mask = None 

            # 2. Converter de BGR para HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # 3. Calcular o Histograma 2D ignorando o canal de brilho (Value)
            hist = cv2.calcHist(
                [hsv], 
                channels=[0, 1], 
                mask=mask, 
                histSize=[self.h_bins, self.s_bins], 
                ranges=[0, 180, 0, 256]
            )

            # 4. Normalizar o histograma 
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L2)
            
            # 5. Achatar o histograma 2D em um vetor 1D
            features.append(hist.flatten())

        return np.array(features, dtype=np.float32)

    def fit(self, crops: List[np.ndarray]) -> None:
        if len(crops) < self.n_clusters:
            raise ValueError("Numero de crops insuficiente para treinar o agrupamento de times.")

        features = self._extract_features(crops)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans.fit(features)

    def predict_with_confidence(self, crops: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna a predicao e um score de confianca baseado na distancia do cluster KMeans."""
        if self.kmeans is None:
            raise RuntimeError("TeamClassifierHSV ainda nao foi treinado. Chame fit antes de predict.")

        if not crops:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        features = self._extract_features(crops)
        
        # KMeans.transform retorna as distancias para todos os centros de cluster
        distances = self.kmeans.transform(features)
        
        # O cluster escolhido é aquele com a menor distância
        predictions = np.argmin(distances, axis=1).astype(np.int32)
        
        # Converte a distância em uma métrica de confiança (0.0 a 1.0)
        min_distances = np.min(distances, axis=1)
        confidences = 1.0 / (1.0 + min_distances)

        return predictions, confidences

    def save(self, filepath: str) -> None:
        """Salva o modelo treinado (KMeans) e parametros relevantes."""
        if self.kmeans is None:
            raise RuntimeError("Nao ha modelo treinado para salvar.")
        
        state = {
            'kmeans': self.kmeans,
            'h_bins': self.h_bins,
            's_bins': self.s_bins
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        """Restaura o modelo a partir do arquivo."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.kmeans = state.get('kmeans')
        self.h_bins = state.get('h_bins', self.h_bins)
        self.s_bins = state.get('s_bins', self.s_bins)