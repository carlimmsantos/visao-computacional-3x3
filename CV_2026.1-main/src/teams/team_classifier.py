from __future__ import annotations

import pickle
from typing import List, Tuple

import cv2
import numpy as np
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel


class TeamClassifier:
    """Classifica jogadores em 2 times com SigLIP Backbone + UMAP + KMeans."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
        batch_size: int = 32,
        n_clusters: int = 2,
        umap_components: int = 3,
        random_state: int = 42,
    ):
        if n_clusters != 2:
            raise ValueError("Este classificador foi configurado para exatamente 2 times.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.umap_components = umap_components
        self.random_state = random_state

        self.processor = AutoProcessor.from_pretrained(model_name)
        # Load strictly the Vision Backbone, skipping the text projection head!
        self.model = SiglipVisionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.reducer = None
        self.kmeans = None

    @staticmethod
    def _to_pil_image(crop: np.ndarray) -> Image.Image:
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_crop)

    def _extract_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch_crops = crops[start:start + self.batch_size]
                images = [self._to_pil_image(crop) for crop in batch_crops]

                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                
                # Average the raw visual patches (dim=1) to keep texture/color data
                image_features = torch.mean(outputs.last_hidden_state, dim=1)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
                
                all_embeddings.append(image_features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def fit(self, crops: List[np.ndarray]) -> None:
        if len(crops) < self.n_clusters:
            raise ValueError("Numero de crops insuficiente.")

        embeddings = self._extract_embeddings(crops)

        n_neighbors = max(2, min(15, len(crops) - 1))
        self.reducer = umap.UMAP(
            n_components=self.umap_components,
            n_neighbors=n_neighbors,
            metric="cosine",
            random_state=self.random_state,
        )
        reduced_embeddings = self.reducer.fit_transform(embeddings)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans.fit(reduced_embeddings)

    def predict_with_confidence(self, crops: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna a predicao e um score de confianca baseado na distancia do centro do cluster."""
        if self.reducer is None or self.kmeans is None:
            raise RuntimeError("TeamClassifier nao treinado/carregado.")

        if not crops:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        embeddings = self._extract_embeddings(crops)
        reduced_embeddings = self.reducer.transform(embeddings)
        
        # Calculate distance to cluster centers
        distances = self.kmeans.transform(reduced_embeddings)
        predictions = np.argmin(distances, axis=1).astype(np.int32)
        
        # Convert minimum distance to a confidence weight (closer to 0 = higher confidence)
        min_distances = np.min(distances, axis=1)
        confidences = 1.0 / (1.0 + min_distances)

        return predictions, confidences

    def save(self, filepath: str) -> None:
        if self.reducer is None or self.kmeans is None:
            raise RuntimeError("Nao ha modelo para salvar.")
        with open(filepath, 'wb') as f:
            pickle.dump({'reducer': self.reducer, 'kmeans': self.kmeans}, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.reducer = data['reducer']
            self.kmeans = data['kmeans']