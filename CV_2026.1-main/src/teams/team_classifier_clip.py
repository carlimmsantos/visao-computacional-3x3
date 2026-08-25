from __future__ import annotations

import pickle
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .base_team_classifier import BaseTeamClassifier
import open_clip


class TeamClassifierCLIP(BaseTeamClassifier):
    """Classifica jogadores baseado em text-prompts CLIP matching.

    Este classificador utiliza o modelo CLIP pré-treinado para extrair
    embeddings de alta dimensionalidade das camisas dos jogadores e
    compara-os com embeddings textuais de prompts que descrevem cores
    de times. A classificação é feita por votação dos resultados de
    múltiplos prompts textuais, eliminando a necessidade de clusterização.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Inicializa o classificador CLIP com matching textual.

        Args:
            model_name: Nome do modelo CLIP (ex: "ViT-B/32", "ViT-L/14")
            device: Dispositivo para computação ("cuda" ou "cpu")
        """
        self.model_name = model_name
        self.device = device

        # Carregar modelo CLIP pré-treinado
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai"
        )
        self.model: Any = model
        self.preprocess: Any = preprocess
        self.tokenizer: Any = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Prompts textuais para descrição de cores de times
        # Será inicializado no método fit()
        self.team0_prompts: List[str] = []
        self.team1_prompts: List[str] = []
        self.team0_embeddings: torch.Tensor | None = None
        self.team1_embeddings: torch.Tensor | None = None

    @property
    def n_clusters(self) -> int:
        """Propriedade para compatibilidade com calibrador.

        O novo método baseado em prompts não precisa de clustering,
        mas mantemos esta propriedade para compatibilidade.
        """
        return 1

    def _get_default_prompts(self) -> Tuple[List[str], List[str]]:
        """Retorna prompts padrão para cores de times.

        Returns:
            Tupla (team0_prompts, team1_prompts) com descrições variadas
        """
        # Prompts para time 0 (white and black)
        team0_prompts = [
            "a white and black jersey",
            "a white jersey with black stripes",
            "black and white striped uniform",
            "a striped white and black shirt",
            "black stripes on white jersey",
            "white and black sports uniform",
            "a white shirt with black",
            "white dominant with black accents",
        ]

        # Prompts para time 1 (purple)
        team1_prompts = [
            "a purple jersey shirt",
            "purple sports uniform",
            "purple athletic wear",
            "a purple striped jersey",
            "a shirt in purple color",
            "dark purple jersey",
            "purple team clothing",
            "purple sports shirt",
        ]

        return team0_prompts, team1_prompts

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Codifica lista de prompts textuais em embeddings CLIP.

        Args:
            prompts: Lista de descrições textuais

        Returns:
            Tensor com embeddings normalizados (shape: [n_prompts, embedding_dim])
        """
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def _extract_image_embedding(self, crop: np.ndarray) -> torch.Tensor | None:
        """Extrai embedding CLIP de uma imagem de camisa.

        Args:
            crop: Imagem da camisa (BGR, numpy array)

        Returns:
            Tensor com embedding normalizado (shape: [embedding_dim]) ou None
        """
        # Máscara (fundo preto)
        mask = np.any(crop > 0, axis=-1).astype(np.uint8)

        if cv2.countNonZero(mask) == 0:
            return None

        # Recorte pela bounding box
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        crop = crop[y:y+h, x:x+w]

        # Pequeno padding preto
        pad = 5
        crop = cv2.copyMakeBorder(
            crop, pad, pad, pad, pad,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # BGR -> RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # CLIP preprocess
        pil_image = Image.fromarray(rgb_crop)
        image_tensor: torch.Tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # Embedding
        with torch.no_grad():
            embedding: torch.Tensor = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.squeeze(0)

    def fit(
        self,
        crops: List[np.ndarray],
        team0_prompts: List[str] | None = None,
        team1_prompts: List[str] | None = None,
    ) -> None:
        """Inicializa o classificador com prompts textuais.

        Este método não treina nada no sentido tradicional, apenas
        codifica os prompts textuais que serão usados para classificação.

        Args:
            crops: Lista de imagens de camisas (não usada no novo método)
            team0_prompts: Prompts customizados para time 0 ou None para usar padrão
            team1_prompts: Prompts customizados para time 1 ou None para usar padrão
        """
        if team0_prompts is None or team1_prompts is None:
            default_team0, default_team1 = self._get_default_prompts()
            team0_prompts = team0_prompts or default_team0
            team1_prompts = team1_prompts or default_team1

        self.team0_prompts = team0_prompts
        self.team1_prompts = team1_prompts

        # Codificar prompts
        self.team0_embeddings = self._encode_prompts(team0_prompts)
        self.team1_embeddings = self._encode_prompts(team1_prompts)

    def predict_with_confidence(
        self, crops: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediz a classificação usando voting sobre prompts textuais.

        Args:
            crops: Lista de imagens para classificar

        Returns:
            Tupla (predictions, confidences) onde:
            - predictions: Array com IDs dos times (0 ou 1)
            - confidences: Array com scores de confiança (0.0 a 1.0)

        Raises:
            RuntimeError: Se o modelo não foi inicializado (fit não chamado)
        """
        if self.team0_embeddings is None or self.team1_embeddings is None:
            raise RuntimeError(
                "TeamClassifierCLIP nao foi inicializado. Chame fit antes de predict."
            )

        if not crops:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        predictions = []
        confidences = []

        for crop in crops:
            image_embedding = self._extract_image_embedding(crop)

            if image_embedding is None:
                # Camisa vazia - classificar como time 0 com baixa confiança
                predictions.append(0)
                confidences.append(0.0)
                continue

            # Calcular similaridades com prompts de cada time
            # (cosine similarity = dot product já que embeddings são normalizados)
            similarities_team0 = torch.nn.functional.cosine_similarity(
                image_embedding.unsqueeze(0),
                self.team0_embeddings
            )  # shape: [n_prompts_team0]

            similarities_team1 = torch.nn.functional.cosine_similarity(
                image_embedding.unsqueeze(0),
                self.team1_embeddings
            )  # shape: [n_prompts_team1]

            # Classificar cada prompt: aquele com maior similaridade vence
            team0_votes = (similarities_team0 > similarities_team1).sum().item()
            team1_votes = (similarities_team1 >= similarities_team0).sum().item()

            # Votação por moda
            if team0_votes > team1_votes:
                prediction = 0
                confidence = float(team0_votes) / (team0_votes + team1_votes)
            else:
                prediction = 1
                confidence = float(team1_votes) / (team0_votes + team1_votes)

            predictions.append(prediction)
            confidences.append(confidence)

        return np.array(predictions, dtype=np.int32), np.array(confidences, dtype=np.float32)

    def save(self, filepath: str) -> None:
        """Salva os prompts textuais em disco.

        Args:
            filepath: Caminho para salvar os prompts

        Raises:
            RuntimeError: Se não há prompts inicializados
        """
        if not self.team0_prompts or not self.team1_prompts:
            raise RuntimeError("Nao ha prompts inicializados para salvar.")

        state = {
            'team0_prompts': self.team0_prompts,
            'team1_prompts': self.team1_prompts,
            'model_name': self.model_name,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        """Carrega prompts textuais previamente salvos do disco.

        Args:
            filepath: Caminho do arquivo com os prompts
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.team0_prompts = state.get('team0_prompts', [])
        self.team1_prompts = state.get('team1_prompts', [])
        self.model_name = state.get('model_name', self.model_name)

        # Codificar prompts carregados
        if self.team0_prompts and self.team1_prompts:
            self.team0_embeddings = self._encode_prompts(self.team0_prompts)
            self.team1_embeddings = self._encode_prompts(self.team1_prompts)
