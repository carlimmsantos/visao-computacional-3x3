from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from .base_team_classifier import BaseTeamClassifier


class TeamClassifierFashionCLIP(BaseTeamClassifier):
    """Classificador de times usando FashionCLIP fine-tuned."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.class_mapping = None
        self.class_texts = None  # Store class descriptions for inference
        self.num_classes = None

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load fine-tuned FashionCLIP model."""
        # Load CLIP model, NOT AutoModelForImageClassification
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Load class mapping
        mapping_path = Path(model_path).parent / "class_mapping.json"

        if mapping_path.exists():
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
                self.class_mapping = {int(k): v for k, v in mapping.items()}
                self.num_classes = len(self.class_mapping)

                # Pre-compute text embeddings for each class
                self.class_texts = {
                    idx: color for idx, color in self.class_mapping.items()
                }
                self._precompute_text_embeddings()
        else:
            print(f"Warning: class_mapping.json not found at {mapping_path}")
            self.class_mapping = {0: "white", 1: "purple and black"}
            self.class_texts = {0: "white", 1: "purple and black"}
            self.num_classes = 2
            self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        """Pre-compute normalized text embeddings for all classes."""
        self.text_embeddings = {}

        with torch.no_grad():
            for class_id, text in self.class_texts.items():
                text_inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    padding=True
                )
                text_inputs = {
                    k: v.to(self.device)
                    for k, v in text_inputs.items()
                    if k in ['input_ids', 'attention_mask']
                }
                text_outputs = self.model.get_text_features(**text_inputs)
                self.text_embeddings[class_id] = F.normalize(text_outputs, dim=-1)

    @property
    def n_clusters(self) -> int:
        return 1

    def _extract_and_classify_image(
        self,
        crop: np.ndarray,
    ) -> Tuple[int, float] | None:
        """Classify image using fine-tuned FashionCLIP."""

        if self.model is None:
            raise RuntimeError("Model not loaded.")

        # Mask of valid pixels
        mask = np.any(crop > 0, axis=-1).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            return None

        # Bounding box
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        crop = crop[y:y+h, x:x+w]

        # Crop focused on torso (same as training preprocessing)
        h_crop, w_crop = crop.shape[:2]
        y1 = int(h_crop * 0.15)
        y2 = int(h_crop * 0.65)
        x1 = int(w_crop * 0.15)
        x2 = int(w_crop * 0.85)
        crop = crop[y1:y2, x1:x2]

        # Padding (same as training)
        pad = 5
        crop = cv2.copyMakeBorder(
            crop, pad, pad, pad, pad,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # BGR -> RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)

        # Process image (same as training)
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt",
        ).to(self.device)

        # Inference using CLIP contrastive matching
        with torch.no_grad():
            # Get image embedding
            image_outputs = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            image_embedding = F.normalize(image_outputs, dim=-1)

            # Compare with all class text embeddings
            logits = []
            for class_id in sorted(self.text_embeddings.keys()):
                text_embedding = self.text_embeddings[class_id]
                similarity = torch.mm(image_embedding, text_embedding.t())
                logits.append(similarity)

            logits = torch.cat(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        # Map to team ID (0 or 1)
        class_name = self.class_mapping.get(pred_class, "").lower()
        team_id = 1 if ("purple" in class_name or "black" in class_name) else 0

        return team_id, confidence

    def fit(self, crops: List[np.ndarray], **kwargs) -> None:
        """Initialize model (compatibility)."""
        if self.model is None and self.model_path is not None:
            self._load_model(self.model_path)

    def predict_with_confidence(
        self,
        crops: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict team and confidence."""

        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Provide model_path in constructor or call fit() first."
            )

        if not crops:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        predictions = []
        confidences = []

        for crop in crops:
            result = self._extract_and_classify_image(crop)

            if result is None:
                predictions.append(0)
                confidences.append(0.0)
                continue

            team_id, confidence = result
            predictions.append(team_id)
            confidences.append(confidence)

        return np.array(predictions, dtype=np.int32), np.array(confidences, dtype=np.float32)

    def save(self, filepath: str) -> None:
        """Save model path and configuration."""
        state = {
            "model_path": self.model_path,
            "class_mapping": self.class_mapping,
            "class_texts": self.class_texts,
            "num_classes": self.num_classes
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        """Load model path and initialize with weights."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.model_path = state.get("model_path")
        self.class_mapping = state.get("class_mapping")
        self.class_texts = state.get("class_texts", self.class_mapping)
        self.num_classes = state.get("num_classes")

        if self.model_path:
            self._load_model(self.model_path)
