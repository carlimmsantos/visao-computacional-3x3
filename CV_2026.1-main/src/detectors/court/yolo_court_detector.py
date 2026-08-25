import os
from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO

from .base_court_detector import BaseCourtDetector


class YOLOCourtDetector(BaseCourtDetector):
    """Detecta pontos-chave da quadra usando um modelo YOLO (Pose) treinado localmente."""

    def __init__(
        self,
        weights_path: str = "models/court_keypoint/yolo26l-court-v1a-teste.pt",
        confidence: float = 0.7,
    ):
        """
        Parâmetros
        ----------
        weights_path : str
            Caminho para o arquivo de pesos (.pt) do modelo YOLO treinado.
        confidence : float
            Limiar mínimo de confiança para filtrar keypoints (padrão: 0.7).
        """
        resolved_path = Path(weights_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Pesos do modelo da quadra não encontrados: {resolved_path}")

        self.confidence = confidence
        
        # Inicializa o modelo YOLO localmente
        print(f"[*] Carregando YOLOCourtDetector ({weights_path})...")
        self.model = YOLO(str(resolved_path))

    def extract_keypoints(self, image: np.ndarray) -> List[tuple]:
        """Executa inferência local e retorna os keypoints da melhor detecção."""
        
        results = self.model.predict(source=image, conf=self.confidence, verbose=False)

        # Retorna vazio se nada for detectado ou se não for um modelo de pose
        if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return []

        # 1. Avalia as confianças gerais dos objetos (Bounding Boxes) detectados
        box_confs = results[0].boxes.conf.cpu().numpy()
        
        if len(box_confs) == 0:
            return []

        # 2. Pega o índice da detecção com a MAIOR confiança
        best_idx = np.argmax(box_confs)

        # 3. Extrai apenas os keypoints correspondentes à melhor quadra
        kpts = results[0].keypoints
        best_xy = kpts.xy[best_idx].cpu().numpy()
        best_kpt_confs = kpts.conf[best_idx].cpu().numpy() if kpts.conf is not None else np.ones(len(best_xy))

        extracted_keypoints = []
        
        # print('-='*30 + 'EXTRAINDO KEYPOINTS DA MELHOR QUADRA' + '-='*30)
        for class_id, ((x, y), kpt_conf) in enumerate(zip(best_xy, best_kpt_confs)):
            # print(class_id, (x, y), kpt_conf)
            # Usa a mesma lógica de filtro, mas agora garantidamente na melhor quadra
            if kpt_conf > self.confidence and (x > 0.0 or y > 0.0):
                extracted_keypoints.append((float(x), float(y), int(class_id), float(kpt_conf)))

        return extracted_keypoints