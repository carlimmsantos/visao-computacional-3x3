import os
from typing import List

import numpy as np
from inference_sdk import InferenceHTTPClient

from .base_court_detector import BaseCourtDetector


class RoboflowCourtDetector(BaseCourtDetector):
    """Detecta pontos-chave da quadra usando o modelo Roboflow basketball-court-detection-2/19.

    TEMPORÁRIO: esta classe será substituída por YOLOCourtDetector quando o modelo
    YOLO próprio estiver pronto. Ver src/detectors/yolo_court_detector.py.

    Usa o InferenceHTTPClient da inference_sdk para chamadas à API em nuvem,
    evitando conflitos de dependência com opencv/numpy.

    A chave da API pode ser passada diretamente ou lida da variável de ambiente
    ROBOFLOW_API_KEY.

    Exemplo:
        import os
        detector = RoboflowCourtDetector(api_key=os.environ["ROBOFLOW_API_KEY"])
    """

    MODEL_ID = "basketball-court-detection-2/19"
    API_URL = "https://detect.roboflow.com"

    def __init__(
        self,
        api_key: str | None = None,
        confidence: float = 0.7,
    ):
        """
        Parâmetros
        ----------
        api_key : str | None
            Chave da API Roboflow. Se None, lê de os.environ["ROBOFLOW_API_KEY"].
        confidence : float
            Limiar mínimo de confiança para filtrar keypoints (padrão: 0.7).
        """
        resolved_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key não fornecida. Passe api_key= ou defina a variável de "
                "ambiente ROBOFLOW_API_KEY."
            )

        self.confidence = confidence
        self._client = InferenceHTTPClient(api_url=self.API_URL, api_key=resolved_key)

    def extract_keypoints(self, image: np.ndarray) -> List[tuple]:
        """Executa inferência e retorna os keypoints filtrados por confiança.

        Parâmetros
        ----------
        image : np.ndarray
            Frame BGR no formato OpenCV.

        Retorna
        -------
        List[tuple]
            Lista de tuplas (x, y, class_id) onde class_id corresponde ao ID
            do keypoint no template da quadra (dst_homo.json).
        """
        try:
            result = self._client.infer(image, model_id=self.MODEL_ID)
        except Exception as exc:
            print(f"[RoboflowCourtDetector] falha na inferência: {exc}")
            return []

        predictions = result.get("predictions", [])
        if not predictions:
            return []

        keypoints = predictions[0].get("keypoints", [])

        return [
            (float(kp["x"]), float(kp["y"]), int(kp["class_id"]), float(kp.get("confidence", 0.0)))
            for kp in keypoints
            if kp.get("confidence", 0) > self.confidence
        ]
