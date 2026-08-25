import os
import cv2
import numpy as np
from pathlib import Path

from tqdm import tqdm

from src.detectors.players.base_detector import BaseDetector
from src.segmentation.player_segmentation import PlayerSegmenter

class GameContextCalibrator:
    """Responsável por escanear a pasta do jogo e treinar o modelo SigLIP + KMeans."""
    def __init__(
        self, 
        game_context_folder: str, 
        detector: BaseDetector, 
        segmenter: PlayerSegmenter,
        player_class_ids: set[int]
    ):
        self.folder = game_context_folder
        self.detector = detector
        self.segmenter = segmenter
        self.player_class_ids = player_class_ids
        self.is_calibrated = False

    def calibrate(self, classifier) -> None:
        if self.is_calibrated:
            return

        print(f"[GameContextCalibrator] Calibrando times na pasta: {self.folder}")
        golden_crops = []
        
        if not os.path.exists(self.folder):
            raise FileNotFoundError(f"Pasta de calibração não encontrada: {self.folder}")

        for filename in tqdm(os.listdir(self.folder)):
            if Path(filename).suffix.lower() not in {".mp4", ".avi", ".mov"}:
                continue
                
            video_path = os.path.join(self.folder, filename)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            

            for progress in np.arange(0, 1, 0.1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * progress))
                ret, frame = cap.read()
                
                if not ret: continue
                
                detections = self.detector.detect(frame)
                segments = self.segmenter.get_segments(frame)
                
                for det in detections:
                    if det[4] < 0.65 or int(det[5]) not in self.player_class_ids:
                        continue
                        
                    box = np.array(det[:4], dtype=np.float32)
                    if (box[2] - box[0]) < 15 or (box[3] - box[1]) < 30:
                        continue
                        
                    scaled_box = self.segmenter._scale_box(box)
                    mask = self.segmenter.find_best_mask(scaled_box, segments)
                    crop = self.segmenter.crop_masked_image(frame, scaled_box, mask)
                    
                    if crop is not None:
                        golden_crops.append(crop)
                        
            cap.release()

        print(f"[GameContextCalibrator] Extraídos {len(golden_crops)} crops dourados. Treinando...")
        if len(golden_crops) >= classifier.n_clusters:
            classifier.fit(golden_crops)
            self.is_calibrated = True
            print("[GameContextCalibrator] Calibração concluída com sucesso.")
        else:
            self.is_calibrated = True 
            print("[GameContextCalibrator] AVISO CRÍTICO: Crops insuficientes. O modelo não vai classificar os times.")