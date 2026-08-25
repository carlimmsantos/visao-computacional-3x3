import cv2
from tqdm import tqdm

from typing import List, Any

from .pipeline_step import PipelineStep
from .frame_data import FrameData



class VideoProcessor:
    def __init__(self, pipeline_steps: List[PipelineStep]):
        self.pipeline_steps = pipeline_steps

    def run(self, input_path: str, output_path: str, show_video: bool = False):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Erro ao abrir o vídeo.")

        # Configurações do VideoWriter
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_to_process = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count  = 0
        for i in tqdm(range(frames_to_process), desc="Processando vídeo", total=frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Empacota o frame no nosso Objeto de Estado
            frame_data = FrameData(frame_id=frame_count, image=frame, fps=fps)

            # 2. Passa o frame pela Chain of Responsibility
            for step in self.pipeline_steps:
                frame_data = step.process(frame_data)

            # 3. Grava o frame processado (anotado pelo DrawStep)
            out.write(frame_data.image)
            frame_count += 1

            if show_video:
                cv2.imshow('Processing', frame_data.image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 4. Finaliza cada etapa do pipeline (lifecycle hooks)
        for step in self.pipeline_steps:
            if hasattr(step, "finish") and callable(step.finish):
                step.finish()