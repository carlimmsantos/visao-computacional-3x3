import argparse
import time
from typing import Any, List

import cv2

from src.core.draw_step import DrawStep
from src.core.frame_data import FrameData
from src.detectors.players.rfdetr_detector import RFDETRDetector
from src.detectors.players.yolo26_detector import YOLO26Detector


DISPLAY_ZOOM = 0.5


def annotate_frame(
    frame: Any,
    frame_id: int,
    detections: List[Any],
    class_names: Any,
    title: str,
    elapsed_ms: float,
) -> Any:
    data = FrameData(
        frame_id=frame_id,
        image=frame.copy(),
        detections=detections,
        class_names=class_names,
    )
    DrawStep().process(data)

    cv2.putText(
        data.image,
        f"{title}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        data.image,
        f"Deteccoes: {len(detections)} | Tempo: {elapsed_ms:.1f} ms",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return data.image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara YOLO26 e RF-DETR lado a lado, frame por frame.",
    )
    parser.add_argument(
        "--input",
        default="videos/basquete.mp4",
        help="Caminho do video de entrada.",
    )
    parser.add_argument(
        "--window-name",
        default="Comparacao: YOLO26 x RF-DETR",
        help="Nome da janela de exibicao.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    yolo_detector = YOLO26Detector()
    rfdetr_detector = RFDETRDetector()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise ValueError(f"Erro ao abrir o video: {args.input}")

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        yolo_detections = yolo_detector.detect(frame)
        yolo_time = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        rfdetr_detections = rfdetr_detector.detect(frame)
        rfdetr_time = (time.perf_counter() - start) * 1000.0

        yolo_frame = annotate_frame(
            frame=frame,
            frame_id=frame_id,
            detections=yolo_detections,
            class_names=yolo_detector.class_names,
            title="YOLO26",
            elapsed_ms=yolo_time,
        )

        rfdetr_frame = annotate_frame(
            frame=frame,
            frame_id=frame_id,
            detections=rfdetr_detections,
            class_names=rfdetr_detector.class_names,
            title="RF-DETR",
            elapsed_ms=rfdetr_time,
        )

        combined = cv2.hconcat([yolo_frame, rfdetr_frame])
        combined_display = cv2.resize(
            combined,
            None,
            fx=DISPLAY_ZOOM,
            fy=DISPLAY_ZOOM,
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow(args.window_name, combined_display)

        key = cv2.waitKey(0) & 0xFF
        # q ou ESC para sair; qualquer outra tecla avanca para o proximo frame.
        if key in (ord("q"), 27):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
