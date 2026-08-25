"""Roda Detector(RF-DETR) + ByteTrack + ShootingDetection + DrawStep num video.

Pipeline para validar visualmente a deteccao de movimento de arremesso:
esqueletos de pose por jogador, destaque amarelo (SHOOTING) quando o
classificador temporal liga, e overlay opcional de features/score para
calibrar os thresholds do ShootingRuleConfig.

Uso:
    uv run python scripts/run_shooting_detection_on_video.py --input videos/meu_video.mp4
    uv run python scripts/run_shooting_detection_on_video.py --input videos/meu_video.mp4 --debug-features
    uv run python scripts/run_shooting_detection_on_video.py --input videos/meu_video.mp4 --output out.mp4 --enter 0.55

Sem --output, o resultado sai ao lado do input com sufixo _sd
(videos/meu_video.mp4 -> videos/meu_video_sd.mp4).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# scripts/ esta fora do PYTHONPATH default — adiciona a raiz do projeto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.actions import (
    PoseFeatureExtractor,
    RuleBasedShootingClassifier,
    ShootingDetectionStep,
    ShootingStateManager,
)
from src.core.draw_step import DrawStep
from src.core.frame_data import FrameData
from src.detectors.players.classes import CLASS_NAMES, PLAYER_CLASS_IDS
from src.detectors.players.detection_step import DetectionStep
from src.detectors.players.rfdetr_detector import RFDETRDetector
from src.pose import DEFAULT_POSE_WEIGHTS, PoseEstimator
from src.trackers.bytetrack_tracker import ByteTrackTracker
from src.trackers.tracking_step import TrackingStep

# Sufixo do arquivo de saida derivado do input (video.mp4 -> video_sd.mp4)
OUTPUT_SUFFIX = "_sd"

CHECKPOINT_CANDIDATES = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
)


def resolve_weights(run_dir: Path) -> Path:
    for candidate in CHECKPOINT_CANDIDATES:
        path = run_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Nenhum checkpoint encontrado em {run_dir}. "
        f"Esperado um de: {', '.join(CHECKPOINT_CANDIDATES)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: mesmo local do input com sufixo _sd (video.mp4 -> video_sd.mp4).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Pesos do RF-DETR. Default: melhor checkpoint em --run-dir.",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/rfdetr_maniacos"))
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate passado para o ByteTrack (afeta lost_track_buffer).",
    )
    parser.add_argument(
        "--pose-weights", type=Path, default=Path(DEFAULT_POSE_WEIGHTS)
    )
    parser.add_argument(
        "--window", type=int, default=24, help="Janela temporal em frames (~0.8s @ 30fps)."
    )
    parser.add_argument("--enter", type=float, default=0.6, help="Score para ligar o estado SHOOTING.")
    parser.add_argument("--exit", type=float, default=0.35, help="Score para desligar o estado SHOOTING.")
    parser.add_argument(
        "--debug-features",
        action="store_true",
        help="Overlay de angulos/altura do punho/score por jogador (calibracao).",
    )
    return parser.parse_args()


def draw_feature_overlay(
    data: FrameData, extractor: PoseFeatureExtractor, state_manager: ShootingStateManager
) -> None:
    """Escreve features e score ao lado do bbox de cada jogador com pose."""
    for idx, det in enumerate(data.detections):
        if idx >= len(data.poses) or data.poses[idx] is None:
            continue
        f = extractor.extract(data.poses[idx])
        if not f.valid:
            continue

        track_id = int(det[6]) if len(det) > 6 else -1
        flag, score = state_manager.get_last_state(track_id)

        def nanreduce(fn, values):
            return np.nan if np.isnan(values).all() else float(fn(values))

        elbow = nanreduce(np.nanmax, [f.elbow_angle_left, f.elbow_angle_right])
        knee = nanreduce(np.nanmin, [f.knee_angle_left, f.knee_angle_right])
        wrist = nanreduce(np.nanmax, [f.wrist_rel_height_left, f.wrist_rel_height_right])
        lines = [
            f"EL {elbow:.0f} KN {knee:.0f}" if np.isfinite(elbow) and np.isfinite(knee) else "EL/KN --",
            f"WR {wrist:+.2f}" if np.isfinite(wrist) else "WR --",
            f"S {score:.2f} {'ON' if flag else 'off'}",
        ]

        x2, y1 = int(det[2]), int(det[1])
        for i, line in enumerate(lines):
            y_text = y1 + 14 + i * 14
            cv2.putText(data.image, line, (x2 + 4, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(data.image, line, (x2 + 4, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: video de entrada nao encontrado: {args.input}", file=sys.stderr)
        return 1

    if args.output is None:
        args.output = args.input.with_stem(args.input.stem + OUTPUT_SUFFIX)

    weights = args.weights or resolve_weights(args.run_dir)
    print(f"[*] weights:      {weights}")
    print(f"[*] pose weights: {args.pose_weights}")
    print(f"[*] input:        {args.input}")
    print(f"[*] output:       {args.output}")
    print(f"[*] confidence:   {args.conf}")
    print(f"[*] window/enter/exit: {args.window}/{args.enter}/{args.exit}")

    detector = RFDETRDetector(weights_path=str(weights), confidence=args.conf)
    # RF-DETR carregado a partir de .pth nao garante class_names persistido —
    # injetamos as 11 classes do dataset para o DrawStep.
    detector.class_names = CLASS_NAMES

    tracker = ByteTrackTracker(frame_rate=args.fps)

    pose_estimator = PoseEstimator(weights_path=str(args.pose_weights))
    feature_extractor = PoseFeatureExtractor()
    state_manager = ShootingStateManager(
        classifier=RuleBasedShootingClassifier(),
        window_size=args.window,
        enter_score=args.enter,
        exit_score=args.exit,
    )
    shooting_detection = ShootingDetectionStep(
        pose_estimator=pose_estimator,
        state_manager=state_manager,
        feature_extractor=feature_extractor,
        player_class_ids=list(PLAYER_CLASS_IDS),
    )

    pipeline = [
        DetectionStep(detector),
        TrackingStep(tracker),
        shooting_detection,
        DrawStep(draw_segmentation=False, minimap_image_path=None, draw_poses=True),
    ]

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"ERROR: nao foi possivel abrir {args.input}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or float(args.fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV no WSL geralmente nao tem encoder H.264; escrevemos com mp4v num
    # arquivo temporario e re-encodamos para H.264 via ffmpeg ao final (se
    # disponivel). H.264 e necessario para previews em VS Code / navegadores.
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        tmp_output = args.output.with_suffix(".mp4v.tmp.mp4")
    else:
        tmp_output = args.output
        print("[!] ffmpeg nao encontrado — saida ficara em mp4v (use VLC para abrir).")
    writer = cv2.VideoWriter(
        str(tmp_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        print("ERROR: cv2.VideoWriter nao abriu.", file=sys.stderr)
        return 1

    print(f"[*] processando {total_frames} frames ({width}x{height} @ {fps:.1f} fps)")

    frame_id = 0
    shooting_frames = 0
    start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = FrameData(frame_id=frame_id, image=frame)
        for step in pipeline:
            data = step.process(data)

        if any(data.shooting_flags):
            shooting_frames += 1
        if args.debug_features:
            draw_feature_overlay(data, feature_extractor, state_manager)

        writer.write(data.image)
        frame_id += 1

        if frame_id % 30 == 0 or frame_id == total_frames:
            elapsed = time.perf_counter() - start
            print(
                f"    frame {frame_id}/{total_frames}  "
                f"({elapsed:.1f}s elapsed, {frame_id / elapsed:.1f} fps)"
            )

    cap.release()
    writer.release()

    total_elapsed = time.perf_counter() - start
    print(f"[*] inferencia concluida em {total_elapsed:.1f}s ({frame_id / total_elapsed:.1f} fps)")
    print(f"[*] frames com SHOOTING ativo: {shooting_frames}/{frame_id}")

    if ffmpeg_path and tmp_output != args.output:
        print(f"[*] re-encodando para H.264 com ffmpeg")
        cmd = [
            ffmpeg_path, "-y", "-loglevel", "error",
            "-i", str(tmp_output),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart",
            str(args.output),
        ]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            tmp_output.unlink(missing_ok=True)
            print(f"[*] saida final: {args.output}")
        else:
            print(
                f"[!] ffmpeg falhou (rc={result.returncode}); mantendo "
                f"{tmp_output} em mp4v.",
                file=sys.stderr,
            )
    else:
        print(f"[*] saida final: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
