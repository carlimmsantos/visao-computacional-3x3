"""Avalia o score_frame do classificador de arremesso com labels fracos do dataset.

Usa os bounding boxes anotados do basquete_combined: classes 5/6
(player-jump-shot / player-layup-dunk) como positivos e 3/4
(player / player-in-possession) como negativos. Para cada box: crop com
padding, pose (YOLO-pose), features geometricas e score_frame do
RuleBasedShootingClassifier; varre thresholds e imprime precision/recall/F1.

CAVEAT: labels fracos baseados em aparencia, anotados em ~340 frames esparsos
(~0.33s entre frames). Isto calibra o score INSTANTANEO e valida o encanamento
pose->features->score; NAO mede o classificador temporal de janela, que e o
que decide o estado SHOOTING em producao.

Uso:
    uv run python scripts/eval_shooting_on_dataset.py
    uv run python scripts/eval_shooting_on_dataset.py --split train --positive-classes 5
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# scripts/ esta fora do PYTHONPATH default — adiciona a raiz do projeto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.actions import PoseFeatureExtractor, RuleBasedShootingClassifier
from src.detectors.players.classes import PLAYER, PLAYER_IN_POSSESSION, SHOOTING_CLASS_IDS
from src.pose import DEFAULT_POSE_WEIGHTS, PoseEstimator

# Labels fracos: classes de arremesso vs jogadores "neutros". Shot-block e
# box-out ficam de fora por ambiguidade (bracos levantados sem arremesso).
DEFAULT_POSITIVE_CLASSES = ",".join(str(c) for c in sorted(SHOOTING_CLASS_IDS))
DEFAULT_NEGATIVE_CLASSES = ",".join(str(c) for c in (PLAYER, PLAYER_IN_POSSESSION))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/datasets/basquete_combined")
    )
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument(
        "--pose-weights", type=Path, default=Path(DEFAULT_POSE_WEIGHTS)
    )
    parser.add_argument(
        "--positive-classes", type=str, default=DEFAULT_POSITIVE_CLASSES,
        help="Classes tratadas como arremesso (CSV de ids).",
    )
    parser.add_argument(
        "--negative-classes", type=str, default=DEFAULT_NEGATIVE_CLASSES,
        help="Classes tratadas como nao-arremesso (CSV de ids).",
    )
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument(
        "--min-crop-height", type=int, default=64,
        help="Boxes menores sao pulados (mesmo guard do ShootingDetectionStep).",
    )
    return parser.parse_args()


def yolo_to_xyxy(line: str, width: int, height: int) -> tuple[int, np.ndarray]:
    parts = line.split()
    class_id = int(parts[0])
    cx, cy, w, h = (float(v) for v in parts[1:5])
    x1 = (cx - w / 2.0) * width
    y1 = (cy - h / 2.0) * height
    x2 = (cx + w / 2.0) * width
    y2 = (cy + h / 2.0) * height
    return class_id, np.array([x1, y1, x2, y2], dtype=np.float32)


def pad_and_clamp(box: np.ndarray, padding: float, width: int, height: int):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * (1.0 + padding), (y2 - y1) * (1.0 + padding)
    x1, y1 = max(0, int(cx - w / 2.0)), max(0, int(cy - h / 2.0))
    x2, y2 = min(width - 1, int(cx + w / 2.0)), min(height - 1, int(cy + h / 2.0))
    return x1, y1, x2, y2


def main() -> int:
    args = parse_args()
    positive_ids = {int(v) for v in args.positive_classes.split(",")}
    negative_ids = {int(v) for v in args.negative_classes.split(",")}
    wanted_ids = positive_ids | negative_ids

    labels_dir = args.dataset_dir / args.split / "labels"
    images_dir = args.dataset_dir / args.split / "images"
    if not labels_dir.exists():
        print(f"ERROR: labels nao encontrados em {labels_dir}", file=sys.stderr)
        return 1

    pose_estimator = PoseEstimator(weights_path=str(args.pose_weights))
    feature_extractor = PoseFeatureExtractor()
    classifier = RuleBasedShootingClassifier()

    scores, labels = [], []
    skipped_small = defaultdict(int)
    pose_failed = defaultdict(int)
    box_counts = defaultdict(int)

    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"[*] {len(label_files)} arquivos de label em {labels_dir}")
    print(f"[*] positivos: {sorted(positive_ids)} | negativos: {sorted(negative_ids)}")

    for label_file in label_files:
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / (label_file.stem + ext)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        crops, origins, inner_boxes, crop_labels = [], [], [], []
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            class_id, box = yolo_to_xyxy(line, img_w, img_h)
            if class_id not in wanted_ids:
                continue

            group = 1 if class_id in positive_ids else 0
            box_counts[group] += 1

            x1, y1, x2, y2 = pad_and_clamp(box, args.crop_padding, img_w, img_h)
            if (y2 - y1) < args.min_crop_height or x2 <= x1:
                skipped_small[group] += 1
                continue

            crops.append(image[y1:y2, x1:x2])
            origins.append((x1, y1))
            inner_boxes.append(box)
            crop_labels.append(group)

        if not crops:
            continue

        poses = pose_estimator.estimate_on_crops(crops, origins, inner_boxes=inner_boxes)
        for kpts, group in zip(poses, crop_labels):
            features = feature_extractor.extract(kpts)
            if kpts is None or not features.valid:
                pose_failed[group] += 1
                # Sem pose nao ha evidencia: conta como score 0 (negativo)
                scores.append(0.0)
                labels.append(group)
                continue
            scores.append(classifier.score_frame(features))
            labels.append(group)

    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    print(f"\n[*] boxes anotados: {box_counts[1]} positivos / {box_counts[0]} negativos")
    print(f"[*] pulados por tamanho (<{args.min_crop_height}px): "
          f"{skipped_small[1]} positivos / {skipped_small[0]} negativos")
    print(f"[*] pose falhou (None/torso invisivel): "
          f"{pose_failed[1]} positivos / {pose_failed[0]} negativos")
    print(f"[*] avaliados: {n_pos} positivos / {n_neg} negativos")
    if n_pos == 0 or n_neg == 0:
        print("ERROR: sem amostras suficientes para avaliar.", file=sys.stderr)
        return 1

    print(f"\n[*] score medio: positivos={scores[labels == 1].mean():.3f} "
          f"negativos={scores[labels == 0].mean():.3f}")

    print(f"\n{'thr':>5} {'prec':>6} {'rec':>6} {'f1':>6}   TP/FP/FN/TN")
    for threshold in np.arange(0.1, 0.95, 0.1):
        predictions = (scores >= threshold).astype(np.int32)
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        print(f"{threshold:>5.1f} {precision:>6.2f} {recall:>6.2f} {f1:>6.2f}   "
              f"{tp}/{fp}/{fn}/{tn}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
