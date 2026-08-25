"""
test_stats_pipeline.py
======================
Smoke-test for the full stats-tracking pipeline.
 
Usage
-----
    python test_stats_pipeline.py --video path/to/video.mp4
 
Optional overrides
------------------
    --detector-weights   models/detector/rfdetr-m-basketball.pth
    --court-weights      models/court_keypoint/best.pt
    --template           assets/dst_homo.json
    --court-width        28.0      (metres, FIBA full court)
    --court-length       15.0
    --fps                30
    --confidence         0.4
    --max-frames         300       (0 = process entire video)
    --output-csv         stats_output.csv
    --no-display                   (skip cv2 preview window)
"""
 
import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd  
import cv2
import numpy as np
from src.homography.homography_step import HomographyStep          # your HomographyStep
from src.detectors.players.rfdetr_detector   import RFDETRDetector           # your RFDETRDetector
from src.detectors.court.yolo_court_detector  import YOLOCourtDetector        # your YOLOCourtDetector
from src.stats.player_heatmap_stats      import PlayerHeatmapTracker     # the fixed tracker
from src.stats.stats_step import StatsStep
from src.core.frame_data import FrameData
 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline stats smoke-test")
    p.add_argument("--video",            required=True,  help="Path to input video")
    p.add_argument("--detector-weights", default="models/rfdetr-m-basketball.pth")
    p.add_argument("--court-weights",    default="models/court_keypoint/yolo26m-court-v8a.pt")
    p.add_argument("--template",         default="assets/dst_homo.json")
    p.add_argument("--court-width",      type=float, default=28.0,  help="Court width  (m)")
    p.add_argument("--court-length",     type=float, default=15.0,  help="Court length (m)")
    p.add_argument("--fps",              type=float, default=0.0,
                   help="Override FPS (0 = read from video)")
    p.add_argument("--confidence",       type=float, default=0.4)
    p.add_argument("--max-frames",       type=int,   default=0,
                   help="Stop after N frames (0 = full video)")
    p.add_argument("--output-csv",       default="outputs/stats/stats_summary.csv")
    p.add_argument("--no-display",       action="store_true",
                   help="Disable OpenCV preview window")
    return p.parse_args()
 
 
def check_file(path: str, label: str):
    if not Path(path).exists():
        print(f"[ERROR] {label} not found: {path}")
        sys.exit(1)
 
 
def draw_overlay(
    frame: np.ndarray,
    detections: list,
    homography_ok: bool,
    frame_id: int,
    fps_proc: float,
) -> np.ndarray:
    """Draw bounding boxes, track IDs and a status bar on the frame."""
    vis = frame.copy()
 
    color_ok  = (0, 200, 0)
    color_bad = (0, 0, 220)
 
    for det in detections:
        if len(det) < 7:
            continue
        x1, y1, x2, y2, conf, _, tid = det[:7]
        tid = int(tid)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color_ok, 2)
        cv2.putText(vis, f"#{tid}  {conf:.2f}",
                    (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_ok, 1, cv2.LINE_AA)
 
    hom_label = "H: OK" if homography_ok else "H: NONE"
    hom_color = color_ok if homography_ok else color_bad
    cv2.putText(vis, f"Frame {frame_id}  |  {hom_label}  |  {fps_proc:.1f} fps",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hom_color, 2, cv2.LINE_AA)
 
    return vis
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    args = parse_args()
 
    # ---- validate inputs ---------------------------------------------------
    check_file(args.video,            "Video")
    check_file(args.detector_weights, "Detector weights")
    check_file(args.court_weights,    "Court-keypoint weights")
    check_file(args.template,         "Homography template (dst_homo.json)")
 
    # ---- lazy imports (heavy ML libs) -------------------------------------
    print("[*] Importing pipeline modules …")
 
    # ---- open video --------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)
 
    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = args.fps if args.fps > 0 else video_fps
    max_frames   = args.max_frames if args.max_frames > 0 else total_frames
 
    print(f"[*] Video  : {args.video}")
    print(f"[*] FPS    : {fps:.2f}  |  Total frames: {total_frames}  |  Will process: {max_frames}")
 
    # ---- build pipeline components ----------------------------------------
    print("[*] Loading detector …")
    detector = RFDETRDetector(
        weights_path=args.detector_weights,
        confidence=args.confidence,
        frame_rate=int(fps),
    )
 
    print("[*] Loading court-keypoint detector …")
    court_detector = YOLOCourtDetector(weights_path=args.court_weights)
 
    print("[*] Building HomographyStep …")
    homo_step = HomographyStep(template_path=args.template)
 
    print("[*] Building PlayerHeatmapTracker …")
    tracker = PlayerHeatmapTracker(fps=fps)
    # Auto-detect bounds from the same template HomographyStep uses,
    # so world-point coords are guaranteed to fall within bounds.
    tracker.set_court_dimensions_from_template(homo_step._template)
    # If your template is in pixels, set scale so distances are in metres:
    # tracker.set_world_scale(pixels_per_metre=XX.X)
    stats_step = StatsStep(tracker=tracker)
 
    # ---- counters ----------------------------------------------------------
    frame_id          = 0
    n_homo_ok         = 0
    n_homo_none       = 0
    n_detections_total = 0
    t_start           = time.perf_counter()
 
    print("\n[*] Starting processing loop …  (press Q to quit)\n")
 
    # ---- processing loop ---------------------------------------------------
    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("[*] End of video reached.")
            break
 
        t_frame = time.perf_counter()
 
        # 1. Detect + track players
        detections = detector.track(frame)
        n_detections_total += len(detections)
 
        # 2. Detect court keypoints
        keypoints = court_detector.extract_keypoints(frame)
 
        # 3. Build a proper FrameData and run steps through it
        fd = FrameData(
            frame_id=frame_id,
            image=frame,
            detections=detections,
            tracks=detections,
            court_keypoints=keypoints,
        )
        fd = homo_step.process(fd)
        fd = stats_step.process(fd)
        H = fd.homography_matrix
 
        # ---- keypoint / homography diagnostics (printed every 30 frames) ---
        if frame_id % 30 == 0:
            src_pts, dst_pts = homo_step._build_point_pairs(keypoints)
            print(
                f"  [DIAG] keypoints={len(keypoints):<3d}"
                f"  matched_pairs={len(src_pts):<3d}"
                f"  need>={homo_step.min_keypoints}"
                f"  H={'OK  ' if H is not None else 'NONE'}"
                + (f"  det(H)={np.linalg.det(H):.4f}" if H is not None else "")
            )
 
        homography_ok = H is not None
        if homography_ok:
            n_homo_ok += 1
        else:
            n_homo_none += 1
 
        # 5. Per-frame console report (every 30 frames)
        if frame_id % 30 == 0:
            elapsed = time.perf_counter() - t_start
            fps_proc = frame_id / elapsed if elapsed > 0 else 0.0
            df_snap  = tracker.get_stats_dataframe()
            n_tracked = len(df_snap)
            print(
                f"  frame {frame_id:>5d} / {max_frames}"
                f"  |  tracked players={n_tracked}"
                f"  |  proc {fps_proc:.1f} fps"
            )
 
        # 6. Optional display
        if not args.no_display:
            elapsed_frame = time.perf_counter() - t_frame
            fps_proc = 1.0 / elapsed_frame if elapsed_frame > 0 else 0.0
            vis = draw_overlay(frame, detections, homography_ok, frame_id, fps_proc)
            cv2.imshow("Pipeline test — press Q to quit", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[*] User quit.")
                break
 
        frame_id += 1
 
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
 
    # ---- final report ------------------------------------------------------
    elapsed_total = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print("  PIPELINE TEST REPORT")
    print("=" * 60)
    print(f"  Frames processed      : {frame_id}")
    print(f"  Total time            : {elapsed_total:.2f}s  ({frame_id/elapsed_total:.1f} fps avg)")
    print(f"  Homography OK         : {n_homo_ok}  ({100*n_homo_ok/max(frame_id,1):.1f}%)")
    print(f"  Homography NONE       : {n_homo_none}  ({100*n_homo_none/max(frame_id,1):.1f}%)")
    print(f"  Total detections      : {n_detections_total}")
 
    # Configura e executa o lifecycle hook de finalização do step de estatísticas
    csv_path = Path(args.output_csv)
    stats_step.output_dir = str(csv_path.parent)
    stats_step.output_csv = str(csv_path)
    stats_step.output_heatmap = str(csv_path.with_name("heatmap_whole.png"))
    stats_step.finish()
 
 
 
if __name__ == "__main__":
    main()