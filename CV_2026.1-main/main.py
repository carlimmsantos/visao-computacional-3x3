import os

from dotenv import load_dotenv

from src.core import VideoProcessor
from src.core import DrawStep
from src.core.player_crop_export_step import PlayerCropExportStep

from src.actions import RuleBasedShootingClassifier, ShootingDetectionStep, ShootingStateManager
from src.detectors.players.classes import PLAYER_CLASS_IDS, TEAM_PLAYER_CLASS_IDS
from src.detectors.players.yolo26_detector import YOLO26Detector
from src.detectors.players.rfdetr_detector import RFDETRDetector
from src.detectors.players.detection_step import DetectionStep
from src.detectors.court.yolo_court_detector import YOLOCourtDetector
from src.detectors.court.court_detector_step import CourtDetectionStep
from src.homography import HomographyStep
from src.pose import PoseEstimator
from src.segmentation.player_segmentation import PlayerSegmenter
from src.teams import (
    TeamAssignmentStep,
    TeamClassifier,
    TeamVoteManager,
    TeamClassifierCLIP,
    TeamClassifierFashionCLIP,
    team_classifier_hsv,
)
import cv2
from src.trackers.bytetrack_tracker import ByteTrackTracker
from src.trackers.tracking_step import TrackingStep
from src.stats.player_heatmap_stats import PlayerHeatmapTracker
from src.stats.stats_step import StatsStep

load_dotenv()

if __name__ == "__main__":
    input_video = "videos/basquete.mp4"
    output_video = "videos/resultado_Homography_Final.mp4"
    input_video = "videos/Posse_IFPB_3.mp4"
    output_video = "videos/resultado_varios_videos.mp4"

    detector = RFDETRDetector(weights_path="models/detector/rfdetr-m-basketball.pth",confidence=0.5)

    classifier = TeamClassifierFashionCLIP('fashionclip_trained/best_model')
    classifier.fit(crops=[])

    segmenter = PlayerSegmenter(weights_path="models/seg/yolo26m-seg.pt", crop_scale=1.0)
    vote_manager = TeamVoteManager()

    # Inicializa a atribuição de times sem o calibrator
    team_assignment = TeamAssignmentStep(
        classifier=classifier,
        segmenter=segmenter,
        vote_manager=vote_manager,
        player_class_ids=list(TEAM_PLAYER_CLASS_IDS)
    )

    court_detector = YOLOCourtDetector(weights_path="models/court_keypoint/yolo26l-court-v1a-teste.pt",confidence=0.5)

    pose_estimator = PoseEstimator()
    shooting_state = ShootingStateManager(classifier=RuleBasedShootingClassifier())
    shooting_detection = ShootingDetectionStep(
        pose_estimator=pose_estimator,
        state_manager=shooting_state,
        player_class_ids=list(PLAYER_CLASS_IDS)
    )

    # Inicializa o rastreador de estatísticas e o pipeline step
    stats_tracker = PlayerHeatmapTracker()
    stats_step = StatsStep(
        tracker=stats_tracker,
        court_length_meters=28.0,
        output_dir="outputs/stats/"
    )

    pipeline = [
        DetectionStep(detector, use_tracking=True),
        shooting_detection,
        team_assignment,
        CourtDetectionStep(court_detector),
        HomographyStep(
            template_path="assets/dst_homo_2 bkp.json",
            min_keypoints=4
        ),
        stats_step,
        DrawStep(
            draw_segmentation=True,
            draw_frame_keypoints=False,
            draw_minimap_keypoints=False, 
            draw_poses=True
        )
    ]

    processor = VideoProcessor(pipeline_steps=pipeline)
    processor.run(input_path=input_video, output_path=output_video, show_video=True)