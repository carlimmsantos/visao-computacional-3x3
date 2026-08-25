from .base_court_detector import BaseCourtDetector

from ...core.frame_data import FrameData
from ...core.pipeline_step import PipelineStep


class CourtDetectionStep(PipelineStep):
    def __init__(self, court_detector: BaseCourtDetector):
        self.court_detector: BaseCourtDetector = court_detector
        
    def process(self, data: FrameData) -> FrameData:
        data.court_keypoints = self.court_detector.extract_keypoints(data.image)
        return data