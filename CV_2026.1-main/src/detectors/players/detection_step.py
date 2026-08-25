from .base_detector import BaseDetector

from ...core.frame_data import FrameData
from ...core.pipeline_step import PipelineStep


class DetectionStep(PipelineStep):
    def __init__(self, detector: BaseDetector, use_tracking: bool = False):
        self.detector = detector
        self.use_tracking = use_tracking
        
    def process(self, data: FrameData) -> FrameData:
        if self.use_tracking:
            data.detections = self.detector.track(data.image)
        else:
            data.detections = self.detector.detect(data.image)

        if hasattr(self.detector, "class_names"):
            data.class_names = self.detector.class_names
        return data