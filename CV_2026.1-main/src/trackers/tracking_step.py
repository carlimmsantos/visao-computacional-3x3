from .base_tracker import BaseTracker

from ..core.frame_data import FrameData
from ..core.pipeline_step import PipelineStep

class TrackingStep(PipelineStep):
    def __init__(self, tracker: BaseTracker):
        self.tracker = tracker
        
    def process(self, data: FrameData) -> FrameData:
        data.tracks = self.tracker.update(data.detections, data.image)
        # Downstream (TeamAssignmentStep, DrawStep) consome data.detections;
        # sobrescreve para que o track_id (det[6]) fique acessivel sem branchs.
        data.detections = data.tracks
        return data