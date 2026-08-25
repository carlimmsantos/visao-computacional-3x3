from .frame_data import FrameData

from abc import ABC, abstractmethod

class PipelineStep(ABC):
    @abstractmethod
    def process(self, data: FrameData) -> FrameData:
        pass