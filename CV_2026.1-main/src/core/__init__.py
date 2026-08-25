"""Core pipeline building blocks.

This package exposes the main components used to build and run the video
processing pipeline.

Usage:
    from src.core import VideoProcessor, DrawStep, FrameData, PipelineStep
"""

from .draw_step import DrawStep
from .frame_data import FrameData
from .pipeline_step import PipelineStep
from .video_processor import VideoProcessor

__all__ = [
    "DrawStep",
    "FrameData",
    "PipelineStep",
    "VideoProcessor",
]
