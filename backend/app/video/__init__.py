"""Video analysis and overlay generation module."""

from .video_analyzer import VideoAnalyzer
from .coordinate_utils import (
    validate_coordinates,
    validate_point,
    box_to_pixel_coordinates,
    to_pixel_coordinates,
    clamp_coordinates,
    fix_inverted_coordinates,
)

__all__ = [
    "VideoAnalyzer",
    "validate_coordinates",
    "validate_point",
    "box_to_pixel_coordinates",
    "to_pixel_coordinates",
    "clamp_coordinates",
    "fix_inverted_coordinates",
]
