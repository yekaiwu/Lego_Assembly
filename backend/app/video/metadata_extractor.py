"""Video metadata extraction utilities using OpenCV."""

import cv2
import logging
from typing import Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_video_metadata(video_path: str) -> Dict:
    """
    Extract metadata from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing:
        - duration_sec: Video duration in seconds
        - fps: Frames per second
        - resolution: [width, height]
        - frame_count: Total number of frames
        - codec: Video codec identifier
    """
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Extract metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Calculate duration
        duration_sec = frame_count / fps if fps > 0 else 0

        # Get codec string
        codec = fourcc_to_string(fourcc)

        cap.release()

        metadata = {
            "duration_sec": round(duration_sec, 2),
            "fps": round(fps, 2),
            "resolution": [width, height],
            "frame_count": frame_count,
            "codec": codec,
            "size_mb": round(Path(video_path).stat().st_size / (1024 * 1024), 2)
        }

        logger.info(f"Extracted metadata from {video_path}: {metadata}")
        return metadata

    except Exception as e:
        logger.error(f"Error extracting video metadata: {e}")
        raise


def fourcc_to_string(fourcc: int) -> str:
    """
    Convert OpenCV fourcc integer to codec string.

    Args:
        fourcc: OpenCV fourcc value

    Returns:
        Codec string (e.g., 'H264', 'mp4v')
    """
    return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])


def validate_video_format(video_path: str, allowed_formats: list = None) -> bool:
    """
    Validate video file format.

    Args:
        video_path: Path to video file
        allowed_formats: List of allowed extensions (default: ['mp4', 'mov', 'avi'])

    Returns:
        True if valid, False otherwise
    """
    if allowed_formats is None:
        allowed_formats = ['mp4', 'mov', 'avi', 'mkv']

    path = Path(video_path)
    extension = path.suffix.lower().lstrip('.')

    if extension not in allowed_formats:
        logger.warning(f"Invalid video format: {extension}. Allowed: {allowed_formats}")
        return False

    # Try to open with OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        is_valid = cap.isOpened()
        cap.release()
        return is_valid
    except Exception as e:
        logger.error(f"Error validating video: {e}")
        return False


def get_frame_at_timestamp(video_path: str, timestamp_sec: float) -> Tuple[bool, any]:
    """
    Extract a single frame at a specific timestamp.

    Args:
        video_path: Path to video file
        timestamp_sec: Time in seconds

    Returns:
        (success, frame) tuple where frame is numpy array if success=True
    """
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return False, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        cap.release()
        return success, frame

    except Exception as e:
        logger.error(f"Error extracting frame at {timestamp_sec}s: {e}")
        return False, None
