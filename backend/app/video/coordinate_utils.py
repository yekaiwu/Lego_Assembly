"""
Coordinate utilities for VLM output processing.

This module handles validation and transformation of coordinates between:
- Normalized 0-1000 scale (VLM output format)
- Pixel coordinates (for rendering)

Coordinate format: [ymin, xmin, ymax, xmax] - note Y comes first!
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Default normalization scale used by Gemini and other VLMs
DEFAULT_SCALE = 1000


def validate_coordinates(
    box_2d: List[int],
    scale: int = DEFAULT_SCALE,
    strict: bool = True
) -> bool:
    """
    Validate that box_2d coordinates are within [0, scale] range.

    Args:
        box_2d: [ymin, xmin, ymax, xmax] coordinates
        scale: Maximum value for coordinates (default 1000)
        strict: If True, also validates ymin < ymax and xmin < xmax

    Returns:
        True if valid, False otherwise
    """
    if not box_2d or len(box_2d) != 4:
        logger.warning(f"Invalid box_2d format: {box_2d}")
        return False

    ymin, xmin, ymax, xmax = box_2d

    # Check all values are within range
    for i, val in enumerate(box_2d):
        if not isinstance(val, (int, float)):
            logger.warning(f"Non-numeric value at index {i}: {val}")
            return False
        if val < 0 or val > scale:
            logger.warning(f"Coordinate out of range [0, {scale}]: {val} at index {i}")
            return False

    # Check logical ordering
    if strict:
        if ymin >= ymax:
            logger.warning(f"Invalid y range: ymin={ymin} >= ymax={ymax}")
            return False
        if xmin >= xmax:
            logger.warning(f"Invalid x range: xmin={xmin} >= xmax={xmax}")
            return False

    return True


def validate_point(
    point: List[int],
    scale: int = DEFAULT_SCALE
) -> bool:
    """
    Validate that a [y, x] point is within [0, scale] range.

    Args:
        point: [y, x] coordinates
        scale: Maximum value (default 1000)

    Returns:
        True if valid, False otherwise
    """
    if not point or len(point) != 2:
        logger.warning(f"Invalid point format: {point}")
        return False

    y, x = point
    for i, val in enumerate([y, x]):
        if not isinstance(val, (int, float)):
            logger.warning(f"Non-numeric point value: {val}")
            return False
        if val < 0 or val > scale:
            logger.warning(f"Point coordinate out of range: {val}")
            return False

    return True


def get_center_from_box(box_2d: List[int]) -> Tuple[float, float]:
    """
    Extract center point from box_2d coordinates.

    Args:
        box_2d: [ymin, xmin, ymax, xmax] coordinates

    Returns:
        (y_center, x_center) tuple - note Y comes first to match VLM format
    """
    ymin, xmin, ymax, xmax = box_2d
    y_center = (ymin + ymax) / 2
    x_center = (xmin + xmax) / 2
    return (y_center, x_center)


def to_pixel_coordinates(
    y_norm: float,
    x_norm: float,
    img_width: int,
    img_height: int,
    scale: int = DEFAULT_SCALE
) -> Tuple[int, int]:
    """
    Convert normalized coordinates to pixel coordinates.

    Args:
        y_norm: Y coordinate in normalized scale (0-1000)
        x_norm: X coordinate in normalized scale (0-1000)
        img_width: Image width in pixels
        img_height: Image height in pixels
        scale: Normalization scale (default 1000)

    Returns:
        (x_pixel, y_pixel) tuple - note order is (x, y) for PIL/CV2 compatibility
    """
    x_pixel = int(x_norm / scale * img_width)
    y_pixel = int(y_norm / scale * img_height)
    return (x_pixel, y_pixel)


def box_to_pixel_coordinates(
    box_2d: List[int],
    img_width: int,
    img_height: int,
    scale: int = DEFAULT_SCALE
) -> Tuple[int, int, int, int]:
    """
    Convert box_2d from normalized to pixel coordinates.

    Args:
        box_2d: [ymin, xmin, ymax, xmax] in normalized scale
        img_width: Image width in pixels
        img_height: Image height in pixels
        scale: Normalization scale (default 1000)

    Returns:
        (x1, y1, x2, y2) tuple in pixel coordinates for PIL/CV2 rectangles
    """
    ymin, xmin, ymax, xmax = box_2d

    x1 = int(xmin / scale * img_width)
    y1 = int(ymin / scale * img_height)
    x2 = int(xmax / scale * img_width)
    y2 = int(ymax / scale * img_height)

    return (x1, y1, x2, y2)


def normalize_from_pixels(
    x_pixel: int,
    y_pixel: int,
    img_width: int,
    img_height: int,
    scale: int = DEFAULT_SCALE
) -> Tuple[int, int]:
    """
    Convert pixel coordinates to normalized scale.

    Args:
        x_pixel: X coordinate in pixels
        y_pixel: Y coordinate in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
        scale: Target normalization scale (default 1000)

    Returns:
        (y_norm, x_norm) tuple - note Y comes first to match VLM format
    """
    x_norm = int(x_pixel / img_width * scale)
    y_norm = int(y_pixel / img_height * scale)
    return (y_norm, x_norm)


def clamp_coordinates(
    box_2d: List[int],
    scale: int = DEFAULT_SCALE
) -> List[int]:
    """
    Clamp coordinates to valid range [0, scale].

    Args:
        box_2d: [ymin, xmin, ymax, xmax] coordinates
        scale: Maximum value (default 1000)

    Returns:
        Clamped coordinates
    """
    return [max(0, min(scale, int(v))) for v in box_2d]


def fix_inverted_coordinates(box_2d: List[int]) -> List[int]:
    """
    Fix inverted coordinates where min > max.

    Args:
        box_2d: [ymin, xmin, ymax, xmax] coordinates

    Returns:
        Fixed coordinates with proper min/max ordering
    """
    ymin, xmin, ymax, xmax = box_2d

    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    return [ymin, xmin, ymax, xmax]


def detect_coordinate_scale(
    box_2d: List[int],
    img_width: int,
    img_height: int
) -> Optional[str]:
    """
    Attempt to detect what scale the coordinates are in.

    Args:
        box_2d: [ymin, xmin, ymax, xmax] coordinates
        img_width: Expected image width
        img_height: Expected image height

    Returns:
        'normalized' if 0-1000, 'pixels' if matching image dimensions,
        'unknown' otherwise
    """
    ymin, xmin, ymax, xmax = box_2d
    max_val = max(box_2d)

    if max_val <= 1000:
        return 'normalized'
    elif max_val <= max(img_width, img_height):
        return 'pixels'
    else:
        return 'unknown'


def auto_normalize_coordinates(
    box_2d: List[int],
    img_width: int,
    img_height: int,
    scale: int = DEFAULT_SCALE
) -> List[int]:
    """
    Auto-detect and normalize coordinates if they appear to be in pixel space.

    Args:
        box_2d: [ymin, xmin, ymax, xmax] coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        scale: Target normalization scale (default 1000)

    Returns:
        Normalized coordinates in 0-scale range
    """
    detected = detect_coordinate_scale(box_2d, img_width, img_height)

    if detected == 'normalized':
        return box_2d
    elif detected == 'pixels':
        ymin, xmin, ymax, xmax = box_2d
        return [
            int(ymin / img_height * scale),
            int(xmin / img_width * scale),
            int(ymax / img_height * scale),
            int(xmax / img_width * scale)
        ]
    else:
        logger.warning(f"Unknown coordinate scale for {box_2d}, returning as-is")
        return box_2d
