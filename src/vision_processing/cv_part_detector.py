"""
Computer Vision-based Part Detector for Lego Instruction Manuals

This module uses traditional computer vision techniques (color segmentation, contour detection)
to detect Lego parts and assembled results in instruction manual images.

Designed specifically for Lego instruction manuals which have:
- Clean, light blue backgrounds
- Distinct part colors (brown, tan, gray, etc.)
- Structured layout (parts box top-left, assembled result center/right)
- High-quality rendered images

Approach:
1. Color-based segmentation to find parts by color
2. Contour detection to find distinct objects
3. Spatial filtering based on Lego manual layout patterns
4. Size filtering to distinguish parts from UI elements
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
from loguru import logger
from dataclasses import dataclass


@dataclass
class DetectedPart:
    """Represents a detected Lego part or component."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # (x, y)
    area: int
    contour: np.ndarray
    color_name: str  # "brown", "tan", "gray", etc.
    region_type: str  # "parts_box" or "assembled_result"
    confidence: float  # 0.0 to 1.0


class CVPartDetector:
    """
    Computer Vision-based detector for Lego parts in instruction manuals.

    Uses color segmentation and contour detection to find parts without
    relying on VLM bounding boxes.
    """

    # Lego part color ranges in HSV color space
    # Format: {color_name: [(lower_bound, upper_bound), ...]}
    # Multiple ranges handle color variations
    COLOR_RANGES = {
        "brown": [
            (np.array([0, 50, 50]), np.array([20, 255, 200])),  # Reddish brown
            (np.array([10, 100, 80]), np.array([20, 255, 180])),  # Standard brown
        ],
        "tan": [
            (np.array([15, 30, 150]), np.array([35, 150, 255])),  # Light tan/beige
            (np.array([20, 20, 180]), np.array([30, 100, 255])),  # Very light tan
        ],
        "gray": [
            (np.array([0, 0, 80]), np.array([180, 30, 180])),  # Medium gray
        ],
        "dark_gray": [
            (np.array([0, 0, 40]), np.array([180, 30, 100])),  # Dark gray
        ],
        "red": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255])),
        ],
        "blue": [
            (np.array([100, 100, 100]), np.array([130, 255, 255])),
        ],
        "yellow": [
            (np.array([20, 100, 100]), np.array([30, 255, 255])),
        ],
        "green": [
            (np.array([40, 100, 100]), np.array([80, 255, 255])),
        ],
    }

    # Background color range (light blue) - to exclude
    BACKGROUND_COLOR = [
        (np.array([90, 10, 180]), np.array([120, 80, 255])),  # Light blue background
    ]

    def __init__(
        self,
        min_part_area: int = 500,
        max_part_area: int = 100000,
        parts_box_region: Tuple[float, float, float, float] = (0.0, 0.0, 0.45, 0.35),
        assembled_region: Tuple[float, float, float, float] = (0.3, 0.2, 1.0, 1.0)
    ):
        """
        Initialize CV part detector.

        Args:
            min_part_area: Minimum area in pixels for a valid part
            max_part_area: Maximum area in pixels for a valid part
            parts_box_region: Region where parts box typically appears (x1, y1, x2, y2) as fractions
            assembled_region: Region where assembled result appears (x1, y1, x2, y2) as fractions
        """
        self.min_part_area = min_part_area
        self.max_part_area = max_part_area
        self.parts_box_region = parts_box_region
        self.assembled_region = assembled_region

        logger.info(
            f"Initialized CVPartDetector: "
            f"area_range=[{min_part_area}, {max_part_area}]px, "
            f"parts_box={parts_box_region}, "
            f"assembled={assembled_region}"
        )

    def detect_parts(
        self,
        image_path: str,
        target_colors: Optional[List[str]] = None,
        include_assembled: bool = True
    ) -> List[DetectedPart]:
        """
        Detect Lego parts in an instruction manual image.

        Args:
            image_path: Path to the instruction page image
            target_colors: List of colors to detect (None = detect all)
            include_assembled: Whether to detect assembled result

        Returns:
            List of detected parts
        """
        # Load image
        img_pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        height, width = img_bgr.shape[:2]

        # Determine which colors to detect
        colors_to_detect = target_colors if target_colors else list(self.COLOR_RANGES.keys())

        all_detections = []

        # Detect parts for each color
        for color_name in colors_to_detect:
            if color_name not in self.COLOR_RANGES:
                logger.warning(f"Unknown color: {color_name}, skipping")
                continue

            detections = self._detect_color(
                img_hsv, img_bgr, color_name, width, height, include_assembled
            )
            all_detections.extend(detections)

        # Remove duplicates (same object detected in multiple colors)
        all_detections = self._remove_duplicates(all_detections)

        # Merge assembled result detections (tan + brown should be one object)
        all_detections = self._merge_assembled_results(all_detections, img_bgr.shape[:2])

        # Sort by area (largest first)
        all_detections.sort(key=lambda d: d.area, reverse=True)

        logger.info(
            f"Detected {len(all_detections)} parts in {image_path}: "
            f"{sum(1 for d in all_detections if d.region_type == 'parts_box')} in parts box, "
            f"{sum(1 for d in all_detections if d.region_type == 'assembled_result')} assembled results"
        )

        return all_detections

    def detect_at_center_point(
        self,
        image_path: str,
        center_point: Tuple[int, int],
        search_radius: int = 150,
        min_area: int = 300
    ) -> Optional[DetectedPart]:
        """
        Detect a part at a given center point using edge-based detection.

        This method is color-agnostic and uses edge detection + contour finding
        to locate parts around the VLM-provided center point.

        Args:
            image_path: Path to the instruction page image
            center_point: (x, y) pixel coordinates of the approximate part center
            search_radius: Radius in pixels to search around the center point
            min_area: Minimum contour area to consider

        Returns:
            DetectedPart if found, None otherwise
        """
        # Load image
        img_pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        height, width = img_bgr.shape[:2]
        cx, cy = center_point

        # Validate center point is within image bounds
        if not (0 <= cx < width and 0 <= cy < height):
            logger.warning(f"Center point {center_point} is outside image bounds {width}x{height}")
            return None

        # Define search region (with bounds checking)
        x1 = max(0, cx - search_radius)
        y1 = max(0, cy - search_radius)
        x2 = min(width, cx + search_radius)
        y2 = min(height, cy + search_radius)

        # Crop to search region
        search_region = img_gray[y1:y2, x1:x2]
        search_region_bgr = img_bgr[y1:y2, x1:x2]

        # Edge detection using Canny
        blurred = cv2.GaussianBlur(search_region, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning(f"No contours found near center point {center_point}")
            return None

        # Find contour containing or nearest to the center point (adjusted to search region coords)
        region_cx = cx - x1
        region_cy = cy - y1

        best_contour = None
        best_distance = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Check if center point is inside this contour
            dist = cv2.pointPolygonTest(contour, (region_cx, region_cy), True)

            # Positive distance = inside, negative = outside
            # We want the contour that contains the point, or is closest to it
            if dist >= 0:
                # Point is inside - prioritize this
                best_contour = contour
                best_distance = 0
                break
            else:
                # Point is outside - track closest contour
                abs_dist = abs(dist)
                if abs_dist < best_distance:
                    best_distance = abs_dist
                    best_contour = contour

        if best_contour is None:
            logger.warning(f"No suitable contour found near center point {center_point}")
            return None

        # Convert contour coordinates back to full image space
        best_contour = best_contour + np.array([x1, y1])

        # Get bounding box
        x, y, w, h = cv2.boundingRect(best_contour)
        bbox = (x, y, x + w, y + h)

        # Calculate center
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None
        detected_cx = int(M["m10"] / M["m00"])
        detected_cy = int(M["m01"] / M["m00"])
        center = (detected_cx, detected_cy)

        # Calculate area
        area = cv2.contourArea(best_contour)

        # Determine dominant color at this location
        color_name = self._determine_color(img_bgr, best_contour)

        # Calculate confidence based on area and distance from hint
        distance_from_hint = np.sqrt((detected_cx - cx)**2 + (detected_cy - cy)**2)
        proximity_score = max(0, 1 - (distance_from_hint / search_radius))

        # Compactness score
        perimeter = cv2.arcLength(best_contour, True)
        if perimeter == 0:
            return None
        compactness = 4 * np.pi * area / (perimeter * perimeter)

        confidence = proximity_score * 0.5 + compactness * 0.5

        detection = DetectedPart(
            bbox=bbox,
            center=center,
            area=int(area),
            contour=best_contour,
            color_name=color_name,
            region_type="unknown",  # No longer using hardcoded regions
            confidence=confidence
        )

        logger.info(
            f"Detected part at {center_point}: color={color_name}, "
            f"area={area:.0f}, distance={distance_from_hint:.1f}px"
        )

        return detection

    def _determine_color(self, img_bgr: np.ndarray, contour: np.ndarray) -> str:
        """
        Determine the dominant color within a contour region.

        Args:
            img_bgr: Image in BGR format
            contour: Contour to analyze

        Returns:
            Color name (e.g., "brown", "tan") or "unknown"
        """
        # Create mask for this contour
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Get mean color in BGR
        mean_bgr = cv2.mean(img_bgr, mask=mask)[:3]

        # Convert to HSV for color classification
        bgr_pixel = np.uint8([[mean_bgr]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]

        # Match to known color ranges
        for color_name, ranges in self.COLOR_RANGES.items():
            for lower, upper in ranges:
                if np.all(hsv_pixel >= lower) and np.all(hsv_pixel <= upper):
                    return color_name

        # If no match, describe as general color category
        h, s, v = hsv_pixel
        if s < 30:
            if v > 180:
                return "white"
            elif v < 80:
                return "black"
            else:
                return "gray"
        else:
            # Return hue-based description
            if h < 15 or h > 165:
                return "red"
            elif h < 30:
                return "orange/tan"
            elif h < 80:
                return "green/yellow"
            elif h < 130:
                return "blue"
            else:
                return "purple/magenta"

    def _detect_color(
        self,
        img_hsv: np.ndarray,
        img_bgr: np.ndarray,
        color_name: str,
        width: int,
        height: int,
        include_assembled: bool
    ) -> List[DetectedPart]:
        """
        Detect parts of a specific color.

        Args:
            img_hsv: Image in HSV color space
            img_bgr: Original image in BGR
            color_name: Name of color to detect
            width: Image width
            height: Image height
            include_assembled: Whether to include assembled region

        Returns:
            List of detected parts for this color
        """
        # Create mask for this color (combine all ranges)
        color_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)

        for lower, upper in self.COLOR_RANGES[color_name]:
            mask_range = cv2.inRange(img_hsv, lower, upper)
            color_mask = cv2.bitwise_or(color_mask, mask_range)

        # Exclude background color
        background_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.BACKGROUND_COLOR:
            bg_range = cv2.inRange(img_hsv, lower, upper)
            background_mask = cv2.bitwise_or(background_mask, bg_range)

        color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(background_mask))

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size
            if area < self.min_part_area or area > self.max_part_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)

            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)

            # Determine region type
            region_type = self._classify_region(cx, cy, width, height)

            # Skip assembled result if not requested
            if not include_assembled and region_type == "assembled_result":
                continue

            # Calculate confidence based on area and compactness
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            confidence = min(1.0, compactness * 0.7 + 0.3)  # Favor compact shapes

            detection = DetectedPart(
                bbox=bbox,
                center=center,
                area=area,
                contour=contour,
                color_name=color_name,
                region_type=region_type,
                confidence=confidence
            )

            detections.append(detection)

        return detections

    def _classify_region(self, cx: int, cy: int, width: int, height: int) -> str:
        """
        Classify whether a point is in the parts box or assembled result region.

        Args:
            cx: Center x coordinate
            cy: Center y coordinate
            width: Image width
            height: Image height

        Returns:
            "parts_box" or "assembled_result"
        """
        # Normalize coordinates to 0-1
        nx = cx / width
        ny = cy / height

        # Check if in parts box region
        px1, py1, px2, py2 = self.parts_box_region
        if px1 <= nx <= px2 and py1 <= ny <= py2:
            return "parts_box"

        # Check if in assembled result region
        ax1, ay1, ax2, ay2 = self.assembled_region
        if ax1 <= nx <= ax2 and ay1 <= ny <= ay2:
            return "assembled_result"

        # Default to parts box if uncertain
        return "parts_box"

    def _remove_duplicates(self, detections: List[DetectedPart]) -> List[DetectedPart]:
        """
        Remove duplicate detections (same object detected in multiple colors).

        Uses IoU (Intersection over Union) to identify overlapping detections.

        Args:
            detections: List of all detections

        Returns:
            Filtered list with duplicates removed
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (keep higher confidence detections)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        filtered = []

        for detection in detections:
            # Check if this detection overlaps significantly with any existing detection
            is_duplicate = False

            for existing in filtered:
                iou = self._calculate_iou(detection.bbox, existing.bbox)
                if iou > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(detection)

        return filtered

    def _merge_assembled_results(
        self,
        detections: List[DetectedPart],
        image_shape: Tuple[int, int]
    ) -> List[DetectedPart]:
        """
        Merge overlapping/adjacent detections in the assembled_result region.

        This handles cases where a multi-colored assembled model is detected as
        separate parts (e.g., tan base + brown top = one assembled model).

        Args:
            detections: List of all detections
            image_shape: (height, width) of the image

        Returns:
            List with assembled results merged
        """
        # Separate assembled results from parts
        assembled_results = [d for d in detections if d.region_type == "assembled_result"]
        parts_box = [d for d in detections if d.region_type == "parts_box"]

        if len(assembled_results) <= 1:
            return detections  # Nothing to merge

        # Merge overlapping assembled results
        merged_assembled = []
        used = set()

        for i, detection in enumerate(assembled_results):
            if i in used:
                continue

            # Find all detections that overlap or are adjacent to this one
            group = [detection]
            used.add(i)

            for j, other in enumerate(assembled_results):
                if j in used:
                    continue

                # Check if overlapping or adjacent (within 10 pixels)
                if self._are_adjacent(detection.bbox, other.bbox, threshold=10):
                    group.append(other)
                    used.add(j)

            # Merge the group into one detection
            if len(group) > 1:
                merged = self._merge_detection_group(group, image_shape)
                merged_assembled.append(merged)
                logger.debug(f"Merged {len(group)} assembled result detections into one")
            else:
                merged_assembled.append(detection)

        # Return parts_box + merged assembled results
        return parts_box + merged_assembled

    def _are_adjacent(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        threshold: int = 10
    ) -> bool:
        """
        Check if two bounding boxes overlap or are adjacent (within threshold pixels).

        Args:
            bbox1: First box (x1, y1, x2, y2)
            bbox2: Second box (x1, y1, x2, y2)
            threshold: Maximum distance in pixels to consider adjacent

        Returns:
            True if boxes overlap or are adjacent
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Expand boxes by threshold
        x1_1_exp = x1_1 - threshold
        y1_1_exp = y1_1 - threshold
        x2_1_exp = x2_1 + threshold
        y2_1_exp = y2_1 + threshold

        # Check if expanded boxes overlap
        if (x1_1_exp <= x2_2 and x2_1_exp >= x1_2 and
            y1_1_exp <= y2_2 and y2_1_exp >= y1_2):
            return True

        return False

    def _merge_detection_group(
        self,
        group: List[DetectedPart],
        image_shape: Tuple[int, int]
    ) -> DetectedPart:
        """
        Merge a group of detections into a single detection.

        Args:
            group: List of detections to merge
            image_shape: (height, width) of image

        Returns:
            Merged detection
        """
        height, width = image_shape

        # Find bounding box that encompasses all detections
        x1 = min(d.bbox[0] for d in group)
        y1 = min(d.bbox[1] for d in group)
        x2 = max(d.bbox[2] for d in group)
        y2 = max(d.bbox[3] for d in group)
        merged_bbox = (x1, y1, x2, y2)

        # Calculate center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        merged_center = (cx, cy)

        # Sum areas
        merged_area = sum(d.area for d in group)

        # Merge contours (concatenate all points)
        merged_contour = np.vstack([d.contour for d in group])

        # Use the most common color name
        color_counts = {}
        for d in group:
            color_counts[d.color_name] = color_counts.get(d.color_name, 0) + 1
        merged_color = max(color_counts, key=color_counts.get)

        # Average confidence
        merged_confidence = sum(d.confidence for d in group) / len(group)

        return DetectedPart(
            bbox=merged_bbox,
            center=merged_center,
            area=merged_area,
            contour=merged_contour,
            color_name=f"multi-color({','.join(sorted(set(d.color_name for d in group)))})",
            region_type="assembled_result",
            confidence=merged_confidence
        )

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            bbox1: First box (x1, y1, x2, y2)
            bbox2: Second box (x1, y1, x2, y2)

        Returns:
            IoU score (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def extract_part_image(
        self,
        image_path: str,
        detection: DetectedPart,
        output_path: str,
        padding: int = 10
    ) -> Optional[Image.Image]:
        """
        Extract a part image using the CV detection contour as a mask.

        Args:
            image_path: Path to the source image
            detection: DetectedPart object
            output_path: Path to save the extracted part image
            padding: Pixels to add around the bounding box

        Returns:
            PIL Image of the extracted part with transparent background, or None if failed
        """
        try:
            # Load the source image
            img_pil = Image.open(image_path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            height, width = img_bgr.shape[:2]

            # Get bounding box with padding
            x1, y1, x2, y2 = detection.bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            # Create mask from contour
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [detection.contour], -1, 255, -1)

            # Crop the image and mask
            img_crop = img_bgr[y1:y2, x1:x2]
            mask_crop = mask[y1:y2, x1:x2]

            # Convert to RGB
            img_crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

            # Create RGBA image with transparency
            rgba = np.dstack([img_crop_rgb, mask_crop])
            part_image = Image.fromarray(rgba, mode="RGBA")

            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            part_image.save(output_path)
            logger.debug(f"Extracted part image to {output_path}")

            return part_image

        except Exception as e:
            logger.error(f"Error extracting part image: {e}")
            return None

    def visualize_detections(
        self,
        image_path: str,
        detections: List[DetectedPart],
        output_path: str
    ):
        """
        Visualize detected parts on the image.

        Args:
            image_path: Path to original image
            detections: List of detected parts
            output_path: Path to save visualization
        """
        # Load image
        img_pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Color map for different part types
        color_map = {
            "parts_box": (0, 255, 0),  # Green
            "assembled_result": (255, 0, 0),  # Blue
        }

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = color_map.get(detection.region_type, (0, 255, 255))

            # Draw bounding box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cx, cy = detection.center
            cv2.circle(img_bgr, (cx, cy), 5, color, -1)

            # Draw label
            label = f"{detection.color_name} ({detection.region_type})"
            cv2.putText(
                img_bgr, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Save visualization
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(output_path)
        logger.info(f"Saved detection visualization to {output_path}")


def create_cv_detector_from_env() -> CVPartDetector:
    """
    Create a CV part detector from environment configuration.

    Returns:
        CVPartDetector instance
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Get configuration
    min_part_area = int(os.getenv("CV_MIN_PART_AREA", "500"))
    max_part_area = int(os.getenv("CV_MAX_PART_AREA", "100000"))

    logger.info(f"Creating CVPartDetector: area_range=[{min_part_area}, {max_part_area}]")

    return CVPartDetector(
        min_part_area=min_part_area,
        max_part_area=max_part_area
    )


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cv_part_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Create detector
    detector = create_cv_detector_from_env()

    # Detect parts
    detections = detector.detect_parts(image_path)

    # Print results
    print(f"\nDetected {len(detections)} parts:")
    for i, detection in enumerate(detections, 1):
        print(
            f"  {i}. {detection.color_name} {detection.region_type}: "
            f"bbox={detection.bbox}, center={detection.center}, "
            f"area={detection.area}px, conf={detection.confidence:.2f}"
        )

    # Visualize
    output_path = image_path.replace(".", "_cv_detections.")
    detector.visualize_detections(image_path, detections, output_path)
    print(f"\nVisualization saved to: {output_path}")
