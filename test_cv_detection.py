"""
Test CV-based part detection to see what it actually detects for parts and assembled results.

This test demonstrates:
1. What the CV detector finds in the parts callout boxes
2. What the CV detector finds in the assembled result area
3. The challenge: assembled results contain colors from previous steps
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.vision_processing.cv_part_detector import CVPartDetector
from PIL import Image
import cv2
import numpy as np
from loguru import logger

def test_cv_detection(image_path: str):
    """
    Test CV detection on a LEGO instruction page.

    Args:
        image_path: Path to instruction page image
    """
    logger.info(f"Testing CV detection on: {image_path}")

    # Create detector
    detector = CVPartDetector(
        min_part_area=500,
        max_part_area=100000,
        parts_box_region=(0.0, 0.0, 0.45, 0.35),  # Top-left area
        assembled_region=(0.3, 0.2, 1.0, 1.0)     # Center-right area
    )

    # Detect all parts (both in parts box and assembled result)
    all_detections = detector.detect_parts(image_path, target_colors=None, include_assembled=True)

    # Get callout boxes for visualization
    img_pil = Image.open(image_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    callout_boxes = detector._detect_callout_boxes(img_bgr, img_hsv)

    # Separate by region type
    parts_box_detections = [d for d in all_detections if d.region_type == "parts_box"]
    assembled_detections = [d for d in all_detections if d.region_type == "assembled_result"]

    logger.info("=" * 80)
    logger.info(f"PARTS BOX DETECTIONS ({len(parts_box_detections)} found):")
    logger.info("=" * 80)
    for i, detection in enumerate(parts_box_detections, 1):
        logger.info(
            f"  {i}. Color: {detection.color_name:15s} | "
            f"Center: {str(detection.center):15s} | "
            f"Area: {detection.area:6.0f}px | "
            f"Bbox: {detection.bbox} | "
            f"Conf: {detection.confidence:.2f}"
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"ASSEMBLED RESULT DETECTIONS ({len(assembled_detections)} found):")
    logger.info("=" * 80)
    for i, detection in enumerate(assembled_detections, 1):
        logger.info(
            f"  {i}. Color: {detection.color_name:15s} | "
            f"Center: {str(detection.center):15s} | "
            f"Area: {detection.area:6.0f}px | "
            f"Bbox: {detection.bbox} | "
            f"Conf: {detection.confidence:.2f}"
        )

    # Create visualization
    output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_cv_test.png")
    visualize_detections_with_labels(image_path, parts_box_detections, assembled_detections, callout_boxes, output_path)

    logger.info("")
    logger.info(f"✓ Visualization saved to: {output_path}")

    return parts_box_detections, assembled_detections


def visualize_detections_with_labels(
    image_path: str,
    parts_detections: list,
    assembled_detections: list,
    callout_boxes: list,
    output_path: str
):
    """
    Create a detailed visualization showing parts vs assembled detections.
    """
    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Draw callout boxes (YELLOW)
    for callout_bbox in callout_boxes:
        x1, y1, x2, y2 = callout_bbox
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 4)

    # Draw parts box detections (GREEN)
    for i, detection in enumerate(parts_detections, 1):
        x1, y1, x2, y2 = detection.bbox
        cx, cy = detection.center

        # Draw bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw center point
        cv2.circle(img_bgr, (cx, cy), 8, (0, 255, 0), -1)

        # Draw label
        label = f"P{i}: {detection.color_name}"
        cv2.putText(
            img_bgr, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # Draw assembled result detections (BLUE)
    for i, detection in enumerate(assembled_detections, 1):
        x1, y1, x2, y2 = detection.bbox
        cx, cy = detection.center

        # Draw bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Draw center point
        cv2.circle(img_bgr, (cx, cy), 8, (255, 0, 0), -1)

        # Draw label
        label = f"A{i}: {detection.color_name}"
        cv2.putText(
            img_bgr, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

    # Add legend
    legend_y = 30
    cv2.putText(img_bgr, "GREEN = Parts Box", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img_bgr, "BLUE = Assembled Result", (10, legend_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Save
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(output_path)


if __name__ == "__main__":
    import glob

    # Test on pages 13-18
    test_pages = sorted(glob.glob("/Users/jay/Desktop/CS480/Lego_Assembly/output/temp_pages/page_01[3-8].png"))

    print("\n" + "=" * 80)
    print("TESTING CV DETECTION ON PAGES 13-18")
    print("=" * 80)

    all_results = []

    for page_path in test_pages:
        page_name = Path(page_path).stem
        print(f"\n{'='*80}")
        print(f"Testing: {page_name}")
        print(f"{'='*80}")

        parts, assembled = test_cv_detection(page_path)
        all_results.append({
            "page": page_name,
            "parts": len(parts),
            "assembled": len(assembled)
        })

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    for result in all_results:
        print(f"{result['page']}: {result['parts']} parts, {result['assembled']} assembled")
