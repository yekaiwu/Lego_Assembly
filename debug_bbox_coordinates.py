#!/usr/bin/env python3
"""
Debug script to check if normalized -> pixel coordinate conversion is correct.
"""

import os
from pathlib import Path
from PIL import Image

os.environ.setdefault("CACHE_ENABLED", "false")

from src.vision_processing.vlm_step_extractor import VLMStepExtractor
from loguru import logger


def debug_coordinates():
    """Check raw VLM output vs converted coordinates."""

    test_image = "output/temp_pages/page_013.png"

    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        return

    # Get image dimensions
    img = Image.open(test_image)
    img_width, img_height = img.size
    print(f"Image dimensions: {img_width} x {img_height}")
    print("=" * 80)

    # Extract with VLM
    extractor = VLMStepExtractor(enable_bbox_visualization=False)
    results = extractor.extract_step(
        image_paths=[test_image],
        step_number=None,
        use_primary=True
    )

    # Analyze results
    for step in results:
        step_num = step.get("step_number", "?")
        print(f"\nStep {step_num}:")

        parts = step.get("parts_required", [])
        for i, part in enumerate(parts):
            desc = part.get("description", "Unknown")
            bbox = part.get("bbox", None)

            print(f"\n  Part {i+1}: {desc}")
            print(f"  Bbox (after conversion): {bbox}")

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                # Check if bbox is within image bounds
                within_bounds = (
                    0 <= x1 < img_width and
                    0 <= x2 <= img_width and
                    0 <= y1 < img_height and
                    0 <= y2 <= img_height and
                    x2 > x1 and y2 > y1
                )

                # Calculate bbox properties
                width = x2 - x1
                height = y2 - y1
                area = width * height

                # Calculate position as percentage
                x_pct = (x1 / img_width) * 100
                y_pct = (y1 / img_height) * 100

                print(f"  Size: {width}px x {height}px (area: {area} px²)")
                print(f"  Position: {x_pct:.1f}% from left, {y_pct:.1f}% from top")
                print(f"  Within bounds: {within_bounds}")

                if not within_bounds:
                    print(f"  ⚠️  ISSUE: Bbox extends outside image!")
                    if x1 < 0 or y1 < 0:
                        print(f"     - Negative coordinates: x1={x1}, y1={y1}")
                    if x2 > img_width or y2 > img_height:
                        print(f"     - Exceeds bounds: x2={x2} (max={img_width}), y2={y2} (max={img_height})")

                # Calculate what the normalized coords should have been
                # (reverse calculation to verify)
                norm_x_min = int((x1 / img_width) * 1000)
                norm_y_min = int((y1 / img_height) * 1000)
                norm_x_max = int((x2 / img_width) * 1000)
                norm_y_max = int((y2 / img_height) * 1000)

                print(f"  Implied normalized coords: [y_min={norm_y_min}, x_min={norm_x_min}, y_max={norm_y_max}, x_max={norm_x_max}]")

                # Check if bbox is too small (likely wrong)
                if area < 100:
                    print(f"  ⚠️  WARNING: Bbox very small (area < 100px²) - likely incorrect")

                # Check if bbox is too large (entire image)
                img_area = img_width * img_height
                if area > img_area * 0.5:
                    print(f"  ⚠️  WARNING: Bbox covers >50% of image - likely too large")


if __name__ == "__main__":
    debug_coordinates()
