#!/usr/bin/env python3
"""
Diagnostic script to check if Gemini API is resizing images.
"""

import os
from pathlib import Path
from PIL import Image

os.environ.setdefault("CACHE_ENABLED", "false")

from src.api.litellm_vlm import UnifiedVLMClient
from src.utils.config import get_config


def diagnose_image_dimensions():
    """Check if Gemini sees the same image dimensions we use for conversion."""

    test_image = "output/temp_pages/page_013.png"

    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        return

    # Get actual image dimensions
    img = Image.open(test_image)
    actual_width, actual_height = img.size

    print("=" * 80)
    print("IMAGE DIMENSION DIAGNOSTIC")
    print("=" * 80)
    print(f"\nTest image: {test_image}")
    print(f"Actual dimensions (from PIL): {actual_width} x {actual_height}")

    # Ask Gemini what dimensions it sees
    config = get_config()
    client = UnifiedVLMClient(config.models.ingestion_vlm)

    prompt = f"""What are the exact dimensions (width x height in pixels) of this image?

Please respond ONLY with the dimensions in this format:
WIDTH x HEIGHT

For example: 1920 x 1080

Do not include any other text in your response."""

    print("\nAsking Gemini what dimensions it sees...")

    response = client.generate_text_description(
        image_path=test_image,
        prompt=prompt
    )

    print(f"Gemini's response: {response}")

    # Try to parse dimensions from response
    try:
        if 'x' in response.lower():
            parts = response.strip().split('x')
            gemini_width = int(parts[0].strip())
            gemini_height = int(parts[1].strip())

            print(f"\nParsed Gemini dimensions: {gemini_width} x {gemini_height}")

            if gemini_width != actual_width or gemini_height != actual_height:
                print("\n" + "!" * 80)
                print("⚠️  DIMENSION MISMATCH DETECTED!")
                print("!" * 80)
                print(f"\nActual image:  {actual_width} x {actual_height}")
                print(f"Gemini sees:   {gemini_width} x {gemini_height}")

                # Calculate scaling factors
                width_ratio = actual_width / gemini_width
                height_ratio = actual_height / gemini_height

                print(f"\nScaling factors:")
                print(f"  Width:  {width_ratio:.4f}x")
                print(f"  Height: {height_ratio:.4f}x")

                print("\n⚠️  THIS IS THE PROBLEM!")
                print("Gemini's bboxes are relative to the RESIZED image, not the original.")
                print(f"We need to apply a {width_ratio:.4f}x scaling factor to X coordinates")
                print(f"and a {height_ratio:.4f}x scaling factor to Y coordinates.")

            else:
                print("\n✓ Dimensions match! Gemini sees the same dimensions we're using.")
                print("  The bbox issue is NOT caused by image resizing.")

    except Exception as e:
        print(f"\nCouldn't parse dimensions from response: {e}")
        print("Please check the response manually.")

    # Additional test: Ask Gemini about a specific object location
    print("\n" + "=" * 80)
    print("BBOX TEST")
    print("=" * 80)

    bbox_prompt = f"""Look at this LEGO instruction page.

Find the tan/beige 2x4 brick shown in the parts inventory callout (usually in the top-right area with a quantity like "1x").

Return ONLY a JSON object with this format:
{{
  "bbox": [y_min, x_min, y_max, x_max],
  "description": "brief description of what you found"
}}

Use normalized coordinates from 0 to 1000.
IMPORTANT: Draw the bbox around the PART ICON itself, not the entire callout box."""

    print("\nAsking Gemini for bbox of the part...")

    bbox_response = client.generate_text_description(
        image_path=test_image,
        prompt=bbox_prompt
    )

    print(f"\nGemini's bbox response:\n{bbox_response}")


if __name__ == "__main__":
    diagnose_image_dimensions()
