"""
Test script to debug VLM bounding box extraction.
Directly calls the VLM and shows raw response.
"""

import os
import sys
import json
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.litellm_vlm import UnifiedVLMClient
from dotenv import load_dotenv

load_dotenv()

# Configure logging to show everything
logger.remove()
logger.add(sys.stderr, level="DEBUG")

def test_bbox_extraction():
    """Test if VLM returns bboxes for a simple test image."""

    # Use a page with actual step instructions (not the cover)
    test_image = "output/temp_pages/page_013.png"

    if not os.path.exists(test_image):
        print(f"Error: Test image not found: {test_image}")
        print("Please run the main extraction first to generate temp_pages/")
        return

    print("=" * 80)
    print("BBOX EXTRACTION TEST")
    print("=" * 80)
    print(f"Test Image: {test_image}")
    print(f"VLM Model: gemini/gemini-robotics-er-1.5-preview")
    print()

    # Initialize VLM client
    client = UnifiedVLMClient("gemini/gemini-robotics-er-1.5-preview")

    # Test 1: Standard extraction with bbox prompt
    print("-" * 80)
    print("TEST 1: Standard extraction (should include bboxes)")
    print("-" * 80)

    results = client.extract_step_info(
        image_paths=[test_image],
        step_number=1,
        use_json_mode=True,
        cache_context="bbox_test"
    )

    print("\n📊 RESULTS:")
    print(json.dumps(results, indent=2))

    # Analyze results
    print("\n🔍 ANALYSIS:")
    for i, step in enumerate(results):
        print(f"\nStep {i+1}:")
        parts = step.get("parts_required", [])
        print(f"  Total parts: {len(parts)}")

        for j, part in enumerate(parts):
            has_bbox = "bbox" in part
            bbox_value = part.get("bbox", "NOT PRESENT")
            print(f"  Part {j+1}: {part.get('color', '?')} {part.get('shape', '?')}")
            print(f"    - Has bbox field: {has_bbox}")
            print(f"    - Bbox value: {bbox_value}")

    # Test 2: Check raw LiteLLM response
    print("\n" + "=" * 80)
    print("TEST 2: Direct LiteLLM call (inspect raw response)")
    print("=" * 80)

    import litellm
    import base64

    # Build the same prompt our code uses
    prompt = """IMPORTANT: This page may contain ONE or MORE assembly steps. Analyze carefully and extract ALL steps shown.

Return a JSON ARRAY containing ALL steps found on this page:

[
  {
    "step_number": 1,
    "parts_required": [
      {
        "description": "part description",
        "color": "color name",
        "shape": "brick type and dimensions",
        "part_id": "LEGO part ID if visible",
        "quantity": 1,
        "bbox": [x1, y1, x2, y2]
      }
    ]
  }
]

CRITICAL: For EVERY part, you MUST provide bbox coordinates [x1, y1, x2, y2].
Example: "bbox": [120, 80, 300, 200]
NEVER omit the bbox field!"""

    # Load and encode image
    with open(test_image, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"}
            }
        ]
    }]

    print("Calling LiteLLM directly...")
    response = litellm.completion(
        model="gemini/gemini-robotics-er-1.5-preview",
        messages=messages,
        temperature=0.7,
        max_tokens=60000
    )

    raw_text = response.choices[0].message.content

    print("\n📝 RAW VLM RESPONSE:")
    print("-" * 80)
    print(raw_text)
    print("-" * 80)

    # Try to parse it
    try:
        # Handle markdown code blocks
        if "```json" in raw_text:
            start = raw_text.find("```json") + 7
            end = raw_text.find("```", start)
            json_str = raw_text[start:end].strip()
        elif "```" in raw_text:
            start = raw_text.find("```") + 3
            end = raw_text.find("```", start)
            json_str = raw_text[start:end].strip()
        else:
            json_str = raw_text.strip()

        parsed = json.loads(json_str)

        print("\n✅ Successfully parsed JSON")
        print("\n🔍 PARSED STRUCTURE:")

        # Check for bbox in parsed response
        if isinstance(parsed, list):
            for step in parsed:
                parts = step.get("parts_required", [])
                for part in parts:
                    if "bbox" in part:
                        print(f"  ✅ Found bbox in part: {part.get('bbox')}")
                    else:
                        print(f"  ❌ NO bbox in part: {part.get('color')} {part.get('shape')}")

    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to parse JSON: {e}")
        print("The VLM might not be following the JSON schema correctly")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_bbox_extraction()
