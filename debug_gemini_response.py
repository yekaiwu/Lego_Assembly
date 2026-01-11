"""
Debug script to capture and analyze Gemini API response structure.
This will help us understand why JSON responses are truncated.
"""

import json
from src.api.gemini_api import GeminiVisionClient
from pathlib import Path
from loguru import logger

# Configure logger to show all details
logger.add("debug_gemini.log", level="DEBUG")

def test_gemini_response():
    """Test Gemini response with a sample image to see the full API response structure."""

    client = GeminiVisionClient()

    # Find a sample step image from temp_pages
    temp_pages_dir = Path("temp_pages")
    if not temp_pages_dir.exists():
        logger.error("temp_pages directory not found. Run main.py first to generate test images.")
        return

    # Get first few images
    image_files = sorted(temp_pages_dir.glob("page_*.png"))[:1]

    if not image_files:
        logger.error("No images found in temp_pages/")
        return

    image_paths = [str(f) for f in image_files]
    logger.info(f"Testing with image: {image_paths[0]}")

    # Build a simple test request
    parts = [{"text": """Analyze this LEGO instruction step. Return ONLY valid JSON:

{
  "step_number": 1,
  "parts_required": [
    {
      "description": "part description",
      "color": "color name",
      "shape": "brick type",
      "part_id": null,
      "quantity": 2
    }
  ],
  "actions": ["action description"],
  "notes": "any notes"
}

Be concise and complete the JSON."""}]

    # Add image
    image_data, mime_type = client._encode_image_to_base64(image_paths[0])
    parts.append({
        "inline_data": {
            "mime_type": mime_type,
            "data": image_data
        }
    })

    # Make API call
    logger.info("Making API call...")
    response = client._call_api_with_retry(parts, use_json_mode=True)

    # Log full response structure
    logger.info("=" * 80)
    logger.info("FULL API RESPONSE:")
    logger.info("=" * 80)
    print(json.dumps(response, indent=2))

    # Check finish reason
    candidates = response.get("candidates", [])
    if candidates:
        finish_reason = candidates[0].get("finishReason", "UNKNOWN")
        logger.info(f"\nFinish Reason: {finish_reason}")

        # Check safety ratings
        safety_ratings = candidates[0].get("safetyRatings", [])
        if safety_ratings:
            logger.info(f"Safety Ratings: {json.dumps(safety_ratings, indent=2)}")

        # Check usage metadata
        usage_metadata = response.get("usageMetadata", {})
        if usage_metadata:
            logger.info(f"\nToken Usage:")
            logger.info(f"  Prompt tokens: {usage_metadata.get('promptTokenCount', 'N/A')}")
            logger.info(f"  Candidates tokens: {usage_metadata.get('candidatesTokenCount', 'N/A')}")
            logger.info(f"  Total tokens: {usage_metadata.get('totalTokenCount', 'N/A')}")

        # Extract text content
        content = candidates[0].get("content", {})
        parts_response = content.get("parts", [])
        if parts_response:
            text_content = parts_response[0].get("text", "")
            logger.info(f"\nResponse Text ({len(text_content)} characters):")
            logger.info(text_content)

            # Try to parse as JSON
            try:
                parsed = json.loads(text_content)
                logger.info("\n✓ JSON is valid!")
                logger.info(json.dumps(parsed, indent=2))
            except json.JSONDecodeError as e:
                logger.error(f"\n✗ JSON parsing failed: {e}")
                logger.error(f"Character at error position: {repr(text_content[e.pos:e.pos+20])}")

if __name__ == "__main__":
    test_gemini_response()
