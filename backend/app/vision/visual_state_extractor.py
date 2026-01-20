"""
Visual State Extractor - Analyzes user photos to extract assembly state.
Uses VLM to detect parts, subassemblies, and spatial relationships.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from ..llm.litellm_client import UnifiedLLMClient


class VisualStateExtractor:
    """
    Extracts structured assembly state from user photos.

    Output format:
    {
        "detected_parts": [
            {
                "description": "red 2x4 brick",
                "color": "red",
                "shape": "brick",
                "size": "2x4",
                "quantity": 2,
                "confidence": 0.95
            }
        ],
        "subassemblies": [
            {
                "description": "dog head with eyes",
                "components": ["white brick", "black dots"],
                "confidence": 0.88
            }
        ],
        "spatial_info": {
            "layout": "parts arranged left to right",
            "orientation": "front view"
        },
        "metadata": {
            "photo_quality": "good",
            "lighting": "adequate",
            "occlusion": "minimal"
        }
    }
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize extractor with VLM model.

        Args:
            model_name: Optional model override. Defaults to config setting.
        """
        from ..config import get_settings

        settings = get_settings()

        # Use provided model or fall back to config
        if not model_name:
            model_name = getattr(settings, 'state_detection_vlm_model', 'gemini/gemini-2.0-flash-exp')

        # Get API keys
        api_keys = settings.get_api_keys_dict()

        # Initialize VLM client
        self.vlm_client = UnifiedLLMClient(
            model=model_name,
            api_keys=api_keys
        )

        logger.info(f"VisualStateExtractor initialized with {model_name}")

    def extract_state(
        self,
        image_path: str,
        manual_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract assembly state from photo.

        Args:
            image_path: Path to user's photo
            manual_context: Optional context about the manual (theme, known parts)

        Returns:
            Structured state dictionary
        """
        # Validate image exists
        if not Path(image_path).exists():
            logger.error(f"Image not found: {image_path}")
            return self._empty_state(error="image_not_found")

        # Build extraction prompt
        prompt = self._build_extraction_prompt(manual_context)

        try:
            # Use VLM to analyze photo
            response = self.vlm_client.generate_with_vision(
                prompt=prompt,
                image_paths=[image_path],
                response_format="json"
            )

            # Parse response
            if response:
                import json
                try:
                    state = json.loads(response) if isinstance(response, str) else response
                    self._validate_state(state)

                    num_parts = len(state.get('detected_parts', []))
                    num_subasm = len(state.get('subassemblies', []))

                    logger.info(f"✓ Extracted state: {num_parts} parts, {num_subasm} subassemblies")
                    return state

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    return self._empty_state(error="json_parse_error")
            else:
                logger.warning("VLM returned empty response")
                return self._empty_state(error="empty_response")

        except Exception as e:
            logger.error(f"State extraction error: {e}")
            return self._empty_state(error=str(e))

    def _build_extraction_prompt(self, manual_context: Optional[Dict]) -> str:
        """Build VLM prompt for state extraction."""
        base_prompt = """Analyze this LEGO assembly photo and extract detailed information.

TASK: Identify all visible LEGO parts and subassemblies.

OUTPUT FORMAT (JSON):
{
  "detected_parts": [
    {
      "description": "short part description",
      "color": "exact color name",
      "shape": "brick|plate|tile|slope|round|special",
      "size": "dimensions like 2x4, 1x2",
      "quantity": number_visible,
      "confidence": 0.0-1.0
    }
  ],
  "subassemblies": [
    {
      "description": "what is assembled",
      "components": ["list", "of", "parts"],
      "confidence": 0.0-1.0
    }
  ],
  "spatial_info": {
    "layout": "how parts are arranged",
    "orientation": "viewing angle"
  },
  "metadata": {
    "photo_quality": "poor|fair|good|excellent",
    "lighting": "poor|adequate|good",
    "occlusion": "none|minimal|moderate|severe"
  }
}

IMPORTANT:
- Be precise with colors (red, blue, yellow, white, black, gray, tan, etc.)
- Include ALL visible parts, even small ones
- Identify completed subassemblies separately from loose parts
- Use standard LEGO terminology
- For confidence: 1.0 = certain, 0.5 = unsure, 0.0 = guess
"""

        # Add manual context if available
        if manual_context:
            theme = manual_context.get("theme", "")
            known_parts = manual_context.get("common_parts", [])
            manual_id = manual_context.get("manual_id", "")

            if theme:
                base_prompt += f"\n\nCONTEXT: This is from a {theme} LEGO set."
            if manual_id:
                base_prompt += f"\nManual: {manual_id}"
            if known_parts:
                parts_list = ", ".join(known_parts[:10])
                base_prompt += f"\nCommon parts in this set: {parts_list}"

        base_prompt += "\n\nOutput ONLY the JSON, no other text."

        return base_prompt

    def _validate_state(self, state: Dict[str, Any]) -> None:
        """Validate and normalize extracted state."""
        # Ensure required keys exist
        state.setdefault("detected_parts", [])
        state.setdefault("subassemblies", [])
        state.setdefault("spatial_info", {})
        state.setdefault("metadata", {})

        # Normalize part descriptions
        for part in state["detected_parts"]:
            part.setdefault("confidence", 0.5)
            part.setdefault("quantity", 1)
            part.setdefault("color", "unknown")
            part.setdefault("shape", "brick")

        # Normalize subassemblies
        for subasm in state["subassemblies"]:
            subasm.setdefault("confidence", 0.5)
            subasm.setdefault("components", [])

    def _empty_state(self, error: str = "unknown") -> Dict[str, Any]:
        """Return empty state structure."""
        return {
            "detected_parts": [],
            "subassemblies": [],
            "spatial_info": {},
            "metadata": {"error": error}
        }

    def extract_parts_only(
        self,
        image_path: str,
        expected_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract only parts (no subassemblies) from image.
        Useful for inventory checking.

        Args:
            image_path: Path to image
            expected_count: Optional expected part count for validation

        Returns:
            List of detected parts
        """
        state = self.extract_state(image_path)
        parts = state.get("detected_parts", [])

        if expected_count and len(parts) != expected_count:
            logger.warning(f"Expected {expected_count} parts, found {len(parts)}")

        return parts


# Singleton instance
_extractor_instance = None


def get_visual_state_extractor(model_name: Optional[str] = None) -> VisualStateExtractor:
    """Get VisualStateExtractor singleton instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = VisualStateExtractor(model_name)
    return _extractor_instance
