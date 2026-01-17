"""
Unified VLM Client using LiteLLM.
Supports 100+ vision models from different providers with a single interface.
"""

import json
import os
from typing import List, Dict, Any, Optional
from loguru import logger
import litellm

from ..utils.config import get_config
from ..utils.cache import get_cache


class UnifiedVLMClient:
    """Unified VLM client using LiteLLM for any vision model."""

    def __init__(self, model_name: str):
        """
        Initialize with LiteLLM model name.

        Args:
            model_name: LiteLLM model identifier (e.g., "gemini/gemini-2.5-flash", "gpt-4o")
        """
        config = get_config()
        self.model = model_name
        self.cache = get_cache()

        # Set API keys in environment for LiteLLM
        if config.api.openai_api_key:
            os.environ["OPENAI_API_KEY"] = config.api.openai_api_key
        if config.api.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = config.api.anthropic_api_key
        if config.api.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = config.api.gemini_api_key
        if config.api.dashscope_api_key:
            os.environ["DASHSCOPE_API_KEY"] = config.api.dashscope_api_key
        if config.api.deepseek_api_key:
            os.environ["DEEPSEEK_API_KEY"] = config.api.deepseek_api_key
        if config.api.moonshot_api_key:
            os.environ["MOONSHOT_API_KEY"] = config.api.moonshot_api_key

        # Configure LiteLLM
        litellm.drop_params = True  # Drop unsupported parameters

        logger.info(f"UnifiedVLMClient initialized with model: {self.model}")

    def extract_step_info(
        self,
        image_paths: List[str],
        step_number: Optional[int] = None,
        use_json_mode: bool = True,
        cache_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract structured information from LEGO instruction step(s).

        Args:
            image_paths: List of paths to step images
            step_number: Optional step number for context
            use_json_mode: Whether to request JSON formatted output
            cache_context: Optional context string to differentiate cache entries

        Returns:
            List of extracted step information dictionaries
        """
        # Check cache first
        cache_suffix = f":{cache_context}" if cache_context else ""
        cache_key = f"{self.model}:{','.join(image_paths)}:{step_number}{cache_suffix}"
        cached = self.cache.get(self.model, cache_key, image_paths)
        if cached:
            logger.debug(f"Cache hit for {self.model}")
            return cached

        # Build prompt
        prompt = self._build_extraction_prompt(step_number, use_json_mode)

        # DEBUG: Log the prompt to verify bbox instructions are included
        logger.debug(f"Extraction prompt (first 800 chars): {prompt[:800]}...")

        # Prepare messages with images
        content = [{"type": "text", "text": prompt}]

        # Add images
        for img_path in image_paths:
            # Read and encode image
            import base64
            from pathlib import Path

            path = Path(img_path)
            if not path.exists():
                logger.error(f"Image not found: {img_path}")
                continue

            # Determine image format
            suffix = path.suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.webp': 'image/webp',
                '.gif': 'image/gif'
            }.get(suffix, 'image/png')

            # Read and encode
            with open(img_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            })

        messages = [{"role": "user", "content": content}]

        # Call LiteLLM
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=60000
            )

            result_text = response.choices[0].message.content

            # Handle None content
            if result_text is None:
                logger.error("LiteLLM returned None content for step extraction")
                return [{"error": "LiteLLM returned None content"}]

            # DEBUG: Log raw response to check if bbox is present
            logger.debug(f"VLM raw response preview: {result_text[:500]}...")

            # Parse JSON response
            result = self._parse_response(result_text, use_json_mode)

            # Normalize to array format
            normalized_result = self._normalize_to_array(result)

            # Validate bbox presence and add defaults if missing
            normalized_result = self._ensure_bbox_fields(normalized_result)

            # Cache result
            self.cache.set(self.model, cache_key, normalized_result, image_paths)

            return normalized_result

        except Exception as e:
            logger.error(f"LiteLLM API call failed: {e}")
            return [{"error": str(e)}]

    def extract_step_info_with_context(
        self,
        image_paths: List[str],
        step_number: Optional[int],
        custom_prompt: str,
        cache_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract step info with custom context-aware prompt.

        Args:
            image_paths: List of paths to step images
            step_number: Optional step number
            custom_prompt: Custom prompt with context
            cache_context: Optional cache context string

        Returns:
            List of extracted step information dictionaries
        """
        # Check cache
        cache_suffix = f":{cache_context}" if cache_context else ""
        cache_key = f"{self.model}:context:{','.join(image_paths)}:{step_number}{cache_suffix}"
        cached = self.cache.get(self.model, cache_key, image_paths)
        if cached:
            return cached

        # Prepare messages with images
        content = [{"type": "text", "text": custom_prompt}]

        # Add images
        for img_path in image_paths:
            import base64
            from pathlib import Path

            path = Path(img_path)
            if not path.exists():
                continue

            suffix = path.suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.webp': 'image/webp',
                '.gif': 'image/gif'
            }.get(suffix, 'image/png')

            with open(img_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            })

        messages = [{"role": "user", "content": content}]

        # Call LiteLLM
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=60000
            )

            result_text = response.choices[0].message.content

            # Handle None content
            if result_text is None:
                logger.error("LiteLLM returned None content for context-aware extraction")
                return [{"error": "LiteLLM returned None content"}]

            result = self._parse_response(result_text, use_json_mode=True)
            normalized_result = self._normalize_to_array(result)

            # Validate bbox presence and add defaults if missing
            normalized_result = self._ensure_bbox_fields(normalized_result)

            # Cache result
            self.cache.set(self.model, cache_key, normalized_result, image_paths)

            return normalized_result

        except Exception as e:
            logger.error(f"LiteLLM context-aware call failed: {e}")
            return [{"error": str(e)}]

    def _build_extraction_prompt(self, step_number: Optional[int], use_json_mode: bool) -> str:
        """Build extraction prompt that handles multiple steps per page."""
        step_hint = f"Look for step {step_number} specifically. " if step_number else ""

        prompt = f"""IMPORTANT: This page may contain ONE or MORE assembly steps. Analyze carefully and extract ALL steps shown.

{step_hint}Return a JSON ARRAY containing ALL steps found on this page:

[
  {{
    "step_number": <number or null>,
    "parts_required": [
      {{
        "description": "part description",
        "color": "color name",
        "shape": "brick type and dimensions",
        "part_id": "LEGO part ID if visible",
        "quantity": <number>,
        "bbox": [x1, y1, x2, y2]
      }}
    ],
    "existing_assembly": "description of already assembled parts shown",
    "new_parts_to_add": [
      "description of each new part being added in this step"
    ],
    "actions": [
      {{
        "action_verb": "attach|connect|place|align|rotate",
        "target": "what is being attached",
        "destination": "where it's being attached",
        "orientation": "directional cues"
      }}
    ],
    "spatial_relationships": {{
      "position": "top|bottom|left|right|front|back|center",
      "rotation": "rotation description if any",
      "alignment": "alignment instructions"
    }},
    "dependencies": "which previous steps are prerequisites",
    "notes": "any special instructions or warnings"
  }}
]

CRITICAL INSTRUCTIONS:
1. Look carefully - the page may show 1, 2, or more steps
2. Each step typically has a step number visible in the image
3. Return ALL steps as an array, even if there's only one
4. Keep descriptions concise (max 10-15 words per field)
5. Return ONLY the JSON array, no additional text

BOUNDING BOX INSTRUCTIONS (MANDATORY):
6. For EVERY part in parts_required, you MUST provide bbox coordinates in PIXEL format:
   "bbox": [x1, y1, x2, y2]

   WHERE:
   - x1, y1 = top-left corner (pixel coordinates)
   - x2, y2 = bottom-right corner (pixel coordinates)
   - Use the actual pixel coordinates from the image

   EXAMPLES:
   - A small part at top-left: "bbox": [50, 50, 150, 150]
   - A part in the center: "bbox": [400, 300, 600, 500]
   - A large part: "bbox": [100, 100, 800, 600]

   IMPORTANT:
   - NEVER omit the bbox field - it is REQUIRED for every part
   - If you cannot see the part clearly, estimate the bounding box
   - Always use integer pixel coordinates

FORMAT EXAMPLES:
- Single step: [{{"step_number": 1, "parts_required": [{{"bbox": [100, 150, 300, 400], ...}}], ...}}]
- Two steps: [{{"step_number": 1, ...}}, {{"step_number": 2, ...}}]"""

        return prompt

    def _parse_response(self, response_text: str, use_json_mode: bool) -> Any:
        """Parse JSON response from VLM."""
        try:
            # Handle None or empty response
            if response_text is None:
                logger.error("Cannot parse None response")
                return {"error": "Response text is None"}

            if not response_text.strip():
                logger.error("Cannot parse empty response")
                return {"error": "Response text is empty"}

            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                json_str = response_text.strip()

            # Parse JSON
            result = json.loads(json_str)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response text: {response_text[:500] if response_text else 'None'}")
            return {"error": f"JSON parse error: {e}", "raw_response": response_text[:500] if response_text else None}

    def _normalize_to_array(self, result: Any) -> List[Dict[str, Any]]:
        """Normalize result to array format."""
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return [result]
        else:
            return [{"error": "Invalid result format", "raw": str(result)}]

    def _convert_box_2d_to_bbox(self, box_2d: List[int], image_width: int = None, image_height: int = None) -> List[int]:
        """
        Convert Gemini Robotics ER box_2d format to pixel bbox.

        Gemini format: [y_min, x_min, y_max, x_max] in normalized coords (0-1000)
        Output format: [x1, y1, x2, y2] in pixel coords

        Args:
            box_2d: Normalized coordinates [y_min, x_min, y_max, x_max]
            image_width: Image width in pixels (if None, assumes 1000x1000)
            image_height: Image height in pixels (if None, assumes 1000x1000)

        Returns:
            Pixel coordinates [x1, y1, x2, y2]
        """
        if not box_2d or len(box_2d) != 4:
            return None

        y_min, x_min, y_max, x_max = box_2d

        # Use default dimensions if not provided
        width = image_width or 1000
        height = image_height or 1000

        # Convert from normalized (0-1000) to pixels
        x1 = int((x_min / 1000.0) * width)
        y1 = int((y_min / 1000.0) * height)
        x2 = int((x_max / 1000.0) * width)
        y2 = int((y_max / 1000.0) * height)

        return [x1, y1, x2, y2]

    def _ensure_bbox_fields(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure all parts have bbox fields.

        Args:
            results: List of step extraction results

        Returns:
            Updated results with bbox fields guaranteed for all parts
        """
        bbox_missing_count = 0
        bbox_present_count = 0

        for step in results:
            if "parts_required" in step:
                for part in step["parts_required"]:
                    if "bbox" in part and part["bbox"]:
                        # Validate bbox format
                        if isinstance(part["bbox"], list) and len(part["bbox"]) == 4:
                            bbox_present_count += 1
                        else:
                            logger.warning(f"Invalid bbox format: {part['bbox']}")
                            part["bbox"] = None
                            bbox_missing_count += 1
                    else:
                        # No bbox - add null placeholder
                        part["bbox"] = None
                        bbox_missing_count += 1

        total_parts = bbox_missing_count + bbox_present_count
        if total_parts > 0:
            if bbox_missing_count > 0:
                logger.warning(
                    f"VLM did not provide bboxes for {bbox_missing_count}/{total_parts} parts "
                    f"({bbox_missing_count/total_parts*100:.1f}%). SAM will use auto-segmentation fallback."
                )
            else:
                logger.info(f"✓ VLM provided bboxes for all {bbox_present_count} parts")

        return results

    def generate_text_description(
        self,
        image_path: str,
        prompt: str,
        cache_context: Optional[str] = None
    ) -> str:
        """
        Generate text description of an image using the VLM.

        Args:
            image_path: Path to the image
            prompt: Description prompt
            cache_context: Optional cache context string

        Returns:
            Text description of the image
        """
        # Check cache
        cache_key = f"{self.model}:desc:{image_path}:{cache_context}" if cache_context else f"{self.model}:desc:{image_path}"
        cached = self.cache.get(self.model, cache_key, [image_path])
        if cached:
            logger.debug(f"Cache hit for description: {self.model}")
            return cached if isinstance(cached, str) else str(cached)

        # Prepare message with image
        import base64
        from pathlib import Path

        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image not found: {image_path}")
            return ""

        # Determine image format
        suffix = path.suffix.lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }.get(suffix, 'image/png')

        # Read and encode
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                }
            ]
        }]

        # Call LiteLLM
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=60000
            )

            # Handle case where content is None
            content = response.choices[0].message.content
            if content is None:
                logger.warning(f"LiteLLM returned None content for text description")
                return ""

            description = content.strip()

            # Cache result
            self.cache.set(self.model, cache_key, description, [image_path])

            return description

        except Exception as e:
            logger.error(f"LiteLLM text description failed: {e}")
            return ""
