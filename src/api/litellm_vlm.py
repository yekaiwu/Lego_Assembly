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

            # Parse JSON response
            result = self._parse_response(result_text, use_json_mode)

            # Normalize to array format
            normalized_result = self._normalize_to_array(result)

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
        "quantity": <number>
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

INSTRUCTIONS:
1. Look carefully - the page may show 1, 2, or more steps
2. Each step typically has a step number visible in the image
3. Return ALL steps as an array, even if there's only one
4. Keep descriptions concise (max 10-15 words per field)
5. Return ONLY the JSON array, no additional text

If only ONE step exists on the page, return an array with one element: [{{...}}]
If TWO steps exist, return an array with two elements: [{{...}}, {{...}}]"""

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
