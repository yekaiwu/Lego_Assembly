"""
Anthropic Claude 3 Vision API integration.
Supports Claude 3 Opus, Sonnet, and Haiku with vision capabilities.
"""

import json
import time
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from loguru import logger

from ..utils.config import get_config
from ..utils.cache import get_cache

class AnthropicVisionClient:
    """Client for Anthropic Claude 3 Vision API."""
    
    def __init__(self):
        config = get_config()
        self.api_key = config.api.anthropic_api_key
        self.endpoint = config.api.anthropic_endpoint
        self.model = config.api.anthropic_model
        self.max_retries = config.models.max_retries
        self.timeout = config.models.request_timeout
        self.cache = get_cache()
        
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set. Claude Vision will not be available.")
    
    def extract_step_info(
        self, 
        image_paths: List[str], 
        step_number: Optional[int] = None,
        use_json_mode: bool = True,
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured information from a LEGO instruction step.
        
        Args:
            image_paths: List of paths to step images
            step_number: Optional step number for context
            use_json_mode: Whether to request JSON formatted output
            cache_context: Optional context string to differentiate cache entries
        
        Returns:
            Extracted step information including parts, actions, spatial relationships
        """
        # Check cache first
        cache_suffix = f":{cache_context}" if cache_context else ""
        cache_key = f"{self.model}:{','.join(image_paths)}:{step_number}{cache_suffix}"
        cached = self.cache.get(self.model, cache_key, image_paths)
        if cached:
            return cached
        
        # Build prompt
        prompt = self._build_extraction_prompt(step_number, use_json_mode)
        
        # Prepare content with images
        content = []
        
        # Add images first (Claude prefers images before text)
        for img_path in image_paths:
            if img_path.startswith('http://') or img_path.startswith('https://'):
                # Claude API doesn't support URL images directly
                logger.warning(f"Claude API requires base64 images. Downloading URL: {img_path}")
                # For now, skip URL images or download them
                continue
            else:
                # Local file - encode to base64
                image_data, media_type = self._encode_image_to_base64(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Make API call with retry logic
        response = self._call_api_with_retry(content, use_json_mode)
        
        # Parse response
        result = self._parse_response(response, use_json_mode)
        
        # Cache result
        self.cache.set(self.model, cache_key, result, image_paths)
        
        return result
    
    def _build_extraction_prompt(self, step_number: Optional[int], use_json_mode: bool) -> str:
        """Build the extraction prompt for Claude."""
        step_context = f"Step {step_number}: " if step_number else ""
        
        if use_json_mode:
            prompt = f"""Analyze this LEGO instruction manual step carefully. {step_context}
Extract the following information and return ONLY valid JSON (no other text):

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

Provide accurate, detailed extraction. If information is unclear, mark as null or "unclear"."""
        else:
            prompt = f"""Analyze this LEGO instruction manual step carefully. {step_context}

Please extract:
1. Parts Required: List all LEGO parts needed (color, shape, dimensions, part ID if visible)
2. Existing Assembly: Describe parts already assembled from previous steps
3. New Parts to Add: What new parts are being added in this step
4. Actions: What actions to perform (attach, connect, place, etc.)
5. Spatial Relationships: Where to place parts (top, bottom, left, right, front, back)
6. Orientation: Any rotation or alignment requirements
7. Dependencies: Which previous steps must be completed first
8. Special Notes: Any warnings or special instructions

Be detailed and precise."""
        
        return prompt
    
    def _call_api_with_retry(self, content: List[Dict], use_json_mode: bool) -> Dict[str, Any]:
        """Call Anthropic API with retry logic."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Anthropic API (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    f"{self.endpoint}/messages",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info("Anthropic API call successful")
                return result
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. API call failed.")
                    raise
        
        raise RuntimeError("Failed to call Anthropic API")
    
    def _parse_response(self, response: Dict[str, Any], use_json_mode: bool) -> Dict[str, Any]:
        """Parse Anthropic API response."""
        try:
            content = response.get("content", [])
            if not content:
                raise ValueError("No content in API response")
            
            # Extract text from content blocks
            text_content = ""
            for block in content:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
            
            if use_json_mode:
                # Parse JSON response
                if text_content:
                    return json.loads(text_content)
                else:
                    return {"error": "Empty response"}
            else:
                # Return as-is for text responses
                return {"raw_text": text_content}
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse API response: {e}")
            logger.debug(f"Response structure: {response}")
            return {"error": str(e), "raw_response": response}
    
    def _encode_image_to_base64(self, image_path: str) -> tuple[str, str]:
        """
        Encode a local image file to base64.
        
        Args:
            image_path: Path to local image file
        
        Returns:
            Tuple of (base64 string, media_type)
        """
        try:
            path = Path(image_path)
            
            # Determine MIME type from extension
            ext = path.suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = mime_types.get(ext, 'image/jpeg')
            
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                base64_str = base64.b64encode(image_bytes).decode('utf-8')
            
            return base64_str, media_type
        
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

