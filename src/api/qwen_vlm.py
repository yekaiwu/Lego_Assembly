"""
Qwen-VL API integration via Alibaba Cloud DashScope.
Primary VLM for vision-language tasks.
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

class QwenVLMClient:
    """Qwen VLM client with vision capabilities."""
    """Client for Qwen-VL-Max/Plus via DashScope API."""
    
    def __init__(self):
        config = get_config()
        self.api_key = config.api.dashscope_api_key
        self.endpoint = config.api.qwen_vl_endpoint
        self.max_retries = config.models.max_retries
        self.timeout = config.models.request_timeout
        self.cache = get_cache()
        
        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY not set. Qwen-VL will not be available.")
    
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
            cache_context: Optional context string to differentiate cache entries (e.g., manual ID)
        
        Returns:
            Extracted step information including parts, actions, spatial relationships
        """
        # Check cache first - include cache_context to prevent collisions across different manuals
        cache_suffix = f":{cache_context}" if cache_context else ""
        cache_key = f"qwen-vl-max:{','.join(image_paths)}:{step_number}{cache_suffix}"
        cached = self.cache.get("qwen-vl-max", cache_key, image_paths)
        if cached:
            return cached
        
        # Build prompt
        prompt = self._build_extraction_prompt(step_number, use_json_mode)
        
        # Prepare multi-modal content
        content = [{"text": prompt}]
        for img_path in image_paths:
            # Convert local file path to base64 if it's not a URL
            if img_path.startswith('http://') or img_path.startswith('https://'):
                content.append({"image": img_path})
            else:
                # Encode local file as base64
                image_data = self._encode_image_to_base64(img_path)
                content.append({"image": image_data})
        
        # Make API call with retry logic
        response = self._call_api_with_retry(content, use_json_mode)
        
        # Parse response
        result = self._parse_response(response, use_json_mode)
        
        # Cache result
        self.cache.set("qwen-vl-max", cache_key, result, image_paths)

        return result

    def extract_step_info_with_context(
        self,
        image_paths: List[str],
        step_number: Optional[int] = None,
        custom_prompt: Optional[str] = None,
        use_json_mode: bool = True,
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract step info with custom context-aware prompt.

        Args:
            image_paths: List of paths to step images
            step_number: Optional step number for context
            custom_prompt: Custom prompt (overrides default)
            use_json_mode: Whether to request JSON formatted output
            cache_context: Optional context string to differentiate cache entries

        Returns:
            Extracted step information
        """
        # Check cache first
        cache_suffix = f":{cache_context}" if cache_context else ""
        cache_key = f"qwen-vl-max:{','.join(image_paths)}:{step_number}{cache_suffix}"
        cached = self.cache.get("qwen-vl-max", cache_key, image_paths)
        if cached:
            return cached

        # Use custom prompt if provided, otherwise build default
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_extraction_prompt(step_number, use_json_mode)

        # Prepare multi-modal content
        content = [{"text": prompt}]
        for img_path in image_paths:
            if img_path.startswith('http://') or img_path.startswith('https://'):
                content.append({"image": img_path})
            else:
                image_data = self._encode_image_to_base64(img_path)
                content.append({"image": image_data})

        # Make API call with retry logic
        response = self._call_api_with_retry(content, use_json_mode)

        # Parse response
        result = self._parse_response(response, use_json_mode)

        # Cache result
        self.cache.set("qwen-vl-max", cache_key, result, image_paths)

        return result

    def generate_text_description(
        self,
        image_path: str,
        prompt: str,
        cache_context: Optional[str] = None
    ) -> str:
        """
        Generate a text description from an image using the VLM.

        Generic method for generating text descriptions (not structured JSON).
        Each VLM client handles its own image format internally.

        Args:
            image_path: Path to image file
            prompt: Text prompt for description
            cache_context: Optional context for caching

        Returns:
            Generated text description
        """
        # Check cache
        cache_suffix = f":{cache_context}" if cache_context else ""
        cache_key = f"qwen-vl-max:text_desc:{image_path}{cache_suffix}"
        cached = self.cache.get("qwen-vl-max", cache_key, [image_path])
        if cached:
            return cached.get('raw_text', '')

        # Prepare content with image in Qwen format
        content = [{"text": prompt}]

        if image_path.startswith('http://') or image_path.startswith('https://'):
            content.append({"image": image_path})
        else:
            image_data = self._encode_image_to_base64(image_path)
            content.append({"image": image_data})

        # Call API
        response = self._call_api_with_retry(content, use_json_mode=False)
        result = self._parse_response(response, use_json_mode=False)

        # Cache result
        self.cache.set("qwen-vl-max", cache_key, result, [image_path])

        return result.get('raw_text', '')

    def _build_extraction_prompt(self, step_number: Optional[int], use_json_mode: bool) -> str:
        """Build the extraction prompt for VLM."""
        step_context = f"Step {step_number}: " if step_number else ""
        
        if use_json_mode:
            prompt = f"""Analyze this LEGO instruction manual step carefully. {step_context}
Extract the following information in JSON format:

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
        """Call Qwen-VL API with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen-vl-max",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        
        if use_json_mode:
            payload["parameters"]["response_format"] = {"type": "json_object"}
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Qwen-VL API (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info("Qwen-VL API call successful")
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
        
        raise RuntimeError("Failed to call Qwen-VL API")
    
    def _parse_response(self, response: Dict[str, Any], use_json_mode: bool) -> Dict[str, Any]:
        """Parse API response and extract structured data."""
        try:
            # Extract message content from DashScope response format
            output = response.get("output", {})
            choices = output.get("choices", [])
            
            if not choices:
                raise ValueError("No choices in API response")
            
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            # Handle different content formats
            if isinstance(content, list):
                # Content is a list of objects (e.g., [{"text": "..."}])
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_content += item["text"]
                content = text_content
            
            if use_json_mode:
                # Parse JSON response
                if isinstance(content, str):
                    return json.loads(content)
                else:
                    # Already a dict/list
                    return content if isinstance(content, dict) else {"raw_content": content}
            else:
                # Return as-is for text responses
                return {"raw_text": content}
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse API response: {e}")
            logger.debug(f"Response structure: {response}")
            return {"error": str(e), "raw_response": response}
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode a local image file to base64 data URI.
        
        Args:
            image_path: Path to local image file
        
        Returns:
            Base64 data URI string
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
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                base64_str = base64.b64encode(image_bytes).decode('utf-8')
            
            # Return as data URI
            return f"data:{mime_type};base64,{base64_str}"

        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
