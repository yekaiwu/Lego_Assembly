"""
OpenAI GPT-4 Vision API integration.
Supports GPT-4 Vision, GPT-4o, and other vision-capable models.
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

class OpenAIVisionClient:
    """Client for OpenAI GPT-4 Vision API."""
    
    def __init__(self):
        config = get_config()
        self.api_key = config.api.openai_api_key
        self.endpoint = config.api.openai_endpoint
        self.model = config.api.openai_model
        self.max_retries = config.models.max_retries
        self.timeout = config.models.request_timeout
        self.cache = get_cache()
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. OpenAI Vision will not be available.")
    
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
        
        # Prepare messages with images
        content = [{"type": "text", "text": prompt}]
        
        for img_path in image_paths:
            if img_path.startswith('http://') or img_path.startswith('https://'):
                # URL image
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_path}
                })
            else:
                # Local file - encode to base64
                image_data = self._encode_image_to_base64(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })
        
        # Make API call with retry logic
        response = self._call_api_with_retry(content, use_json_mode)
        
        # Parse response
        result = self._parse_response(response, use_json_mode)
        
        # Cache result
        self.cache.set(self.model, cache_key, result, image_paths)
        
        return result
    
    def _build_extraction_prompt(self, step_number: Optional[int], use_json_mode: bool) -> str:
        """Build the extraction prompt for OpenAI."""
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
        """Call OpenAI API with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000
        }
        
        # Add JSON mode if supported and requested
        if use_json_mode and "gpt-4" in self.model:
            payload["response_format"] = {"type": "json_object"}
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info("OpenAI API call successful")
                return result
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")

                # Log response details for debugging
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        logger.error(f"API Error Details: {error_detail}")
                    except:
                        logger.error(f"API Response Text: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")

                # For rate limit errors (429), wait longer
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait_time = 60  # Wait 1 minute before retry
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                elif attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. API call failed.")
                    raise
        
        raise RuntimeError("Failed to call OpenAI API")
    
    def _parse_response(self, response: Dict[str, Any], use_json_mode: bool) -> Dict[str, Any]:
        """Parse OpenAI API response."""
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices in API response")
            
            content = choices[0].get("message", {}).get("content", "")
            
            if use_json_mode:
                # Parse JSON response
                if isinstance(content, str):
                    return json.loads(content)
                else:
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
                '.webp': 'image/webp'
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

