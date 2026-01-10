"""
Google Gemini Vision API integration.
Supports Gemini Pro Vision and other vision-capable Gemini models.
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

class GeminiVisionClient:
    """Client for Google Gemini Vision API."""
    
    def __init__(self):
        config = get_config()
        self.api_key = config.api.gemini_api_key
        self.endpoint = config.api.gemini_endpoint
        self.model = config.api.gemini_model
        self.max_retries = config.models.max_retries
        self.timeout = config.models.request_timeout
        self.cache = get_cache()
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. Gemini Vision will not be available.")
    
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
        parts = []
        
        # Add text first
        parts.append({"text": prompt})
        
        # Add images
        for img_path in image_paths:
            if img_path.startswith('http://') or img_path.startswith('https://'):
                # Gemini supports URLs, but we'll download for consistency
                logger.warning(f"URL images not fully supported. Using local file instead: {img_path}")
                continue
            else:
                # Local file - encode to base64
                image_data, mime_type = self._encode_image_to_base64(img_path)
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data
                    }
                })
        
        # Make API call with retry logic
        response = self._call_api_with_retry(parts, use_json_mode)
        
        # Parse response
        result = self._parse_response(response, use_json_mode)
        
        # Cache result
        self.cache.set(self.model, cache_key, result, image_paths)
        
        return result
    
    def _build_extraction_prompt(self, step_number: Optional[int], use_json_mode: bool) -> str:
        """Build the extraction prompt for Gemini."""
        step_context = f"Step {step_number}: " if step_number else ""
        
        if use_json_mode:
            prompt = f"""Analyze this LEGO instruction manual step carefully. {step_context}
Extract the following information and return ONLY valid JSON:

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
    
    def _call_api_with_retry(self, parts: List[Dict], use_json_mode: bool) -> Dict[str, Any]:
        """Call Gemini API with retry logic."""
        # Gemini API uses API key as query parameter
        url = f"{self.endpoint}/{self.model}:generateContent?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048
            }
        }
        
        # Add JSON mode if requested
        if use_json_mode:
            payload["generationConfig"]["response_mime_type"] = "application/json"
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Gemini API (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info("Gemini API call successful")
                
                # Add delay to respect rate limits (15 RPM = 4 seconds between requests)
                time.sleep(4)
                
                return result
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")

                # Log response details for debugging
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        logger.error(f"Gemini API Error Details: {error_detail}")
                    except:
                        logger.error(f"Gemini API Response Text: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")

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
        
        raise RuntimeError("Failed to call Gemini API")
    
    def _parse_response(self, response: Dict[str, Any], use_json_mode: bool) -> Dict[str, Any]:
        """Parse Gemini API response."""
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in API response")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            if not parts:
                raise ValueError("No parts in response content")

            # Extract text from parts
            text_content = ""
            for part in parts:
                if "text" in part:
                    text_content += part["text"]

            if use_json_mode:
                # Parse JSON response
                if text_content:
                    try:
                        return json.loads(text_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing failed: {e}")
                        logger.error(f"Raw text content (first 500 chars): {text_content[:500]}")
                        logger.error(f"Raw text content (last 500 chars): {text_content[-500:]}")

                        # Try to fix common JSON issues
                        try:
                            # Sometimes Gemini returns JSON with markdown code blocks
                            if "```json" in text_content:
                                text_content = text_content.split("```json")[1].split("```")[0].strip()
                                return json.loads(text_content)
                            elif "```" in text_content:
                                text_content = text_content.split("```")[1].split("```")[0].strip()
                                return json.loads(text_content)
                        except:
                            pass

                        # Return error with the raw text for debugging
                        return {"error": f"JSON parse error: {str(e)}", "raw_text": text_content}
                else:
                    return {"error": "Empty response"}
            else:
                # Return as-is for text responses
                return {"raw_text": text_content}

        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse API response: {e}")
            logger.debug(f"Response structure: {response}")
            return {"error": str(e), "raw_response": response}
    
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

        NEW: Accepts custom_prompt that includes context from build memory.

        Args:
            image_paths: List of paths to step images
            step_number: Optional step number for context
            custom_prompt: Custom prompt with context (overrides default)
            use_json_mode: Whether to request JSON formatted output
            cache_context: Optional context string to differentiate cache entries

        Returns:
            Extracted step information
        """
        # Check cache first
        cache_suffix = f":{cache_context}" if cache_context else ""
        cache_key = f"{self.model}:{','.join(image_paths)}:{step_number}{cache_suffix}"
        cached = self.cache.get(self.model, cache_key, image_paths)
        if cached:
            return cached

        # Use custom prompt if provided, otherwise build default
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_extraction_prompt(step_number, use_json_mode)

        # Prepare content with images
        parts = []
        parts.append({"text": prompt})

        # Add images
        for img_path in image_paths:
            if img_path.startswith('http://') or img_path.startswith('https://'):
                logger.warning(f"URL images not fully supported. Using local file instead: {img_path}")
                continue
            else:
                # Local file - encode to base64
                image_data, mime_type = self._encode_image_to_base64(img_path)
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data
                    }
                })

        # Make API call with retry logic
        response = self._call_api_with_retry(parts, use_json_mode)

        # Parse response
        result = self._parse_response(response, use_json_mode)

        # Cache result
        self.cache.set(self.model, cache_key, result, image_paths)

        return result

    def _encode_image_to_base64(self, image_path: str) -> tuple[str, str]:
        """
        Encode a local image file to base64.

        Args:
            image_path: Path to local image file

        Returns:
            Tuple of (base64 string, mime_type)
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

            return base64_str, mime_type

        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

