"""
DeepSeek-V2 API integration.
Secondary VLM option with good structured output capabilities.
"""

import json
import time
from typing import List, Dict, Any, Optional
import requests
from loguru import logger

from ..utils.config import get_config
from ..utils.cache import get_cache

class DeepSeekClient:
    """Client for DeepSeek-V2 with vision capabilities."""
    
    def __init__(self):
        config = get_config()
        self.api_key = config.api.deepseek_api_key
        self.endpoint = config.api.deepseek_endpoint
        self.max_retries = config.models.max_retries
        self.timeout = config.models.request_timeout
        self.cache = get_cache()
        
        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY not set. DeepSeek will not be available.")
    
    def extract_step_info(
        self, 
        image_paths: List[str], 
        step_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract structured information from a LEGO instruction step."""
        # Check cache
        cache_key = f"deepseek-v2:{','.join(image_paths)}:{step_number}"
        cached = self.cache.get("deepseek-v2", cache_key, image_paths)
        if cached:
            return cached
        
        # Build prompt
        prompt = self._build_extraction_prompt(step_number)
        
        # Make API call
        response = self._call_api_with_retry(image_paths, prompt)
        
        # Parse response
        result = self._parse_response(response)
        
        # Cache result
        self.cache.set("deepseek-v2", cache_key, result, image_paths)
        
        return result
    
    def _build_extraction_prompt(self, step_number: Optional[int]) -> str:
        """Build extraction prompt."""
        step_context = f"Step {step_number}: " if step_number else ""
        
        return f"""Analyze this LEGO instruction step. {step_context}

Extract and return a JSON object with:
- step_number: The step number
- parts_required: Array of parts needed (color, shape, dimensions, part_id, quantity)
- existing_assembly: Description of already assembled parts
- new_parts_to_add: Array of new parts being added
- actions: Array of actions (verb, target, destination, orientation)
- spatial_relationships: Position, rotation, alignment details
- dependencies: Prerequisites from previous steps
- notes: Special instructions

Return ONLY valid JSON, no additional text."""
    
    def _call_api_with_retry(self, image_paths: List[str], prompt: str) -> Dict[str, Any]:
        """Call DeepSeek API with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # DeepSeek uses OpenAI-compatible format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Add images (assuming base64 encoding or URLs)
        for img_path in image_paths:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"file://{img_path}"}
            })
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling DeepSeek API (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info("DeepSeek API call successful")
                return result
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. API call failed.")
                    raise
        
        raise RuntimeError("Failed to call DeepSeek API")
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse DeepSeek API response."""
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices in API response")
            
            content = choices[0].get("message", {}).get("content", "")
            
            # Try to parse as JSON
            return json.loads(content)
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse response: {e}")
            return {"error": str(e), "raw_content": content if 'content' in locals() else None}

