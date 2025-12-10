"""
Kimi (Moonshot AI) API integration.
Fallback VLM with strong multi-modal and context understanding.
"""

import json
import time
from typing import List, Dict, Any, Optional
import requests
from loguru import logger

from ..utils.config import get_config
from ..utils.cache import get_cache

class KimiClient:
    """Client for Kimi vision capabilities via Moonshot AI."""
    
    def __init__(self):
        config = get_config()
        self.api_key = config.api.moonshot_api_key
        self.endpoint = config.api.moonshot_endpoint
        self.max_retries = config.models.max_retries
        self.timeout = config.models.request_timeout
        self.cache = get_cache()
        
        if not self.api_key:
            logger.warning("MOONSHOT_API_KEY not set. Kimi will not be available.")
    
    def extract_step_info(
        self, 
        image_paths: List[str], 
        step_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract structured information from a LEGO instruction step."""
        # Check cache
        cache_key = f"kimi-vision:{','.join(image_paths)}:{step_number}"
        cached = self.cache.get("kimi-vision", cache_key, image_paths)
        if cached:
            return cached
        
        # Build prompt
        prompt = self._build_extraction_prompt(step_number)
        
        # Make API call
        response = self._call_api_with_retry(image_paths, prompt)
        
        # Parse response
        result = self._parse_response(response)
        
        # Cache result
        self.cache.set("kimi-vision", cache_key, result, image_paths)
        
        return result
    
    def _build_extraction_prompt(self, step_number: Optional[int]) -> str:
        """Build extraction prompt for Kimi."""
        step_context = f"Step {step_number}: " if step_number else ""
        
        return f"""请分析这个乐高说明书步骤。{step_context}

提取以下信息并以JSON格式返回：
{{
  "step_number": <步骤编号>,
  "parts_required": [
    {{"description": "零件描述", "color": "颜色", "shape": "形状尺寸", "part_id": "零件ID", "quantity": 数量}}
  ],
  "existing_assembly": "已组装部分描述",
  "new_parts_to_add": ["要添加的新零件"],
  "actions": [
    {{"action_verb": "连接|放置|对齐|旋转", "target": "目标物", "destination": "放置位置", "orientation": "方向"}}
  ],
  "spatial_relationships": {{"position": "位置", "rotation": "旋转", "alignment": "对齐"}},
  "dependencies": "依赖的前置步骤",
  "notes": "特殊说明"
}}

请仅返回有效的JSON，不要额外文字。"""
    
    def _call_api_with_retry(self, image_paths: List[str], prompt: str) -> Dict[str, Any]:
        """Call Kimi API with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Moonshot uses OpenAI-compatible format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Add images
        for img_path in image_paths:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"file://{img_path}"}
            })
        
        payload = {
            "model": "moonshot-v1-vision",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Kimi API (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info("Kimi API call successful")
                return result
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. API call failed.")
                    raise
        
        raise RuntimeError("Failed to call Kimi API")
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Kimi API response."""
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

