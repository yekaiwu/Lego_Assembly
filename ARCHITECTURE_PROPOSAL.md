# VLM Client Architecture Proposal

## Problem
Currently, each VLM client implements `extract_step_info_with_context()` separately, leading to code duplication.

## Better Design: Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseVLMClient(ABC):
    """Abstract base class for all VLM providers."""

    def __init__(self):
        self.cache = get_cache()

    # UNIVERSAL METHOD (implemented once in base class)
    def extract_step_info_with_context(
        self,
        image_paths: List[str],
        step_number: Optional[int] = None,
        custom_prompt: Optional[str] = None,
        use_json_mode: bool = True,
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Universal extraction method - works for ALL providers."""

        # Check cache
        cache_key = self._build_cache_key(image_paths, step_number, cache_context)
        cached = self.cache.get(self.model_name, cache_key, image_paths)
        if cached:
            return cached

        # Build prompt (universal)
        prompt = custom_prompt or self._build_extraction_prompt(step_number, use_json_mode)

        # Format request (provider-specific via abstract method)
        request_payload = self._format_multimodal_request(prompt, image_paths, use_json_mode)

        # Make API call (provider-specific)
        response = self._call_api(request_payload)

        # Parse response (provider-specific)
        result = self._parse_response(response, use_json_mode)

        # Cache and return
        self.cache.set(self.model_name, cache_key, result, image_paths)
        return result

    # ABSTRACT METHODS (each provider implements differently)
    @abstractmethod
    def _format_multimodal_request(
        self,
        prompt: str,
        image_paths: List[str],
        use_json_mode: bool
    ) -> Dict[str, Any]:
        """Format the API request payload. Provider-specific."""
        pass

    @abstractmethod
    def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make the actual API call. Provider-specific."""
        pass

    @abstractmethod
    def _parse_response(self, response: Dict[str, Any], use_json_mode: bool) -> Dict[str, Any]:
        """Parse the API response. Provider-specific."""
        pass

    # SHARED HELPER METHODS
    def _build_cache_key(self, image_paths, step_number, cache_context):
        """Build cache key - same for all providers."""
        cache_suffix = f":{cache_context}" if cache_context else ""
        return f"{self.model_name}:{','.join(image_paths)}:{step_number}{cache_suffix}"


# CONCRETE IMPLEMENTATIONS (only implement the differences)

class GeminiVisionClient(BaseVLMClient):
    """Gemini-specific implementation."""

    def _format_multimodal_request(self, prompt, image_paths, use_json_mode):
        """Gemini's inline_data format."""
        parts = [{"text": prompt}]
        for img_path in image_paths:
            image_data, mime_type = self._encode_image_to_base64(img_path)
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            })
        return {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 65536,
                "response_mime_type": "application/json" if use_json_mode else None
            }
        }

    def _call_api(self, payload):
        """Gemini API call logic."""
        url = f"{self.endpoint}/{self.model}:generateContent?key={self.api_key}"
        response = requests.post(url, json=payload, timeout=self.timeout)
        return response.json()


class OpenAIVisionClient(BaseVLMClient):
    """OpenAI-specific implementation."""

    def _format_multimodal_request(self, prompt, image_paths, use_json_mode):
        """OpenAI's image_url format."""
        content = [{"type": "text", "text": prompt}]
        for img_path in image_paths:
            image_data = self._encode_image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"} if use_json_mode else None
        }

    def _call_api(self, payload):
        """OpenAI API call logic."""
        url = f"{self.endpoint}/chat/completions"
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout
        )
        return response.json()
```

## Benefits of This Approach

### 1. **No Code Duplication**
- `extract_step_info_with_context()` is implemented **once** in base class
- Caching logic is universal
- Cache key generation is universal

### 2. **Easy to Add New Providers**
Only implement 3 methods:
- `_format_multimodal_request()` - API format
- `_call_api()` - HTTP request
- `_parse_response()` - Response parsing

### 3. **Consistent Behavior**
All providers handle:
- Caching identically
- Custom prompts identically
- Error handling identically

### 4. **Single Source of Truth**
Change caching logic once → affects all providers

## Migration Path

1. Create `src/api/base_vlm_client.py` with abstract base class
2. Refactor each provider to inherit from `BaseVLMClient`
3. Move duplicated code into base class
4. Each provider only implements provider-specific methods

## Example: Adding a New Provider

```python
class NewProviderClient(BaseVLMClient):
    def _format_multimodal_request(self, prompt, image_paths, use_json_mode):
        # Just implement the API format
        return {"prompt": prompt, "images": image_paths}

    def _call_api(self, payload):
        # Just implement the HTTP call
        return requests.post(self.endpoint, json=payload).json()

    def _parse_response(self, response, use_json_mode):
        # Just parse the response
        return response["result"]
```

That's it! No need to reimplement caching, prompt building, etc.
