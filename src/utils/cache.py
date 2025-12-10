"""
Caching utilities for VLM API responses to reduce costs and improve performance.
"""

import hashlib
import json
from typing import Any, Optional
from pathlib import Path
from diskcache import Cache
from loguru import logger

from .config import get_config

class ResponseCache:
    """Cache for VLM API responses."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        config = get_config()
        self.cache_dir = cache_dir or config.paths.cache_dir
        self.cache = Cache(str(self.cache_dir))
        self.enabled = config.cache_enabled
        logger.info(f"Response cache initialized at {self.cache_dir}")
    
    def _generate_key(self, model: str, prompt: str, images: list) -> str:
        """Generate a unique cache key based on model, prompt, and images."""
        # Create hash from model + prompt + image hashes
        hash_input = f"{model}:{prompt}"
        
        # Add image hashes if present
        if images:
            for img in images:
                if isinstance(img, str):
                    hash_input += f":{img}"
                elif isinstance(img, bytes):
                    hash_input += f":{hashlib.md5(img).hexdigest()}"
        
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def get(self, model: str, prompt: str, images: list = None) -> Optional[Any]:
        """Retrieve cached response if available."""
        if not self.enabled:
            return None
        
        images = images or []
        key = self._generate_key(model, prompt, images)
        
        cached_value = self.cache.get(key)
        if cached_value:
            logger.debug(f"Cache hit for key {key[:16]}...")
            return cached_value
        
        logger.debug(f"Cache miss for key {key[:16]}...")
        return None
    
    def set(self, model: str, prompt: str, response: Any, images: list = None, expire: int = 86400):
        """Store response in cache with expiration (default 24 hours)."""
        if not self.enabled:
            return
        
        images = images or []
        key = self._generate_key(model, prompt, images)
        self.cache.set(key, response, expire=expire)
        logger.debug(f"Cached response for key {key[:16]}...")
    
    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "volume": self.cache.volume()
        }

# Global cache instance
_cache_instance = None

def get_cache() -> ResponseCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache()
    return _cache_instance

