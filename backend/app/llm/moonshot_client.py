"""
Moonshot (Kimi) LLM client for RAG generation.
Uses OpenAI-compatible API.
"""

from openai import OpenAI
from typing import List, Dict, Any
from loguru import logger


class MoonshotClient:
    """Client for Moonshot (Kimi) LLM."""
    
    def __init__(self, api_key: str, model: str = "moonshot-v1-32k"):
        """
        Initialize Moonshot client.
        
        Args:
            api_key: Moonshot API key
            model: Model name
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate text completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling Moonshot: {e}")
            raise


