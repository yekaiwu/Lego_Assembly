"""
Qwen LLM client for RAG generation.
Uses Alibaba Cloud DashScope API.
"""

import dashscope
from typing import List, Dict, Any
from loguru import logger


class QwenClient:
    """Client for Qwen LLM via DashScope."""
    
    def __init__(self, api_key: str, model: str = "qwen-max"):
        """
        Initialize Qwen client.
        
        Args:
            api_key: DashScope API key
            model: Model name (qwen-max, qwen-plus, qwen-turbo)
        """
        self.api_key = api_key
        self.model = model
        dashscope.api_key = api_key
    
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
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                result_format='message',
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                logger.error(f"Qwen API error: {response.message}")
                raise Exception(f"Qwen API error: {response.message}")
                
        except Exception as e:
            logger.error(f"Error calling Qwen: {e}")
            raise
    
    def get_embeddings(self, texts: List[str], model: str = "text-embedding-v2") -> List[List[float]]:
        """
        Get embeddings for texts.
        Batches requests to respect Qwen's 25-text limit per API call.
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
        
        Returns:
            List of embedding vectors
        """
        try:
            # Qwen has a limit of 25 texts per batch
            batch_size = 25
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = dashscope.TextEmbedding.call(
                    model=model,
                    input=batch
                )
                
                if response.status_code == 200:
                    batch_embeddings = [item['embedding'] for item in response.output['embeddings']]
                    all_embeddings.extend(batch_embeddings)
                    logger.debug(f"Generated embeddings for batch {i//batch_size + 1} ({len(batch)} texts)")
                else:
                    logger.error(f"Qwen embedding error: {response.message}")
                    raise Exception(f"Qwen embedding error: {response.message}")
            
            return all_embeddings
                
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

