"""
Unified LLM Client using LiteLLM.
Supports 100+ LLM providers with a single interface.
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger
import litellm


class UnifiedLLMClient:
    """
    Unified client for text generation and embeddings using LiteLLM.

    Supports all major providers:
    - OpenAI: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
    - Anthropic: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"
    - Google: "gemini/gemini-2.5-flash", "gemini/gemini-pro"
    - Alibaba: "dashscope/qwen-max", "dashscope/qwen-plus"
    - DeepSeek: "deepseek/deepseek-chat"
    - And 100+ more...

    For embeddings:
    - OpenAI: "text-embedding-3-large", "text-embedding-ada-002"
    - Google: "gemini/text-embedding-004"
    - Cohere: "embed-english-v3.0"
    - And more...
    """

    def __init__(
        self,
        model: str,
        api_keys: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize unified LLM client.

        Args:
            model: Model identifier (e.g., "gpt-4", "gemini/gemini-2.5-flash")
            api_keys: Optional dict of API keys (e.g., {"OPENAI_API_KEY": "sk-..."})
            **kwargs: Additional LiteLLM configuration
        """
        self.model = model

        # Set API keys in environment if provided
        if api_keys:
            for key, value in api_keys.items():
                if value:  # Only set if not None/empty
                    os.environ[key] = value

        # Configure LiteLLM
        litellm.drop_params = True  # Auto-drop unsupported params
        litellm.set_verbose = kwargs.get("verbose", False)

        logger.info(f"UnifiedLLMClient initialized with model: {model}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            content = response.choices[0].message.content
            logger.debug(f"Generated {len(content)} chars with {self.model}")
            return content

        except Exception as e:
            logger.error(f"Error calling {self.model}: {e}")
            raise

    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Get embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Optional embedding model (defaults to self.model)
            **kwargs: Additional provider-specific parameters

        Returns:
            List of embedding vectors
        """
        embedding_model = model or self.model

        try:
            # LiteLLM's embedding function
            response = litellm.embedding(
                model=embedding_model,
                input=texts,
                **kwargs
            )

            # Extract embeddings from response
            embeddings = [item["embedding"] for item in response.data]

            logger.debug(f"Generated {len(embeddings)} embeddings with {embedding_model}")
            return embeddings

        except Exception as e:
            logger.error(f"Error getting embeddings from {embedding_model}: {e}")
            raise

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Async text generation.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in async generation with {self.model}: {e}")
            raise

    async def aget_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Async embedding generation.

        Args:
            texts: List of texts to embed
            model: Optional embedding model
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors
        """
        embedding_model = model or self.model

        try:
            response = await litellm.aembedding(
                model=embedding_model,
                input=texts,
                **kwargs
            )

            return [item["embedding"] for item in response.data]

        except Exception as e:
            logger.error(f"Error in async embeddings with {embedding_model}: {e}")
            raise


def get_llm_client(
    model: str,
    api_keys: Optional[Dict[str, str]] = None,
    **kwargs
) -> UnifiedLLMClient:
    """
    Factory function to create LLM client.

    Args:
        model: Model identifier
        api_keys: Optional API keys dict
        **kwargs: Additional configuration

    Returns:
        UnifiedLLMClient instance
    """
    return UnifiedLLMClient(model=model, api_keys=api_keys, **kwargs)
