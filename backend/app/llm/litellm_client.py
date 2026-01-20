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

    def generate_with_vision(
        self,
        prompt: str,
        image_paths: List[str],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        response_format: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text with vision input (multimodal).

        Args:
            prompt: Text prompt
            image_paths: List of paths to images
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional format ("json" for JSON mode)
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        import base64
        from pathlib import Path

        # Build message content with text + images
        content = [{"type": "text", "text": prompt}]

        # Add images
        for img_path in image_paths:
            path = Path(img_path)
            if not path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            # Read and encode image
            with open(img_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Determine mime type
            suffix = path.suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.webp': 'image/webp',
                '.gif': 'image/gif'
            }.get(suffix, 'image/png')

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            })

        messages = [{"role": "user", "content": content}]

        try:
            # Add response format if specified
            extra_params = {}
            if response_format == "json":
                extra_params["response_format"] = {"type": "json_object"}

            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_params,
                **kwargs
            )

            result = response.choices[0].message.content
            logger.debug(f"Vision generation: {len(result)} chars with {self.model}")
            return result

        except Exception as e:
            logger.error(f"Error in vision generation with {self.model}: {e}")
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
