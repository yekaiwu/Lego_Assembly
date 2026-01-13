"""
Gemini LLM client for RAG generation and embeddings.
Uses Google Generative AI API.
"""

import requests
import time
from typing import List, Dict, Any
from loguru import logger


class GeminiClient:
    """Client for Gemini LLM via Google Generative AI API."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", embedding_model: str = "text-embedding-004"):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key
            model: Model name for generation (gemini-2.5-flash, gemini-2.0-flash-exp, etc.)
            embedding_model: Model name for embeddings (text-embedding-004)
        """
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.timeout = 120

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
            # Convert messages to Gemini format (concatenate user/assistant messages)
            contents = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })

            url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }

            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            # Add delay to respect rate limits
            time.sleep(1)

            # Extract text from response
            candidates = result.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in Gemini response")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            if not parts:
                raise ValueError("No parts in Gemini response content")

            text = parts[0].get("text", "")
            return text

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error details: {error_detail}")
                except:
                    logger.error(f"Response text: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
            raise Exception(f"Gemini API error: {e}")
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            raise

    def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        Get embeddings for texts using Gemini embedding model.

        Gemini text-embedding-004:
        - Supports up to ~50 texts per batch
        - Input text length up to ~20K characters
        - Returns 768-dimensional embeddings

        Args:
            texts: List of texts to embed
            model: Embedding model name (defaults to self.embedding_model)

        Returns:
            List of embedding vectors
        """
        if model is None:
            model = self.embedding_model

        try:
            # Gemini supports larger batches than Qwen (up to 100, but we use 50 for safety)
            batch_size = 50
            all_embeddings = []

            # Gemini text-embedding supports up to ~20K characters per text
            MAX_CHARS = 20000

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Truncate texts that are too long
                truncated_batch = []
                for text in batch:
                    if len(text) > MAX_CHARS:
                        # Truncate and add indicator
                        truncated_text = text[:MAX_CHARS-20] + "... [truncated]"
                        truncated_batch.append(truncated_text)
                        logger.warning(f"Text truncated from {len(text)} to {len(truncated_text)} chars for embedding")
                    else:
                        truncated_batch.append(text)

                # Build embedding request for batch
                url = f"{self.base_url}/models/{model}:batchEmbedContents?key={self.api_key}"

                payload = {
                    "requests": [
                        {
                            "model": f"models/{model}",
                            "content": {
                                "parts": [{"text": text}]
                            }
                        }
                        for text in truncated_batch
                    ]
                }

                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()

                # Extract embeddings from response
                embeddings_data = result.get("embeddings", [])
                if not embeddings_data:
                    logger.error(f"Gemini embedding error: No embeddings returned")
                    raise Exception("Gemini embedding error: No embeddings returned")

                batch_embeddings = [item["values"] for item in embeddings_data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated embeddings for batch {i//batch_size + 1} ({len(batch)} texts)")

                # Add delay to respect rate limits
                time.sleep(1)

            return all_embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini embedding error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error details: {error_detail}")
                except:
                    logger.error(f"Response text: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
            raise Exception(f"Gemini embedding error: {e}")
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
