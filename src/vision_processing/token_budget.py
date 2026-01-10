"""
Token Budget Manager for context-aware extraction.
Manages token usage to stay within context window limits and provides
dynamic adjustment of memory window size.
"""

from typing import Dict, Any
from loguru import logger


class TokenBudgetManager:
    """
    Manages token usage to stay within context window limits.
    Dynamically adjusts memory window size if approaching limits.
    """

    def __init__(self, max_tokens: int = 1_000_000):
        """
        Args:
            max_tokens: Maximum context window size (default: 1M for Gemini 2.5 Flash)
        """
        self.max_tokens = max_tokens
        self.safety_margin = 0.8  # Use only 80% of max to be safe
        self.available_tokens = int(max_tokens * self.safety_margin)

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens in text (rough: 1 token ≈ 4 characters)."""
        return len(text) // 4

    def estimate_image_tokens(self, image_path: str) -> int:
        """Estimate tokens for an image (rough: 2000-3000 per image)."""
        return 2500

    def check_budget(
        self,
        image_count: int,
        sliding_window_size: int,
        long_term_memory_size: int,
        prompt_size: int
    ) -> Dict[str, Any]:
        """
        Check if current configuration fits in token budget.

        Args:
            image_count: Number of images
            sliding_window_size: Number of steps in sliding window
            long_term_memory_size: Size of long-term memory context
            prompt_size: Size of base prompt

        Returns:
            {
                "fits": true/false,
                "estimated_usage": 6500,
                "available": 800000,
                "utilization": 0.008,
                "recommendations": {
                    "reduce_window_to": 3,
                    "compress_long_term": false
                }
            }
        """
        estimated_usage = (
            image_count * 2500 +
            sliding_window_size * 300 +
            long_term_memory_size +
            prompt_size +
            1000  # Response tokens
        )

        fits = estimated_usage <= self.available_tokens

        recommendations = {}
        if not fits:
            # Calculate how much to reduce
            excess = estimated_usage - self.available_tokens

            # Option 1: Reduce sliding window
            tokens_per_step = 300
            reduce_by = (excess // tokens_per_step) + 1
            recommendations["reduce_window_to"] = max(2, sliding_window_size - reduce_by)

            # Option 2: Compress long-term memory
            if long_term_memory_size > 1000:
                recommendations["compress_long_term"] = True

        return {
            "fits": fits,
            "estimated_usage": estimated_usage,
            "available": self.available_tokens,
            "utilization": estimated_usage / self.available_tokens,
            "recommendations": recommendations
        }

    def auto_adjust_window_size(
        self,
        current_window_size: int,
        token_estimate: int
    ) -> int:
        """
        Automatically adjust window size to fit budget.

        Args:
            current_window_size: Current window size
            token_estimate: Estimated tokens for current configuration

        Returns:
            New window size (may be smaller than current)
        """
        if token_estimate <= self.available_tokens:
            return current_window_size

        # Calculate reduction needed
        excess = token_estimate - self.available_tokens
        tokens_per_step = 300
        reduce_by = (excess // tokens_per_step) + 1

        new_size = max(2, current_window_size - reduce_by)
        logger.warning(
            f"Token budget exceeded. Reducing window size: {current_window_size} → {new_size}"
        )

        return new_size
