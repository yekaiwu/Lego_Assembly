"""
Base Query Handler - Abstract base class for all query handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger


class BaseQueryHandler(ABC):
    """
    Abstract base class for query handlers.

    All handlers must implement the handle() method.
    """

    def __init__(self, *args, **kwargs):
        """Initialize handler with dependencies."""
        self.logger = logger
        logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def handle(
        self,
        query: str,
        manual_id: str,
        image_path: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle user query.

        Args:
            query: User's text query
            manual_id: Manual identifier
            image_path: Optional path to user photo
            context: Additional context (current_step, conversation history, etc.)

        Returns:
            Response dictionary matching QueryResponse schema
        """
        pass

    def _format_response(
        self,
        answer: str,
        confidence: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Helper to format response dictionary.

        Args:
            answer: Text response
            confidence: Confidence score (0.0-1.0)
            **kwargs: Additional fields (current_step, parts_needed, etc.)

        Returns:
            Formatted response dict
        """
        response = {
            "answer": answer,
            "confidence": confidence,
        }
        response.update(kwargs)
        return response
