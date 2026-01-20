"""
Help Handler - Handles general "how to" and help queries.
Uses RAG to retrieve relevant manual content.
"""

from typing import Dict, Any, Optional
from loguru import logger

from .base_handler import BaseQueryHandler


class HelpHandler(BaseQueryHandler):
    """
    Handles general help queries.

    Flow:
    1. Augment query with available context
    2. Retrieve relevant information from RAG
    3. Generate helpful response
    """

    def __init__(self, rag_pipeline):
        super().__init__()
        self.rag_pipeline = rag_pipeline

    def handle(
        self,
        query: str,
        manual_id: str,
        image_path: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle help query.

        Returns:
            {
                "answer": "To attach pieces...",
                "confidence": 0.75,
                "metadata": {...}
            }
        """
        context = context or {}

        # Use RAG to answer the query
        try:
            # Query RAG system
            rag_response = self.rag_pipeline.query(
                query=query,
                manual_id=manual_id,
                current_step=context.get("current_step"),
                history=context.get("conversation_history", [])
            )

            if rag_response and "answer" in rag_response:
                answer = rag_response["answer"]
                confidence = rag_response.get("confidence", 0.7)

                return self._format_response(
                    answer=answer,
                    confidence=confidence,
                    metadata={
                        "handler": "help",
                        "rag_used": True,
                        "sources": rag_response.get("sources", [])
                    }
                )
            else:
                # RAG failed, provide generic help
                return self._generic_help_response(query)

        except Exception as e:
            logger.error(f"RAG query failed: {e}", exc_info=True)
            return self._generic_help_response(query)

    def _generic_help_response(self, query: str) -> Dict[str, Any]:
        """Provide generic help when RAG fails."""
        return self._format_response(
            answer="I'm having trouble finding specific information about that. "
                   "Could you be more specific? For example:\n"
                   "- How do I attach piece X to piece Y?\n"
                   "- What does 'align studs' mean?\n"
                   "- How do I fix a mistake on step N?",
            confidence=0.3,
            metadata={"handler": "help", "fallback": True}
        )
