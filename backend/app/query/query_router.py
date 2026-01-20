"""
Query Router - Routes user queries to specialized handlers.
Determines query intent and selects appropriate processing pipeline.
"""

from typing import Dict, Any, Optional
from loguru import logger

from .query_types import QueryType, QueryRequest, QueryResponse


class QueryRouter:
    """
    Routes queries to appropriate handlers based on intent.
    """

    def __init__(self, graph_manager=None, rag_pipeline=None):
        """
        Args:
            graph_manager: GraphManager instance (optional, will use singleton)
            rag_pipeline: RAG pipeline instance (optional, will use singleton)
        """
        # Lazy import to avoid circular dependencies
        if graph_manager is None:
            from ..graph.graph_manager import get_graph_manager
            graph_manager = get_graph_manager()

        if rag_pipeline is None:
            from ..rag.pipeline import get_rag_pipeline
            rag_pipeline = get_rag_pipeline()

        self.graph_manager = graph_manager
        self.rag_pipeline = rag_pipeline

        # Initialize handlers (lazy loaded)
        self._handlers = {}

        logger.info("QueryRouter initialized")

    def _get_handler(self, query_type: QueryType):
        """Lazy load handler for query type."""
        if query_type not in self._handlers:
            if query_type == QueryType.NEXT_STEP:
                from .handlers.next_step_handler import NextStepHandler
                self._handlers[query_type] = NextStepHandler(
                    self.graph_manager,
                    self.rag_pipeline
                )
            elif query_type == QueryType.PARTS_NEEDED:
                from .handlers.parts_handler import PartsHandler
                self._handlers[query_type] = PartsHandler(
                    self.graph_manager,
                    self.rag_pipeline
                )
            elif query_type == QueryType.CURRENT_STEP:
                from .handlers.current_step_handler import CurrentStepHandler
                self._handlers[query_type] = CurrentStepHandler(
                    self.graph_manager
                )
            elif query_type == QueryType.HELP:
                from .handlers.help_handler import HelpHandler
                self._handlers[query_type] = HelpHandler(
                    self.rag_pipeline
                )

        return self._handlers.get(query_type)

    def route_query(
        self,
        user_query: str,
        manual_id: str,
        image_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route query to appropriate handler.

        Args:
            user_query: User's text query
            manual_id: Manual identifier
            image_path: Optional path to user photo
            context: Additional context (conversation history, current_step, etc.)

        Returns:
            Response dictionary (matches QueryResponse schema)
        """
        # 1. Classify query type
        query_type = self._classify_query(user_query)
        logger.info(f"Query classified as: {query_type.value}")

        # 2. Get appropriate handler
        handler = self._get_handler(query_type)
        if not handler:
            logger.warning(f"No handler for query type: {query_type}")
            return self._fallback_response(user_query)

        # 3. Delegate to handler
        try:
            response = handler.handle(
                query=user_query,
                manual_id=manual_id,
                image_path=image_path,
                context=context or {}
            )
            response["query_type"] = query_type.value
            return response

        except Exception as e:
            logger.error(f"Handler error for {query_type.value}: {e}", exc_info=True)
            return self._error_response(str(e))

    def _classify_query(self, query: str) -> QueryType:
        """
        Classify query intent using keyword matching.

        For production, could use LLM-based classification for better accuracy.
        """
        query_lower = query.lower()

        # Next step patterns
        if any(pattern in query_lower for pattern in [
            "what's next", "what next", "now what",
            "what do i do now", "next step", "what should i do"
        ]):
            return QueryType.NEXT_STEP

        # Parts needed patterns
        if any(pattern in query_lower for pattern in [
            "what parts", "which parts", "which pieces",
            "what do i need", "parts needed", "parts required",
            "what pieces", "need what"
        ]):
            return QueryType.PARTS_NEEDED

        # Current step patterns
        if any(pattern in query_lower for pattern in [
            "what step", "which step", "where am i",
            "current step", "step am i on", "what step am i"
        ]):
            return QueryType.CURRENT_STEP

        # Verification patterns
        if any(pattern in query_lower for pattern in [
            "did i do", "is this correct", "is this right",
            "verify", "check this", "correct"
        ]):
            return QueryType.VERIFICATION

        # Help patterns (catch-all for "how to")
        if any(pattern in query_lower for pattern in [
            "how do i", "how to", "help", "stuck", "confused",
            "can't", "cannot", "problem", "issue"
        ]):
            return QueryType.HELP

        return QueryType.UNKNOWN

    def _fallback_response(self, query: str) -> Dict[str, Any]:
        """Generate fallback response for unknown queries."""
        return {
            "answer": "I'm not sure what you're asking. Try questions like:\n"
                     "- What's the next step?\n"
                     "- What parts do I need?\n"
                     "- What step am I on?\n"
                     "- How do I attach this piece?",
            "query_type": "unknown",
            "confidence": 0.0,
            "metadata": {"original_query": query}
        }

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            "answer": f"Sorry, I encountered an error: {error_msg}",
            "query_type": "error",
            "confidence": 0.0,
            "error": error_msg
        }


# Singleton instance
_query_router_instance = None


def get_query_router(graph_manager=None, rag_pipeline=None) -> QueryRouter:
    """Get QueryRouter singleton instance."""
    global _query_router_instance

    if _query_router_instance is None:
        _query_router_instance = QueryRouter(graph_manager, rag_pipeline)

    return _query_router_instance
