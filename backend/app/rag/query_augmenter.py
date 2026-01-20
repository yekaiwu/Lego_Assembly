"""
Query Augmenter for RAG Pipeline.
Uses LLM to expand vague queries into precise, retrieval-optimized queries.
"""

from typing import Dict, Any, Optional
from loguru import logger

from ..config import get_settings
from ..llm.litellm_client import UnifiedLLMClient


class QueryAugmenter:
    """
    Augments user queries with context for better retrieval.

    Handles:
    - Vague queries like "What's next?" or "How do I fix this?"
    - Image analysis integration (detected parts, estimated step)
    - Manual-specific context
    - Query expansion with LEGO-specific terminology
    """

    def __init__(self):
        """Initialize query augmenter with LLM client."""
        self.settings = get_settings()

        # Initialize LiteLLM client
        api_keys = self.settings.get_api_keys_dict()
        self.llm_client = UnifiedLLMClient(
            model=self.settings.rag_llm_model,
            api_keys=api_keys
        )

        logger.info(f"QueryAugmenter initialized with model: {self.settings.rag_llm_model}")
    
    def augment_query(
        self,
        user_query: str,
        manual_id: str,
        image_analysis: Optional[Dict[str, Any]] = None,
        current_step: Optional[int] = None
    ) -> str:
        """
        Augment vague user query with context for better retrieval.
        
        Examples:
        - "What's next?" → "What is the next assembly step after step 17 using red 2x4 brick?"
        - "Fix this" → "How to correct assembly error with blue plate in step 12?"
        
        Args:
            user_query: Original user query (may be vague)
            manual_id: Manual identifier
            image_analysis: Optional image analysis results from StateAnalyzer
            current_step: Optional current step number
        
        Returns:
            Augmented query optimized for retrieval
        """
        try:
            # If query is already specific, minimal augmentation needed
            if self._is_query_specific(user_query):
                logger.debug("Query already specific, minimal augmentation")
                return user_query
            
            # Build augmentation prompt
            prompt = self._build_augmentation_prompt(
                user_query,
                manual_id,
                image_analysis,
                current_step
            )
            
            # Call LLM to augment query
            messages = [{"role": "user", "content": prompt}]
            augmented_query = self.llm_client.generate(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent augmentation
                max_tokens=200
            )
            
            # Clean up response
            augmented_query = augmented_query.strip().strip('"').strip("'")
            
            logger.info(f"Query augmented: '{user_query}' → '{augmented_query}'")
            return augmented_query
            
        except Exception as e:
            logger.error(f"Error augmenting query: {e}")
            # Return original query on error
            return user_query
    
    def _is_query_specific(self, query: str) -> bool:
        """
        Check if query is already specific enough.
        
        Specific queries contain:
        - Step numbers
        - Part descriptions (colors, shapes)
        - Specific actions (attach, connect, place)
        - Detailed questions
        
        Args:
            query: User query
        
        Returns:
            True if query is specific, False if vague
        """
        query_lower = query.lower()
        
        # Check for vague patterns
        vague_patterns = [
            "what's next", "what next", "now what",
            "help", "stuck", "confused",
            "fix this", "wrong", "error",
            "how do i", "what should i",
            "?",  # Single word question
        ]
        
        # If contains vague pattern and is short, it's vague
        for pattern in vague_patterns:
            if pattern in query_lower and len(query) < 50:
                return False
        
        # Check for specific indicators
        specific_indicators = [
            "step",  # "step 5", "step number"
            "attach", "connect", "place", "align", "rotate",  # Specific actions
            "red", "blue", "yellow", "black", "white", "green",  # Colors
            "brick", "plate", "tile", "part",  # Part types
        ]
        
        has_specific = any(indicator in query_lower for indicator in specific_indicators)
        is_long = len(query) > 30
        
        # Specific if has indicators or is long enough
        return has_specific or is_long
    
    def _build_augmentation_prompt(
        self,
        user_query: str,
        manual_id: str,
        image_analysis: Optional[Dict[str, Any]],
        current_step: Optional[int]
    ) -> str:
        """
        Build LLM prompt for query augmentation.
        
        Args:
            user_query: Original query
            manual_id: Manual identifier
            image_analysis: Image analysis results
            current_step: Current step number
        
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a LEGO assembly assistant. Rewrite the user's query to be more specific and retrieval-friendly.

User Query: "{user_query}"
Manual ID: {manual_id}
"""
        
        # Add image analysis context if available
        if image_analysis:
            detected_parts = image_analysis.get('detected_parts', [])
            step_guess = image_analysis.get('step_guess', None)
            confidence = image_analysis.get('confidence', 0.0)
            
            if step_guess:
                prompt += f"\nEstimated Current Step Range: {step_guess}"
            
            if detected_parts:
                prompt += "\nDetected Parts in User's Photo:"
                for part in detected_parts[:5]:  # Top 5 parts
                    color = part.get('color', 'unknown')
                    shape = part.get('shape', 'brick')
                    desc = part.get('description', '')
                    prompt += f"\n  - {color} {shape} ({desc})"
            
            if confidence:
                prompt += f"\nDetection Confidence: {confidence:.1%}"
        
        # Add current step context if available
        if current_step:
            prompt += f"\nCurrent Step: {current_step}"
        
        prompt += """

Requirements:
1. Keep the original intent of the query
2. Add specific context from the information above
3. Include step numbers if available
4. Mention detected parts if relevant to the query
5. Use LEGO-specific terminology (brick, plate, stud, connection, etc.)
6. Make it optimized for semantic search in instruction manuals
7. Output ONLY the rewritten query in English, nothing else

Rewritten Query:"""
        
        return prompt
    
    def augment_with_conversation_history(
        self,
        user_query: str,
        conversation_history: list[Dict[str, str]],
        manual_id: str
    ) -> str:
        """
        Augment query using conversation history for context.
        
        Useful for follow-up questions like:
        - "And then?" (after asking about step 5)
        - "What about the red one?" (after discussing parts)
        
        Args:
            user_query: Current query
            conversation_history: Previous Q&A pairs
            manual_id: Manual identifier
        
        Returns:
            Augmented query with conversation context
        """
        try:
            if not conversation_history or self._is_query_specific(user_query):
                return user_query
            
            # Build context from last 3 exchanges
            recent_history = conversation_history[-3:]
            history_text = "\n".join([
                f"Q: {item['question']}\nA: {item['answer'][:150]}..."
                for item in recent_history
            ])
            
            prompt = f"""Given the conversation history, rewrite the user's current query to be self-contained.

Conversation History:
{history_text}

Current Query: "{user_query}"
Manual ID: {manual_id}

Rewrite the query to include necessary context from the conversation history.
Output ONLY the rewritten query in English.

Rewritten Query:"""
            
            messages = [{"role": "user", "content": prompt}]
            augmented = self.llm_client.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=150
            )
            
            augmented = augmented.strip().strip('"').strip("'")
            logger.info(f"Query augmented with history: '{user_query}' → '{augmented}'")
            return augmented
            
        except Exception as e:
            logger.error(f"Error augmenting with history: {e}")
            return user_query

    def augment_with_graph_context(
        self,
        user_query: str,
        manual_id: str,
        current_step: Optional[int] = None,
        graph_manager = None
    ) -> str:
        """
        Augment query with graph-based context.

        Example:
        "What parts?" + current_step=5 + graph context
        → "What parts are needed for step 6 after completing the red base assembly?"

        Args:
            user_query: Original user query
            manual_id: Manual identifier
            current_step: Current step number if known
            graph_manager: GraphManager instance (optional)

        Returns:
            Augmented query with graph context
        """
        if not graph_manager or not current_step:
            return user_query

        try:
            # Get step context from graph
            step_context = graph_manager.get_step_context(
                manual_id,
                current_step,
                include_history=True
            )

            # Build context string
            context_parts = []

            # Add current step info
            if "step_state" in step_context:
                state = step_context["step_state"]
                assembly_desc = state.get("assembly_description", "")
                if assembly_desc:
                    context_parts.append(f"Current assembly: {assembly_desc}")

            # Add history summary
            if "history" in step_context and len(step_context["history"]) > 0:
                recent_steps = step_context["history"][-3:]  # Last 3 steps
                history_desc = ", ".join([
                    s.get("assembly_description", f"step {s.get('step_number')}")
                    for s in recent_steps if "assembly_description" in s
                ])
                if history_desc:
                    context_parts.append(f"Previous steps: {history_desc}")

            # Build augmented query
            if context_parts:
                context_str = "; ".join(context_parts)
                augmented = f"{user_query} (Context: {context_str})"
                logger.info(f"Graph-augmented query: {augmented[:100]}...")
                return augmented

        except Exception as e:
            logger.error(f"Graph augmentation error: {e}")

        return user_query


# Singleton instance
_query_augmenter_instance = None


def get_query_augmenter() -> QueryAugmenter:
    """Get QueryAugmenter singleton instance."""
    global _query_augmenter_instance
    if _query_augmenter_instance is None:
        _query_augmenter_instance = QueryAugmenter()
    return _query_augmenter_instance



