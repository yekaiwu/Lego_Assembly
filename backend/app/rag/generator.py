"""
Generator component for RAG pipeline.
Handles LLM-based response generation with retrieved context.
"""

from typing import List, Dict, Any
from loguru import logger

from ..config import get_settings
from ..llm import QwenClient, DeepSeekClient, MoonshotClient


class GeneratorService:
    """Handles LLM-based response generation."""
    
    def __init__(self):
        """Initialize generator with configured LLM client."""
        self.settings = get_settings()
        
        # Initialize appropriate LLM client
        api_key = self.settings.get_llm_api_key()
        provider = self.settings.rag_llm_provider
        model = self.settings.rag_llm_model
        
        if provider == "qwen":
            self.client = QwenClient(api_key, model)
        elif provider == "deepseek":
            self.client = DeepSeekClient(api_key, model)
        elif provider == "moonshot":
            self.client = MoonshotClient(api_key, model)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        logger.info(f"Initialized {provider} LLM for generation")
    
    def generate_response(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        manual_id: str
    ) -> str:
        """
        Generate a response using retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved context chunks
            manual_id: Manual identifier
        
        Returns:
            Generated response text
        """
        try:
            # Build context string
            context_str = self._format_contexts(contexts)
            
            # Create system prompt
            system_prompt = """You are a helpful LEGO assembly assistant. Your role is to provide clear, 
            step-by-step guidance for building LEGO models based on instruction manuals.
            
            Use the provided context to answer questions accurately. If you reference specific steps, 
            mention the step number. If parts are mentioned, describe them clearly with color and shape.
            
            Be concise but thorough. If the user asks about a specific step, provide detailed instructions 
            for that step and mention what comes before/after if relevant.
            
            If the context doesn't contain enough information to answer the question, say so clearly."""
            
            # Create user prompt
            user_prompt = f"""Manual ID: {manual_id}
            
User Question: {query}

Relevant Context:
{context_str}

Please provide a helpful answer based on the context above."""
            
            # Call LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            answer = self.client.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            logger.info(f"Generated response for query: {query[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format retrieved contexts into a readable string.
        
        Args:
            contexts: List of context dicts
        
        Returns:
            Formatted context string
        """
        if not contexts:
            return "No relevant context found."
        
        formatted_parts = []
        
        for i, ctx in enumerate(contexts, 1):
            step_num = ctx.get('step_number', 'unknown')
            content = ctx.get('content', '')
            similarity = ctx.get('similarity_score', 0)
            
            formatted_parts.append(
                f"[Step {step_num}] (Relevance: {similarity:.2f})\n{content}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def generate_next_step_guidance(
        self,
        current_step: int,
        current_step_context: Dict[str, Any],
        next_step_context: Dict[str, Any]
    ) -> str:
        """
        Generate guidance for what to do next.
        
        Args:
            current_step: Current step number
            current_step_context: Context for current step
            next_step_context: Context for next step
        
        Returns:
            Guidance text
        """
        try:
            system_prompt = """You are a LEGO assembly guide. Given information about the current step 
            and the next step, provide clear, encouraging guidance about what the user should do next."""
            
            user_prompt = f"""Current Step {current_step}:
{current_step_context.get('content', 'No information')}

Next Step {current_step + 1}:
{next_step_context.get('content', 'No information')}

Provide brief, clear guidance on what to do next."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return self.client.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
        except Exception as e:
            logger.error(f"Error generating guidance: {e}")
            return "Please proceed to the next step in your manual."


def get_generator_service() -> GeneratorService:
    """Get GeneratorService singleton."""
    return GeneratorService()

