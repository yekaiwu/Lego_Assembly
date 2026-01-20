"""
RAG Pipeline orchestration.
Combines retrieval and generation for end-to-end query processing.
Supports image-aware queries with user assembly photos.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .retrieval import get_retriever_service
from .generator import get_generator_service
from ..models.schemas import QueryResponse, RetrievalResult, PartInfo
from ..vision.state_analyzer import get_state_analyzer
from ..graph.graph_manager import get_graph_manager


class RAGPipeline:
    """Orchestrates the RAG pipeline for query processing."""

    def __init__(self):
        """Initialize RAG pipeline components."""
        self.retriever = get_retriever_service()
        self.generator = get_generator_service()
        self.state_analyzer = get_state_analyzer()
        self.graph_manager = get_graph_manager()
    
    def process_text_query(
        self,
        manual_id: str,
        question: str,
        max_results: int = 5,
        include_images: bool = True,
        user_images: Optional[List[str]] = None
    ) -> QueryResponse:
        """
        Process a text-based query through the RAG pipeline.
        Supports optional user assembly images for context-aware responses.
        
        Args:
            manual_id: Manual identifier
            question: User's question
            max_results: Maximum number of context chunks to retrieve
            include_images: Whether to include image paths in response
            user_images: Optional list of paths to user's assembly photos
        
        Returns:
            QueryResponse with answer and sources
        """
        try:
            logger.info(f"Processing query for manual {manual_id}: {question}")
            if user_images:
                logger.info(f"Query includes {len(user_images)} user images")
            
            # Step 0: Classify query intent
            query_intent = self._classify_query_intent(question)
            logger.info(f"Query intent: {query_intent}")

            # Step 1: Analyze user images if provided (VLM analysis)
            image_analysis = None
            current_step = None
            step_confidence = 0.0

            if user_images:
                logger.info("🔍 Analyzing user assembly images with VLM...")
                image_analysis = self.state_analyzer.analyze_assembly_state(
                    image_paths=user_images,
                    manual_id=manual_id,
                    context=question
                )

                # Use GRAPH-based matching for precise step detection
                detected_parts = image_analysis.get('detected_parts', [])
                if detected_parts:
                    logger.info(f"📊 Matching {len(detected_parts)} detected parts to graph...")
                    graph_matches = self.state_analyzer.match_state_to_graph(
                        analysis_result=image_analysis,
                        manual_id=manual_id,
                        top_k=1
                    )

                    if graph_matches:
                        current_step = graph_matches[0]['step_number']
                        step_confidence = graph_matches[0]['confidence']
                        image_analysis['current_step'] = current_step
                        image_analysis['step_confidence'] = step_confidence
                        logger.info(f"✓ Graph match: Step {current_step} (confidence: {step_confidence:.2f})")

            # Step 2: Extract step number if mentioned in query (overrides detection)
            query_step = self._extract_step_number(question)
            if query_step:
                current_step = query_step
                step_confidence = 1.0
                logger.info(f"Using step from query: {current_step}")

            # Step 3: Use comprehensive RAG approach for ALL queries
            # The LLM will handle different query types using the enriched context
            logger.info("📚 Using comprehensive RAG for answer generation...")

            # Step 3: Retrieve relevant context (with image analysis if available)
            contexts = self.retriever.retrieve_context(
                query=question,
                manual_id=manual_id,
                top_k=max_results,
                step_number=current_step,
                image_analysis=image_analysis
            )

            if not contexts:
                return QueryResponse(
                    answer="I couldn't find relevant information for your question. Please make sure the manual is loaded and try rephrasing your question.",
                    sources=[],
                    current_step=current_step
                )

            # Step 4: Generate response with LLM using comprehensive context
            answer = self.generator.generate_response(
                query=question,
                contexts=contexts,
                manual_id=manual_id,
                current_step=current_step,
                image_analysis=image_analysis
            )

            # Step 5: Format sources
            sources = []
            for ctx in contexts:
                source = RetrievalResult(
                    step_number=ctx.get('step_number', 0),
                    content=ctx.get('content', ''),
                    similarity_score=ctx.get('similarity_score', 0.0),
                    metadata=ctx.get('metadata', {}),
                    image_path=ctx.get('image_path', '') if include_images else None
                )
                sources.append(source)

            # Step 6: Return comprehensive response
            # The LLM handles next step guidance and parts information based on comprehensive context
            return QueryResponse(
                answer=answer,
                sources=sources,
                current_step=current_step
            )
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return QueryResponse(
                answer="I encountered an error processing your query. Please try again.",
                sources=[]
            )
    
    def get_step_details(
        self,
        manual_id: str,
        step_number: int
    ) -> QueryResponse:
        """
        Get detailed information for a specific step.
        
        Args:
            manual_id: Manual identifier
            step_number: Step number
        
        Returns:
            QueryResponse with step details
        """
        return self.process_text_query(
            manual_id=manual_id,
            question=f"What are the detailed instructions for step {step_number}?",
            max_results=3,
            include_images=True
        )
    
    def get_next_step_info(
        self,
        manual_id: str,
        current_step: int
    ) -> QueryResponse:
        """
        Get information about the next step.
        
        Args:
            manual_id: Manual identifier
            current_step: Current step number
        
        Returns:
            QueryResponse with next step info
        """
        next_step = current_step + 1
        return self.process_text_query(
            manual_id=manual_id,
            question=f"What do I do in step {next_step}?",
            max_results=3,
            include_images=True
        )
    
    def _extract_step_number(self, question: str) -> Optional[int]:
        """
        Extract step number from query if present.
        
        Args:
            question: User question
        
        Returns:
            Step number if found, None otherwise
        """
        import re
        
        # Look for patterns like "step 5", "step number 5", "step#5"
        patterns = [
            r'step\s*#?\s*(\d+)',
            r'step\s+number\s+(\d+)',
            r'(?:^|\s)(\d+)(?:st|nd|rd|th)\s+step'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_parts_from_context(self, context: Dict[str, Any]) -> Optional[list[PartInfo]]:
        """
        Extract parts information from context metadata.
        
        Args:
            context: Context dictionary
        
        Returns:
            List of PartInfo or None
        """
        try:
            # This would need to be enhanced to parse parts from the context
            # For now, return None
            # In a full implementation, we'd parse the parts_required from the metadata
            return None
        except Exception:
            return None
    
    def _classify_query_intent(self, query: str) -> str:
        """
        Classify user query intent.

        Returns one of:
        - "next_step": What's next? What do I do now?
        - "parts_needed": What parts do I need?
        - "current_step": What step am I on?
        - "help": How do I...? General questions
        """
        query_lower = query.lower()

        # Next step patterns
        if any(p in query_lower for p in [
            "what's next", "what next", "now what", "what do i do now",
            "next step", "what should i do"
        ]):
            return "next_step"

        # Parts needed patterns
        if any(p in query_lower for p in [
            "what parts", "which parts", "what pieces", "which pieces",
            "parts needed", "parts do i need", "what do i need"
        ]):
            return "parts_needed"

        # Current step patterns
        if any(p in query_lower for p in [
            "what step", "which step", "where am i", "current step",
            "step am i on"
        ]):
            return "current_step"

        # Default: help/general query
        return "help"

    def _handle_next_step_query(
        self,
        manual_id: str,
        current_step: int,
        step_confidence: float,
        include_images: bool
    ) -> QueryResponse:
        """Handle 'what's next' queries using graph traversal."""
        logger.info(f"🔄 Graph query: Next step after {current_step}")

        next_step_num = current_step + 1
        next_step_state = self.graph_manager.get_step_state(manual_id, next_step_num)

        if not next_step_state:
            # Check if assembly is complete
            return QueryResponse(
                answer=f"🎉 Great job! You're on step {current_step} and you've completed the assembly!",
                sources=[],
                current_step=current_step,
                is_complete=True
            )

        # Build answer from graph data
        parts_needed = next_step_state.get("parts_needed", [])
        actions = next_step_state.get("actions", [])
        assembly_desc = next_step_state.get("assembly_description", "")

        answer_lines = []
        if step_confidence < 0.7:
            answer_lines.append(f"I think you're on step {current_step} (confidence: {step_confidence:.0%}).\n")

        answer_lines.append(f"**Next: Step {next_step_num}**\n")

        if assembly_desc:
            answer_lines.append(f"{assembly_desc}\n")

        if parts_needed:
            answer_lines.append("**Parts needed:**")
            for part in parts_needed:
                desc = part.get("description", "unknown")
                qty = part.get("quantity", 1)
                answer_lines.append(f"  • {qty}x {desc}")
            answer_lines.append("")

        if actions:
            answer_lines.append("**Instructions:**")
            for i, action in enumerate(actions, 1):
                answer_lines.append(f"  {i}. {action}")

        return QueryResponse(
            answer="\n".join(answer_lines),
            sources=[],
            current_step=current_step,
            next_step=next_step_num,
            parts_needed=[PartInfo(**p) for p in parts_needed] if parts_needed else None
        )

    def _handle_parts_query(
        self,
        manual_id: str,
        target_step: int,
        current_step: Optional[int],
        include_images: bool
    ) -> QueryResponse:
        """Handle 'what parts' queries using graph."""
        logger.info(f"🔧 Graph query: Parts for step {target_step}")

        step_state = self.graph_manager.get_step_state(manual_id, target_step)

        if not step_state:
            return QueryResponse(
                answer=f"Step {target_step} not found in this manual.",
                sources=[],
                current_step=current_step
            )

        parts_needed = step_state.get("parts_needed", [])
        assembly_desc = step_state.get("assembly_description", "")

        answer_lines = []
        if current_step:
            answer_lines.append(f"You're on step {current_step}.\n")

        answer_lines.append(f"**Parts needed for step {target_step}:**\n")

        if assembly_desc:
            answer_lines.append(f"_{assembly_desc}_\n")

        if parts_needed:
            for part in parts_needed:
                desc = part.get("description", "unknown")
                qty = part.get("quantity", 1)
                answer_lines.append(f"  • **{qty}x {desc}**")
        else:
            answer_lines.append("  _No specific parts listed for this step._")

        return QueryResponse(
            answer="\n".join(answer_lines),
            sources=[],
            current_step=current_step,
            target_step=target_step,
            parts_needed=[PartInfo(**p) for p in parts_needed] if parts_needed else None
        )

    def _handle_current_step_query(
        self,
        current_step: int,
        step_confidence: float,
        image_analysis: Dict[str, Any]
    ) -> QueryResponse:
        """Handle 'what step am I on' queries."""
        logger.info(f"📍 Current step query: Step {current_step}")

        answer_lines = []

        if step_confidence >= 0.8:
            answer_lines.append(f"You're on **step {current_step}**.")
        elif step_confidence >= 0.5:
            answer_lines.append(f"You appear to be on **step {current_step}** (confidence: {step_confidence:.0%}).")
        else:
            answer_lines.append(f"I think you're on **step {current_step}**, but I'm not very confident (confidence: {step_confidence:.0%}).")

        # Add detected parts info
        detected_parts = image_analysis.get('detected_parts', [])
        if detected_parts:
            answer_lines.append(f"\nI detected {len(detected_parts)} parts in your image.")

        return QueryResponse(
            answer="\n".join(answer_lines),
            sources=[],
            current_step=current_step
        )


def get_rag_pipeline() -> RAGPipeline:
    """Get RAGPipeline singleton."""
    return RAGPipeline()


