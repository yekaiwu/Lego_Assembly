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


class RAGPipeline:
    """Orchestrates the RAG pipeline for query processing."""
    
    def __init__(self):
        """Initialize RAG pipeline components."""
        self.retriever = get_retriever_service()
        self.generator = get_generator_service()
        self.state_analyzer = get_state_analyzer()
    
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
            
            # Step 1: Analyze user images if provided
            image_analysis = None
            if user_images:
                logger.info("Analyzing user assembly images...")
                image_analysis = self.state_analyzer.analyze_assembly_state(
                    image_paths=user_images,
                    manual_id=manual_id,
                    context=question
                )
                
                # Estimate step from detected parts
                detected_parts = image_analysis.get('detected_parts', [])
                if detected_parts:
                    step_guess = self._estimate_step_from_parts(detected_parts, manual_id)
                    if step_guess:
                        image_analysis['step_guess'] = step_guess
                        logger.info(f"Estimated step range from parts: {step_guess}")
            
            # Step 2: Extract step number if mentioned in query
            step_number = self._extract_step_number(question)
            
            # Step 3: Retrieve relevant context (with image analysis if available)
            contexts = self.retriever.retrieve_context(
                query=question,
                manual_id=manual_id,
                top_k=max_results,
                step_number=step_number,
                image_analysis=image_analysis
            )
            
            if not contexts:
                return QueryResponse(
                    answer="I couldn't find relevant information for your question. Please make sure the manual is loaded and try rephrasing your question.",
                    sources=[],
                    current_step=step_number
                )
            
            # Step 3: Generate response
            answer = self.generator.generate_response(
                query=question,
                contexts=contexts,
                manual_id=manual_id
            )
            
            # Step 4: Format sources
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
            
            # Step 5: Get next step guidance if applicable
            next_step = None
            guidance = None
            if step_number is not None:
                next_step = step_number + 1
                current_ctx = self.retriever.get_step_info(manual_id, step_number)
                next_ctx = self.retriever.get_step_info(manual_id, next_step)
                
                if current_ctx and next_ctx:
                    guidance = self.generator.generate_next_step_guidance(
                        step_number,
                        current_ctx,
                        next_ctx
                    )
            
            # Step 6: Extract parts information from top result
            parts_needed = None
            if contexts:
                parts_needed = self._extract_parts_from_context(contexts[0])
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                current_step=step_number,
                next_step=next_step,
                guidance=guidance,
                parts_needed=parts_needed
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
    
    def _estimate_step_from_parts(
        self,
        detected_parts: List[Dict[str, Any]],
        manual_id: str
    ) -> Optional[str]:
        """
        Estimate step range from detected parts in user's assembly photo.
        
        Uses semantic search to find steps that use similar parts,
        then returns the range of likely steps.
        
        Args:
            detected_parts: List of detected parts from image analysis
            manual_id: Manual identifier
        
        Returns:
            Step range string (e.g., "17-19") or None
        """
        try:
            if not detected_parts:
                return None
            
            # Build query from top detected parts
            part_keywords = []
            for part in detected_parts[:5]:  # Top 5 parts
                color = part.get('color', '')
                shape = part.get('shape', '')
                if color and shape:
                    part_keywords.append(f"{color} {shape}")
            
            if not part_keywords:
                return None
            
            query_text = "Parts: " + ", ".join(part_keywords)
            
            # Search for steps with these parts
            results = self.retriever.chroma.query(
                query_text=query_text,
                n_results=10,
                where={
                    "$and": [
                        {"manual_id": manual_id},
                        {"chunk_type": "step"}
                    ]
                }
            )
            
            if results and 'metadatas' in results and len(results['metadatas']) > 0:
                # Extract step numbers
                step_numbers = [
                    m.get('step_number', 0)
                    for m in results['metadatas'][0]
                    if m.get('step_number', 0) > 0
                ]
                
                if step_numbers:
                    min_step = min(step_numbers)
                    max_step = max(step_numbers)
                    
                    # If range is too wide, narrow it down
                    if max_step - min_step > 5:
                        # Use top 3 results for narrower range
                        step_numbers = step_numbers[:3]
                        min_step = min(step_numbers)
                        max_step = max(step_numbers)
                    
                    return f"{min_step}-{max_step}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error estimating step from parts: {e}")
            return None


def get_rag_pipeline() -> RAGPipeline:
    """Get RAGPipeline singleton."""
    return RAGPipeline()


