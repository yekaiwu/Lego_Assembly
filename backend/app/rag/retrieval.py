"""
Retrieval component for RAG pipeline.
Handles semantic search and context assembly with sophisticated matching.
Supports image-aware retrieval and query augmentation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from ..vector_store.chroma_manager import get_chroma_manager
from ..config import get_settings
from .query_augmenter import get_query_augmenter
from ..graph.graph_manager import get_graph_manager


class RetrieverService:
    """Handles retrieval of relevant context from vector store with hybrid search."""
    
    def __init__(self):
        """Initialize retriever."""
        self.settings = get_settings()
        self.chroma = get_chroma_manager()
        self.query_augmenter = get_query_augmenter()
        self.graph_manager = get_graph_manager()
        
        # Keywords that indicate important queries
        self.step_keywords = ['step', 'instruction', 'how', 'what', 'where', 'build']
        self.part_keywords = ['part', 'piece', 'brick', 'plate', 'color']
        self.action_keywords = ['attach', 'connect', 'place', 'add', 'remove', 'rotate']
        
        # Keywords that indicate graph-relevant queries
        self.graph_keywords = [
            'subassembly', 'assembly', 'structure', 'section',
            'contains', 'made of', 'part of', 'what is',
            'where does', 'hierarchy', 'relationship'
        ]
    
    def retrieve_context(
        self,
        query: str,
        manual_id: str,
        top_k: int = 5,
        step_number: Optional[int] = None,
        image_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context using hybrid search with sophisticated matching.
        
        Combines:
        - Semantic similarity (vector search)
        - Keyword matching
        - Step number awareness
        - Query expansion with LLM
        - Image analysis integration
        - Re-ranking
        
        Args:
            query: User query text
            manual_id: Manual to search within
            top_k: Number of results to return
            step_number: Optional specific step to focus on
            image_analysis: Optional image analysis results from StateAnalyzer
        
        Returns:
            List of retrieval results with content and metadata
        """
        try:
            # Augment query if image analysis is available or query is vague
            augmented_query = self.query_augmenter.augment_query(
                user_query=query,
                manual_id=manual_id,
                image_analysis=image_analysis,
                current_step=step_number
            )
            
            # Use augmented query for retrieval
            if augmented_query != query:
                logger.info(f"Using augmented query for retrieval")
                query = augmented_query
            
            # Check if query is graph-relevant (structural query)
            is_graph_query = self._is_graph_relevant_query(query)
            
            # Extract step range from image analysis if available
            step_range = None
            if image_analysis and 'step_guess' in image_analysis:
                step_range = image_analysis['step_guess']
                logger.info(f"Image analysis suggests step range: {step_range}")
            
            # Extract step number from query if not provided
            if step_number is None:
                step_number = self._extract_step_number(query)
            
            # If graph-relevant query, use graph-enhanced retrieval
            if is_graph_query:
                logger.info("Using graph-enhanced retrieval for structural query")
                return self._retrieve_with_graph_context(
                    query, manual_id, top_k, image_analysis
                )
            
            # If we have step range from image, use step range retrieval
            if step_range and not step_number:
                logger.info(f"Using step range retrieval for {step_range}")
                return self._retrieve_step_range(query, manual_id, step_range, top_k, image_analysis)
            
            # If step number is specified, use step-aware retrieval
            if step_number is not None:
                logger.info(f"Using step-aware retrieval for step {step_number}")
                return self._retrieve_step_aware(query, manual_id, step_number, top_k)
            
            # Otherwise use hybrid retrieval with part boosting if image analysis available
            return self._retrieve_hybrid(query, manual_id, top_k, image_analysis)
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def _retrieve_step_aware(
        self,
        query: str,
        manual_id: str,
        step_number: int,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context focused on a specific step and its neighbors.
        
        Strategy:
        1. Try exact step match with metadata filter
        2. Get semantic results and filter for nearby steps
        3. Combine and rank
        
        Args:
            query: User query
            manual_id: Manual ID
            step_number: Target step number
            top_k: Max results
        
        Returns:
            List of contexts
        """
        contexts = []
        seen_ids = set()
        
        # Priority order: exact step first, then immediate neighbors
        step_priority = [
            (step_number, 1.0),      # Exact match - highest priority
            (step_number - 1, 0.6),  # Previous step
            (step_number + 1, 0.6),  # Next step
            (step_number - 2, 0.3),  # Two steps back
            (step_number + 2, 0.3),  # Two steps ahead
        ]
        
        for target_step, boost in step_priority:
            if target_step < 1:
                continue
            
            try:
                # Query with step number filter
                results = self.chroma.query(
                    query_text=query,
                    n_results=5,  # Get more candidates
                    where={
                        "$and": [
                            {"manual_id": manual_id},
                            {"step_number": target_step}
                        ]
                    }
                )
                
                if results and 'documents' in results and len(results['documents']) > 0:
                    documents = results['documents'][0]
                    metadatas = results['metadatas'][0]
                    distances = results['distances'][0]
                    ids = results['ids'][0]
                    
                    for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
                        if doc_id in seen_ids:
                            continue
                        seen_ids.add(doc_id)
                        
                        doc_step = metadata.get('step_number', 0)
                        
                        # Accept all results from exact step filter
                        similarity = max(0, 1 - (distance / 2))
                        
                        contexts.append({
                            "id": doc_id,
                            "content": doc,
                            "metadata": metadata,
                            "similarity_score": round(similarity, 3),
                            "step_number": doc_step,
                            "image_path": metadata.get('image_path', ''),
                            "relevance_boost": boost,
                            "final_score": round(similarity * boost, 3)
                        })
                        
                        logger.debug(f"Step-aware: found step {doc_step} (target: {target_step}), boost: {boost}, score: {similarity}")
                
            except Exception as e:
                # If step_number filter fails, try without it
                logger.warning(f"Step filter failed for step {target_step}: {e}")
                continue
            
            # If we have enough high-priority results, stop
            if len(contexts) >= top_k and boost > 0.5:
                break
        
        # If no results with step filter, fall back to semantic search
        if not contexts:
            logger.info(f"No step-specific results, falling back to semantic search")
            results = self.chroma.query(
                query_text=f"step {step_number} {query}",
                n_results=top_k * 2,
                where={"manual_id": manual_id}
            )
            
            if results and 'documents' in results and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
                    doc_step = metadata.get('step_number', 0)
                    
                    # Only include nearby steps
                    step_diff = abs(doc_step - step_number)
                    if step_diff > 3:
                        continue
                    
                    similarity = max(0, 1 - (distance / 2))
                    boost = 1.0 / (step_diff + 1)  # Closer steps get higher boost
                    
                    contexts.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": round(similarity, 3),
                        "step_number": doc_step,
                        "image_path": metadata.get('image_path', ''),
                        "relevance_boost": boost,
                        "final_score": round(similarity * boost, 3)
                    })
        
        # Sort by final score (similarity * boost)
        contexts.sort(key=lambda x: (x['relevance_boost'], x['similarity_score']), reverse=True)
        
        logger.info(f"Step-aware retrieval: {len(contexts[:top_k])} contexts for step {step_number}")
        return contexts[:top_k]
    
    def _retrieve_step_range(
        self,
        query: str,
        manual_id: str,
        step_range: str,
        top_k: int,
        image_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context focused on a range of steps (from image analysis).
        
        Args:
            query: User query
            manual_id: Manual ID
            step_range: Step range string (e.g., "17-19")
            top_k: Max results
            image_analysis: Image analysis results for part boosting
        
        Returns:
            List of contexts
        """
        try:
            # Parse step range
            min_step, max_step = map(int, step_range.split('-'))
            
            # Query with step range filter
            results = self.chroma.query(
                query_text=query,
                n_results=top_k * 2,
                where={
                    "manual_id": manual_id,
                    "$and": [
                        {"step_number": {"$gte": min_step}},
                        {"step_number": {"$lte": max_step}}
                    ]
                }
            )
            
            contexts = []
            if results and 'documents' in results and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
                    similarity = max(0, 1 - (distance / 2))
                    
                    # Apply part-based boosting if image analysis available
                    part_boost = 0.0
                    if image_analysis:
                        part_boost = self._calculate_part_match_boost(doc, image_analysis)
                    
                    final_score = similarity + part_boost
                    
                    contexts.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": round(similarity, 3),
                        "step_number": metadata.get('step_number', 0),
                        "image_path": metadata.get('image_path', ''),
                        "part_boost": round(part_boost, 3),
                        "final_score": round(final_score, 3)
                    })
            
            # Sort by final score
            contexts.sort(key=lambda x: x['final_score'], reverse=True)
            
            logger.info(f"Step range retrieval: {len(contexts[:top_k])} contexts for steps {step_range}")
            return contexts[:top_k]
            
        except Exception as e:
            logger.error(f"Error in step range retrieval: {e}")
            # Fall back to hybrid retrieval
            return self._retrieve_hybrid(query, manual_id, top_k, image_analysis)
    
    def _retrieve_hybrid(
        self,
        query: str,
        manual_id: str,
        top_k: int,
        image_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining semantic search with keyword and heuristic boosting.
        
        Args:
            query: User query
            manual_id: Manual ID
            top_k: Max results
            image_analysis: Optional image analysis for part-based boosting
        
        Returns:
            List of contexts
        """
        # Expand query with variations
        expanded_queries = self._expand_query(query)
        
        all_results = []
        seen_ids = set()
        
        # Query with each variation
        for expanded_query in expanded_queries[:2]:  # Use top 2 variations
            results = self.chroma.query(
                query_text=expanded_query,
                n_results=top_k * 2,
                where={"manual_id": manual_id}
            )
            
            if results and 'documents' in results and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    
                    # Calculate base similarity
                    similarity = max(0, 1 - (distance / 2))
                    
                    # Skip metadata chunks for general queries
                    if metadata.get('chunk_type') == 'metadata':
                        continue
                    
                    # Apply keyword boosting
                    keyword_boost = self._calculate_keyword_boost(query, doc, metadata)
                    
                    # Apply step relevance boost
                    step_boost = self._calculate_step_boost(query, metadata)
                    
                    # Apply part-based boost if image analysis available
                    part_boost = 0.0
                    if image_analysis:
                        part_boost = self._calculate_part_match_boost(doc, image_analysis)
                    
                    # Combined score
                    final_score = similarity + (keyword_boost * 0.2) + (step_boost * 0.1) + (part_boost * 0.3)
                    
                    all_results.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": round(similarity, 3),
                        "final_score": round(final_score, 3),
                        "step_number": metadata.get('step_number', 0),
                        "image_path": metadata.get('image_path', ''),
                        "keyword_boost": round(keyword_boost, 3),
                        "step_boost": round(step_boost, 3),
                        "part_boost": round(part_boost, 3)
                    })
        
        # Sort by final score and take top_k
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Apply minimum relevance threshold (relaxed)
        filtered_results = [r for r in all_results if r['final_score'] > 0.2]
        
        logger.info(f"Hybrid retrieval: {len(filtered_results[:top_k])} contexts (from {len(all_results)} candidates)")
        return filtered_results[:top_k]
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with variations for better matching.
        
        Args:
            query: Original query
        
        Returns:
            List of query variations
        """
        expansions = [query]  # Original query first
        
        query_lower = query.lower()
        
        # Add step-focused variation if asking about steps
        if 'step' in query_lower:
            expansions.append(f"assembly instructions for {query}")
        
        # Add parts-focused variation if asking about parts
        if any(kw in query_lower for kw in ['part', 'piece', 'brick', 'need']):
            expansions.append(f"parts required {query}")
        
        # Add action-focused variation if asking about actions
        if any(kw in query_lower for kw in ['how', 'attach', 'connect', 'build']):
            expansions.append(f"building instructions {query}")
        
        return expansions
    
    def _calculate_keyword_boost(
        self,
        query: str,
        document: str,
        metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate boost score based on keyword matches.
        
        Args:
            query: User query
            document: Document content
            metadata: Document metadata
        
        Returns:
            Boost score (0.0-1.0)
        """
        boost = 0.0
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Check for step keywords
        if any(kw in query_lower for kw in self.step_keywords):
            boost += 0.2
        
        # Check for part keywords
        if any(kw in query_lower for kw in self.part_keywords):
            if metadata.get('has_parts', False):
                boost += 0.3
        
        # Check for action keywords
        if any(kw in query_lower for kw in self.action_keywords):
            if 'Actions:' in document or 'attach' in doc_lower:
                boost += 0.2
        
        # Check for color mentions
        colors = ['red', 'blue', 'yellow', 'green', 'black', 'white', 'gray', 'orange', 'brown']
        query_colors = [c for c in colors if c in query_lower]
        if query_colors:
            doc_colors = [c for c in colors if c in doc_lower]
            if any(c in doc_colors for c in query_colors):
                boost += 0.3
        
        return min(boost, 1.0)
    
    def _calculate_step_boost(
        self,
        query: str,
        metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate boost based on step relevance.
        
        Args:
            query: User query
            metadata: Document metadata
        
        Returns:
            Boost score (0.0-1.0)
        """
        # Extract step number from query
        query_step = self._extract_step_number(query)
        doc_step = metadata.get('step_number', 0)
        
        if query_step is None:
            return 0.0
        
        # Exact match gets highest boost
        if doc_step == query_step:
            return 1.0
        
        # Adjacent steps get partial boost
        step_diff = abs(doc_step - query_step)
        if step_diff == 1:
            return 0.5
        elif step_diff == 2:
            return 0.2
        
        return 0.0
    
    def _extract_step_number(self, text: str) -> Optional[int]:
        """
        Extract step number from text, including ordinal words.
        
        Args:
            text: Text to search
        
        Returns:
            Step number if found, None otherwise
        """
        text_lower = text.lower()
        
        # Ordinal word mapping
        ordinal_map = {
            'first': 1, '1st': 1,
            'second': 2, '2nd': 2,
            'third': 3, '3rd': 3,
            'fourth': 4, '4th': 4,
            'fifth': 5, '5th': 5,
            'sixth': 6, '6th': 6,
            'seventh': 7, '7th': 7,
            'eighth': 8, '8th': 8,
            'ninth': 9, '9th': 9,
            'tenth': 10, '10th': 10
        }
        
        # Check for ordinal words (e.g., "first step", "the second step")
        for ordinal, number in ordinal_map.items():
            if re.search(rf'\b{ordinal}\s+step', text_lower):
                logger.debug(f"Extracted step {number} from ordinal '{ordinal}'")
                return number
        
        # Numeric patterns
        patterns = [
            r'step\s*#?\s*(\d+)',
            r'step\s+number\s+(\d+)',
            r'(?:^|\s)(\d+)(?:st|nd|rd|th)\s+step'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    number = int(match.group(1))
                    logger.debug(f"Extracted step {number} from pattern '{pattern}'")
                    return number
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def get_step_info(self, manual_id: str, step_number: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific step.
        
        Args:
            manual_id: Manual identifier
            step_number: Step number
        
        Returns:
            Step information dict or None
        """
        try:
            results = self.chroma.query(
                query_text=f"Step {step_number} assembly instructions",
                n_results=1,
                where={
                    "$and": [
                        {"manual_id": manual_id},
                        {"step_number": step_number}
                    ]
                }
            )
            
            if results and 'documents' in results and len(results['documents'][0]) > 0:
                return {
                    "content": results['documents'][0][0],
                    "metadata": results['metadatas'][0][0],
                    "step_number": step_number
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving step info: {e}")
            return None
    
    def get_contextual_steps(
        self,
        manual_id: str,
        step_number: int,
        context_window: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get current step plus surrounding steps for context.
        
        Args:
            manual_id: Manual identifier
            step_number: Current step number
            context_window: Number of steps before/after to include
        
        Returns:
            List of step contexts
        """
        contexts = []
        
        for offset in range(-context_window, context_window + 1):
            target_step = step_number + offset
            if target_step < 1:
                continue
            
            step_info = self.get_step_info(manual_id, target_step)
            if step_info:
                contexts.append(step_info)
        
        return contexts
    
    def _calculate_part_match_boost(
        self,
        document: str,
        image_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate boost score based on detected parts matching document content.
        
        Used for image-aware retrieval: boosts results that mention parts
        detected in the user's assembly photo.
        
        Args:
            document: Document content
            image_analysis: Image analysis results with detected_parts
        
        Returns:
            Boost score (0.0-1.0)
        """
        detected_parts = image_analysis.get('detected_parts', [])
        if not detected_parts:
            return 0.0
        
        content_lower = document.lower()
        matches = 0
        
        for part in detected_parts[:10]:  # Top 10 detected parts
            color = part.get('color', '').lower()
            shape = part.get('shape', '').lower()
            description = part.get('description', '').lower()
            
            # Check if part color/shape/description appears in document
            if color and color in content_lower:
                matches += 1
            if shape and shape in content_lower:
                matches += 1
            if description and len(description) > 3 and description in content_lower:
                matches += 0.5
        
        # Normalize by number of detected parts
        boost = min(matches / len(detected_parts), 1.0)
        
        if boost > 0:
            logger.debug(f"Part match boost: {boost:.3f} ({matches} matches)")
        
        return boost
    
    def _is_graph_relevant_query(self, query: str) -> bool:
        """
        Determine if query is graph-relevant (structural/relationship query).
        
        Args:
            query: User query
        
        Returns:
            True if query should use graph-enhanced retrieval
        """
        query_lower = query.lower()
        
        # Check for graph-specific keywords
        return any(keyword in query_lower for keyword in self.graph_keywords)
    
    def _retrieve_with_graph_context(
        self,
        query: str,
        manual_id: str,
        top_k: int,
        image_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using both graph structure and vector embeddings.
        
        Process:
        1. Parse query for structural entities (subassemblies, parts)
        2. Query graph to find related nodes and steps
        3. Query vector store for semantic matches
        4. Combine and re-rank results
        
        Args:
            query: User query
            manual_id: Manual identifier
            top_k: Max results
            image_analysis: Optional image analysis for additional context
        
        Returns:
            Combined and re-ranked results
        """
        try:
            # Step 1: Extract entities from query
            entities = self._extract_entities_from_query(query, manual_id)
            
            # Step 2: Query graph for relevant steps
            graph_step_numbers = set()
            
            for entity in entities:
                entity_type = entity.get('type')
                entity_name = entity.get('name')
                
                if entity_type == 'subassembly':
                    # Find subassembly nodes
                    nodes = self.graph_manager.get_node_by_name(
                        manual_id=manual_id,
                        name=entity_name,
                        fuzzy=True
                    )
                    
                    # Get steps where these subassemblies are used
                    for node in nodes:
                        if node.get('type') == 'subassembly':
                            steps = self.graph_manager.get_steps_for_node(
                                manual_id=manual_id,
                                node_id=node['node_id']
                            )
                            graph_step_numbers.update(steps)
                
                elif entity_type == 'part':
                    # Find part nodes
                    nodes = self.graph_manager.get_node_by_name(
                        manual_id=manual_id,
                        name=entity_name,
                        fuzzy=True
                    )
                    
                    # Get steps where these parts are used
                    for node in nodes:
                        if node.get('type') == 'part':
                            steps = self.graph_manager.get_steps_for_node(
                                manual_id=manual_id,
                                node_id=node['node_id']
                            )
                            graph_step_numbers.update(steps)
            
            logger.info(f"Graph query found {len(graph_step_numbers)} relevant steps")
            
            # Step 3: Get vector search results
            vector_results = self.chroma.query(
                query_text=query,
                n_results=top_k * 2,
                where={"manual_id": manual_id}
            )
            
            # Step 4: Combine and re-rank results
            contexts = []
            seen_ids = set()
            
            if vector_results and 'documents' in vector_results and len(vector_results['documents']) > 0:
                documents = vector_results['documents'][0]
                metadatas = vector_results['metadatas'][0]
                distances = vector_results['distances'][0]
                ids = vector_results['ids'][0]
                
                for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    
                    doc_step = metadata.get('step_number', 0)
                    
                    # Calculate base similarity
                    similarity = max(0, 1 - (distance / 2))
                    
                    # Calculate graph relevance boost
                    graph_boost = 0.0
                    if doc_step in graph_step_numbers:
                        graph_boost = 0.4  # High boost for graph matches
                    
                    # Calculate intersection boost
                    intersection_boost = 0.0
                    if doc_step in graph_step_numbers and similarity > 0.5:
                        intersection_boost = 0.2  # Extra boost for being in both
                    
                    # Apply part-based boost if image analysis available
                    part_boost = 0.0
                    if image_analysis:
                        part_boost = self._calculate_part_match_boost(doc, image_analysis)
                    
                    # Combined score
                    final_score = (
                        similarity * 0.4 +
                        graph_boost +
                        intersection_boost +
                        part_boost * 0.2
                    )
                    
                    contexts.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": round(similarity, 3),
                        "graph_boost": round(graph_boost, 3),
                        "intersection_boost": round(intersection_boost, 3),
                        "part_boost": round(part_boost, 3),
                        "final_score": round(final_score, 3),
                        "step_number": doc_step,
                        "image_path": metadata.get('image_path', '')
                    })
            
            # Sort by final score
            contexts.sort(key=lambda x: x['final_score'], reverse=True)
            
            logger.info(f"Graph-enhanced retrieval: {len(contexts[:top_k])} contexts "
                       f"(graph steps: {len(graph_step_numbers)})")
            return contexts[:top_k]
            
        except Exception as e:
            logger.error(f"Error in graph-enhanced retrieval: {e}")
            # Fall back to hybrid retrieval
            return self._retrieve_hybrid(query, manual_id, top_k, image_analysis)
    
    def retrieve_for_step(
        self,
        query: str,
        manual_id: str,
        step_number: int,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents specifically for a step.
        Applies step_number filter for precision.

        Args:
            query: Search query
            manual_id: Manual identifier
            step_number: Specific step number to filter by
            top_k: Number of results to return

        Returns:
            List of retrieved documents
        """
        # Add step filter to metadata
        filters = {"manual_id": manual_id, "step_number": step_number}

        logger.info(f"Retrieving for step {step_number} in {manual_id}")

        try:
            results = self.retrieve(
                query=query,
                manual_id=manual_id,
                top_k=top_k,
                filters=filters
            )

            logger.info(f"Retrieved {len(results)} docs for step {step_number}")
            return results

        except Exception as e:
            logger.error(f"Step retrieval error: {e}")
            return []

    def retrieve_step_range(
        self,
        query: str,
        manual_id: str,
        step_range: Tuple[int, int],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a range of steps.
        Useful when current step is uncertain.

        Args:
            query: Search query
            manual_id: Manual identifier
            step_range: Tuple of (start_step, end_step) inclusive
            top_k: Number of results to return

        Returns:
            List of retrieved documents, deduplicated and sorted by score
        """
        start_step, end_step = step_range
        logger.info(f"Retrieving for step range {start_step}-{end_step}")

        all_results = []
        for step in range(start_step, end_step + 1):
            results = self.retrieve_for_step(
                query, manual_id, step, top_k=2
            )
            all_results.extend(results)

        # Deduplicate based on document ID
        unique_results = {r.get("id", str(i)): r for i, r in enumerate(all_results)}.values()

        # Sort by score (if available)
        sorted_results = sorted(
            unique_results,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        return list(sorted_results)[:top_k]

    def _extract_entities_from_query(
        self,
        query: str,
        manual_id: str
    ) -> List[Dict[str, str]]:
        """
        Extract structural entities (parts, subassemblies) from query.

        Args:
            query: User query
            manual_id: Manual identifier

        Returns:
            List of entities with type and name
        """
        entities = []
        query_lower = query.lower()
        
        # Simple keyword-based extraction
        # In full implementation, could use NER or LLM
        
        # Check for subassembly mentions
        if 'assembly' in query_lower or 'subassembly' in query_lower:
            # Extract potential subassembly name
            # Simple heuristic: words between "the" and "assembly/subassembly"
            import re
            pattern = r'(?:the\s+)?(\w+(?:\s+\w+)*?)\s+(?:sub)?assembly'
            matches = re.findall(pattern, query_lower)
            
            for match in matches:
                entities.append({
                    'type': 'subassembly',
                    'name': match.strip()
                })
        
        # Check for part mentions (colors + shapes)
        colors = ['red', 'blue', 'yellow', 'green', 'black', 'white', 'gray', 'grey', 'orange', 'brown']
        shapes = ['brick', 'plate', 'tile', 'wheel', 'axle', 'pin', 'connector']
        
        for color in colors:
            for shape in shapes:
                if color in query_lower and shape in query_lower:
                    entities.append({
                        'type': 'part',
                        'name': f"{color} {shape}"
                    })
        
        logger.debug(f"Extracted {len(entities)} entities from query")
        return entities


def get_retriever_service() -> RetrieverService:
    """Get RetrieverService singleton."""
    return RetrieverService()

