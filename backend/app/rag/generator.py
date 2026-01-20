"""
Generator component for RAG pipeline.
Handles LLM-based response generation with retrieved context.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..config import get_settings
from ..llm.litellm_client import UnifiedLLMClient
from ..graph.graph_manager import get_graph_manager


class GeneratorService:
    """Handles LLM-based response generation."""

    def __init__(self):
        """Initialize generator with configured LLM client."""
        self.settings = get_settings()
        self.graph_manager = get_graph_manager()

        # Initialize LiteLLM client
        api_keys = self.settings.get_api_keys_dict()
        self.client = UnifiedLLMClient(
            model=self.settings.rag_llm_model,
            api_keys=api_keys
        )

        logger.info(f"Initialized LiteLLM generator with model: {self.settings.rag_llm_model}")
    
    def generate_response(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        manual_id: str,
        include_graph_context: bool = True,
        current_step: Optional[int] = None,
        next_step: Optional[int] = None,
        image_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response using retrieved contexts with comprehensive enrichment.

        Args:
            query: User query
            contexts: Retrieved context chunks
            manual_id: Manual identifier
            include_graph_context: Whether to enrich with graph context
            current_step: Optional current step number (step completed/matched)
            next_step: Optional next step number (step to do next)
            image_analysis: Optional image analysis results for context

        Returns:
            Generated response text
        """
        try:
            # Enrich contexts with graph information if available
            if include_graph_context:
                contexts = self.enrich_with_graph_context(contexts, manual_id)

            # Build COMPREHENSIVE context (includes metadata, state, etc.)
            comprehensive_context = self._build_comprehensive_context(
                manual_id=manual_id,
                query=query,
                contexts=contexts,
                current_step=current_step,
                next_step=next_step,
                image_analysis=image_analysis
            )

            # Enhanced system prompt for flexible query handling
            system_prompt = """You are a helpful LEGO assembly assistant with access to comprehensive assembly information. Your role is to answer questions about LEGO model assembly.

You have access to:
- Model information (what is being built, total parts, steps, components)
- Step-by-step assembly instructions
- Assembly structure (subassemblies, hierarchies, part relationships)
- Current assembly state (if provided)
- Detected parts from user photos (if provided)

**Your capabilities:**
- Answer questions about what model is being built
- Provide step-by-step guidance for assembly
- Identify parts and their locations
- Explain assembly structure and relationships
- Help with troubleshooting and corrections
- Estimate current progress and next steps

**Guidelines:**
- For "what am I building" questions: Use model metadata and subassembly clues
- For step questions: Reference specific step numbers
- For part questions: Mention color and shape clearly
- For structural questions: Reference subassemblies and hierarchies
- Be confident when you have clear information
- If information is missing, say so clearly

Use the provided context to answer accurately. Reference specific details (step numbers, part colors/shapes, subassembly names) to support your answers."""

            # Create user prompt with comprehensive context
            user_prompt = f"""Manual ID: {manual_id}

User Question: {query}

{comprehensive_context}

Please provide a helpful, accurate answer based on all the context above."""

            # Call LLM with increased max_tokens for comprehensive answers
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            answer = self.client.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=1500
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

    def _build_comprehensive_context(
        self,
        manual_id: str,
        query: str,
        contexts: List[Dict[str, Any]],
        current_step: Optional[int] = None,
        next_step: Optional[int] = None,
        image_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build comprehensive context string with all available information.

        Includes:
        - Model metadata (model name, total steps, parts, subassemblies)
        - Graph overview (key components, structure)
        - Current assembly state (progress, completion, next step)
        - Detected state from images (visible parts)
        - Retrieved step contexts

        Args:
            manual_id: Manual identifier
            query: User query
            contexts: Retrieved context chunks
            current_step: Optional current step number (step completed/matched)
            next_step: Optional next step number (step to do next)
            image_analysis: Optional image analysis results

        Returns:
            Comprehensive context string
        """
        context_parts = []

        # 1. MODEL METADATA (for "what am I building" queries)
        try:
            graph = self.graph_manager.load_graph(manual_id)
            if graph:
                metadata = graph.get("metadata", {})
                nodes = graph.get("nodes", [])

                # Get model node
                model_node = next((n for n in nodes if n.get("type") == "model"), None)

                model_info_parts = ["## Model Information"]
                model_info_parts.append(f"- Model: {model_node.get('name', 'LEGO Model') if model_node else 'Unknown'}")
                model_info_parts.append(f"- Total Steps: {metadata.get('total_steps', 'Unknown')}")
                model_info_parts.append(f"- Total Parts: {metadata.get('total_parts', 'Unknown')}")
                model_info_parts.append(f"- Total Subassemblies: {metadata.get('total_subassemblies', 'Unknown')}")

                # Get descriptive subassembly names for model clues
                subassemblies = [n for n in nodes if n.get("type") == "subassembly"]
                if subassemblies:
                    early_subassemblies = [s for s in subassemblies if s.get("step_created", 999) <= 5]
                    if early_subassemblies:
                        subasm_names = [s.get("name", "") for s in early_subassemblies[:5]]
                        model_info_parts.append(f"- Key Components: {', '.join(subasm_names)}")

                context_parts.append("\n".join(model_info_parts))

        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")

        # 2. CURRENT ASSEMBLY STATE (if available)
        if current_step:
            state_info_parts = ["## Current Assembly State"]
            state_info_parts.append(f"- Completed Step: {current_step}")

            if next_step:
                state_info_parts.append(f"- Next Step: {next_step}")

            # Get step state from graph
            try:
                graph = self.graph_manager.load_graph(manual_id)
                if graph:
                    step_state = self.graph_manager.get_step_state(manual_id, current_step)
                    if step_state:
                        final_state = step_state.get("final_state", {})
                        completion = final_state.get("completion_percentage", 0)
                        state_info_parts.append(f"- Progress: {completion:.1f}% complete")
            except Exception as e:
                logger.warning(f"Could not load step state: {e}")

            context_parts.append("\n".join(state_info_parts))

        # 3. DETECTED STATE FROM IMAGES (if available)
        if image_analysis:
            detected_parts = image_analysis.get("detected_parts", [])
            if detected_parts:
                detected_info_parts = ["## Detected Assembly State"]
                part_summary = ", ".join([
                    f"{p.get('quantity', 1)}x {p.get('color', '')} {p.get('shape', '')}"
                    for p in detected_parts[:5]
                ])
                detected_info_parts.append(f"- Visible Parts: {part_summary}")
                context_parts.append("\n".join(detected_info_parts))

        # 4. RETRIEVED STEP CONTEXTS
        if contexts:
            context_parts.append("## Assembly Instructions (Retrieved Steps)")
            context_parts.append(self._format_contexts(contexts))

        return "\n\n".join(context_parts)
    
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
    
    def enrich_with_graph_context(
        self,
        contexts: List[Dict[str, Any]],
        manual_id: str
    ) -> List[Dict[str, Any]]:
        """
        Add graph structural context to retrieved contexts.
        
        For each context:
        - Find associated graph nodes (parts, subassemblies)
        - Get parent subassemblies and hierarchical relationships
        - Get part roles and functional descriptions
        - Add this information to context
        
        Args:
            contexts: Retrieved context chunks
            manual_id: Manual identifier
        
        Returns:
            Enriched contexts with graph information
        """
        try:
            enriched_contexts = []
            
            for ctx in contexts:
                enriched_ctx = ctx.copy()
                step_num = ctx.get('step_number', 0)
                
                if not step_num or step_num < 1:
                    enriched_contexts.append(enriched_ctx)
                    continue
                
                # Get step state from graph
                step_state = self.graph_manager.get_step_state(manual_id, step_num)
                
                if step_state:
                    # Extract graph context
                    graph_context_parts = []
                    
                    # Add subassembly information
                    new_subassemblies = step_state.get('new_subassemblies_created', [])
                    if new_subassemblies:
                        subasm_names = [s['name'] for s in new_subassemblies]
                        graph_context_parts.append(
                            f"Subassemblies formed: {', '.join(subasm_names)}"
                        )
                    
                    # Add parts information with roles
                    new_parts = step_state.get('new_parts_added', [])
                    if new_parts:
                        # Get detailed part info including roles
                        part_details = []
                        for part_info in new_parts[:5]:  # Limit to top 5
                            part_node = self.graph_manager.get_node(
                                manual_id, 
                                part_info['node_id']
                            )
                            if part_node:
                                role = part_node.get('role', 'component')
                                name = part_node.get('name', '')
                                part_details.append(f"{name} ({role})")
                        
                        if part_details:
                            graph_context_parts.append(
                                f"Parts added: {', '.join(part_details)}"
                            )
                    
                    # Add completion progress
                    final_state = step_state.get('final_state', {})
                    completion = final_state.get('completion_percentage', 0)
                    if completion > 0:
                        graph_context_parts.append(
                            f"Assembly progress: {completion:.0f}% complete"
                        )
                    
                    # Combine graph context
                    if graph_context_parts:
                        graph_context = "\n[Structural Context] " + " | ".join(graph_context_parts)
                        
                        # Add to content
                        original_content = enriched_ctx.get('content', '')
                        enriched_ctx['content'] = original_content + "\n\n" + graph_context
                        enriched_ctx['graph_enriched'] = True
                        
                        logger.debug(f"Enriched step {step_num} with graph context")
                
                enriched_contexts.append(enriched_ctx)
            
            return enriched_contexts
            
        except Exception as e:
            logger.error(f"Error enriching with graph context: {e}")
            # Return original contexts if enrichment fails
            return contexts
    
    def generate_relationship_answer(
        self,
        query: str,
        manual_id: str,
        contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer that explains relationships and hierarchical context.
        
        Specialized for queries about:
        - "What is this part for?"
        - "What does this subassembly become?"
        - "What parts make up X?"
        - "Where does this attach?"
        
        Args:
            query: User query
            manual_id: Manual identifier
            contexts: Retrieved contexts
        
        Returns:
            Answer with structural/relationship explanation
        """
        try:
            # Extract entity mentioned in query (simplified)
            query_lower = query.lower()
            
            # Try to find mentioned node
            graph_info = None
            
            # Check for subassembly mentions
            if 'assembly' in query_lower or 'subassembly' in query_lower:
                # Extract potential name and search graph
                words = query_lower.split()
                for i, word in enumerate(words):
                    if 'assembly' in word and i > 0:
                        potential_name = words[i-1]
                        nodes = self.graph_manager.get_node_by_name(
                            manual_id, potential_name, fuzzy=True
                        )
                        if nodes and nodes[0].get('type') == 'subassembly':
                            node = nodes[0]
                            
                            # Get children (parts that make it up)
                            children = self.graph_manager.get_children(
                                manual_id, node['node_id']
                            )
                            
                            # Get parents (what it becomes part of)
                            parents = self.graph_manager.get_parents(
                                manual_id, node['node_id']
                            )
                            
                            graph_info = {
                                'type': 'subassembly',
                                'node': node,
                                'children': children,
                                'parents': parents
                            }
                            break
            
            # Build enhanced prompt with graph relationships
            if graph_info:
                relationship_context = self._format_relationship_context(graph_info)
                
                system_prompt = """You are a LEGO assembly expert explaining structural relationships.
                Use the provided relationship information to give a clear, hierarchical explanation."""
                
                user_prompt = f"""Query: {query}

Relationship Information:
{relationship_context}

Context from Manual:
{self._format_contexts(contexts)}

Explain the relationships and how this component fits into the overall assembly."""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                return self.client.generate(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
            else:
                # Fall back to regular generation
                return self.generate_response(query, contexts, manual_id)
                
        except Exception as e:
            logger.error(f"Error generating relationship answer: {e}")
            return self.generate_response(query, contexts, manual_id)
    
    def _format_relationship_context(self, graph_info: Dict[str, Any]) -> str:
        """Format graph relationship information."""
        parts = []
        
        node = graph_info.get('node', {})
        node_type = graph_info.get('type', 'unknown')
        
        parts.append(f"Type: {node_type}")
        parts.append(f"Name: {node.get('name', 'Unknown')}")
        parts.append(f"Created in: Step {node.get('step_created', '?')}")
        
        # Children (what it's made from)
        children = graph_info.get('children', [])
        if children:
            child_names = [c.get('name', 'Unknown') for c in children[:5]]
            parts.append(f"Made from: {', '.join(child_names)}")
        
        # Parents (what it becomes part of)
        parents = graph_info.get('parents', [])
        if parents:
            parent_names = [p.get('name', 'Unknown') for p in parents]
            parts.append(f"Becomes part of: {', '.join(parent_names)}")
        
        return "\n".join(parts)


def get_generator_service() -> GeneratorService:
    """Get GeneratorService singleton."""
    return GeneratorService()

