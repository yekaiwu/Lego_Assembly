"""
Multimodal Processor for LEGO Manual Ingestion.
Generates diagram descriptions using Qwen-VL for multimodal embeddings.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

# Import Phase 1 VLM client
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.litellm_vlm import UnifiedVLMClient
from src.utils.config import get_config as get_phase1_config


class MultimodalProcessor:
    """
    Processes diagram images to generate text descriptions for embedding.

    Uses configured VLM to generate rich text descriptions of diagrams, then embed those descriptions.
    """

    def __init__(self, vlm_client=None, diagram_vlm: str = None):
        """
        Initialize with configured VLM client.

        Args:
            vlm_client: Optional pre-initialized VLM client
            diagram_vlm: Optional VLM model name for diagram descriptions (from backend config)
        """
        if vlm_client is None:
            # Use DIAGRAM_VLM from Phase 1 config
            from src.vision_processing.vlm_step_extractor import VLMStepExtractor
            config = get_phase1_config()
            extractor = VLMStepExtractor()

            # Use diagram_vlm parameter if provided, otherwise fall back to config
            if diagram_vlm is None:
                diagram_vlm = config.models.diagram_vlm

            self.vlm_client = extractor._get_client(diagram_vlm)
            logger.info(f"MultimodalProcessor initialized with DIAGRAM_VLM: {diagram_vlm}")
        else:
            self.vlm_client = vlm_client
            logger.info("MultimodalProcessor initialized with provided VLM client")
    
    def generate_diagram_description(
        self,
        image_path: str,
        step_number: int,
        manual_id: str,
        step_context: Optional[str] = None
    ) -> str:
        """
        Generate rich text description of a LEGO instruction diagram using Qwen-VL.
        
        This description will be embedded alongside the text content to create
        multimodal embeddings that capture both textual and visual information.
        
        Args:
            image_path: Path to step diagram image
            step_number: Step number for context
            manual_id: Manual identifier for caching
            step_context: Optional text context from the step
        
        Returns:
            Rich text description of the diagram suitable for embedding
        """
        try:
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                return ""
            
            logger.debug(f"Generating diagram description for step {step_number}: {image_path}")
            
            # Build specialized prompt for diagram description
            prompt = self._build_diagram_description_prompt(step_number, step_context)

            # Use the generic text description method - each VLM client handles its own format
            cache_key = f"{manual_id}_step{step_number}"
            description = self.vlm_client.generate_text_description(
                image_path=image_path,
                prompt=prompt,
                cache_context=cache_key
            )
            
            if not description:
                logger.warning(f"Empty description for step {step_number}")
                return f"LEGO instruction diagram for step {step_number}"
            
            logger.debug(f"Generated {len(description)} char description for step {step_number}")
            return description
            
        except Exception as e:
            logger.error(f"Error generating diagram description: {e}")
            # Return fallback description
            return f"LEGO instruction diagram for step {step_number}"
    
    def _build_diagram_description_prompt(
        self,
        step_number: int,
        step_context: Optional[str] = None
    ) -> str:
        """
        Build specialized prompt for diagram description.
        
        Focuses on visual elements that are important for retrieval:
        - Parts shown (colors, shapes, sizes)
        - Assembly actions depicted
        - Spatial relationships
        - Visual cues and indicators
        
        Args:
            step_number: Step number
            step_context: Optional text context
        
        Returns:
            Formatted prompt
        """
        prompt = f"""Describe this LEGO instruction diagram for step {step_number} in detail.

Focus on the following for accurate retrieval:

1. **Parts Visible**: List all LEGO parts shown in the diagram
   - Colors (red, blue, yellow, black, etc.)
   - Shapes and types (brick, plate, tile, technic, etc.)
   - Sizes (2x4, 1x2, 1x1, etc.)
   - Quantities

2. **Assembly Actions**: Describe what actions are being shown
   - What is being attached/connected
   - Where new parts are being placed
   - Direction of assembly (top, bottom, left, right)
   - Rotation or orientation

3. **Visual Indicators**: Note any special visual cues
   - Arrows showing direction
   - Circles or highlights indicating focus areas
   - Dotted lines showing hidden connections
   - Zoom-in details or callouts

4. **Spatial Relationships**: Describe the layout
   - Relative positions of parts
   - How parts connect together
   - Overall structure being built

5. **Context**: If this step builds on previous work
   - What existing assembly is shown
   - How it relates to the full model

"""
        
        if step_context:
            prompt += f"\n**Text Context**: {step_context[:200]}\n"
        
        prompt += "\nProvide a detailed, searchable description that captures both what is shown and what action is being performed."
        
        return prompt
    
    def generate_fused_embedding(
        self,
        text_content: str,
        diagram_description: str,
        embedding_client: Any,
        text_weight: float = 0.4,
        diagram_weight: float = 0.6
    ) -> List[float]:
        """
        Generate fused embedding from text and diagram description.

        Combines text embedding and diagram description embedding using weighted average.
        Diagram weight is higher (0.6) because visual information is often more important
        for LEGO assembly instructions.

        Args:
            text_content: Text content of the step
            diagram_description: VLM-generated description of diagram
            embedding_client: Embedding client (Gemini, Qwen, etc.) for generating embeddings
            text_weight: Weight for text embedding (default: 0.4)
            diagram_weight: Weight for diagram embedding (default: 0.6)

        Returns:
            Fused embedding vector
        """
        try:
            # Generate embeddings for both text and diagram description
            embeddings = embedding_client.get_embeddings([text_content, diagram_description])

            if not embeddings or len(embeddings) < 2:
                logger.warning("Insufficient embeddings returned, returning None")
                return None

            text_embedding = embeddings[0]
            diagram_embedding = embeddings[1]

            # Weighted fusion
            fused = [
                text_weight * t + diagram_weight * d
                for t, d in zip(text_embedding, diagram_embedding)
            ]

            logger.debug(f"Generated fused embedding (dim: {len(fused)})")
            return fused

        except Exception as e:
            logger.error(f"Error generating fused embedding: {e}")
            # Return None to signal failure - caller should handle fallback
            return None
    
    def process_step_multimodal(
        self,
        step_text: str,
        image_path: Optional[str],
        step_number: int,
        manual_id: str,
        embedding_client: Any
    ) -> tuple[str, List[float], bool]:
        """
        Process a single step to generate multimodal content and embedding.

        Args:
            step_text: Text content of the step
            image_path: Path to step diagram (or None)
            step_number: Step number
            manual_id: Manual identifier
            embedding_client: Embedding client (Gemini, Qwen, etc.) for embeddings

        Returns:
            Tuple of (combined_text, fused_embedding, has_diagram)
        """
        has_diagram = bool(image_path and Path(image_path).exists())
        
        if has_diagram:
            # Generate diagram description
            diagram_description = self.generate_diagram_description(
                image_path,
                step_number,
                manual_id,
                step_context=step_text[:300]  # First 300 chars as context
            )

            # Combine text and diagram description for storage
            combined_text = f"{step_text}\n\nDiagram Description: {diagram_description}"

            # Generate fused embedding
            fused_embedding = self.generate_fused_embedding(
                step_text,
                diagram_description,
                embedding_client
            )

            # Check if embedding generation succeeded
            if fused_embedding is not None:
                logger.info(f"Step {step_number}: Multimodal processing complete (text + diagram)")
                return combined_text, fused_embedding, True
            else:
                logger.warning(f"Step {step_number}: Fused embedding failed, falling back to text-only")
                # Fall through to text-only processing below

        # Text-only fallback (either no diagram or embedding generation failed)
        try:
            embeddings = embedding_client.get_embeddings([step_text])
            if embeddings and len(embeddings) > 0:
                text_only_embedding = embeddings[0]
                logger.info(f"Step {step_number}: Text-only processing")
                return step_text, text_only_embedding, False
            else:
                logger.error(f"Step {step_number}: Empty embeddings returned")
                return step_text, None, False
        except Exception as e:
            logger.error(f"Step {step_number}: Text embedding failed: {e}")
            return step_text, None, False


def get_multimodal_processor(diagram_vlm: str = None) -> MultimodalProcessor:
    """
    Get MultimodalProcessor instance.

    Args:
        diagram_vlm: Optional VLM model name for diagram descriptions

    Returns:
        MultimodalProcessor instance
    """
    return MultimodalProcessor(diagram_vlm=diagram_vlm)



