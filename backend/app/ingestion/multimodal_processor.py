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

from src.api.qwen_vlm import QwenVLMClient


class MultimodalProcessor:
    """
    Processes diagram images to generate text descriptions for embedding.
    
    Since Qwen doesn't provide direct image embedding API, we use Qwen-VL
    to generate rich text descriptions of diagrams, then embed those descriptions.
    """
    
    def __init__(self):
        """Initialize with Qwen-VL client."""
        self.vlm_client = QwenVLMClient()
        logger.info("MultimodalProcessor initialized with Qwen-VL client")
    
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
            
            # Prepare content for VLM
            content = [{"text": prompt}]
            
            # Add image
            if image_path.startswith('http://') or image_path.startswith('https://'):
                content.append({"image": image_path})
            else:
                image_data = self.vlm_client._encode_image_to_base64(image_path)
                content.append({"image": image_data})
            
            # Call VLM with retry logic
            response = self.vlm_client._call_api_with_retry(content, use_json_mode=False)
            result = self.vlm_client._parse_response(response, use_json_mode=False)
            
            # Extract text description
            description = result.get('raw_text', '')
            
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
        qwen_client: Any,
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
            qwen_client: Qwen client for generating embeddings
            text_weight: Weight for text embedding (default: 0.4)
            diagram_weight: Weight for diagram embedding (default: 0.6)
        
        Returns:
            Fused embedding vector
        """
        try:
            # Generate embeddings for both text and diagram description
            embeddings = qwen_client.get_embeddings([text_content, diagram_description])
            
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
            # Fallback to text-only embedding
            return qwen_client.get_embeddings([text_content])[0]
    
    def process_step_multimodal(
        self,
        step_text: str,
        image_path: Optional[str],
        step_number: int,
        manual_id: str,
        qwen_client: Any
    ) -> tuple[str, List[float], bool]:
        """
        Process a single step to generate multimodal content and embedding.
        
        Args:
            step_text: Text content of the step
            image_path: Path to step diagram (or None)
            step_number: Step number
            manual_id: Manual identifier
            qwen_client: Qwen client for embeddings
        
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
                qwen_client
            )
            
            logger.info(f"Step {step_number}: Multimodal processing complete (text + diagram)")
            return combined_text, fused_embedding, True
        else:
            # Text-only fallback
            text_only_embedding = qwen_client.get_embeddings([step_text])[0]
            logger.info(f"Step {step_number}: Text-only processing (no diagram)")
            return step_text, text_only_embedding, False


def get_multimodal_processor() -> MultimodalProcessor:
    """Get MultimodalProcessor singleton."""
    return MultimodalProcessor()



