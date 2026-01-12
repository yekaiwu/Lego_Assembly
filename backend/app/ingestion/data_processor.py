"""
Data Processing and Chunking for LEGO Manual Ingestion.
Processes extracted.json, plan.json, dependencies.json, and images.
Supports multimodal embedding generation with diagram descriptions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger


class ManualDataProcessor:
    """Processes LEGO manual data for ingestion into vector store."""
    
    def __init__(self, output_dir: Path, use_multimodal: bool = True):
        """
        Initialize data processor.
        
        Args:
            output_dir: Path to output directory containing manual data
            use_multimodal: Whether to use multimodal processing (default: True)
        """
        self.output_dir = Path(output_dir)
        self.use_multimodal = use_multimodal
        
        # Import multimodal processor if needed
        if use_multimodal:
            from .multimodal_processor import get_multimodal_processor
            self.multimodal_processor = get_multimodal_processor()
            logger.info("DataProcessor initialized with multimodal support")
        else:
            self.multimodal_processor = None
            logger.info("DataProcessor initialized (text-only mode)")
    
    def load_manual_data(
        self,
        manual_id: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
        """
        Load all data files for a manual.
        
        Args:
            manual_id: Manual identifier
        
        Returns:
            Tuple of (extracted_data, plan_data, dependencies_data, plan_text)
        """
        try:
            # Load JSON files
            extracted_path = self.output_dir / f"{manual_id}_extracted.json"
            plan_path = self.output_dir / f"{manual_id}_plan.json"
            dependencies_path = self.output_dir / f"{manual_id}_dependencies.json"
            plan_txt_path = self.output_dir / f"{manual_id}_plan.txt"
            
            with open(extracted_path, 'r', encoding='utf-8') as f:
                extracted_data = json.load(f)
            
            with open(plan_path, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)
            
            with open(dependencies_path, 'r', encoding='utf-8') as f:
                dependencies_data = json.load(f)
            
            with open(plan_txt_path, 'r', encoding='utf-8') as f:
                plan_text = f.read()
            
            logger.info(f"Loaded data for manual {manual_id}")
            return extracted_data, plan_data, dependencies_data, plan_text
            
        except Exception as e:
            logger.error(f"Error loading manual data: {e}")
            raise
    
    def create_step_chunks(
        self,
        manual_id: str,
        extracted_data: List[Dict[str, Any]],
        plan_data: Dict[str, Any],
        dependencies_data: Dict[str, Any],
        qwen_client: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Create document chunks for each step with optional multimodal embeddings.
        
        Args:
            manual_id: Manual identifier
            extracted_data: Extracted step information
            plan_data: 3D plan data
            dependencies_data: Dependency graph
            qwen_client: Optional Qwen client for generating embeddings (required for multimodal)
        
        Returns:
            List of chunk dictionaries with content, metadata, and optional pre-computed embeddings
        """
        chunks = []
        
        # Get steps from plan data
        plan_steps = plan_data.get('steps', [])
        dependency_nodes = dependencies_data.get('nodes', {})
        
        for idx, extracted_step in enumerate(extracted_data):
            step_number = extracted_step.get('step_number')
            
            # Skip non-step pages (cover, intro, etc.)
            if step_number is None:
                continue
            
            # Find corresponding plan step
            plan_step = next(
                (s for s in plan_steps if s.get('step_number') == step_number),
                None
            )
            
            # Find dependency info
            dep_info = dependency_nodes.get(str(step_number), {})
            
            # Create rich text content for embedding
            content_parts = []
            
            # Add step description
            if extracted_step.get('existing_assembly'):
                content_parts.append(f"Existing assembly: {extracted_step['existing_assembly']}")
            
            # Add parts information
            parts_required = extracted_step.get('parts_required', [])
            if parts_required:
                parts_text = "Parts needed: "
                part_descriptions = []
                for part in parts_required:
                    desc = f"{part.get('color', '')} {part.get('shape', '')}".strip()
                    qty = part.get('quantity', 1)
                    if desc:
                        part_descriptions.append(f"{qty}x {desc}")
                parts_text += ", ".join(part_descriptions)
                content_parts.append(parts_text)
            
            # Add new parts to add
            new_parts = extracted_step.get('new_parts_to_add', [])
            if new_parts and isinstance(new_parts, list) and len(new_parts) > 0:
                if isinstance(new_parts[0], str):
                    content_parts.append(f"New parts: {', '.join(new_parts)}")
            
            # Add actions
            actions = extracted_step.get('actions', [])
            if actions:
                action_texts = []
                for action in actions:
                    if isinstance(action, dict):
                        verb = action.get('action_verb', '')
                        target = action.get('target', '')
                        dest = action.get('destination', '')
                        if verb and target:
                            action_texts.append(f"{verb} {target} to {dest}")
                if action_texts:
                    content_parts.append(f"Actions: {'; '.join(action_texts)}")
            
            # Add spatial relationships
            spatial = extracted_step.get('spatial_relationships', {})
            if spatial and isinstance(spatial, dict):
                spatial_text = []
                for key, value in spatial.items():
                    if value and value != "unclear":
                        spatial_text.append(f"{key}: {value}")
                if spatial_text:
                    content_parts.append(f"Spatial info: {', '.join(spatial_text)}")
            
            # Add dependencies
            dependencies = dep_info.get('dependencies', [])
            if dependencies:
                content_parts.append(f"Requires steps: {', '.join(map(str, dependencies))}")
            
            # Add notes
            notes = extracted_step.get('notes', '')
            if notes:
                content_parts.append(f"Notes: {notes}")
            
            # Combine all content
            content = "\n".join(content_parts)
            
            # Determine image path
            image_path = None
            temp_pages_dir = self.output_dir / "temp_pages"
            # Pages are typically step_number + offset for cover pages
            # Try to find the corresponding page
            for page_num in range(1, 300):  # reasonable upper bound
                page_path = temp_pages_dir / f"page_{page_num:03d}.png"
                if page_path.exists():
                    # This is a heuristic - may need adjustment
                    if page_num >= step_number:
                        image_path = str(page_path)
                        break
            
            # Process multimodal if enabled and client provided
            has_diagram = False
            pre_computed_embedding = None
            
            if self.use_multimodal and self.multimodal_processor and qwen_client and image_path:
                try:
                    # Generate multimodal content and embedding
                    multimodal_content, fused_embedding, has_diagram = \
                        self.multimodal_processor.process_step_multimodal(
                            step_text=content,
                            image_path=image_path,
                            step_number=step_number,
                            manual_id=manual_id,
                            qwen_client=qwen_client
                        )
                    
                    # Use multimodal content and embedding if successful
                    content = multimodal_content
                    # Only use embedding if it's not None
                    if fused_embedding is not None:
                        pre_computed_embedding = fused_embedding
                    else:
                        logger.warning(f"Step {step_number}: No embedding generated, will skip or use default")
                        pre_computed_embedding = None

                except Exception as e:
                    logger.error(f"Multimodal processing failed for step {step_number}: {e}")
                    # Fall back to text-only
                    has_diagram = False
                    pre_computed_embedding = None
            
            # Create chunk metadata
            metadata = {
                "manual_id": manual_id,
                "step_number": step_number,
                "chunk_type": "step",
                "has_parts": len(parts_required) > 0,
                "parts_count": len(parts_required),
                "has_dependencies": len(dependencies) > 0,
                "image_path": image_path or "",
                "has_diagram": has_diagram
            }
            
            # Add plan data if available
            if plan_step:
                # Add 3D position info if available
                if plan_step.get('parts'):
                    first_part = plan_step['parts'][0]
                    if first_part.get('position'):
                        pos = first_part['position']
                        metadata['position_x'] = pos.get('x', 0)
                        metadata['position_y'] = pos.get('y', 0)
                        metadata['position_z'] = pos.get('z', 0)
            
            # Create chunk (use idx to ensure unique IDs even if step_numbers repeat)
            chunk = {
                "id": f"{manual_id}_step_{idx}",
                "content": content,
                "metadata": metadata,
                "step_data": {
                    "step_number": step_number,
                    "parts": parts_required,
                    "actions": actions,
                    "spatial_relationships": spatial,
                    "dependencies": dependencies,
                    "notes": notes,
                    "image_path": image_path
                }
            }
            
            # Add pre-computed embedding if available
            if pre_computed_embedding:
                chunk["embedding"] = pre_computed_embedding
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} step chunks for manual {manual_id}")
        return chunks
    
    def create_manual_metadata_chunk(
        self,
        manual_id: str,
        plan_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create metadata chunk for the entire manual.
        
        Args:
            manual_id: Manual identifier
            plan_data: Plan data containing metadata
        
        Returns:
            Metadata chunk dictionary
        """
        metadata_content = f"""
        LEGO Manual {manual_id}
        Total Steps: {plan_data.get('total_steps', 'unknown')}
        Generated: {plan_data.get('generated_at', 'unknown')}
        
        This manual contains step-by-step instructions for assembling a LEGO model.
        Use this manual to guide the assembly process.
        """
        
        chunk = {
            "id": f"{manual_id}_metadata",
            "content": metadata_content.strip(),
            "metadata": {
                "manual_id": manual_id,
                "chunk_type": "metadata",
                "total_steps": plan_data.get('total_steps', 0),
                "generated_at": plan_data.get('generated_at', ''),
                "image_path": ""
            }
        }
        
        return chunk

