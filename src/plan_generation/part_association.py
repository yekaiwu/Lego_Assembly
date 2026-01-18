"""
Part Association Module (Stage 1 of Manual2Skill hierarchical graph construction).
Maps physical LEGO parts to their manual representations and assigns roles.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.litellm_vlm import UnifiedVLMClient
from src.utils.config import get_config

# Import PromptManager from backend
backend_path = project_root / "backend"
sys.path.insert(0, str(backend_path))
from app.vision.prompt_manager import PromptManager


class PartCatalog:
    """Manages part registry with lookup and tracking capabilities."""
    
    def __init__(self):
        self.parts = {}  # part_id -> part data
        self.parts_by_signature = {}  # "color:shape" -> list of part_ids
        self.parts_by_role = {}  # role -> list of part_ids
        
    def add_part(self, part_data: Dict[str, Any]) -> str:
        """Add part to catalog and return its ID."""
        part_id = part_data["label_id"]
        self.parts[part_id] = part_data
        
        # Index by signature
        color = part_data.get("color", "").lower()
        shape = part_data.get("shape", "").lower()
        signature = f"{color}:{shape}"
        
        if signature not in self.parts_by_signature:
            self.parts_by_signature[signature] = []
        self.parts_by_signature[signature].append(part_id)
        
        # Index by role
        role = part_data.get("role", "unknown")
        if role not in self.parts_by_role:
            self.parts_by_role[role] = []
        self.parts_by_role[role].append(part_id)
        
        return part_id
    
    def get_part(self, part_id: str) -> Optional[Dict[str, Any]]:
        """Get part by ID."""
        return self.parts.get(part_id)
    
    def find_by_signature(self, color: str, shape: str) -> List[Dict[str, Any]]:
        """Find parts by color and shape."""
        signature = f"{color.lower()}:{shape.lower()}"
        part_ids = self.parts_by_signature.get(signature, [])
        return [self.parts[pid] for pid in part_ids]
    
    def find_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Find parts by role."""
        part_ids = self.parts_by_role.get(role, [])
        return [self.parts[pid] for pid in part_ids]
    
    def get_all_parts(self) -> List[Dict[str, Any]]:
        """Get all parts in catalog."""
        return list(self.parts.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Export catalog to dictionary."""
        return {
            "parts_catalog": self.parts,
            "total_parts": len(self.parts),
            "parts_by_role": {
                role: len(part_ids) 
                for role, part_ids in self.parts_by_role.items()
            }
        }


class PartAssociationModule:
    """
    Stage 1: Part Association Module.
    Analyzes manual pages to identify all unique parts and assign roles.
    """
    
    def __init__(self, vlm_client=None):
        # Use provided VLM client or get from VLMStepExtractor's client registry
        if vlm_client is None:
            # Import here to avoid circular dependency
            from src.vision_processing.vlm_step_extractor import VLMStepExtractor
            config = get_config()
            extractor = VLMStepExtractor()
            ingestion_vlm = config.models.ingestion_vlm
            self.vlm_client = extractor._get_client(ingestion_vlm)
            logger.info(f"PartAssociationModule initialized with INGESTION_VLM: {ingestion_vlm}")
        else:
            self.vlm_client = vlm_client
            logger.info("PartAssociationModule initialized with provided VLM client")
        
        self.prompt_manager = PromptManager()
    
    def build_part_catalog(
        self,
        extracted_steps: List[Dict[str, Any]],
        manual_pages: List[str],
        assembly_id: str,
        component_extractor=None
    ) -> Dict[str, Any]:
        """
        Build complete part catalog from manual pages.

        Args:
            extracted_steps: Previously extracted step data
            manual_pages: List of paths to manual page images
            assembly_id: Assembly identifier for caching
            component_extractor: Optional ComponentExtractor for extracting part images

        Returns:
            Dictionary with parts_catalog and confidence
        """
        logger.info(f"Building part catalog for assembly {assembly_id}")
        logger.info(f"Analyzing {len(manual_pages)} pages, {len(extracted_steps)} steps")

        catalog = PartCatalog()

        # Phase 1: Extract parts from already-extracted step data
        parts_from_steps = self._extract_parts_from_steps(extracted_steps)
        logger.info(f"Found {len(parts_from_steps)} unique part types from step data")

        # Phase 2: VLM analysis for part roles and detailed descriptions
        # Filter relevant pages (skip cover pages, ads, etc.)
        relevant_pages = self._filter_relevant_pages(manual_pages, extracted_steps)
        logger.info(f"Filtered to {len(relevant_pages)} relevant pages")

        # Analyze parts in batches to determine roles
        enriched_parts = self._enrich_parts_with_roles(
            parts_from_steps,
            relevant_pages[:10],  # Analyze first 10 relevant pages
            assembly_id
        )

        # Phase 2.5: Extract component images (if SAM is enabled)
        if component_extractor and component_extractor.is_enabled():
            logger.info("Extracting part images with SAM")
            enriched_parts = self._extract_part_images(
                enriched_parts,
                extracted_steps,
                manual_pages,
                component_extractor
            )

        # Phase 3: Build catalog
        for i, part_data in enumerate(enriched_parts):
            part_data["label_id"] = f"part_{i}"
            catalog.add_part(part_data)

        result = catalog.to_dict()
        result["confidence"] = self._calculate_catalog_confidence(enriched_parts)

        logger.info(f"Part catalog built: {result['total_parts']} unique parts")
        logger.info(f"  Role breakdown: {result['parts_by_role']}")
        logger.info(f"  Confidence: {result['confidence']:.2f}")

        return result
    
    def _extract_parts_from_steps(
        self, 
        extracted_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract unique parts from step data."""
        parts_map = {}  # signature -> part data
        
        for step_idx, step in enumerate(extracted_steps):
            parts_required = step.get("parts_required", [])
            
            for part in parts_required:
                color = part.get("color", "unknown").lower()
                shape = part.get("shape", "unknown").lower()
                
                if not color or not shape or color == "unknown":
                    continue
                
                signature = f"{color}:{shape}"
                
                if signature not in parts_map:
                    parts_map[signature] = {
                        "name": f"{color} {shape}",
                        "color": color,
                        "shape": shape,
                        "appearance_description": part.get("description", ""),
                        "first_appears_step": step.get("step_number", step_idx + 1),
                        "total_quantity": part.get("quantity", 1),
                        "equivalent_parts": []
                    }
                else:
                    # Update quantity and track multiple appearances
                    parts_map[signature]["total_quantity"] += part.get("quantity", 1)
        
        return list(parts_map.values())
    
    def _filter_relevant_pages(
        self,
        manual_pages: List[str],
        extracted_steps: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Filter out irrelevant pages (ads, covers, etc.).
        Uses extracted step data + optional VLM verification.
        """
        logger.info(f"Filtering {len(manual_pages)} manual pages for relevance")
        relevant_pages = []
        
        for i, step in enumerate(extracted_steps):
            # Check if this step has actual parts
            has_parts = len(step.get("parts_required", [])) > 0
            has_actions = len(step.get("actions", [])) > 0
            notes = step.get("notes") or ""
            is_cover = "cover" in notes.lower()
            is_ad = any(keyword in notes.lower()
                       for keyword in ["advertisement", "promotional", "lego.com", "visit"])

            if (has_parts or has_actions) and not is_cover and not is_ad:
                # This is a relevant build step page - use actual source page
                source_pages = step.get("_source_page_paths", [])
                if source_pages:
                    page_path = source_pages[0]
                    logger.debug(f"✓ Step {i+1}: Relevant (has parts or actions), page: {page_path}")
                    relevant_pages.append(page_path)
            else:
                # Log why page was filtered out
                if is_cover:
                    logger.debug(f"✗ Page {i+1}: Filtered (cover page)")
                elif is_ad:
                    logger.debug(f"✗ Page {i+1}: Filtered (advertisement)")
                elif not has_parts and not has_actions:
                    logger.debug(f"✗ Page {i+1}: Filtered (no parts or actions)")
        
        logger.info(f"Filtered to {len(relevant_pages)} relevant pages")
        return relevant_pages
    
    def _enrich_parts_with_roles(
        self,
        parts: List[Dict[str, Any]],
        sample_pages: List[str],
        assembly_id: str
    ) -> List[Dict[str, Any]]:
        """
        Use VLM to analyze parts and assign roles.
        
        Roles:
        - structural: Base plates, large bricks, frames (load-bearing)
        - decorative: Colored tiles, printed pieces, stickers
        - functional: Wheels, hinges, Technic pins (moving parts)
        - connector: Small plates, brackets (joining pieces)
        """
        logger.info("Enriching parts with role assignments via VLM")
        
        # Build prompt for role assignment
        prompt = self._build_role_assignment_prompt(parts)

        # Use provider-agnostic method to call VLM with custom prompt
        try:
            result = self._call_vlm_with_custom_prompt(
                prompt=prompt,
                image_paths=sample_pages[:6]
            )

            # Handle list return from extract_step_info_with_context
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            # Parse role assignments
            role_assignments = result.get("part_roles", [])
            
            # Match assignments back to parts
            enriched_parts = []
            for part in parts:
                # Find matching assignment
                matched_role = self._match_part_to_role_assignment(part, role_assignments)
                
                part_copy = part.copy()
                part_copy["role"] = matched_role.get("role", "structural")
                part_copy["role_reasoning"] = matched_role.get("reasoning", "")
                
                enriched_parts.append(part_copy)
            
            return enriched_parts
            
        except Exception as e:
            logger.warning(f"VLM role assignment failed: {e}. Using heuristics.")
            # Fallback: use heuristic role assignment
            return [self._assign_role_heuristic(p) for p in parts]

    def _extract_part_images(
        self,
        parts: List[Dict[str, Any]],
        extracted_steps: List[Dict[str, Any]],
        manual_pages: List[str],
        component_extractor
    ) -> List[Dict[str, Any]]:
        """
        Extract component images for parts using SAM.

        Args:
            parts: List of part data dictionaries
            extracted_steps: Previously extracted step data
            manual_pages: List of paths to manual page images
            component_extractor: ComponentExtractor instance

        Returns:
            Parts list with image_path field added
        """
        logger.info(f"Extracting images for {len(parts)} parts")

        parts_with_images = []
        for part in parts:
            part_copy = part.copy()

            # Find the first step where this part appears
            first_appears_step = part.get("first_appears_step", 1)

            # Find the actual page for this step from extracted_steps
            page_path = None
            for step in extracted_steps:
                if step.get("step_number") == first_appears_step:
                    source_pages = step.get("_source_page_paths", [])
                    if source_pages:
                        page_path = source_pages[0]
                    break

            if page_path:
                # Generate a unique part ID for this part
                part_id = f"{part['color']}_{part['shape']}".replace(" ", "_")

                # Get VLM center point hint from extracted_steps
                center_point = None
                for step in extracted_steps:
                    if step.get("step_number") == first_appears_step:
                        parts_required = step.get("parts_required", [])
                        # Try to match this part to a part in parts_required by color and shape
                        for part_req in parts_required:
                            if (part_req.get("color", "").lower() in part["color"].lower() or
                                part["color"].lower() in part_req.get("color", "").lower()):
                                if "center_point" in part_req:
                                    center_point = part_req["center_point"]
                                    logger.debug(f"Found center_point {center_point} for {part['name']}")
                                    break
                        if center_point:
                            break

                # Prepare part data with center point hint
                part_data_with_hint = part.copy()
                if center_point:
                    part_data_with_hint["center_point"] = center_point

                # Extract the part image using CV detection with center point hint
                image_path = component_extractor.extract_part_image(
                    part_id=part_id,
                    page_path=page_path,
                    part_data=part_data_with_hint
                )

                if image_path:
                    part_copy["image_path"] = image_path
                    logger.debug(f"✓ Extracted image for {part['name']}: {image_path}")
                else:
                    logger.debug(f"✗ Failed to extract image for {part['name']}")
            else:
                logger.warning(f"Could not find page for step {first_appears_step} for part {part['name']}")

            parts_with_images.append(part_copy)

        extracted_count = sum(1 for p in parts_with_images if p.get("image_path"))
        logger.info(f"Successfully extracted {extracted_count}/{len(parts)} part images")

        return parts_with_images

    def _find_part_bbox(
        self,
        part: Dict[str, Any],
        step_number: int,
        extracted_steps: List[Dict[str, Any]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find bounding box hint for a part from VLM extraction data.

        Args:
            part: Part dictionary from catalog
            step_number: Step number where part first appears
            extracted_steps: Raw VLM extraction data

        Returns:
            Bounding box (x1, y1, x2, y2) or None if not found
        """
        # Find the step in extracted data
        matching_step = None
        for step in extracted_steps:
            if step.get("step_number") == step_number:
                matching_step = step
                break

        if not matching_step:
            return None

        # Look for matching part in parts_required
        parts_required = matching_step.get("parts_required", [])
        for vlm_part in parts_required:
            # Match by color and shape
            if (vlm_part.get("color", "").lower() == part.get("color", "").lower() and
                vlm_part.get("shape", "").lower() in part.get("shape", "").lower()):

                # Check if bbox exists and is valid
                bbox = vlm_part.get("bbox")
                if bbox and isinstance(bbox, list) and len(bbox) == 4:
                    # Validate bbox values
                    x1, y1, x2, y2 = bbox
                    if all(isinstance(v, (int, float)) for v in bbox) and x2 > x1 and y2 > y1:
                        logger.debug(f"Found VLM bbox for {part['name']}: {bbox}")
                        return tuple(int(v) for v in bbox)

        return None

    def _call_vlm_with_custom_prompt(
        self,
        prompt: str,
        image_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Provider-agnostic VLM call with custom prompt and images.
        Handles different API formats for Gemini, OpenAI, Claude, etc.
        """
        client_type = type(self.vlm_client).__name__

        # Use the public extract_step_info_with_context method if available
        if hasattr(self.vlm_client, 'extract_step_info_with_context'):
            return self.vlm_client.extract_step_info_with_context(
                image_paths=image_paths,
                step_number=None,
                custom_prompt=prompt
            )

        # Fallback: use extract_step_info with step_number=None
        # (Not ideal but works for simple cases)
        elif hasattr(self.vlm_client, 'extract_step_info'):
            logger.warning(f"{client_type} doesn't support custom prompts, using standard extraction")
            return self.vlm_client.extract_step_info(
                image_paths=image_paths,
                step_number=None,
                use_json_mode=True
            )

        else:
            raise NotImplementedError(f"VLM client {client_type} doesn't support required methods")

    def _build_role_assignment_prompt(self, parts: List[Dict[str, Any]]) -> str:
        """Build VLM prompt for role assignment using PromptManager."""
        # Build parts list for context
        parts_list = []
        for i, part in enumerate(parts):
            parts_list.append(
                f'  {i+1}. {part["name"]} - {part.get("appearance_description", "")}'
            )
        
        parts_text = "\n".join(parts_list)
        
        # Use PromptManager to load and format the prompt
        prompt = self.prompt_manager.get_prompt(
            'part_association',
            context={'parts_list': parts_text}
        )
        
        return prompt
    
    def _match_part_to_role_assignment(
        self,
        part: Dict[str, Any],
        role_assignments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Match a part to its role assignment from VLM."""
        part_name = part["name"].lower()
        part_color = part["color"].lower()
        part_shape = part["shape"].lower()
        
        # Try exact match first
        for assignment in role_assignments:
            assign_name = assignment.get("part_name", "").lower()
            assign_color = assignment.get("color", "").lower()
            assign_shape = assignment.get("shape", "").lower()
            
            if (part_color in assign_name or assign_color in part_name) and \
               (part_shape in assign_name or assign_shape in part_name):
                return assignment
        
        # No match found, return default
        return {"role": "structural", "reasoning": "default assignment"}
    
    def _assign_role_heuristic(self, part: Dict[str, Any]) -> Dict[str, Any]:
        """Assign role using heuristics (fallback)."""
        part_copy = part.copy()
        name = part["name"].lower()
        shape = part["shape"].lower()
        
        # Heuristic rules
        if any(kw in name or kw in shape for kw in ["wheel", "axle", "hinge", "technic", "pin"]):
            part_copy["role"] = "functional"
        elif any(kw in shape for kw in ["plate 1x", "plate 2x1", "1x1", "bracket", "clip"]):
            part_copy["role"] = "connector"
        elif any(kw in shape for kw in ["tile", "printed"]):
            part_copy["role"] = "decorative"
        else:
            part_copy["role"] = "structural"
        
        part_copy["role_reasoning"] = "heuristic assignment (fallback)"
        
        return part_copy
    
    def _calculate_catalog_confidence(self, parts: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for catalog."""
        if not parts:
            return 0.0
        
        # Factors:
        # - Do we have descriptions?
        # - Do we have role assignments?
        # - Are parts well-distributed across roles?
        
        has_description = sum(1 for p in parts if p.get("appearance_description"))
        has_role = sum(1 for p in parts if p.get("role"))
        
        description_ratio = has_description / len(parts)
        role_ratio = has_role / len(parts)
        
        # Check role distribution (good if not all the same role)
        roles = [p.get("role", "unknown") for p in parts]
        unique_roles = len(set(roles))
        role_diversity = min(unique_roles / 4, 1.0)  # 4 possible roles
        
        confidence = (description_ratio * 0.3 + role_ratio * 0.4 + role_diversity * 0.3)
        
        return round(confidence, 2)

