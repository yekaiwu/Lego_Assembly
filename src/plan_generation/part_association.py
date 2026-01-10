"""
Part Association Module (Stage 1 of Manual2Skill hierarchical graph construction).
Maps physical LEGO parts to their manual representations and assigns roles.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.qwen_vlm import QwenVLMClient

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
    
    def __init__(self):
        self.vlm_client = QwenVLMClient()
        self.prompt_manager = PromptManager()
        logger.info("PartAssociationModule initialized with PromptManager")
    
    def build_part_catalog(
        self,
        extracted_steps: List[Dict[str, Any]],
        manual_pages: List[str],
        assembly_id: str
    ) -> Dict[str, Any]:
        """
        Build complete part catalog from manual pages.
        
        Args:
            extracted_steps: Previously extracted step data
            manual_pages: List of paths to manual page images
            assembly_id: Assembly identifier for caching
        
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
            is_cover = "cover" in step.get("notes", "").lower()
            is_ad = any(keyword in step.get("notes", "").lower() 
                       for keyword in ["advertisement", "promotional", "lego.com", "visit"])
            
            if (has_parts or has_actions) and not is_cover and not is_ad:
                # This is a relevant build step page
                if i < len(manual_pages):
                    page_path = manual_pages[i]
                    logger.debug(f"✓ Page {i+1}: Relevant (has parts or actions)")
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
        
        # Call VLM with sample pages (limit to 4-6 images)
        content = [{"text": prompt}]
        for page_path in sample_pages[:6]:
            if page_path.startswith('http://') or page_path.startswith('https://'):
                content.append({"image": page_path})
            else:
                image_data = self.vlm_client._encode_image_to_base64(page_path)
                content.append({"image": image_data})
        
        try:
            response = self.vlm_client._call_api_with_retry(content, use_json_mode=True)
            result = self.vlm_client._parse_response(response, use_json_mode=True)
            
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

