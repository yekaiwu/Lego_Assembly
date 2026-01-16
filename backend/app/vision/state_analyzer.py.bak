"""
State Analyzer - Vision-based analysis of user's physical assembly state.
Uses Qwen-VL to detect parts, connections, and structure from photos.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

# Add project root to path to import Phase 1 modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.litellm_vlm import UnifiedVLMClient
from src.utils.config import get_config as get_phase1_config
from ..graph.graph_manager import get_graph_manager
from .prompt_manager import PromptManager


class StateAnalyzer:
    """Analyzes user's physical assembly state from photos."""

    def __init__(self, vlm_client=None):
        """Initialize with Phase 1 VLM client."""
        if vlm_client is None:
            # Use INGESTION_VLM from Phase 1 config (for analyzing user photos)
            from src.vision_processing.vlm_step_extractor import VLMStepExtractor
            config = get_phase1_config()
            extractor = VLMStepExtractor()
            ingestion_vlm = config.models.ingestion_vlm
            self.vlm_client = extractor._get_client(ingestion_vlm)
            logger.info(f"StateAnalyzer initialized with INGESTION_VLM: {ingestion_vlm}")
        else:
            self.vlm_client = vlm_client
            logger.info("StateAnalyzer initialized with provided VLM client")
        
        self.graph_manager = get_graph_manager()
        self.prompt_manager = PromptManager()
    
    def analyze_assembly_state(
        self,
        image_paths: List[str],
        manual_id: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze current assembly state from photos.
        
        Args:
            image_paths: List of paths to user's assembly photos (1-4 images, 2+ recommended)
            manual_id: Manual identifier for context
            context: Optional additional context
        
        Returns:
            Dictionary containing:
                - detected_parts: List of visible parts with details
                - assembled_structures: Description of built structures
                - spatial_relationships: How parts are connected
                - confidence: Overall confidence score (0.0-1.0)
                - raw_analysis: Full VLM response
        """
        try:
            logger.info(f"Analyzing assembly state for manual {manual_id}")
            logger.info(f"Processing {len(image_paths)} images")
            
            # Build analysis prompt for user assembly photos
            prompt = self._build_analysis_prompt(manual_id, context)
            
            # Call VLM with custom prompt for user photo analysis
            content = [{"text": prompt}]
            for img_path in image_paths:
                if img_path.startswith('http://') or img_path.startswith('https://'):
                    content.append({"image": img_path})
                else:
                    # Encode local file as base64
                    image_data = self.vlm_client._encode_image_to_base64(img_path)
                    content.append({"image": image_data})
            
            # Call API with retry logic
            response = self.vlm_client._call_api_with_retry(content, use_json_mode=True)
            result = self.vlm_client._parse_response(response, use_json_mode=True)
            
            # Structure the response
            structured_result = self._structure_analysis_result(result)
            
            logger.info(f"Analysis complete. Detected {len(structured_result.get('detected_parts', []))} parts")
            return structured_result
            
        except Exception as e:
            logger.error(f"Error analyzing assembly state: {e}")
            return {
                "detected_parts": [],
                "assembled_structures": "Error during analysis",
                "spatial_relationships": {},
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _build_analysis_prompt(
        self,
        manual_id: str,
        context: Optional[str] = None
    ) -> str:
        """
        Build VLM prompt for analyzing user's assembly photos using PromptManager.
        
        Args:
            manual_id: Manual identifier
            context: Optional context string
        
        Returns:
            Formatted prompt string
        """
        # Use PromptManager with context variables
        prompt_context = {
            'manual_id': manual_id,
            'expected_step': 'unknown',
            'total_steps': 'unknown'
        }
        
        prompt = self.prompt_manager.get_prompt('state_analysis', context=prompt_context)
        
        if context:
            prompt += f"\n\nAdditional Context: {context}"
        
        return prompt
    
    def _structure_analysis_result(self, vlm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure and validate VLM analysis result.
        
        Args:
            vlm_result: Raw VLM response
        
        Returns:
            Structured analysis result
        """
        # Handle error case
        if "error" in vlm_result:
            return {
                "detected_parts": [],
                "assembled_structures": [],
                "connections": [],
                "spatial_layout": {},
                "confidence": 0.0,
                "error": vlm_result.get("error", "Unknown error")
            }
        
        # Extract and validate fields
        detected_parts = vlm_result.get("detected_parts", [])
        assembled_structures = vlm_result.get("assembled_structures", [])
        connections = vlm_result.get("connections", [])
        spatial_layout = vlm_result.get("spatial_layout", {})
        confidence = vlm_result.get("confidence", 0.5)
        notes = vlm_result.get("notes", "")
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, float(confidence)))
        
        return {
            "detected_parts": detected_parts,
            "assembled_structures": assembled_structures,
            "connections": connections,
            "spatial_layout": spatial_layout,
            "confidence": confidence,
            "notes": notes,
            "raw_analysis": vlm_result
        }
    
    def detect_visible_parts(
        self,
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract list of visible parts from analysis result.
        
        Args:
            analysis_result: Result from analyze_assembly_state
        
        Returns:
            List of detected parts
        """
        return analysis_result.get("detected_parts", [])
    
    def get_assembly_description(
        self,
        analysis_result: Dict[str, Any]
    ) -> str:
        """
        Generate natural language description of current assembly state.
        
        Args:
            analysis_result: Result from analyze_assembly_state
        
        Returns:
            Natural language description
        """
        parts = analysis_result.get("detected_parts", [])
        structures = analysis_result.get("assembled_structures", [])
        confidence = analysis_result.get("confidence", 0.0)
        
        if not parts and not structures:
            return "No assembled parts detected in the images."
        
        description_parts = []
        
        if structures:
            description_parts.append("Current assembly includes:")
            for struct in structures:
                description_parts.append(f"  - {struct.get('description', 'Unknown structure')}")
        
        if parts:
            part_count = len(parts)
            description_parts.append(f"\nVisible parts: {part_count} distinct pieces detected")
            
            # Summarize by color
            color_counts = {}
            for part in parts:
                color = part.get("color", "unknown")
                color_counts[color] = color_counts.get(color, 0) + part.get("quantity", 1)
            
            if color_counts:
                color_summary = ", ".join([f"{count} {color}" for color, count in color_counts.items()])
                description_parts.append(f"Colors: {color_summary}")
        
        description_parts.append(f"\nAnalysis confidence: {confidence * 100:.0f}%")
        
        return "\n".join(description_parts)
    
    def detect_subassemblies(
        self,
        image_paths: List[str],
        manual_id: str
    ) -> Dict[str, Any]:
        """
        Enhanced subassembly detection using part composition matching and completeness validation.
        
        Args:
            image_paths: List of paths to user's assembly photos
            manual_id: Manual identifier for graph lookup
        
        Returns:
            Dictionary containing:
                - detected_subassemblies: List of detected complete subassemblies
                - partial_subassemblies: List of incomplete subassemblies
                - estimated_step: Estimated current step
                - step_confidence: Confidence of step estimation (0.0-1.0)
        """
        try:
            logger.info(f"Detecting subassemblies for manual {manual_id} (enhanced mode)")
            
            # First, analyze the assembly state
            analysis = self.analyze_assembly_state(
                image_paths=image_paths,
                manual_id=manual_id,
                context="Identify distinct subassemblies or functional groups of connected parts. For each structure, list the specific parts that make it up."
            )
            
            # Extract assembled structures
            structures = analysis.get('assembled_structures', [])
            detected_parts = analysis.get('detected_parts', [])
            
            if not structures:
                logger.info("No subassemblies detected in images")
                return {
                    "detected_subassemblies": [],
                    "partial_subassemblies": [],
                    "estimated_step": None,
                    "step_confidence": 0.0
                }
            
            # Load graph
            graph = self.graph_manager.load_graph(manual_id)
            if not graph:
                logger.warning(f"No graph found for manual {manual_id}, using basic matching")
                return self._detect_subassemblies_fallback(structures, manual_id)
            
            # Enhanced matching: use part composition
            detected_subassemblies = []
            partial_subassemblies = []
            
            for structure in structures:
                # Extract parts from structure description
                component_parts = self._extract_parts_from_structure(structure, detected_parts)
                
                # Find matching subassembly in graph
                match_result = self._match_structure_by_composition(
                    structure=structure,
                    component_parts=component_parts,
                    graph=graph
                )
                
                if match_result:
                    subasm_node = match_result["node"]
                    similarity = match_result["similarity"]
                    
                    # Validate completeness
                    completeness_result = self._validate_completeness(
                        structure=structure,
                        component_parts=component_parts,
                        subasm_node=subasm_node,
                        detected_parts=detected_parts
                    )
                    
                    subasm_info = {
                        "subassembly_id": subasm_node["node_id"],
                        "name": subasm_node["name"],
                        "confidence": round(similarity * 0.7 + completeness_result["score"] * 0.3, 2),
                        "completeness": completeness_result["completion_ratio"],
                        "visible_parts": completeness_result["visible_parts"],
                        "missing_parts": completeness_result["missing_parts"]
                    }
                    
                    if completeness_result["completion_ratio"] > 0.9:
                        detected_subassemblies.append(subasm_info)
                        logger.debug(f"Complete subassembly: {subasm_info['name']} ({subasm_info['confidence']})")
                    elif completeness_result["completion_ratio"] > 0.5:
                        partial_subassemblies.append(subasm_info)
                        logger.debug(f"Partial subassembly: {subasm_info['name']} ({completeness_result['completion_ratio']:.0%} complete)")
            
            # Estimate step from detected subassemblies
            estimated_step = None
            step_confidence = 0.0
            
            if detected_subassemblies:
                # Use graph-aware estimation
                max_step = max([
                    self.graph_manager.get_node(manual_id, s["subassembly_id"]).get("step_created", 1)
                    for s in detected_subassemblies
                ])
                estimated_step = max_step
                step_confidence = sum(s["confidence"] for s in detected_subassemblies) / len(detected_subassemblies)
            
            logger.info(f"Enhanced detection: {len(detected_subassemblies)} complete, "
                       f"{len(partial_subassemblies)} partial subassemblies")
            
            return {
                "detected_subassemblies": detected_subassemblies,
                "partial_subassemblies": partial_subassemblies,
                "estimated_step": estimated_step,
                "step_confidence": round(step_confidence, 2)
            }
            
        except Exception as e:
            logger.error(f"Error detecting subassemblies: {e}")
            return {
                "detected_subassemblies": [],
                "partial_subassemblies": [],
                "estimated_step": None,
                "step_confidence": 0.0,
                "error": str(e)
            }
    
    def _extract_parts_from_structure(
        self,
        structure: Dict[str, Any],
        detected_parts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract part signatures from structure description.
        
        Returns:
            List of part signatures (color:shape)
        """
        structure_desc = structure.get("description", "").lower()
        
        # Try to find parts mentioned in description
        component_parts = []
        
        for part in detected_parts:
            color = part.get("color", "").lower()
            shape = part.get("shape", "").lower()
            
            # Check if part is mentioned in structure description
            if color in structure_desc and any(word in structure_desc for word in shape.split()):
                signature = f"{color}:{shape}"
                component_parts.append(signature)
        
        return component_parts
    
    def _match_structure_by_composition(
        self,
        structure: Dict[str, Any],
        component_parts: List[str],
        graph: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Match structure to subassembly using part composition (Jaccard similarity).
        
        Returns:
            Dict with node and similarity score, or None
        """
        nodes = graph.get("nodes", [])
        subassemblies = [n for n in nodes if n.get("type") == "subassembly"]
        
        if not subassemblies:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for subasm in subassemblies:
            # Get part catalog to find component parts of this subassembly
            subasm_component_parts = self._get_subassembly_component_parts(subasm, graph)
            
            if not subasm_component_parts:
                continue
            
            # Calculate Jaccard similarity
            detected_set = set(component_parts)
            expected_set = set(subasm_component_parts)
            
            if not expected_set:
                continue
            
            intersection = len(detected_set & expected_set)
            union = len(detected_set | expected_set)
            
            jaccard = intersection / union if union > 0 else 0.0
            
            # Also check name similarity
            structure_desc = structure.get("description", "").lower()
            subasm_name = subasm.get("name", "").lower()
            
            name_overlap = len(set(structure_desc.split()) & set(subasm_name.split()))
            name_score = min(name_overlap / 3, 1.0)  # Normalize
            
            # Combined score
            similarity = jaccard * 0.7 + name_score * 0.3
            
            if similarity > best_similarity and similarity > 0.6:
                best_similarity = similarity
                best_match = {
                    "node": subasm,
                    "similarity": round(similarity, 2)
                }
        
        return best_match
    
    def _get_subassembly_component_parts(
        self,
        subasm_node: Dict[str, Any],
        graph: Dict[str, Any]
    ) -> List[str]:
        """Get component parts of a subassembly from graph."""
        # Check completeness markers first
        markers = subasm_node.get("completeness_markers", {})
        if "component_parts" in markers:
            return markers["component_parts"]
        
        # Fallback: get children that are parts
        child_ids = subasm_node.get("children", [])
        nodes = graph.get("nodes", [])
        node_map = {n["node_id"]: n for n in nodes}
        
        component_parts = []
        for child_id in child_ids:
            child = node_map.get(child_id)
            if child and child.get("type") == "part":
                color = child.get("color", "")
                shape = child.get("shape", "")
                if color and shape:
                    component_parts.append(f"{color}:{shape}")
        
        return component_parts
    
    def _validate_completeness(
        self,
        structure: Dict[str, Any],
        component_parts: List[str],
        subasm_node: Dict[str, Any],
        detected_parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate subassembly completeness using completeness markers.
        
        Returns:
            Dict with completion_ratio, visible_parts, missing_parts, score
        """
        markers = subasm_node.get("completeness_markers", {})
        required_parts_count = markers.get("required_parts", len(component_parts))
        
        # Get expected component parts
        expected_parts = self._get_subassembly_component_parts(subasm_node, {"nodes": [subasm_node]})
        
        # Check which parts are visible
        detected_signatures = set(f"{p.get('color', '')}:{p.get('shape', '')}" for p in detected_parts)
        expected_set = set(expected_parts) if expected_parts else set(component_parts)
        
        visible_parts = list(detected_signatures & expected_set)
        missing_parts = list(expected_set - detected_signatures)
        
        # Calculate completion ratio
        if expected_set:
            completion_ratio = len(visible_parts) / len(expected_set)
        else:
            completion_ratio = 0.5  # Unknown
        
        # Calculate completeness score
        score = completion_ratio
        
        return {
            "completion_ratio": round(completion_ratio, 2),
            "visible_parts": visible_parts,
            "missing_parts": missing_parts,
            "score": round(score, 2)
        }
    
    def _detect_subassemblies_fallback(
        self,
        structures: List[Dict[str, Any]],
        manual_id: str
    ) -> Dict[str, Any]:
        """Fallback method using fuzzy name matching."""
        detected_subassemblies = []
        partial_subassemblies = []
        
        for structure in structures:
            description = structure.get('description', '').lower()
            completeness = structure.get('completeness', 'unknown').lower()
            
            # Try to match to graph subassemblies
            matched_nodes = self.graph_manager.get_node_by_name(
                manual_id=manual_id,
                name=description,
                fuzzy=True
            )
            
            # Filter to only subassembly nodes
            subasm_nodes = [
                n for n in matched_nodes 
                if n.get('type') == 'subassembly'
            ]
            
            if subasm_nodes:
                matched_node = subasm_nodes[0]
                
                subasm_info = {
                    "subassembly_id": matched_node["node_id"],
                    "name": matched_node["name"],
                    "confidence": 0.6,  # Lower confidence for fallback
                    "completeness": completeness,
                    "visible_parts": [],
                    "missing_parts": []
                }
                
                if 'complete' in completeness:
                    detected_subassemblies.append(subasm_info)
                else:
                    partial_subassemblies.append(subasm_info)
        
        return {
            "detected_subassemblies": detected_subassemblies,
            "partial_subassemblies": partial_subassemblies,
            "estimated_step": None,
            "step_confidence": 0.0
        }


# Singleton instance
_state_analyzer_instance = None


def get_state_analyzer() -> StateAnalyzer:
    """Get StateAnalyzer singleton instance."""
    global _state_analyzer_instance
    if _state_analyzer_instance is None:
        _state_analyzer_instance = StateAnalyzer()
    return _state_analyzer_instance


