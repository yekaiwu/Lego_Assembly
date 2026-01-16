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
    


# Singleton instance
_state_analyzer_instance = None


def get_state_analyzer() -> StateAnalyzer:
    """Get StateAnalyzer singleton instance."""
    global _state_analyzer_instance
    if _state_analyzer_instance is None:
        _state_analyzer_instance = StateAnalyzer()
    return _state_analyzer_instance


