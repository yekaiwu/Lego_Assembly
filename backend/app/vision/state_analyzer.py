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
from .visual_matcher import VisualMatcher


class StateAnalyzer:
    """Analyzes user's physical assembly state from photos."""

    def __init__(self, vlm_client=None, visual_matcher=None):
        """Initialize with VLM client from backend settings."""
        if vlm_client is None:
            # Use STATE_ANALYSIS_VLM from backend settings (configurable via .env)
            from ..config import get_settings
            settings = get_settings()
            state_analysis_vlm = settings.state_analysis_vlm

            # Initialize VLM client directly
            self.vlm_client = UnifiedVLMClient(model_name=state_analysis_vlm)
            logger.info(f"StateAnalyzer initialized with STATE_ANALYSIS_VLM: {state_analysis_vlm}")
        else:
            self.vlm_client = vlm_client
            logger.info("StateAnalyzer initialized with provided VLM client")

        self.graph_manager = get_graph_manager()
        self.prompt_manager = PromptManager()

        # Initialize visual matcher for SAM-based visual matching
        self.visual_matcher = visual_matcher or VisualMatcher()
        if self.visual_matcher.is_available():
            logger.info("Visual matching enabled (SAM available)")
        else:
            logger.info("Visual matching disabled (SAM not available)")
    
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

            # Call VLM with custom prompt for user photo analysis using the new method
            cache_context = f"{manual_id}_state_analysis"
            result = self.vlm_client.analyze_images_json(
                image_paths=image_paths,
                prompt=prompt,
                cache_context=cache_context
            )
            
            # Structure the response
            structured_result = self._structure_analysis_result(result)

            # Store image paths for potential visual matching
            structured_result["image_paths"] = image_paths

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

    def match_state_to_graph(
        self,
        analysis_result: Dict[str, Any],
        manual_id: str,
        top_k: int = 3,
        use_visual_matching: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Match analyzed assembly state to graph nodes using text + visual matching.

        Combines text-based part matching with SAM-based visual similarity
        for more robust and accurate step detection.

        Args:
            analysis_result: Result from analyze_assembly_state()
            manual_id: Manual identifier
            top_k: Number of top matches to return
            use_visual_matching: Whether to use visual matching (default: True)

        Returns:
            List of matching steps with confidence scores:
            [
                {
                    "step_number": 5,
                    "combined_confidence": 0.85,  # Combined score (if visual enabled)
                    "text_confidence": 0.87,
                    "visual_confidence": 0.82,
                    "match_reason": "Combined: 85% (text: 87%, visual: 82%)",
                    "step_state": {...},
                    "next_step": 6
                },
                ...
            ]
        """
        try:
            # Use graph-based state matcher for text-based matching
            from ..graph.state_matcher import StateMatcher
            import os

            matcher = StateMatcher(self.graph_manager)

            # Get weights from environment
            visual_weight = float(os.getenv("VISUAL_MATCH_WEIGHT", "0.5"))
            text_weight = float(os.getenv("TEXT_MATCH_WEIGHT", "0.5"))

            # Normalize weights to ensure they sum to 1.0
            total_weight = visual_weight + text_weight
            if total_weight > 0:
                visual_weight = visual_weight / total_weight
                text_weight = text_weight / total_weight
            else:
                # Default to equal weights if both are 0
                visual_weight = 0.5
                text_weight = 0.5

            logger.info(f"Match weights: Visual={visual_weight:.0%}, Text={text_weight:.0%}")

            # Step 1: Visual matching (if enabled and available) - RUN FIRST
            visual_matches = []
            if use_visual_matching and self.visual_matcher.is_available():
                image_paths = analysis_result.get("image_paths", [])

                if image_paths:
                    logger.info("=" * 60)
                    logger.info("STEP 1: Running ORB visual matching with SAM3...")
                    logger.info("=" * 60)
                    visual_matches = self.visual_matcher.match_user_assembly_to_graph(
                        user_image_paths=image_paths,
                        manual_id=manual_id,
                        graph_manager=self.graph_manager,
                        top_k=top_k * 2  # Get more candidates for combining
                    )

                    logger.info(f"✓ ORB visual matching found {len(visual_matches)} potential steps")
                    if visual_matches:
                        logger.info(
                            f"  Best visual match: Step {visual_matches[0]['step_number']} "
                            f"(similarity: {visual_matches[0]['visual_similarity']:.2%})"
                        )
                        # Log top 3 visual matches
                        for i, match in enumerate(visual_matches[:3], 1):
                            logger.info(
                                f"  #{i}: Step {match['step_number']} - "
                                f"{match['visual_similarity']:.2%} similarity"
                            )
                else:
                    logger.info("No image paths available for visual matching")
            else:
                if not use_visual_matching:
                    logger.info("Visual matching disabled by parameter")
                else:
                    logger.info("Visual matching not available (SAM not loaded)")

            # Step 2: Text-based matching using detected parts - RUN SECOND
            logger.info("=" * 60)
            logger.info("STEP 2: Running text-based part matching...")
            logger.info("=" * 60)
            text_matches = matcher.match_state(
                detected_state=analysis_result,
                manual_id=manual_id,
                top_k=top_k * 2  # Get more candidates for combining
            )

            logger.info(f"✓ Text matching found {len(text_matches)} potential steps")
            if text_matches:
                logger.info(f"  Best text match: Step {text_matches[0]['step_number']} "
                           f"(confidence: {text_matches[0]['confidence']:.2%})")
                # Log top 3 text matches
                for i, match in enumerate(text_matches[:3], 1):
                    logger.info(
                        f"  #{i}: Step {match['step_number']} - "
                        f"{match['confidence']:.2%} confidence"
                    )

            # Step 3: Combine visual and text matches
            logger.info("=" * 60)
            logger.info(f"STEP 3: Combining results (Visual: {visual_weight:.0%}, Text: {text_weight:.0%})...")
            logger.info("=" * 60)

            if visual_matches:
                combined_matches = matcher.combine_text_and_visual_matches(
                    text_matches=text_matches,
                    visual_matches=visual_matches,
                    text_weight=text_weight,
                    visual_weight=visual_weight,
                    top_k=top_k
                )

                logger.info(f"✓ Combined matching complete, returning top {len(combined_matches)} results")
                if combined_matches:
                    logger.info(f"  Best combined match: Step {combined_matches[0]['step_number']} "
                               f"(score: {combined_matches[0].get('combined_confidence', 0):.2%})")
                return combined_matches
            else:
                # Fallback: return text-only matches
                logger.info("⚠ Using text-only matches (no visual matches available)")
                return text_matches[:top_k]

        except Exception as e:
            logger.error(f"Error in state matching: {e}")
            return []


# Singleton instance
_state_analyzer_instance = None


def get_state_analyzer() -> StateAnalyzer:
    """Get StateAnalyzer singleton instance."""
    global _state_analyzer_instance
    if _state_analyzer_instance is None:
        _state_analyzer_instance = StateAnalyzer()
    return _state_analyzer_instance


