"""
Direct Step Analyzer - Simplified VLM-only approach for step detection.

This module replaces the complex pipeline (VLM + SAM3 + ORB + text matching)
with a single VLM call that directly identifies the current assembly step.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

# Add project root to path to import Phase 1 modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.litellm_vlm import UnifiedVLMClient
from ..graph.graph_manager import get_graph_manager
from .prompt_manager import PromptManager


class DirectStepAnalyzer:
    """
    Simplified state analyzer using VLM-only approach for direct step detection.

    Replaces the complex pipeline of:
    1. VLM part detection (StateAnalyzer.analyze_assembly_state)
    2. Text-based matching (StateMatcher.match_state)
    3. Visual matching (VisualMatcher.match_user_assembly_to_graph)

    With a single VLM call that directly identifies the current step.
    """

    def __init__(self, vlm_client=None):
        """Initialize with VLM client from backend settings."""
        if vlm_client is None:
            from ..config import get_settings
            settings = get_settings()
            state_analysis_vlm = settings.state_analysis_vlm

            self.vlm_client = UnifiedVLMClient(model_name=state_analysis_vlm)
            logger.info(f"DirectStepAnalyzer initialized with {state_analysis_vlm}")
        else:
            self.vlm_client = vlm_client
            logger.info("DirectStepAnalyzer initialized with provided VLM client")

        self.graph_manager = get_graph_manager()
        self.prompt_manager = PromptManager()

    def detect_current_step(
        self,
        image_paths: List[str],
        manual_id: str,
        max_reference_images: int = 20,
        detected_parts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Directly detect which step the user is currently on from photos.

        This is the main entry point for the simplified approach.
        Single VLM call replaces the entire complex pipeline.

        Args:
            image_paths: List of paths to user's assembly photos (1-4 images)
            manual_id: Manual identifier for context
            max_reference_images: Maximum number of reference images to include (default: 20)
            detected_parts: Optional list of parts detected by StateAnalyzer
                          (used for parts matching to improve accuracy)

        Returns:
            Dictionary containing:
                - step_number: Detected current step (0 if no parts, total if complete)
                - confidence: VLM confidence score (0.0-1.0)
                - reasoning: Detailed explanation of the detection
                - next_step: Next step to work on (step_number + 1, or None if complete)
                - assembly_status: Additional detected information
        """
        try:
            logger.info(f"Detecting current step for manual {manual_id}")
            logger.info(f"Processing {len(image_paths)} user images")

            # Load reference images from the manual
            reference_images = self._load_reference_images(manual_id, max_reference_images)
            logger.info(f"Loaded {len(reference_images)} reference step images")

            # Build the prompt with manual steps context and detected parts
            prompt = self._build_step_detection_prompt(
                manual_id,
                has_reference_images=len(reference_images) > 0,
                num_user_images=len(image_paths),
                detected_parts=detected_parts
            )

            if detected_parts:
                logger.info(f"Passing {len(detected_parts)} detected parts to VLM for comparison")

            # Combine user images and reference images for VLM
            all_images = image_paths + reference_images

            logger.info(f"Sending to VLM: {len(image_paths)} user images + {len(reference_images)} reference images = {len(all_images)} total")
            logger.debug(f"User images: {image_paths}")
            logger.debug(f"First 3 reference images: {reference_images[:3]}")
            logger.debug(f"Prompt length: {len(prompt)} chars")

            # Call VLM with the direct step detection prompt
            cache_context = f"{manual_id}_step_detection"
            result = self.vlm_client.analyze_images_json(
                image_paths=all_images,
                prompt=prompt,
                cache_context=cache_context
            )

            logger.debug(f"VLM raw response: {json.dumps(result, indent=2)[:500]}...")

            # Validate and structure the result
            structured_result = self._validate_step_detection(result, manual_id)

            logger.info(
                f"Step detection complete. Current step: {structured_result['step_number']}, "
                f"confidence: {structured_result['confidence']:.2%}"
            )

            return structured_result

        except Exception as e:
            logger.error(f"Error detecting current step: {e}")
            return {
                "step_number": 0,
                "confidence": 0.0,
                "reasoning": f"Error during step detection: {str(e)}",
                "next_step": 1,
                "assembly_status": {
                    "total_parts_detected": 0,
                    "key_features": [],
                    "potential_issues": [f"Detection failed: {str(e)}"]
                },
                "error": str(e)
            }

    def _build_step_detection_prompt(
        self,
        manual_id: str,
        has_reference_images: bool = False,
        num_user_images: int = 1,
        detected_parts: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build VLM prompt for direct step detection.

        Loads manual steps from dependencies.json and formats them
        as context for the VLM to compare against the user's assembly.

        Args:
            manual_id: Manual identifier
            has_reference_images: Whether reference images are included
            num_user_images: Number of user assembly images
            detected_parts: Optional list of detected parts to include in prompt

        Returns:
            Formatted prompt string with manual steps context
        """
        # Load dependencies to get manual steps
        dependencies = self._load_dependencies(manual_id)

        if not dependencies:
            logger.warning(f"Could not load dependencies for {manual_id}")
            total_steps = "unknown"
            manual_steps_context = "Manual steps not available"
        else:
            nodes = dependencies.get("nodes", {})
            total_steps = len(nodes)

            # Format manual steps with cumulative parts for VLM
            manual_steps_context = self._format_manual_steps_with_parts(nodes)

        # Format detected parts if available
        detected_parts_text = ""
        if detected_parts:
            detected_parts_text = self._format_detected_parts(detected_parts)

        # Use PromptManager to load the direct_step_detection prompt template
        prompt_context = {
            'manual_id': manual_id,
            'total_steps': str(total_steps),
            'manual_steps_context': manual_steps_context,
            'has_reference_images': 'yes' if has_reference_images else 'no',
            'num_user_images': str(num_user_images),
            'detected_parts_section': detected_parts_text
        }

        prompt = self.prompt_manager.get_prompt(
            'direct_step_detection',
            context=prompt_context
        )

        return prompt

    def _format_detected_parts(self, detected_parts: List[Dict[str, Any]]) -> str:
        """
        Format detected parts into readable text for VLM prompt.

        Args:
            detected_parts: List of parts detected by StateAnalyzer

        Returns:
            Formatted string describing detected parts
        """
        if not detected_parts:
            return ""

        parts_lines = ["\nDETECTED PARTS FROM USER'S ASSEMBLY:"]
        for i, part in enumerate(detected_parts, 1):
            color = part.get("color", "unknown")
            shape = part.get("shape", "unknown")
            desc = part.get("description", "")
            qty = part.get("quantity", 1)

            part_text = f"{i}. {qty}x {color} {shape}"
            if desc and desc not in shape:
                part_text += f" ({desc})"
            parts_lines.append(part_text)

        parts_lines.append("")  # blank line
        return "\n".join(parts_lines)

    def _format_manual_steps_with_parts(self, nodes: Dict[str, Any]) -> str:
        """
        Format manual steps with cumulative parts information for VLM.

        Args:
            nodes: Dictionary of step nodes from dependencies.json

        Returns:
            Formatted string describing each step with cumulative parts
        """
        # Sort steps by step_number
        sorted_steps = sorted(
            nodes.items(),
            key=lambda x: x[1].get('step_number', 0)
        )

        formatted_steps = []

        for step_id, step_data in sorted_steps:
            step_num = step_data.get('step_number', '?')
            parts = step_data.get('parts_required', [])
            cumulative_parts = step_data.get('cumulative_parts', [])
            actions = step_data.get('actions', [])
            existing = step_data.get('existing_assembly', '')

            # Format parts added in this step
            parts_list = []
            for part in parts:
                qty = part.get('quantity', 1)
                color = part.get('color', '')
                desc = part.get('description', '')
                parts_list.append(f"{qty}x {color} {desc}".strip())

            parts_text = ", ".join(parts_list) if parts_list else "no new parts"

            # Get action verb
            action = actions[0].get('action_verb', 'attach') if actions else 'attach'

            # Build step description
            step_desc = f"Step {step_num}: {action} {parts_text}"

            # Add existing assembly context if available
            if existing and existing != "null":
                step_desc += f" (building on: {existing})"

            # Add cumulative parts count
            if cumulative_parts:
                total_parts = sum(p.get('quantity', 1) for p in cumulative_parts)
                step_desc += f" [Total parts up to this step: {total_parts}]"

            formatted_steps.append(step_desc)

        return "\n".join(formatted_steps)

    def _load_dependencies(self, manual_id: str) -> Optional[Dict[str, Any]]:
        """
        Load dependencies.json for the manual.

        Args:
            manual_id: Manual identifier

        Returns:
            Dependencies dict or None if not found
        """
        try:
            from ..config import get_settings
            settings = get_settings()

            # settings.output_dir is now guaranteed to be an absolute path
            dependencies_path = Path(settings.output_dir) / manual_id / f"{manual_id}_dependencies.json"

            logger.debug(f"Looking for dependencies at: {dependencies_path}")

            if not dependencies_path.exists():
                logger.error(f"Dependencies file not found: {dependencies_path}")
                return None

            with open(dependencies_path, 'r', encoding='utf-8') as f:
                dependencies = json.load(f)

            logger.info(f"Loaded dependencies for manual {manual_id}")
            return dependencies

        except Exception as e:
            logger.error(f"Error loading dependencies for {manual_id}: {e}")
            return None

    def _load_reference_images(self, manual_id: str, max_images: int = 20) -> List[str]:
        """
        Load reference step images from the manual.

        Args:
            manual_id: Manual identifier
            max_images: Maximum number of reference images to load

        Returns:
            List of paths to reference step images
        """
        try:
            from ..config import get_settings
            settings = get_settings()

            # settings.output_dir is now guaranteed to be an absolute path
            # Reference images are stored in output/temp_pages/{manual_id}/
            images_dir = Path(settings.output_dir) / "temp_pages" / manual_id

            logger.debug(f"Looking for reference images in: {images_dir}")
            logger.debug(f"settings.output_dir: {settings.output_dir}")

            if not images_dir.exists():
                logger.error(f"Reference images directory not found: {images_dir}")
                return []

            # Get all page images, sorted by name
            image_files = sorted(images_dir.glob("page_*.png"))

            if not image_files:
                logger.warning(f"No reference images found in {images_dir}")
                return []

            # Limit to max_images
            selected_images = image_files[:max_images]

            # Convert to absolute paths as strings
            reference_paths = [str(img.absolute()) for img in selected_images]

            logger.info(f"Found {len(reference_paths)} reference images for manual {manual_id}")
            return reference_paths

        except Exception as e:
            logger.error(f"Error loading reference images for {manual_id}: {e}")
            return []

    def _format_manual_steps(self, nodes: Dict[str, Any]) -> str:
        """
        Format manual steps into readable context for VLM.

        Args:
            nodes: Dictionary of step nodes from dependencies.json

        Returns:
            Formatted string describing each step
        """
        # Sort steps by step_number
        sorted_steps = sorted(
            nodes.items(),
            key=lambda x: x[1].get('step_number', 0)
        )

        formatted_steps = []

        for step_id, step_data in sorted_steps:
            step_num = step_data.get('step_number', '?')
            parts = step_data.get('parts_required', [])
            actions = step_data.get('actions', [])
            existing = step_data.get('existing_assembly', '')

            # Format parts list
            parts_list = []
            for part in parts:
                qty = part.get('quantity', 1)
                color = part.get('color', '')
                desc = part.get('description', '')
                parts_list.append(f"{qty}x {color} {desc}".strip())

            parts_text = ", ".join(parts_list) if parts_list else "no new parts"

            # Get action verb
            action = actions[0].get('action_verb', 'attach') if actions else 'attach'

            # Build step description
            step_desc = f"Step {step_num}: {action} {parts_text}"

            # Add existing assembly context if available
            if existing and existing != "null":
                step_desc += f" (building on: {existing})"

            formatted_steps.append(step_desc)

        return "\n".join(formatted_steps)

    def _validate_step_detection(
        self,
        vlm_result: Dict[str, Any],
        manual_id: str
    ) -> Dict[str, Any]:
        """
        Validate and structure VLM step detection result.

        Ensures all required fields are present and values are in valid ranges.

        Args:
            vlm_result: Raw VLM response
            manual_id: Manual identifier for validation

        Returns:
            Validated and structured result
        """
        # Handle error case
        if "error" in vlm_result:
            return {
                "step_number": 0,
                "confidence": 0.0,
                "reasoning": f"VLM error: {vlm_result.get('error', 'Unknown error')}",
                "next_step": 1,
                "assembly_status": {
                    "total_parts_detected": 0,
                    "key_features": [],
                    "potential_issues": ["VLM analysis failed"]
                },
                "error": vlm_result.get("error")
            }

        # Extract fields with defaults
        step_number = vlm_result.get("step_number", 0)
        confidence = vlm_result.get("confidence", 0.5)
        reasoning = vlm_result.get("reasoning", "No reasoning provided")
        next_step = vlm_result.get("next_step")
        assembly_status = vlm_result.get("assembly_status", {})

        # Validate step_number range
        try:
            step_number = int(step_number)
            # Load dependencies to get total steps for validation
            dependencies = self._load_dependencies(manual_id)
            if dependencies:
                total_steps = len(dependencies.get("nodes", {}))
                step_number = max(0, min(step_number, total_steps))
            else:
                step_number = max(0, step_number)
        except (ValueError, TypeError):
            logger.warning(f"Invalid step_number: {step_number}, defaulting to 0")
            step_number = 0

        # Validate confidence range
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence: {confidence}, defaulting to 0.5")
            confidence = 0.5

        # Validate next_step
        if next_step is not None:
            try:
                next_step = int(next_step)
            except (ValueError, TypeError):
                next_step = step_number + 1

        # Ensure assembly_status has required fields
        if not isinstance(assembly_status, dict):
            assembly_status = {}

        assembly_status.setdefault("total_parts_detected", 0)
        assembly_status.setdefault("key_features", [])
        assembly_status.setdefault("potential_issues", [])

        return {
            "step_number": step_number,
            "confidence": confidence,
            "reasoning": reasoning,
            "next_step": next_step,
            "assembly_status": assembly_status,
            "raw_vlm_response": vlm_result
        }

    def get_step_info(
        self,
        manual_id: str,
        step_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific step.

        Useful for generating guidance after step detection.

        Args:
            manual_id: Manual identifier
            step_number: Step number

        Returns:
            Step information dict or None if not found
        """
        dependencies = self._load_dependencies(manual_id)
        if not dependencies:
            return None

        nodes = dependencies.get("nodes", {})

        # Find the step node
        for step_id, step_data in nodes.items():
            if step_data.get("step_number") == step_number:
                return step_data

        return None


# Singleton instance
_direct_step_analyzer_instance = None


def get_direct_step_analyzer() -> DirectStepAnalyzer:
    """Get DirectStepAnalyzer singleton instance."""
    global _direct_step_analyzer_instance
    if _direct_step_analyzer_instance is None:
        _direct_step_analyzer_instance = DirectStepAnalyzer()
    return _direct_step_analyzer_instance
