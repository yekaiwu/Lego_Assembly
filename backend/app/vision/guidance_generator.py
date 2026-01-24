"""
Guidance Generator - Generates intelligent next-step guidance.
Creates actionable instructions based on current state analysis and comparison.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from ..config import get_settings
from ..llm import QwenClient, DeepSeekClient, MoonshotClient


class GuidanceGenerator:
    """Generates intelligent assembly guidance based on state analysis."""
    
    def __init__(self):
        """Initialize with configured LLM client."""
        self.settings = get_settings()
        
        # Initialize LLM client (same as RAG generator)
        api_key = self.settings.get_llm_api_key()
        provider = self.settings.rag_llm_provider
        model = self.settings.rag_llm_model
        
        if provider == "qwen":
            self.client = QwenClient(api_key, model)
        elif provider == "deepseek":
            self.client = DeepSeekClient(api_key, model)
        elif provider == "moonshot":
            self.client = MoonshotClient(api_key, model)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        logger.info(f"GuidanceGenerator initialized with {provider} LLM")
    
    def generate_guidance(
        self,
        detected_state: Dict[str, Any],
        comparison_result: Dict[str, Any],
        manual_id: str,
        output_dir: str = "./output"
    ) -> Dict[str, Any]:
        """
        Generate actionable guidance based on current state.
        
        Args:
            detected_state: Result from StateAnalyzer
            comparison_result: Result from StateComparator
            manual_id: Manual identifier
            output_dir: Directory containing Phase 1 outputs
        
        Returns:
            Dictionary containing:
                - instruction: Main instruction text
                - next_step_number: Next step to perform
                - parts_needed: List of parts for next step
                - reference_image: Path to reference image
                - error_corrections: List of corrections if errors detected
                - encouragement: Positive feedback message
                - confidence: Confidence in guidance
        """
        try:
            logger.info(f"Generating guidance for manual {manual_id}")
            
            # Load next step information
            next_steps = comparison_result.get("next_steps", [])
            if not next_steps:
                return self._create_completion_guidance(comparison_result)
            
            next_step_number = next_steps[0]
            next_step_info = self._load_step_info(manual_id, next_step_number, output_dir)
            
            if not next_step_info:
                return self._create_error_guidance("Could not load next step information")
            
            # Generate main instruction using LLM
            instruction = self._generate_instruction(
                detected_state,
                comparison_result,
                next_step_info
            )
            
            # Extract parts needed for next step
            parts_needed = self._extract_parts_needed(next_step_info)
            
            # Get reference image path
            reference_image = self._get_reference_image(manual_id, next_step_number, output_dir)
            
            # Generate error corrections if needed
            error_corrections = self._generate_error_corrections(
                comparison_result.get("errors", []),
                comparison_result.get("missing_parts", [])
            )
            
            # Generate encouragement message
            encouragement = self._generate_encouragement(comparison_result)
            
            # Calculate guidance confidence
            confidence = self._calculate_guidance_confidence(
                comparison_result,
                next_step_info
            )
            
            result = {
                "instruction": instruction,
                "next_step_number": next_step_number,
                "parts_needed": parts_needed,
                "reference_image": reference_image,
                "error_corrections": error_corrections,
                "encouragement": encouragement,
                "confidence": confidence,
                "status": "success"
            }
            
            logger.info(f"Guidance generated for step {next_step_number}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating guidance: {e}")
            return self._create_error_guidance(str(e))
    
    def _load_step_info(
        self,
        manual_id: str,
        step_number: int,
        output_dir: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load information for a specific step from Phase 1 data.
        
        Args:
            manual_id: Manual identifier
            step_number: Step number
            output_dir: Output directory
        
        Returns:
            Step information dictionary or None
        """
        try:
            output_path = Path(output_dir)
            extracted_path = output_path / manual_id / f"{manual_id}_extracted.json"

            if not extracted_path.exists():
                logger.error(f"Extracted data not found: {extracted_path}")
                return None
            
            with open(extracted_path, 'r', encoding='utf-8') as f:
                extracted_steps = json.load(f)
            
            # Find step by step_number
            for step in extracted_steps:
                if step.get("step_number") == step_number:
                    return step
            
            # If not found by step_number, try by index
            if 0 < step_number <= len(extracted_steps):
                return extracted_steps[step_number - 1]
            
            logger.warning(f"Step {step_number} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error loading step info: {e}")
            return None
    
    def _generate_instruction(
        self,
        detected_state: Dict[str, Any],
        comparison_result: Dict[str, Any],
        next_step_info: Dict[str, Any]
    ) -> str:
        """
        Generate natural language instruction using LLM.
        
        Args:
            detected_state: Current detected state
            comparison_result: Comparison results
            next_step_info: Information about next step
        
        Returns:
            Instruction text
        """
        try:
            # Build context for LLM
            current_progress = comparison_result.get("progress_percentage", 0)
            completed_steps = comparison_result.get("completed_steps", [])
            
            # Format next step information
            next_step_parts = next_step_info.get("parts_required", [])
            next_step_actions = next_step_info.get("actions", [])
            next_step_notes = next_step_info.get("notes", "")
            
            system_prompt = """You are a helpful LEGO assembly guide. Your role is to provide clear, 
            encouraging, step-by-step instructions to help someone build their LEGO model.
            
            Use simple, direct language. Be specific about colors, part sizes, and positions.
            Break down complex actions into simple steps. Always be encouraging and positive."""
            
            user_prompt = f"""Current Assembly Progress: {current_progress:.0f}%
Completed Steps: {len(completed_steps)}

Next Step Information:
Parts Needed: {json.dumps(next_step_parts, indent=2)}
Actions: {json.dumps(next_step_actions, indent=2)}
Notes: {next_step_notes}

Based on this information, provide clear, step-by-step instructions for what the user should do next.
Focus on:
1. What parts they need to find (describe color and size clearly)
2. Where to place these parts
3. How to connect them to existing assembly
4. Any special tips or things to watch out for

Keep instructions concise but complete. Use simple language."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            instruction = self.client.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return instruction.strip()
            
        except Exception as e:
            logger.error(f"Error generating instruction: {e}")
            # Fallback to basic instruction
            return self._generate_fallback_instruction(next_step_info)
    
    def _generate_fallback_instruction(self, next_step_info: Dict[str, Any]) -> str:
        """Generate basic instruction if LLM fails."""
        parts = next_step_info.get("parts_required", [])
        actions = next_step_info.get("actions", [])
        
        instruction_parts = ["For the next step:"]
        
        if parts:
            instruction_parts.append("\nParts needed:")
            for part in parts:
                color = part.get("color", "unknown")
                shape = part.get("shape", "unknown")
                qty = part.get("quantity", 1)
                instruction_parts.append(f"  - {qty}x {color} {shape}")
        
        if actions:
            instruction_parts.append("\nActions:")
            for action in actions:
                verb = action.get("action_verb", "attach")
                target = action.get("target", "piece")
                destination = action.get("destination", "assembly")
                instruction_parts.append(f"  - {verb.capitalize()} {target} to {destination}")
        
        return "\n".join(instruction_parts)
    
    def _extract_parts_needed(self, step_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract parts needed for next step."""
        parts_required = step_info.get("parts_required", [])
        
        formatted_parts = []
        for part in parts_required:
            formatted_parts.append({
                "description": part.get("description", "Unknown part"),
                "color": part.get("color", "unknown"),
                "shape": part.get("shape", "unknown"),
                "part_id": part.get("part_id"),
                "quantity": part.get("quantity", 1)
            })
        
        return formatted_parts
    
    def _get_reference_image(
        self,
        manual_id: str,
        step_number: int,
        output_dir: str
    ) -> Optional[str]:
        """
        Get path to reference image for step.
        
        Args:
            manual_id: Manual identifier
            step_number: Step number
            output_dir: Output directory
        
        Returns:
            Image path or None
        """
        try:
            output_path = Path(output_dir)
            # New structure: output/temp_pages/{manual_id}/
            temp_pages_dir = output_path / "temp_pages" / manual_id

            # Try common image naming patterns
            patterns = [
                f"page_{step_number:03d}.png",  # page_001.png
                f"page_{step_number}.png",
                f"{manual_id}_page_{step_number}.png",
                f"{manual_id}_step_{step_number}.png",
                f"step_{step_number}.png"
            ]

            for pattern in patterns:
                image_path = temp_pages_dir / pattern
                if image_path.exists():
                    return str(image_path)

            # Try finding any image with step number in name
            if temp_pages_dir.exists():
                for image_path in temp_pages_dir.glob(f"*{step_number}*.png"):
                    return str(image_path)

            logger.warning(f"Reference image not found for step {step_number} in {temp_pages_dir}")
            return None

        except Exception as e:
            logger.error(f"Error finding reference image: {e}")
            return None
    
    def _generate_error_corrections(
        self,
        errors: List[Dict[str, str]],
        missing_parts: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate correction instructions for detected errors."""
        corrections = []
        
        for error in errors:
            error_type = error.get("type", "unknown")
            message = error.get("message", "")
            
            if error_type == "no_parts_detected":
                corrections.append(
                    "⚠️ Tip: Make sure your photos clearly show the assembled parts. "
                    "Try taking photos from multiple angles with good lighting."
                )
        
        if missing_parts:
            corrections.append(
                f"📦 Note: {len(missing_parts)} part(s) appear to be missing for the current step. "
                "Please check that you have all required pieces before proceeding."
            )
        
        return corrections
    
    def _generate_encouragement(self, comparison_result: Dict[str, Any]) -> str:
        """Generate encouraging message based on progress."""
        progress = comparison_result.get("progress_percentage", 0)
        completed_steps = len(comparison_result.get("completed_steps", []))
        
        if progress == 0:
            return "🚀 Let's get started! Follow the instructions to begin building."
        elif progress < 25:
            return f"👍 Great start! You've completed {completed_steps} step(s). Keep going!"
        elif progress < 50:
            return f"🎯 You're making good progress! {progress:.0f}% complete."
        elif progress < 75:
            return f"⭐ Excellent work! You're over halfway there at {progress:.0f}%!"
        elif progress < 100:
            return f"🔥 Almost done! Just {100 - progress:.0f}% to go!"
        else:
            return "🎉 Congratulations! You've completed the build!"
    
    def _calculate_guidance_confidence(
        self,
        comparison_result: Dict[str, Any],
        next_step_info: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for guidance."""
        # Base confidence from comparison
        comparison_confidence = comparison_result.get("confidence", 0.5)
        
        # Step info quality
        has_parts = len(next_step_info.get("parts_required", [])) > 0
        has_actions = len(next_step_info.get("actions", [])) > 0
        step_quality = (0.5 if has_parts else 0.0) + (0.5 if has_actions else 0.0)
        
        # Combined confidence
        confidence = (comparison_confidence * 0.7) + (step_quality * 0.3)
        
        return round(confidence, 2)
    
    def _create_completion_guidance(self, comparison_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create guidance when build is complete."""
        return {
            "instruction": "🎉 Congratulations! You've completed all steps of the assembly!",
            "next_step_number": None,
            "parts_needed": [],
            "reference_image": None,
            "error_corrections": [],
            "encouragement": "Amazing work! Your LEGO model is complete!",
            "confidence": 1.0,
            "status": "complete"
        }
    
    def _create_error_guidance(self, error_message: str) -> Dict[str, Any]:
        """Create error guidance."""
        return {
            "instruction": f"Unable to generate guidance: {error_message}",
            "next_step_number": None,
            "parts_needed": [],
            "reference_image": None,
            "error_corrections": [error_message],
            "encouragement": "Please try uploading clearer photos or check the manual data.",
            "confidence": 0.0,
            "status": "error"
        }

    def generate_guidance_for_step(
        self,
        manual_id: str,
        current_step: int,
        next_step: Optional[int],
        output_dir: str = "./output",
        detection_confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        Generate guidance for a specific step (simplified VLM-only approach).

        This method is used by the DirectStepAnalyzer approach where we already
        know the current step number from VLM detection.

        Args:
            manual_id: Manual identifier
            current_step: Current step number (detected by VLM)
            next_step: Next step number (None if complete)
            output_dir: Directory containing Phase 1 outputs
            detection_confidence: Confidence from step detection

        Returns:
            Dictionary containing guidance information
        """
        try:
            logger.info(f"Generating guidance for step {current_step} -> {next_step}")

            # Load dependencies to get total steps
            from pathlib import Path
            import json

            dependencies_path = Path(output_dir) / manual_id / f"{manual_id}_dependencies.json"
            total_steps = 0

            if dependencies_path.exists():
                with open(dependencies_path, 'r', encoding='utf-8') as f:
                    dependencies = json.load(f)
                    total_steps = len(dependencies.get("nodes", {}))

            # If no next step, build is complete
            if next_step is None or next_step > total_steps:
                return {
                    "instruction": "🎉 Congratulations! You've completed all steps of the assembly!",
                    "next_step_number": None,
                    "parts_needed": [],
                    "reference_image": None,
                    "error_corrections": [],
                    "encouragement": "Amazing work! Your LEGO model is complete!",
                    "confidence": 1.0,
                    "total_steps": total_steps,
                    "status": "complete"
                }

            # Load next step information
            next_step_info = self._load_step_info(manual_id, next_step, output_dir)

            if not next_step_info:
                return self._create_error_guidance(f"Could not load information for step {next_step}")

            # Generate instruction for next step
            instruction = self._generate_instruction_simple(next_step_info, current_step, total_steps)

            # Extract parts needed
            parts_needed = self._extract_parts_needed(next_step_info)

            # Get reference image
            reference_image = self._get_reference_image(manual_id, next_step, output_dir)

            # Generate encouragement
            progress_percentage = (current_step / total_steps * 100) if total_steps > 0 else 0
            encouragement = self._generate_encouragement_simple(current_step, total_steps, progress_percentage)

            return {
                "instruction": instruction,
                "next_step_number": next_step,
                "parts_needed": parts_needed,
                "reference_image": reference_image,
                "error_corrections": [],
                "encouragement": encouragement,
                "confidence": detection_confidence,
                "total_steps": total_steps,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error generating guidance for step: {e}")
            return self._create_error_guidance(str(e))

    def _generate_instruction_simple(
        self,
        next_step_info: Dict[str, Any],
        current_step: int,
        total_steps: int
    ) -> str:
        """
        Generate simplified instruction without LLM (faster).

        Args:
            next_step_info: Information about the next step
            current_step: Current step number
            total_steps: Total number of steps

        Returns:
            Instruction text
        """
        parts = next_step_info.get("parts_required", [])
        actions = next_step_info.get("actions", [])
        notes = next_step_info.get("notes", "")

        instruction_parts = [f"📍 Step {next_step_info.get('step_number', '?')} of {total_steps}"]

        if parts:
            instruction_parts.append("\n🔧 Parts needed:")
            for part in parts:
                color = part.get("color", "unknown")
                desc = part.get("description", part.get("shape", "unknown"))
                qty = part.get("quantity", 1)
                instruction_parts.append(f"  • {qty}x {color} {desc}")

        if actions:
            instruction_parts.append("\n📝 What to do:")
            for i, action in enumerate(actions, 1):
                verb = action.get("action_verb", "attach")
                target = action.get("target", "piece")
                destination = action.get("destination", "assembly")
                orientation = action.get("orientation", "")

                action_text = f"  {i}. {verb.capitalize()} {target} to {destination}"
                if orientation:
                    action_text += f" ({orientation})"
                instruction_parts.append(action_text)

        if notes and notes != "null":
            instruction_parts.append(f"\n💡 Tip: {notes}")

        return "\n".join(instruction_parts)

    def _generate_encouragement_simple(
        self,
        current_step: int,
        total_steps: int,
        progress_percentage: float
    ) -> str:
        """Generate simple encouragement message based on progress."""
        if current_step == 0:
            return "🚀 Let's get started! Follow the instructions to begin building."
        elif progress_percentage < 25:
            return f"👍 Great start! You've completed {current_step} of {total_steps} steps. Keep going!"
        elif progress_percentage < 50:
            return f"🎯 You're making good progress! {progress_percentage:.0f}% complete."
        elif progress_percentage < 75:
            return f"⭐ Excellent work! You're over halfway there at {progress_percentage:.0f}%!"
        elif progress_percentage < 100:
            return f"🔥 Almost done! Just {total_steps - current_step} steps to go!"
        else:
            return "🎉 Congratulations! You've completed the build!"


# Singleton instance
_guidance_generator_instance = None


def get_guidance_generator() -> GuidanceGenerator:
    """Get GuidanceGenerator singleton instance."""
    global _guidance_generator_instance
    if _guidance_generator_instance is None:
        _guidance_generator_instance = GuidanceGenerator()
    return _guidance_generator_instance




