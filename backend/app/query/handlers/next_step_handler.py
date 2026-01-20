"""
Next Step Handler - Handles "What's next?" queries.
Uses graph traversal to determine next assembly step.
"""

from typing import Dict, Any, Optional
from loguru import logger

from .base_handler import BaseQueryHandler


class NextStepHandler(BaseQueryHandler):
    """
    Handles queries about the next assembly step.

    Flow:
    1. Detect current state (from photo or context)
    2. Match to graph node
    3. Get next steps from graph
    4. Format response with next step details
    """

    def __init__(self, graph_manager, rag_pipeline):
        super().__init__()
        self.graph_manager = graph_manager
        self.rag_pipeline = rag_pipeline

        # Lazy load extractors
        self._state_extractor = None
        self._state_matcher = None

    @property
    def state_extractor(self):
        """Lazy load VisualStateExtractor."""
        if self._state_extractor is None:
            from ...vision.visual_state_extractor import get_visual_state_extractor
            self._state_extractor = get_visual_state_extractor()
        return self._state_extractor

    @property
    def state_matcher(self):
        """Lazy load StateMatcher."""
        if self._state_matcher is None:
            from ...graph.state_matcher import get_state_matcher
            self._state_matcher = get_state_matcher(self.graph_manager)
        return self._state_matcher

    def handle(
        self,
        query: str,
        manual_id: str,
        image_path: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle "what's next?" query.

        Returns:
            {
                "answer": "Next, attach the red 2x4 brick to the top...",
                "current_step": 5,
                "next_step": 6,
                "confidence": 0.87,
                "parts_needed": [...],
                "actions": [...],
                "visual_aids": ["step_6_image.png"]
            }
        """
        context = context or {}

        # 1. Determine current state
        current_step = None
        confidence = 0.0
        detection_method = "unknown"

        if image_path:
            # Extract state from photo
            logger.info("Extracting state from photo...")
            detected_state = self.state_extractor.extract_state(
                image_path,
                manual_context={"manual_id": manual_id}
            )

            # Match to graph
            matches = self.state_matcher.match_state(
                detected_state,
                manual_id,
                top_k=1
            )

            if matches:
                current_step = matches[0]["step_number"]
                confidence = matches[0]["confidence"]
                detection_method = "image_analysis"
                logger.info(f"Detected current step: {current_step} (conf: {confidence:.2f})")
            else:
                logger.warning("Could not match detected state to any step")

        elif context and "current_step" in context:
            # Use provided step
            current_step = context["current_step"]
            confidence = 1.0
            detection_method = "user_provided"
            logger.info(f"Using provided current step: {current_step}")

        # If we still don't know current step
        if current_step is None:
            return self._format_response(
                answer="I need to see a photo of your current progress to determine what's next. "
                       "Please provide a photo or tell me what step you're on.",
                confidence=0.0,
                requires_image=True,
                metadata={"reason": "no_current_step"}
            )

        # 2. Get next step from graph
        step_state = self.graph_manager.get_step_state(manual_id, current_step)
        if not step_state:
            return self._format_response(
                answer=f"I found that you're on step {current_step}, but I couldn't load the details for the next step.",
                confidence=confidence,
                current_step=current_step,
                error="step_not_found"
            )

        # Find next step
        graph = self.graph_manager.load_graph(manual_id)
        if not graph:
            return self._format_response(
                answer="Sorry, I couldn't load the assembly manual.",
                confidence=0.0,
                error="graph_not_found"
            )

        next_step_num = current_step + 1
        next_step_state = self.graph_manager.get_step_state(manual_id, next_step_num)

        if not next_step_state:
            # Check if assembly is complete
            all_steps = graph.get("step_states", [])
            max_step = max([s.get("step_number", 0) for s in all_steps])

            if current_step >= max_step:
                return self._format_response(
                    answer="🎉 Congratulations! You've completed the assembly!",
                    confidence=confidence,
                    current_step=current_step,
                    is_complete=True,
                    metadata={"detection_method": detection_method}
                )
            else:
                return self._format_response(
                    answer=f"I detected you're on step {current_step}, but I couldn't find information about the next step.",
                    confidence=confidence,
                    current_step=current_step,
                    error="next_step_not_found"
                )

        # 3. Extract next step details
        parts_needed = next_step_state.get("parts_needed", [])
        actions = next_step_state.get("actions", [])
        assembly_desc = next_step_state.get("assembly_description", "")

        # 4. Format response
        answer = self._build_answer(
            current_step=current_step,
            next_step_num=next_step_num,
            assembly_desc=assembly_desc,
            parts_needed=parts_needed,
            actions=actions,
            confidence=confidence
        )

        return self._format_response(
            answer=answer,
            confidence=confidence,
            current_step=current_step,
            next_step=next_step_num,
            parts_needed=parts_needed,
            actions=actions,
            visual_aids=[f"step_{next_step_num}_image.png"],
            metadata={
                "handler": "next_step",
                "detection_method": detection_method,
                "used_image": image_path is not None
            }
        )

    def _build_answer(
        self,
        current_step: int,
        next_step_num: int,
        assembly_desc: str,
        parts_needed: list,
        actions: list,
        confidence: float
    ) -> str:
        """Build formatted answer text."""
        lines = []

        # Header
        if confidence < 0.7:
            lines.append(f"I think you're on step {current_step}, but I'm not entirely sure (confidence: {confidence:.0%}).\n")
        else:
            lines.append(f"You're on step {current_step}.\n")

        lines.append(f"**Next: Step {next_step_num}**\n")

        # Assembly description
        if assembly_desc:
            lines.append(f"{assembly_desc}\n")

        # Parts needed
        if parts_needed:
            lines.append("**Parts needed:**")
            for part in parts_needed:
                desc = part.get("description", "unknown part")
                qty = part.get("quantity", 1)
                lines.append(f"  • {qty}x {desc}")
            lines.append("")

        # Instructions
        if actions:
            lines.append("**Instructions:**")
            for i, action in enumerate(actions, 1):
                lines.append(f"  {i}. {action}")

        return "\n".join(lines)
