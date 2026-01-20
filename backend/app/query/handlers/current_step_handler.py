"""
Current Step Handler - Handles "What step am I on?" queries.
Uses visual state detection to determine current progress.
"""

from typing import Dict, Any, Optional
from loguru import logger

from .base_handler import BaseQueryHandler


class CurrentStepHandler(BaseQueryHandler):
    """
    Handles queries about current step.

    Flow:
    1. Detect state from photo
    2. Match to graph
    3. Return current step info
    """

    def __init__(self, graph_manager):
        super().__init__()
        self.graph_manager = graph_manager

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
        Handle "what step am I on?" query.

        Returns:
            {
                "answer": "You're on step 5...",
                "current_step": 5,
                "confidence": 0.87,
                ...
            }
        """
        # Need image to detect current step
        if not image_path:
            return self._format_response(
                answer="Please show me a photo of your current progress so I can tell you what step you're on.",
                confidence=0.0,
                requires_image=True
            )

        # Extract state from photo
        logger.info("Extracting state from photo to determine current step...")
        detected_state = self.state_extractor.extract_state(
            image_path,
            manual_context={"manual_id": manual_id}
        )

        # Match to graph
        matches = self.state_matcher.match_state(
            detected_state,
            manual_id,
            top_k=3  # Get top 3 to show alternatives
        )

        if not matches:
            return self._format_response(
                answer="I couldn't determine which step you're on from the photo. "
                       "The image might be unclear, or the assembly might not match this manual. "
                       "Try taking a clearer photo or telling me which step you think you're on.",
                confidence=0.0,
                metadata={"matches_found": 0}
            )

        # Best match
        best_match = matches[0]
        current_step = best_match["step_number"]
        confidence = best_match["confidence"]
        reason = best_match["match_reason"]

        # Build answer
        answer = self._build_answer(current_step, confidence, reason, matches[1:])

        # Get step state details
        step_state = best_match.get("step_state", {})

        return self._format_response(
            answer=answer,
            confidence=confidence,
            current_step=current_step,
            metadata={
                "handler": "current_step",
                "match_reason": reason,
                "alternative_matches": [m["step_number"] for m in matches[1:]],
                "step_state": step_state
            }
        )

    def _build_answer(
        self,
        current_step: int,
        confidence: float,
        reason: str,
        alternatives: list
    ) -> str:
        """Build formatted answer."""
        lines = []

        # Main result
        if confidence >= 0.8:
            lines.append(f"You're on **step {current_step}**.")
        elif confidence >= 0.5:
            lines.append(f"You appear to be on **step {current_step}** (confidence: {confidence:.0%}).")
        else:
            lines.append(f"I think you're on **step {current_step}**, but I'm not very confident (confidence: {confidence:.0%}).")

        # Match reason
        if reason:
            lines.append(f"\n_{reason}_")

        # Alternative matches
        if alternatives and len(alternatives) > 0:
            alt_steps = [str(m["step_number"]) for m in alternatives[:2]]
            lines.append(f"\nAlternatively, you might be on step {' or '.join(alt_steps)}.")

        return "\n".join(lines)
