"""
Parts Handler - Handles "What parts do I need?" queries.
Uses graph + RAG to retrieve detailed part information.
"""

from typing import Dict, Any, Optional
import re
from loguru import logger

from .base_handler import BaseQueryHandler


class PartsHandler(BaseQueryHandler):
    """
    Handles queries about parts needed.

    Flow:
    1. Determine target step (current + 1, or specified)
    2. Query graph for parts_needed
    3. Use RAG to get detailed part descriptions (optional enrichment)
    4. Format response with part details
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
        Handle "what parts?" query.

        Returns:
            {
                "answer": "For step 6, you need: ...",
                "target_step": 6,
                "parts": [...],
                "confidence": 0.85
            }
        """
        context = context or {}

        # 1. Determine target step
        target_step = None
        current_step = None
        confidence = 0.9  # Default confidence

        # Check if query specifies a step ("what parts for step 5?")
        step_match = re.search(r'step\s+(\d+)', query.lower())
        if step_match:
            target_step = int(step_match.group(1))
            confidence = 1.0
            logger.info(f"Target step from query: {target_step}")

        # If not specified, detect current state and use next step
        if not target_step:
            if image_path:
                logger.info("Detecting current step from photo...")
                detected_state = self.state_extractor.extract_state(
                    image_path,
                    manual_context={"manual_id": manual_id}
                )
                matches = self.state_matcher.match_state(detected_state, manual_id, top_k=1)
                if matches:
                    current_step = matches[0]["step_number"]
                    target_step = current_step + 1  # Next step
                    confidence = matches[0]["confidence"]
                    logger.info(f"Detected current step {current_step}, targeting step {target_step}")

            elif context and "current_step" in context:
                current_step = context["current_step"]
                target_step = current_step + 1
                confidence = 1.0
                logger.info(f"Using provided current step {current_step}, targeting step {target_step}")

        # Still no target step?
        if not target_step:
            return self._format_response(
                answer="Which step are you asking about? You can say 'What parts for step 5?' "
                       "or show me your current progress.",
                confidence=0.0,
                requires_clarification=True,
                metadata={"reason": "no_target_step"}
            )

        # 2. Get parts from graph
        step_state = self.graph_manager.get_step_state(manual_id, target_step)

        if not step_state:
            return self._format_response(
                answer=f"Step {target_step} not found in this manual.",
                confidence=0.0,
                error="step_not_found",
                target_step=target_step
            )

        parts_needed = step_state.get("parts_needed", [])
        assembly_desc = step_state.get("assembly_description", "")

        # 3. Build answer
        answer = self._build_answer(
            target_step=target_step,
            parts_needed=parts_needed,
            assembly_desc=assembly_desc,
            current_step=current_step
        )

        return self._format_response(
            answer=answer,
            confidence=confidence,
            target_step=target_step,
            current_step=current_step,
            parts_needed=parts_needed,
            metadata={
                "handler": "parts",
                "used_image": image_path is not None
            }
        )

    def _build_answer(
        self,
        target_step: int,
        parts_needed: list,
        assembly_desc: str,
        current_step: Optional[int]
    ) -> str:
        """Build formatted answer text."""
        lines = []

        # Header
        if current_step:
            lines.append(f"You're on step {current_step}.")
            lines.append(f"\n**Parts needed for step {target_step}:**\n")
        else:
            lines.append(f"**Parts needed for step {target_step}:**\n")

        # Assembly description
        if assembly_desc:
            lines.append(f"_{assembly_desc}_\n")

        # Parts list
        if parts_needed:
            for part in parts_needed:
                desc = part.get("description", "unknown")
                qty = part.get("quantity", 1)
                color = part.get("color", "")

                line = f"  • **{qty}x {desc}**"
                if color and color not in desc.lower():
                    line += f" ({color})"
                lines.append(line)
        else:
            lines.append("  _No specific parts listed for this step._")

        return "\n".join(lines)
