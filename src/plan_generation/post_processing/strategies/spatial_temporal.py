"""
Spatial-Temporal Pattern Mining Strategy.

Detects progressive builds (e.g., base → walls → roof) by analyzing spatial
positions and temporal sequences in step descriptions.
"""

from typing import Dict, Any, List, Optional
from loguru import logger
import re


class SpatialTemporalPatternMiner:
    """
    Detects progressive assembly patterns across step sequences.

    Algorithm:
    1. Analyze spatial positions over time from step descriptions
    2. Detect vertical progressions (keywords: "top", "above", "stack")
    3. Detect horizontal progressions (keywords: "side", "beside", "adjacent")
    4. Group consecutive steps with same direction (≥min_sequence_steps)
    """

    # Spatial keyword patterns
    VERTICAL_UP_KEYWORDS = [
        "top", "above", "upper", "stack", "layer", "level",
        "on top", "place above", "add to top"
    ]

    VERTICAL_DOWN_KEYWORDS = [
        "bottom", "below", "under", "base", "foundation",
        "underneath", "place below"
    ]

    HORIZONTAL_KEYWORDS = [
        "side", "beside", "next to", "adjacent", "alongside",
        "left", "right", "attach side", "connect side"
    ]

    ENCLOSURE_KEYWORDS = [
        "wall", "roof", "door", "window", "enclose",
        "surround", "cover", "close"
    ]

    EXPANSION_KEYWORDS = [
        "extend", "expand", "widen", "lengthen", "grow",
        "continue", "build out"
    ]

    def __init__(
        self,
        min_sequence_steps: int = 3,
        similarity_threshold: float = 0.6
    ):
        """
        Initialize the miner.

        Args:
            min_sequence_steps: Minimum steps for a valid progression
            similarity_threshold: Threshold for pattern consistency
        """
        self.min_sequence_steps = min_sequence_steps
        self.similarity_threshold = similarity_threshold

    def find_progressive_builds(
        self,
        graph: Dict[str, Any],
        extracted_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find progressive build patterns across step sequences.

        Args:
            graph: Current graph structure
            extracted_steps: Extracted step data

        Returns:
            List of discovered progressive patterns
        """
        logger.debug("Running spatial-temporal pattern mining")

        patterns = []

        # Analyze spatial directions for each step
        spatial_sequence = self._analyze_spatial_directions(extracted_steps)

        if not spatial_sequence:
            logger.debug("No spatial directions detected")
            return patterns

        # Find directional progressions
        progressions = self._find_directional_progressions(spatial_sequence)

        # Create patterns from progressions
        for progression in progressions:
            if len(progression["steps"]) >= self.min_sequence_steps:
                pattern = self._create_pattern(
                    progression=progression,
                    extracted_steps=extracted_steps
                )
                patterns.append(pattern)
                logger.debug(
                    f"Found {progression['direction']} progression: "
                    f"{len(progression['steps'])} steps"
                )

        logger.info(f"Spatial-temporal mining found {len(patterns)} progressive builds")
        return patterns

    def _analyze_spatial_directions(
        self,
        extracted_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze spatial direction for each step.

        Args:
            extracted_steps: Step data

        Returns:
            List of step spatial analyses
        """
        spatial_sequence = []

        for step in extracted_steps:
            step_num = step.get("step_number")
            if not step_num:
                continue

            # Get step text content
            notes = step.get("notes", "")
            existing_assembly = step.get("existing_assembly", "")
            actions = step.get("actions", [])

            # Combine text for analysis
            action_text = " ".join([
                f"{a.get('action_verb', '')} {a.get('target', '')} {a.get('destination', '')}"
                for a in actions
            ])
            combined_text = f"{notes} {existing_assembly} {action_text}".lower()

            # Detect direction
            direction = self._detect_direction(combined_text)

            if direction:
                spatial_sequence.append({
                    "step_number": step_num,
                    "direction": direction,
                    "text": combined_text[:100]  # Keep sample for debugging
                })

        return spatial_sequence

    def _detect_direction(self, text: str) -> Optional[str]:
        """
        Detect spatial direction from text.

        Args:
            text: Step description text (lowercase)

        Returns:
            Direction string or None
        """
        # Check for vertical upward
        if any(kw in text for kw in self.VERTICAL_UP_KEYWORDS):
            return "vertical_up"

        # Check for vertical downward
        if any(kw in text for kw in self.VERTICAL_DOWN_KEYWORDS):
            return "vertical_down"

        # Check for horizontal
        if any(kw in text for kw in self.HORIZONTAL_KEYWORDS):
            return "horizontal"

        # Check for enclosure (special case of progression)
        if any(kw in text for kw in self.ENCLOSURE_KEYWORDS):
            return "enclosure"

        # Check for expansion
        if any(kw in text for kw in self.EXPANSION_KEYWORDS):
            return "expansion"

        return None

    def _find_directional_progressions(
        self,
        spatial_sequence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find sequences of steps with consistent direction.

        Args:
            spatial_sequence: List of step spatial analyses

        Returns:
            List of progressions
        """
        progressions = []

        i = 0
        while i < len(spatial_sequence):
            current = spatial_sequence[i]
            current_direction = current["direction"]

            # Start a new progression
            progression_steps = [current["step_number"]]
            j = i + 1

            # Look ahead for same direction (allowing 1 skip)
            skips = 0
            max_skips = 1

            while j < len(spatial_sequence):
                next_item = spatial_sequence[j]

                if next_item["direction"] == current_direction:
                    progression_steps.append(next_item["step_number"])
                    skips = 0  # Reset skips on match
                    j += 1
                elif skips < max_skips:
                    # Allow one skip
                    skips += 1
                    j += 1
                else:
                    # Different direction, end progression
                    break

            # Record progression if long enough
            if len(progression_steps) >= self.min_sequence_steps:
                progressions.append({
                    "direction": current_direction,
                    "steps": progression_steps
                })

                # Move past this progression
                i = j
            else:
                i += 1

        return progressions

    def _create_pattern(
        self,
        progression: Dict[str, Any],
        extracted_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create pattern metadata from progression.

        Args:
            progression: Progression data
            extracted_steps: All extracted steps

        Returns:
            Pattern dictionary
        """
        direction = progression["direction"]
        steps = progression["steps"]

        # Generate name based on direction
        direction_names = {
            "vertical_up": "Vertical Build",
            "vertical_down": "Foundation Build",
            "horizontal": "Horizontal Expansion",
            "enclosure": "Enclosure Build",
            "expansion": "Progressive Expansion"
        }

        base_name = direction_names.get(direction, "Progressive Build")
        name = f"{base_name} ({len(steps)} steps)"

        # Try to refine name from step content
        step_keywords = []
        for step_num in steps[:3]:  # Check first 3 steps
            step_idx = step_num - 1
            if 0 <= step_idx < len(extracted_steps):
                step = extracted_steps[step_idx]
                notes = step.get("notes", "").lower()

                # Look for functional components
                if "wall" in notes:
                    step_keywords.append("wall")
                elif "roof" in notes:
                    step_keywords.append("roof")
                elif "base" in notes or "foundation" in notes:
                    step_keywords.append("base")
                elif "frame" in notes:
                    step_keywords.append("frame")

        # Refine name if we found keywords
        if step_keywords:
            main_keyword = step_keywords[0].title()
            if direction == "vertical_up":
                name = f"{main_keyword} Assembly (vertical)"
            elif direction == "enclosure":
                name = f"{main_keyword} Construction"
            else:
                name = f"{main_keyword} Build ({len(steps)} steps)"

        # Generate description
        description = self._generate_description(direction, steps)

        return {
            "type": "progressive_build",
            "name": name,
            "description": description,
            "steps": steps,
            "parts": [],  # Will be filled by analyzer
            "confidence": self._calculate_confidence(steps, direction),
            "discovery_method": "spatial_temporal",
            "metadata": {
                "direction": direction,
                "step_count": len(steps),
                "first_step": steps[0],
                "last_step": steps[-1]
            }
        }

    def _generate_description(self, direction: str, steps: List[int]) -> str:
        """
        Generate human-readable description.

        Args:
            direction: Spatial direction
            steps: Step numbers

        Returns:
            Description string
        """
        direction_descriptions = {
            "vertical_up": "upward assembly progression",
            "vertical_down": "downward construction from base",
            "horizontal": "horizontal expansion",
            "enclosure": "enclosing structure build",
            "expansion": "progressive structural expansion"
        }

        direction_desc = direction_descriptions.get(
            direction,
            "progressive assembly"
        )

        return (
            f"Progressive build pattern showing {direction_desc} "
            f"across {len(steps)} consecutive steps"
        )

    def _calculate_confidence(self, steps: List[int], direction: str) -> float:
        """
        Calculate confidence score for the pattern.

        Factors:
        - More steps = higher confidence
        - Certain directions more reliable (vertical_up, enclosure)
        - Consecutive steps = higher confidence

        Args:
            steps: Step numbers in pattern
            direction: Spatial direction

        Returns:
            Confidence score (0.0-1.0)
        """
        base_confidence = 0.75

        # Direction reliability bonus
        reliable_directions = {"vertical_up", "enclosure"}
        direction_bonus = 0.1 if direction in reliable_directions else 0.0

        # Length bonus (max +0.1)
        length_bonus = min(0.1, (len(steps) - self.min_sequence_steps) * 0.02)

        # Check if steps are consecutive
        is_consecutive = all(
            steps[i] + 1 == steps[i + 1]
            for i in range(len(steps) - 1)
        )
        consecutive_bonus = 0.05 if is_consecutive else 0.0

        confidence = (
            base_confidence +
            direction_bonus +
            length_bonus +
            consecutive_bonus
        )

        # Cap at 1.0
        return min(1.0, confidence)
