"""
Part Similarity Clustering Strategy.

Identifies repeated/symmetrical structures by comparing part usage patterns
across consecutive steps using Jaccard similarity.
"""

from typing import Dict, Any, List, Set
from loguru import logger


class PartSimilarityClusterer:
    """
    Detects repeated structures by analyzing part usage similarity across steps.

    Algorithm:
    1. Build part-step usage matrix (which parts in which steps)
    2. Compare consecutive steps using Jaccard similarity
    3. Group steps with similarity > threshold
    4. Return multi-step patterns (≥min_pattern_steps)
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.7,
        min_pattern_steps: int = 2
    ):
        """
        Initialize the clusterer.

        Args:
            jaccard_threshold: Minimum similarity to group steps (0.0-1.0)
            min_pattern_steps: Minimum steps for a valid pattern
        """
        self.jaccard_threshold = jaccard_threshold
        self.min_pattern_steps = min_pattern_steps

    def find_repeated_structures(
        self,
        graph: Dict[str, Any],
        extracted_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find repeated assembly structures across steps.

        Args:
            graph: Current graph structure
            extracted_steps: Extracted step data from Stage 1

        Returns:
            List of discovered patterns with metadata
        """
        logger.debug("Running part similarity clustering analysis")

        patterns = []

        # Build part usage matrix: step_num -> {part signatures}
        part_usage = self._build_part_usage_matrix(extracted_steps)

        if not part_usage:
            logger.debug("No part usage data available for clustering")
            return patterns

        # Sliding window to find consecutive similar steps
        steps = sorted(part_usage.keys())
        i = 0

        while i < len(steps):
            current_step = steps[i]
            current_parts = part_usage[current_step]

            if not current_parts:
                i += 1
                continue

            group_steps = [current_step]

            # Look ahead for similar steps
            j = i + 1
            while j < len(steps):
                next_parts = part_usage[steps[j]]

                # Calculate Jaccard similarity
                similarity = self._jaccard_similarity(current_parts, next_parts)

                if similarity >= self.jaccard_threshold:
                    group_steps.append(steps[j])
                    j += 1
                else:
                    break

            # Record multi-step pattern
            if len(group_steps) >= self.min_pattern_steps:
                pattern = self._create_pattern(
                    group_steps=group_steps,
                    parts=current_parts,
                    extracted_steps=extracted_steps
                )
                patterns.append(pattern)
                logger.debug(
                    f"Found repeated structure: {len(group_steps)} steps "
                    f"using {len(current_parts)} part types"
                )

            # Move to next unprocessed step
            i = j if j > i + 1 else i + 1

        logger.info(f"Part similarity clustering found {len(patterns)} repeated structures")
        return patterns

    def _build_part_usage_matrix(
        self,
        extracted_steps: List[Dict[str, Any]]
    ) -> Dict[int, Set[str]]:
        """
        Build matrix of which parts are used in which steps.

        Args:
            extracted_steps: Step data

        Returns:
            Dictionary mapping step_number -> set of part signatures
        """
        part_usage = {}

        for step in extracted_steps:
            step_num = step.get("step_number")
            parts = step.get("parts_required", [])

            if not step_num or not parts:
                continue

            # Create part signatures: "color:shape"
            signatures = {
                f"{p.get('color', 'unknown')}:{p.get('shape', 'unknown')}"
                for p in parts
            }

            part_usage[step_num] = signatures

        return part_usage

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets.

        Args:
            set_a: First set
            set_b: Second set

        Returns:
            Similarity score (0.0-1.0)
        """
        if not set_a and not set_b:
            return 1.0

        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def _create_pattern(
        self,
        group_steps: List[int],
        parts: Set[str],
        extracted_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create pattern metadata from grouped steps.

        Args:
            group_steps: List of step numbers in the pattern
            parts: Set of part signatures used
            extracted_steps: All extracted steps for context

        Returns:
            Pattern dictionary
        """
        # Generate descriptive name
        repetitions = len(group_steps)
        part_count = len(parts)

        name = f"Repeated Assembly ({repetitions}x)"

        # Try to infer more specific name from parts
        part_list = list(parts)
        if part_list:
            # Extract common color or shape
            colors = {p.split(':')[0] for p in part_list}
            shapes = {p.split(':')[1] for p in part_list}

            if len(colors) == 1:
                color = list(colors)[0].title()
                name = f"{color} Structure ({repetitions}x)"
            elif len(shapes) == 1:
                shape = list(shapes)[0].replace('_', ' ').title()
                name = f"{shape} Assembly ({repetitions}x)"

        # Create description
        description = (
            f"Repeated structure using {part_count} part type(s) "
            f"across {repetitions} steps"
        )

        # Add context from first step if available
        first_step_idx = group_steps[0] - 1
        if 0 <= first_step_idx < len(extracted_steps):
            first_step = extracted_steps[first_step_idx]
            notes = first_step.get("notes", "")
            if notes:
                # Extract meaningful keywords
                keywords = ["leg", "wheel", "wing", "arm", "support", "wall"]
                for kw in keywords:
                    if kw.lower() in notes.lower():
                        name = f"{kw.title()} Assembly ({repetitions}x)"
                        break

        return {
            "type": "repeated_structure",
            "name": name,
            "description": description,
            "steps": group_steps,
            "parts": list(parts),
            "confidence": self._calculate_confidence(group_steps, parts),
            "discovery_method": "part_similarity",
            "metadata": {
                "jaccard_threshold_used": self.jaccard_threshold,
                "repetition_count": repetitions,
                "part_types": part_count
            }
        }

    def _calculate_confidence(
        self,
        group_steps: List[int],
        parts: Set[str]
    ) -> float:
        """
        Calculate confidence score for the pattern.

        Factors:
        - More repetitions = higher confidence
        - More parts = higher confidence
        - Consecutive steps = higher confidence

        Args:
            group_steps: Step numbers in pattern
            parts: Part signatures

        Returns:
            Confidence score (0.0-1.0)
        """
        base_confidence = 0.8

        # Bonus for more repetitions (max +0.15)
        repetition_bonus = min(0.15, (len(group_steps) - 2) * 0.03)

        # Bonus for more parts (max +0.05)
        part_bonus = min(0.05, (len(parts) - 1) * 0.01)

        # Check if steps are consecutive
        is_consecutive = all(
            group_steps[i] + 1 == group_steps[i + 1]
            for i in range(len(group_steps) - 1)
        )
        consecutive_bonus = 0.1 if is_consecutive else 0.0

        confidence = base_confidence + repetition_bonus + part_bonus + consecutive_bonus

        # Cap at 1.0
        return min(1.0, confidence)
