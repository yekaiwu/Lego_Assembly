"""
State Matcher - Matches detected assembly state to graph nodes.
Uses hierarchical graph structure + part similarity scoring.
"""

from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from collections import Counter


class StateMatcher:
    """
    Matches visual assembly state to graph nodes.

    Uses multiple scoring methods:
    1. Part overlap (Jaccard similarity)
    2. Subassembly matching
    3. Step sequence heuristics
    4. Color/shape distribution similarity
    """

    def __init__(self, graph_manager):
        """
        Args:
            graph_manager: GraphManager instance with loaded assembly graphs
        """
        self.graph_manager = graph_manager
        logger.info("StateMatcher initialized")

    def match_state(
        self,
        detected_state: Dict[str, Any],
        manual_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Match detected state to graph step nodes.

        Args:
            detected_state: Output from VisualStateExtractor
            manual_id: Manual identifier
            top_k: Number of top matches to return

        Returns:
            List of matches with scores:
            [
                {
                    "step_number": 5,
                    "confidence": 0.87,
                    "match_reason": "8/10 parts matched",
                    "step_state": {...},  # Full step state from graph
                    "next_steps": [6],    # Possible next steps
                },
                ...
            ]
        """
        graph = self.graph_manager.load_graph(manual_id)
        if not graph:
            logger.error(f"No graph found for manual {manual_id}")
            return []

        # Extract parts from detected state
        detected_parts = self._normalize_parts(detected_state.get("detected_parts", []))
        detected_subassemblies = detected_state.get("subassemblies", [])

        if not detected_parts and not detected_subassemblies:
            logger.warning("No parts or subassemblies detected in state")
            return []

        logger.debug(f"Matching {len(detected_parts)} detected parts against graph")

        # Score all step states in graph
        matches = []
        step_states = graph.get("step_states", [])

        for step_state in step_states:
            step_number = step_state.get("step_number")
            if step_number is None:
                continue

            # Compute similarity score
            score, reason = self._compute_similarity(
                detected_parts,
                detected_subassemblies,
                step_state,
                step_number
            )

            if score > 0.1:  # Minimum threshold
                matches.append({
                    "step_number": step_number,
                    "confidence": score,
                    "match_reason": reason,
                    "step_state": step_state,
                    "next_steps": self._get_next_steps(step_number, step_states),
                })

        # Sort by confidence and return top-k
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        top_matches = matches[:top_k]

        logger.info(f"Found {len(matches)} potential matches, returning top {len(top_matches)}")
        if top_matches:
            logger.info(f"Best match: Step {top_matches[0]['step_number']} "
                       f"(confidence: {top_matches[0]['confidence']:.2f})")

        return top_matches

    def _normalize_parts(self, parts: List[Dict]) -> List[str]:
        """
        Normalize parts to canonical descriptions.

        "red 2x4 brick" → "red_2x4_brick"
        """
        normalized = []
        for part in parts:
            desc = part.get("description", "")
            color = part.get("color", "")
            size = part.get("size", "")
            shape = part.get("shape", "")

            # Create canonical form
            # Prefer using the full description if available
            if desc:
                canonical = desc.lower().replace(" ", "_")
            else:
                canonical = f"{color}_{size}_{shape}".lower().replace(" ", "_")

            quantity = part.get("quantity", 1)

            # Add multiple times for quantity
            normalized.extend([canonical] * quantity)

        return normalized

    def _compute_similarity(
        self,
        detected_parts: List[str],
        detected_subassemblies: List[Dict],
        step_state: Dict[str, Any],
        step_number: int
    ) -> Tuple[float, str]:
        """
        Compute similarity score between detected state and step state.

        Returns:
            (score, reason) tuple
        """
        # Extract step state parts
        step_parts = self._extract_step_parts(step_state)
        step_assembly_desc = step_state.get("assembly_description", "")

        # 1. Part overlap score (Jaccard similarity)
        detected_set = set(detected_parts)
        step_set = set(step_parts)

        if not detected_set or not step_set:
            return 0.0, "no_parts"

        intersection = detected_set & step_set
        union = detected_set | step_set

        part_score = len(intersection) / len(union) if union else 0.0

        # 2. Subassembly matching bonus
        subassembly_bonus = 0.0
        if detected_subassemblies and step_assembly_desc:
            # Check if any detected subassembly matches step description
            for subasm in detected_subassemblies:
                subasm_desc = subasm.get("description", "").lower()
                step_desc_lower = step_assembly_desc.lower()

                # Fuzzy match: check for key words
                subasm_words = set(subasm_desc.split())
                step_words = set(step_desc_lower.split())
                word_overlap = subasm_words & step_words

                if len(word_overlap) >= 2:  # At least 2 words match
                    subassembly_bonus = 0.2
                    break

        # 3. Part count similarity
        count_similarity = 1.0 - abs(len(detected_parts) - len(step_parts)) / max(len(detected_parts), len(step_parts))
        count_score = count_similarity * 0.1  # Small weight

        # Combined score
        final_score = (part_score * 0.7) + subassembly_bonus + count_score
        final_score = min(final_score, 1.0)

        # Generate reason
        matched_parts = len(intersection)
        total_detected = len(detected_set)
        total_step = len(step_set)

        reason = f"{matched_parts}/{total_detected} detected parts matched ({matched_parts}/{total_step} step parts)"

        if subassembly_bonus > 0:
            reason += ", subassembly matched"

        return final_score, reason

    def _extract_step_parts(self, step_state: Dict[str, Any]) -> List[str]:
        """Extract normalized parts from step state."""
        parts = []

        # From parts_needed
        for part in step_state.get("parts_needed", []):
            desc = part.get("description", "")
            quantity = part.get("quantity", 1)

            # Normalize description
            canonical = desc.lower().replace(" ", "_")
            parts.extend([canonical] * quantity)

        # Also consider parts_accumulated (total parts so far)
        for part in step_state.get("parts_accumulated", []):
            desc = part.get("description", "")
            quantity = part.get("quantity", 1)
            canonical = desc.lower().replace(" ", "_")
            parts.extend([canonical] * quantity)

        return parts

    def _get_next_steps(self, current_step: int, step_states: List[Dict]) -> List[int]:
        """Get step numbers of next sequential steps."""
        # Simple sequential: next step is current_step + 1
        next_steps = []

        # Find if next step exists
        for state in step_states:
            step_num = state.get("step_number")
            if step_num == current_step + 1:
                next_steps.append(step_num)
                break

        return next_steps

    def match_by_subassembly(
        self,
        detected_subassembly: str,
        manual_id: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Match based on subassembly description alone.

        Useful when user shows a completed subassembly.

        Args:
            detected_subassembly: Description of detected subassembly
            manual_id: Manual identifier
            top_k: Number of top matches

        Returns:
            List of matching step states
        """
        graph = self.graph_manager.load_graph(manual_id)
        if not graph:
            return []

        matches = []
        step_states = graph.get("step_states", [])
        subasm_lower = detected_subassembly.lower()

        for step_state in step_states:
            assembly_desc = step_state.get("assembly_description", "").lower()

            # Fuzzy word matching
            subasm_words = set(subasm_lower.split())
            assembly_words = set(assembly_desc.split())
            overlap = subasm_words & assembly_words

            if len(overlap) >= 2:  # At least 2 matching words
                score = len(overlap) / max(len(subasm_words), len(assembly_words))
                matches.append({
                    "step_number": step_state.get("step_number"),
                    "confidence": score,
                    "match_reason": f"Subassembly: {len(overlap)} words matched",
                    "step_state": step_state,
                    "next_steps": self._get_next_steps(step_state.get("step_number"), step_states)
                })

        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches[:top_k]


# Singleton instance
_state_matcher_instance = None


def get_state_matcher(graph_manager=None) -> StateMatcher:
    """Get StateMatcher singleton instance."""
    global _state_matcher_instance

    if _state_matcher_instance is None:
        if graph_manager is None:
            from .graph_manager import get_graph_manager
            graph_manager = get_graph_manager()

        _state_matcher_instance = StateMatcher(graph_manager)

    return _state_matcher_instance
