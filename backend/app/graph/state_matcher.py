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
        Match detected assembly state using pre-computed step_states.

        OPTIMIZED APPROACH: Uses step_states with existing_nodes for O(n) matching.
        Each step_state contains cumulative list of all nodes up to that step.

        Args:
            detected_state: Output from StateAnalyzer with detected_parts
            manual_id: Manual identifier
            top_k: Number of top matches to return

        Returns:
            List of matches with scores:
            [
                {
                    "step_number": 3,  # The step they completed
                    "confidence": 0.87,
                    "match_reason": "8/9 parts matched (F1: 0.87)",
                    "step_state": {...},  # The matching step state
                    "next_step": 4  # The next step to do
                },
                ...
            ]
        """
        graph = self.graph_manager.load_graph(manual_id)
        if not graph:
            logger.error(f"No graph found for manual {manual_id}")
            return []

        # Extract detected parts
        detected_parts = detected_state.get("detected_parts", [])
        if not detected_parts:
            logger.warning("No parts detected in state")
            return []

        logger.debug(f"Matching {len(detected_parts)} detected parts against step states")

        # Build part signature → node_id mapping from part catalog
        part_catalog = graph.get("part_catalog", {}).get("parts_catalog", {})
        if not part_catalog:
            logger.error("No part catalog found in graph")
            return []

        signature_to_node = self._build_part_signature_map(part_catalog, graph.get("nodes", []))

        # Convert detected parts to node IDs
        detected_node_ids = self._convert_parts_to_node_ids(detected_parts, signature_to_node)

        if not detected_node_ids:
            logger.warning("Could not map detected parts to node IDs")
            return []

        logger.debug(f"Mapped to {len(detected_node_ids)} node IDs")

        # Match against step_states (most complete first)
        step_states = graph.get("step_states", [])
        if not step_states:
            logger.error("No step_states found in graph")
            return []

        matches = self._score_step_states(
            detected_node_ids,
            step_states,
            graph.get("nodes", [])
        )

        # Sort by confidence and return top-k
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        top_matches = matches[:top_k]

        logger.info(f"Found {len(matches)} potential matches, returning top {len(top_matches)}")
        if top_matches:
            logger.info(f"Best match: Step {top_matches[0]['step_number']} "
                       f"(confidence: {top_matches[0]['confidence']:.2f}) "
                       f"- {top_matches[0]['match_reason']}")

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

    def _build_part_signature_map(
        self,
        part_catalog: Dict[str, Any],
        nodes: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Build mapping from part signatures to node IDs.

        A part signature is "color:shape" which uniquely identifies parts.

        Args:
            part_catalog: Part catalog from graph
            nodes: All nodes from graph

        Returns:
            Dict mapping "color:shape" → "part_0", "part_1", etc.
        """
        signature_map = {}

        for node_id, part_data in part_catalog.items():
            color = part_data.get("color", "").lower()
            shape = part_data.get("shape", "").lower()

            if color and shape:
                signature = f"{color}:{shape}"
                signature_map[signature] = node_id

        logger.debug(f"Built signature map with {len(signature_map)} entries")
        return signature_map

    def _convert_parts_to_node_ids(
        self,
        detected_parts: List[Dict[str, Any]],
        signature_map: Dict[str, str]
    ) -> Set[str]:
        """
        Convert detected parts to node IDs using signature matching.

        Args:
            detected_parts: List of detected parts with color, shape, etc.
            signature_map: Mapping from "color:shape" to node_id

        Returns:
            Set of node IDs matching the detected parts
        """
        node_ids = set()

        for part in detected_parts:
            color = part.get("color", "").lower()
            shape = part.get("shape", "").lower()

            if color and shape:
                signature = f"{color}:{shape}"
                if signature in signature_map:
                    node_id = signature_map[signature]
                    # Add multiple times based on quantity
                    quantity = part.get("quantity", 1)
                    for _ in range(quantity):
                        node_ids.add(node_id)
                else:
                    logger.debug(f"No match for signature: {signature}")

        return node_ids

    def _score_step_states(
        self,
        detected_node_ids: Set[str],
        step_states: List[Dict[str, Any]],
        nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score each step_state against detected parts using F1 score.

        Args:
            detected_node_ids: Set of detected part node IDs
            step_states: List of step states with existing_nodes
            nodes: All nodes from graph (to filter parts vs subassemblies)

        Returns:
            List of match results
        """
        matches = []

        # Build set of part node IDs for quick filtering
        part_node_ids = {n["node_id"] for n in nodes if n["type"] == "part"}

        # Score each step state (start from end for better matches first)
        for step_state in reversed(step_states):
            step_number = step_state.get("step_number")
            if step_number is None:
                continue

            # Get existing nodes for this step (cumulative)
            existing_nodes = set(step_state.get("existing_nodes", []))

            # Filter to only part nodes (exclude subassemblies)
            step_part_nodes = existing_nodes & part_node_ids

            if not step_part_nodes:
                continue

            # Calculate overlap
            overlap = detected_node_ids & step_part_nodes

            # Precision: how many detected parts are correct
            precision = len(overlap) / len(detected_node_ids) if detected_node_ids else 0.0

            # Recall: how many expected parts were detected
            recall = len(overlap) / len(step_part_nodes) if step_part_nodes else 0.0

            # F1 score (harmonic mean of precision and recall)
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # Only include reasonable matches
            if f1_score > 0.1:
                reason = (
                    f"{len(overlap)}/{len(detected_node_ids)} parts matched "
                    f"({len(overlap)}/{len(step_part_nodes)} expected, F1: {f1_score:.2f})"
                )

                matches.append({
                    "step_number": step_number,
                    "confidence": f1_score,
                    "match_reason": reason,
                    "step_state": step_state,
                    "next_step": step_number + 1,
                    "precision": precision,
                    "recall": recall,
                    "matched_node_ids": list(overlap)  # Node IDs that matched
                })

        return matches

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
