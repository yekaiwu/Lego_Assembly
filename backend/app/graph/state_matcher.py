"""
State Matcher - Matches detected assembly state to graph nodes.
Uses hierarchical graph structure + part similarity scoring + visual matching.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from loguru import logger
from collections import Counter
from difflib import SequenceMatcher


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

    def combine_text_and_visual_matches(
        self,
        text_matches: List[Dict[str, Any]],
        visual_matches: List[Dict[str, Any]],
        text_weight: float = 0.6,
        visual_weight: float = 0.4,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Combine text-based (part matching) and visual similarity matches.

        Uses weighted combination of both signals for more robust matching.

        Args:
            text_matches: Matches from match_state() (text-based part matching)
            visual_matches: Matches from VisualMatcher (visual similarity)
            text_weight: Weight for text-based matching (default: 0.6)
            visual_weight: Weight for visual matching (default: 0.4)
            top_k: Number of top matches to return

        Returns:
            Combined matches sorted by combined score:
            [
                {
                    "step_number": 3,
                    "combined_confidence": 0.82,
                    "text_confidence": 0.87,
                    "visual_confidence": 0.75,
                    "match_reason": "Combined: 82% (text: 87%, visual: 75%)",
                    "step_state": {...},
                    "next_step": 4,
                    ...
                },
                ...
            ]
        """
        # Build lookup maps
        text_by_step = {m["step_number"]: m for m in text_matches}
        visual_by_step = {m["step_number"]: m for m in visual_matches}

        # Get all unique step numbers
        all_steps = set(text_by_step.keys()) | set(visual_by_step.keys())

        combined_matches = []

        for step_num in all_steps:
            text_match = text_by_step.get(step_num)
            visual_match = visual_by_step.get(step_num)

            # Get confidence scores (default to 0 if not present)
            text_conf = text_match["confidence"] if text_match else 0.0
            visual_conf = visual_match["visual_similarity"] if visual_match else 0.0

            # Compute weighted combined score
            combined_conf = text_weight * text_conf + visual_weight * visual_conf

            # Build combined match result
            combined = {
                "step_number": step_num,
                "combined_confidence": combined_conf,
                "text_confidence": text_conf,
                "visual_confidence": visual_conf,
                "match_reason": (
                    f"Combined: {combined_conf:.0%} "
                    f"(text: {text_conf:.0%}, visual: {visual_conf:.0%})"
                ),
                "next_step": step_num + 1
            }

            # Include step_state from text match (preferred) or visual match
            if text_match and "step_state" in text_match:
                combined["step_state"] = text_match["step_state"]
            elif visual_match and "step_state" in visual_match:
                combined["step_state"] = visual_match["step_state"]

            # Include additional details from both matches
            if text_match:
                combined["text_match_reason"] = text_match.get("match_reason")
                combined["precision"] = text_match.get("precision")
                combined["recall"] = text_match.get("recall")
                combined["matched_node_ids"] = text_match.get("matched_node_ids", [])

            if visual_match:
                combined["visual_match_reason"] = visual_match.get("match_reason")
                combined["reference_image_path"] = visual_match.get("reference_image_path")
                combined["segmentation_confidence"] = visual_match.get("segmentation_confidence")

            combined_matches.append(combined)

        # Sort by combined confidence
        combined_matches.sort(key=lambda x: x["combined_confidence"], reverse=True)
        top_matches = combined_matches[:top_k]

        logger.info(
            f"Combined {len(text_matches)} text + {len(visual_matches)} visual matches "
            f"-> {len(combined_matches)} total, returning top {len(top_matches)}"
        )

        if top_matches:
            best = top_matches[0]
            logger.info(
                f"Best combined match: Step {best['step_number']} "
                f"(combined: {best['combined_confidence']:.2f}, "
                f"text: {best['text_confidence']:.2f}, "
                f"visual: {best['visual_confidence']:.2f})"
            )

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

    def _normalize_shape(self, shape: str) -> str:
        """
        Normalize shape description for better matching.

        Examples:
        - "1x2 slope 30° 1x2x2/3" → "1x2 slope"
        - "round plate with eye" → "round plate"
        - "2x4 rectangular brick" → "2x4 brick"
        """
        import re

        # Remove special characters and normalize
        normalized = shape.lower()

        # Remove degree symbols and fractions
        normalized = re.sub(r'[°×/]', '', normalized)

        # Remove "printed with..." patterns
        normalized = re.sub(r',?\s*printed with.*', '', normalized)

        # Extract core shape keywords
        # Common patterns: "1x2 plate", "2x4 brick", "1x1 round plate"
        core_keywords = ['brick', 'plate', 'tile', 'slope', 'round', 'flat']

        # Keep size prefix (e.g., "1x2", "2x4")
        size_match = re.search(r'\d+x\d+', normalized)
        size_prefix = size_match.group(0) if size_match else ""

        # Find core shape keyword
        shape_keyword = ""
        for keyword in core_keywords:
            if keyword in normalized:
                shape_keyword = keyword
                break

        # Combine size + shape
        if size_prefix and shape_keyword:
            return f"{size_prefix} {shape_keyword}"
        elif shape_keyword:
            return shape_keyword
        else:
            # Fallback: take first 2-3 words
            words = normalized.split()[:3]
            return " ".join(words)

    def _fuzzy_match_signature(
        self,
        detected_signature: str,
        signature_map: Dict[str, str],
        threshold: float = 0.7
    ) -> Optional[str]:
        """
        Find best matching signature using fuzzy string matching.

        Args:
            detected_signature: Signature from detected part (e.g., "black:1x2 slope 30°")
            signature_map: Map of known signatures to node IDs
            threshold: Minimum similarity score (0-1) to consider a match

        Returns:
            node_id of best match, or None if no match above threshold
        """
        detected_color, detected_shape = detected_signature.split(":", 1)

        best_match = None
        best_score = 0.0

        for known_sig, node_id in signature_map.items():
            known_color, known_shape = known_sig.split(":", 1)

            # Color must match exactly
            if known_color != detected_color:
                continue

            # Fuzzy match on shape using normalized forms
            detected_norm = self._normalize_shape(detected_shape)
            known_norm = self._normalize_shape(known_shape)

            # Calculate similarity
            similarity = SequenceMatcher(None, detected_norm, known_norm).ratio()

            # Also check if one contains the other (partial match)
            if detected_norm in known_norm or known_norm in detected_norm:
                similarity = max(similarity, 0.8)  # Boost partial matches

            if similarity > best_score:
                best_score = similarity
                best_match = node_id

        if best_score >= threshold:
            logger.debug(f"Fuzzy match: '{detected_signature}' → matched with score {best_score:.2f}")
            return best_match
        else:
            return None

    def _convert_parts_to_node_ids(
        self,
        detected_parts: List[Dict[str, Any]],
        signature_map: Dict[str, str]
    ) -> Set[str]:
        """
        Convert detected parts to node IDs using fuzzy signature matching.

        Args:
            detected_parts: List of detected parts with color, shape, etc.
            signature_map: Mapping from "color:shape" to node_id

        Returns:
            Set of node IDs matching the detected parts
        """
        node_ids = set()

        for part in detected_parts:
            color = part.get("color", "").lower()

            # Try both 'shape' and 'description' fields
            shape = part.get("shape") or part.get("description", "")
            shape = shape.lower()

            if color and shape:
                signature = f"{color}:{shape}"

                # Try exact match first (fastest)
                if signature in signature_map:
                    node_id = signature_map[signature]
                    quantity = part.get("quantity", 1)
                    for _ in range(quantity):
                        node_ids.add(node_id)
                else:
                    # Fall back to fuzzy matching
                    node_id = self._fuzzy_match_signature(signature, signature_map, threshold=0.6)
                    if node_id:
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
