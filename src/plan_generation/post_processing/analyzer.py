"""
Post-Processing Subassembly Analyzer.

Main analyzer that coordinates all detection strategies and integrates
discovered patterns into the hierarchical assembly graph.
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from copy import deepcopy

from .strategies.part_similarity import PartSimilarityClusterer
from .strategies.dependency_analysis import DependencyAnalyzer
from .strategies.spatial_temporal import SpatialTemporalPatternMiner


class PostProcessingConfig:
    """Configuration for post-processing analysis."""

    def __init__(
        self,
        enabled: bool = True,
        # Part similarity clustering
        jaccard_threshold: float = 0.7,
        min_pattern_steps: int = 2,
        # Dependency analysis
        min_group_steps: int = 3,
        min_isolation_steps: int = 2,
        # Spatial-temporal mining
        min_sequence_steps: int = 3,
        similarity_threshold: float = 0.6,
        # Validation
        min_confidence: float = 0.6,
        max_discovered_patterns: int = 20,
        # Optional LLM refinement
        enable_llm_refinement: bool = False
    ):
        self.enabled = enabled
        self.jaccard_threshold = jaccard_threshold
        self.min_pattern_steps = min_pattern_steps
        self.min_group_steps = min_group_steps
        self.min_isolation_steps = min_isolation_steps
        self.min_sequence_steps = min_sequence_steps
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.max_discovered_patterns = max_discovered_patterns
        self.enable_llm_refinement = enable_llm_refinement


class PostProcessingSubassemblyAnalyzer:
    """
    Main analyzer for post-processing subassembly detection.

    Coordinates three detection strategies:
    1. Part Similarity Clustering - Repeated structures
    2. Dependency Analysis - Independent groups
    3. Spatial-Temporal Mining - Progressive builds

    Workflow:
    1. Run all detection strategies
    2. Validate and filter patterns
    3. Integrate into graph
    """

    def __init__(
        self,
        vlm_client=None,
        config: Optional[PostProcessingConfig] = None
    ):
        """
        Initialize the analyzer.

        Args:
            vlm_client: Optional VLM client for refinement
            config: Configuration object
        """
        self.vlm_client = vlm_client
        self.config = config or PostProcessingConfig()

        # Initialize strategies
        self.part_clusterer = PartSimilarityClusterer(
            jaccard_threshold=self.config.jaccard_threshold,
            min_pattern_steps=self.config.min_pattern_steps
        )

        self.dependency_analyzer = DependencyAnalyzer(
            min_group_steps=self.config.min_group_steps,
            min_isolation_steps=self.config.min_isolation_steps
        )

        self.pattern_miner = SpatialTemporalPatternMiner(
            min_sequence_steps=self.config.min_sequence_steps,
            similarity_threshold=self.config.similarity_threshold
        )

        logger.info("PostProcessingSubassemblyAnalyzer initialized")

    def analyze_and_augment_graph(
        self,
        graph: Dict[str, Any],
        extracted_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Main entry point: analyze and augment graph with discovered patterns.

        Args:
            graph: Current graph structure from Stage 2
            extracted_steps: Extracted step data from Stage 1

        Returns:
            Augmented graph with discovered subassemblies
        """
        if not self.config.enabled:
            logger.info("Post-processing analysis disabled")
            return graph

        logger.info("=" * 60)
        logger.info("Starting Post-Processing Subassembly Analysis")
        logger.info("=" * 60)

        # Make a deep copy to avoid modifying original
        augmented_graph = deepcopy(graph)

        # Phase 1: Run all detection strategies
        logger.info("\n[Phase 1/3] Running Detection Strategies")
        all_patterns = self._run_detection_strategies(augmented_graph, extracted_steps)

        if not all_patterns:
            logger.info("No patterns discovered by post-processing")
            return augmented_graph

        # Phase 2: Validate and filter patterns
        logger.info("\n[Phase 2/3] Validating and Filtering Patterns")
        validated_patterns = self._validate_patterns(all_patterns, augmented_graph)

        if not validated_patterns:
            logger.info("No patterns passed validation")
            return augmented_graph

        # Phase 3: Integrate patterns into graph
        logger.info("\n[Phase 3/3] Integrating Patterns into Graph")
        augmented_graph = self._integrate_patterns_into_graph(
            augmented_graph,
            validated_patterns,
            extracted_steps
        )

        logger.info("\n" + "=" * 60)
        logger.info("✓ Post-Processing Analysis Complete")
        logger.info(f"  Discovered: {len(validated_patterns)} new subassemblies")
        logger.info("=" * 60)

        return augmented_graph

    def _run_detection_strategies(
        self,
        graph: Dict[str, Any],
        extracted_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run all detection strategies.

        Args:
            graph: Current graph
            extracted_steps: Step data

        Returns:
            Combined list of all discovered patterns
        """
        all_patterns = []

        # Strategy 1: Part Similarity Clustering
        logger.info("  Running Strategy 1: Part Similarity Clustering")
        repeated_structures = self.part_clusterer.find_repeated_structures(
            graph, extracted_steps
        )
        all_patterns.extend(repeated_structures)
        logger.info(f"    → Found {len(repeated_structures)} repeated structures")

        # Strategy 2: Dependency Analysis
        logger.info("  Running Strategy 2: Dependency Analysis")
        independent_groups = self.dependency_analyzer.find_independent_groups(
            graph, extracted_steps
        )
        all_patterns.extend(independent_groups)
        logger.info(f"    → Found {len(independent_groups)} independent groups")

        # Strategy 3: Spatial-Temporal Mining
        logger.info("  Running Strategy 3: Spatial-Temporal Pattern Mining")
        progressive_builds = self.pattern_miner.find_progressive_builds(
            graph, extracted_steps
        )
        all_patterns.extend(progressive_builds)
        logger.info(f"    → Found {len(progressive_builds)} progressive builds")

        logger.info(f"\n  Total patterns discovered: {len(all_patterns)}")

        return all_patterns

    def _validate_patterns(
        self,
        patterns: List[Dict[str, Any]],
        graph: Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate and filter discovered patterns.

        Validation criteria:
        - Confidence score above threshold
        - No overlap with existing subassemblies
        - Reasonable step ranges
        - Not too many patterns (prevent over-grouping)

        Args:
            patterns: All discovered patterns
            graph: Current graph

        Returns:
            Validated patterns
        """
        validated = []

        # Get existing subassembly step ranges
        existing_ranges = self._get_existing_subassembly_ranges(graph)

        for pattern in patterns:
            # Check confidence
            confidence = pattern.get("confidence", 0.0)
            if confidence < self.config.min_confidence:
                logger.debug(
                    f"Rejected pattern '{pattern['name']}': "
                    f"confidence {confidence:.2f} < {self.config.min_confidence}"
                )
                continue

            # Check for overlap with existing subassemblies
            pattern_steps = set(pattern.get("steps", []))
            if self._overlaps_with_existing(pattern_steps, existing_ranges):
                logger.debug(
                    f"Rejected pattern '{pattern['name']}': "
                    f"overlaps with existing subassembly"
                )
                continue

            # Check for overlap with other validated patterns
            if self._overlaps_with_validated(pattern_steps, validated):
                logger.debug(
                    f"Rejected pattern '{pattern['name']}': "
                    f"overlaps with another discovered pattern"
                )
                continue

            # Pattern is valid
            validated.append(pattern)
            logger.debug(
                f"Validated pattern '{pattern['name']}': "
                f"confidence {confidence:.2f}, steps {pattern['steps']}"
            )

            # Check if we've reached max patterns
            if len(validated) >= self.config.max_discovered_patterns:
                logger.warning(
                    f"Reached max discovered patterns limit "
                    f"({self.config.max_discovered_patterns})"
                )
                break

        logger.info(f"  Validated {len(validated)}/{len(patterns)} patterns")

        return validated

    def _get_existing_subassembly_ranges(
        self,
        graph: Dict[str, Any]
    ) -> List[set]:
        """
        Get step ranges for existing subassemblies.

        Args:
            graph: Current graph

        Returns:
            List of sets of step numbers
        """
        ranges = []

        nodes = graph.get("nodes", [])
        for node in nodes:
            if node.get("type") == "subassembly":
                # Get step range for this subassembly
                step_created = node.get("step_created")
                children = node.get("children", [])

                # Find all steps involved in this subassembly
                child_steps = set()
                for child_id in children:
                    child_node = next(
                        (n for n in nodes if n["node_id"] == child_id),
                        None
                    )
                    if child_node:
                        child_step = child_node.get("step_created")
                        if child_step:
                            child_steps.add(child_step)

                if step_created:
                    child_steps.add(step_created)

                if child_steps:
                    ranges.append(child_steps)

        return ranges

    def _overlaps_with_existing(
        self,
        pattern_steps: set,
        existing_ranges: List[set]
    ) -> bool:
        """
        Check if pattern overlaps with existing subassemblies.

        Args:
            pattern_steps: Set of step numbers in pattern
            existing_ranges: List of existing step ranges

        Returns:
            True if overlap detected
        """
        for existing in existing_ranges:
            overlap = pattern_steps & existing
            # Allow small overlap (1 step), reject significant overlap
            if len(overlap) > 1:
                return True

        return False

    def _overlaps_with_validated(
        self,
        pattern_steps: set,
        validated_patterns: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if pattern overlaps with already validated patterns.

        Args:
            pattern_steps: Set of step numbers in pattern
            validated_patterns: Already validated patterns

        Returns:
            True if overlap detected
        """
        for validated in validated_patterns:
            validated_steps = set(validated.get("steps", []))
            overlap = pattern_steps & validated_steps

            # Reject any overlap with validated patterns
            if len(overlap) > 0:
                return True

        return False

    def _integrate_patterns_into_graph(
        self,
        graph: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        extracted_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Integrate validated patterns into graph structure.

        Process:
        1. Create new subassembly nodes
        2. Find parts created in pattern's step range
        3. Re-parent parts from model to new subassembly
        4. Add edges from subassembly to model
        5. Update metadata

        Args:
            graph: Current graph
            patterns: Validated patterns
            extracted_steps: All extracted steps

        Returns:
            Updated graph
        """
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        metadata = graph.get("metadata", {})

        # Get model node
        model_node = next((n for n in nodes if n["type"] == "model"), None)
        if not model_node:
            logger.error("No model node found in graph")
            return graph

        # Counter for discovered subassembly IDs
        existing_discovered = [
            n for n in nodes
            if n.get("type") == "subassembly" and
               n.get("discovery_method")
        ]
        discovered_counter = len(existing_discovered)

        for pattern in patterns:
            # Create new subassembly node
            subasm_id = f"subasm_discovered_{discovered_counter}"
            discovered_counter += 1

            pattern_steps = pattern.get("steps", [])
            first_step = min(pattern_steps) if pattern_steps else 1

            # Fill in parts list from graph
            parts_in_pattern = self._find_parts_in_step_range(
                nodes, pattern_steps
            )
            pattern["parts"] = parts_in_pattern

            subasm_node = {
                "node_id": subasm_id,
                "type": "subassembly",
                "name": pattern["name"],
                "description": pattern["description"],
                "children": parts_in_pattern,
                "parents": [model_node["node_id"]],
                "step_created": first_step,
                "layer": 1,  # Will be recalculated
                "discovery_method": pattern["discovery_method"],
                "confidence": pattern["confidence"],
                "step_range": pattern_steps,
                "metadata": pattern.get("metadata", {})
            }

            nodes.append(subasm_node)

            # Re-parent parts to this subassembly
            for part_id in parts_in_pattern:
                part_node = next((n for n in nodes if n["node_id"] == part_id), None)
                if part_node:
                    # Remove from model's children
                    if part_id in model_node["children"]:
                        model_node["children"].remove(part_id)

                    # Add to subassembly's children
                    # (already in subasm_node["children"])

                    # Update part's parents
                    if model_node["node_id"] in part_node["parents"]:
                        part_node["parents"].remove(model_node["node_id"])
                    part_node["parents"].append(subasm_id)

                    # Update edges
                    # Remove old edge (part -> model)
                    edges[:] = [
                        e for e in edges
                        if not (e["from"] == part_id and e["to"] == model_node["node_id"])
                    ]

                    # Add new edge (part -> subassembly)
                    edges.append({
                        "from": part_id,
                        "to": subasm_id,
                        "type": "component",
                        "created_step": part_node.get("step_created", first_step)
                    })

            # Add subassembly to model's children
            model_node["children"].append(subasm_id)

            # Add edge (subassembly -> model)
            edges.append({
                "from": subasm_id,
                "to": model_node["node_id"],
                "type": "attachment",
                "created_step": first_step
            })

            logger.info(
                f"  Created {subasm_id}: '{pattern['name']}' "
                f"with {len(parts_in_pattern)} parts"
            )

        # Recalculate layers
        self._recalculate_layers(nodes)

        # Update metadata
        total_subassemblies = len([n for n in nodes if n["type"] == "subassembly"])
        discovered_subassemblies = len([
            n for n in nodes
            if n.get("type") == "subassembly" and n.get("discovery_method")
        ])

        metadata["total_subassemblies"] = total_subassemblies
        metadata["discovered_subassemblies"] = discovered_subassemblies

        # Update max depth
        max_depth = max([n.get("layer", 0) for n in nodes])
        metadata["max_depth"] = max_depth

        logger.info(
            f"  Updated metadata: {total_subassemblies} total subassemblies "
            f"({discovered_subassemblies} discovered)"
        )

        return graph

    def _find_parts_in_step_range(
        self,
        nodes: List[Dict[str, Any]],
        step_range: List[int]
    ) -> List[str]:
        """
        Find all part node IDs created within step range.

        Args:
            nodes: All graph nodes
            step_range: List of step numbers

        Returns:
            List of part node IDs
        """
        step_set = set(step_range)
        parts = []

        for node in nodes:
            if node.get("type") == "part":
                step_created = node.get("step_created")
                if step_created in step_set:
                    parts.append(node["node_id"])

        return parts

    def _recalculate_layers(self, nodes: List[Dict[str, Any]]):
        """
        Recalculate layer numbers after graph modification.

        Uses BFS from root node.

        Args:
            nodes: All graph nodes (modified in-place)
        """
        node_map = {node["node_id"]: node for node in nodes}

        # BFS from root to assign layers
        queue = [(n["node_id"], 0) for n in nodes if n["type"] == "model"]
        visited = set()

        while queue:
            node_id, layer = queue.pop(0)

            if node_id in visited:
                continue
            visited.add(node_id)

            node = node_map[node_id]
            node["layer"] = layer

            # Add children to queue
            for child_id in node.get("children", []):
                if child_id not in visited:
                    queue.append((child_id, layer + 1))
