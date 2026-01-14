"""
Dependency Analysis Strategy.

Identifies independent/parallel assembly groups by analyzing step dependencies
and finding weakly connected components in the assembly graph.
"""

from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
from loguru import logger


class UnionFind:
    """Union-Find data structure for detecting connected components."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression

        return self.parent[x]

    def union(self, x: int, y: int):
        """Union by rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_components(self) -> Dict[int, List[int]]:
        """Get all connected components."""
        components = defaultdict(list)
        for node in self.parent:
            root = self.find(node)
            components[root].append(node)
        return dict(components)


class DependencyAnalyzer:
    """
    Detects independent assembly groups by analyzing step dependencies.

    Algorithm:
    1. Build dependency graph from step relationships
    2. Use Union-Find to identify weakly connected components
    3. Find components with ≥min_group_steps steps that don't interact until merge
    4. Return independent assembly groups
    """

    def __init__(
        self,
        min_group_steps: int = 3,
        min_isolation_steps: int = 2
    ):
        """
        Initialize the analyzer.

        Args:
            min_group_steps: Minimum steps for a valid group
            min_isolation_steps: Minimum steps a group must remain independent
        """
        self.min_group_steps = min_group_steps
        self.min_isolation_steps = min_isolation_steps

    def find_independent_groups(
        self,
        graph: Dict[str, Any],
        extracted_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find independent assembly groups that can be built in parallel.

        Args:
            graph: Current graph structure
            extracted_steps: Extracted step data

        Returns:
            List of discovered independent assembly patterns
        """
        logger.debug("Running dependency analysis for independent groups")

        patterns = []

        # Build step dependency graph
        step_dependencies = self._build_step_dependencies(extracted_steps)

        if not step_dependencies:
            logger.debug("No step dependencies to analyze")
            return patterns

        # Find connected components using Union-Find
        components = self._find_connected_components(step_dependencies)

        # Filter and validate components
        for component_id, steps in components.items():
            if len(steps) < self.min_group_steps:
                continue

            # Check if this component was truly independent for some time
            isolation_period = self._calculate_isolation_period(
                steps, step_dependencies, extracted_steps
            )

            if isolation_period >= self.min_isolation_steps:
                pattern = self._create_pattern(
                    steps=steps,
                    extracted_steps=extracted_steps,
                    isolation_period=isolation_period
                )
                patterns.append(pattern)
                logger.debug(
                    f"Found independent group: {len(steps)} steps "
                    f"with {isolation_period} isolation steps"
                )

        logger.info(f"Dependency analysis found {len(patterns)} independent groups")
        return patterns

    def _build_step_dependencies(
        self,
        extracted_steps: List[Dict[str, Any]]
    ) -> Dict[int, Set[int]]:
        """
        Build dependency graph between steps.

        A step depends on previous steps if it references existing assembly.

        Args:
            extracted_steps: Step data

        Returns:
            Dictionary mapping step -> set of steps it depends on
        """
        dependencies = defaultdict(set)

        for i, step in enumerate(extracted_steps):
            step_num = step.get("step_number")
            if not step_num:
                continue

            # Check if step references existing assembly
            existing_assembly = step.get("existing_assembly", "")

            # If step adds to existing assembly, it depends on earlier steps
            if existing_assembly and "previous" in existing_assembly.lower():
                # Simple heuristic: depends on immediately previous step
                if step_num > 1:
                    dependencies[step_num].add(step_num - 1)

            # Check for explicit dependencies in actions
            actions = step.get("actions", [])
            for action in actions:
                destination = (action.get("destination") or "").lower()

                # If attaching to existing structure
                if any(kw in destination for kw in ["existing", "previous", "base", "main"]):
                    if step_num > 1:
                        dependencies[step_num].add(step_num - 1)
                    break

            # If step has no dependencies, it starts independently
            if step_num not in dependencies:
                dependencies[step_num] = set()

        return dict(dependencies)

    def _find_connected_components(
        self,
        step_dependencies: Dict[int, Set[int]]
    ) -> Dict[int, List[int]]:
        """
        Find connected components using Union-Find.

        Args:
            step_dependencies: Step dependency graph

        Returns:
            Dictionary mapping component_id -> list of steps
        """
        uf = UnionFind()

        # Add all steps to union-find
        all_steps = set(step_dependencies.keys())
        for step, deps in step_dependencies.items():
            all_steps.update(deps)

        # Initialize all steps
        for step in all_steps:
            uf.find(step)

        # Union steps with their dependencies
        for step, deps in step_dependencies.items():
            for dep in deps:
                uf.union(step, dep)

        # Get components
        components = uf.get_components()

        return components

    def _calculate_isolation_period(
        self,
        component_steps: List[int],
        step_dependencies: Dict[int, Set[int]],
        extracted_steps: List[Dict[str, Any]]
    ) -> int:
        """
        Calculate how long this component remained independent.

        Args:
            component_steps: Steps in this component
            step_dependencies: All step dependencies
            extracted_steps: All steps

        Returns:
            Number of steps the component was isolated
        """
        component_set = set(component_steps)
        sorted_steps = sorted(component_steps)

        if not sorted_steps:
            return 0

        # Find when component first interacts with steps outside itself
        first_step = sorted_steps[0]
        last_independent_step = sorted_steps[-1]

        for step in sorted_steps:
            # Check if this step depends on steps outside component
            deps = step_dependencies.get(step, set())
            external_deps = deps - component_set

            if external_deps:
                # Found where component merges with external assembly
                step_idx = sorted_steps.index(step)
                last_independent_step = sorted_steps[step_idx - 1] if step_idx > 0 else first_step
                break

        # Calculate isolation period
        isolation_period = last_independent_step - first_step + 1

        return isolation_period

    def _create_pattern(
        self,
        steps: List[int],
        extracted_steps: List[Dict[str, Any]],
        isolation_period: int
    ) -> Dict[str, Any]:
        """
        Create pattern metadata from independent group.

        Args:
            steps: List of step numbers in the group
            extracted_steps: All extracted steps
            isolation_period: How long group was isolated

        Returns:
            Pattern dictionary
        """
        sorted_steps = sorted(steps)

        # Generate name based on isolation period
        if isolation_period >= 5:
            name = f"Independent Subassembly ({len(steps)} steps)"
        else:
            name = f"Parallel Assembly Group ({len(steps)} steps)"

        # Try to infer purpose from step descriptions
        step_descriptions = []
        for step_num in sorted_steps[:3]:  # Check first 3 steps
            step_idx = step_num - 1
            if 0 <= step_idx < len(extracted_steps):
                step = extracted_steps[step_idx]
                notes = step.get("notes", "")
                if notes:
                    step_descriptions.append(notes)

        # Look for functional keywords
        combined_text = " ".join(step_descriptions).lower()
        keywords = {
            "wheel": "Wheel Assembly",
            "wing": "Wing Assembly",
            "leg": "Leg Assembly",
            "arm": "Arm Assembly",
            "door": "Door Assembly",
            "window": "Window Assembly",
            "base": "Base Assembly",
            "frame": "Frame Assembly"
        }

        for keyword, assembly_name in keywords.items():
            if keyword in combined_text:
                name = f"{assembly_name} (independent)"
                break

        description = (
            f"Independent assembly group spanning {len(steps)} steps, "
            f"isolated for {isolation_period} steps before merging"
        )

        return {
            "type": "independent_group",
            "name": name,
            "description": description,
            "steps": sorted_steps,
            "parts": [],  # Will be filled by analyzer
            "confidence": self._calculate_confidence(steps, isolation_period),
            "discovery_method": "dependency_analysis",
            "metadata": {
                "isolation_period": isolation_period,
                "step_count": len(steps),
                "first_step": sorted_steps[0],
                "last_step": sorted_steps[-1]
            }
        }

    def _calculate_confidence(
        self,
        steps: List[int],
        isolation_period: int
    ) -> float:
        """
        Calculate confidence score for the pattern.

        Factors:
        - Longer isolation = higher confidence
        - More steps = higher confidence

        Args:
            steps: Step numbers in pattern
            isolation_period: How long group was isolated

        Returns:
            Confidence score (0.0-1.0)
        """
        base_confidence = 0.7

        # Bonus for longer isolation (max +0.2)
        isolation_bonus = min(0.2, isolation_period * 0.04)

        # Bonus for more steps (max +0.1)
        step_bonus = min(0.1, (len(steps) - self.min_group_steps) * 0.02)

        confidence = base_confidence + isolation_bonus + step_bonus

        # Cap at 1.0
        return min(1.0, confidence)
