"""
Dependency Graph Constructor: Builds directed acyclic graph (DAG) showing
step dependencies and assembly hierarchies.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict, deque
from loguru import logger

class DependencyGraph:
    """Represents assembly step dependencies as a directed acyclic graph."""
    
    def __init__(self):
        self.nodes: Dict[int, Dict[str, Any]] = {}  # step_number -> step_info
        self.edges: Dict[int, List[int]] = defaultdict(list)  # step -> [dependent_steps]
        self.reverse_edges: Dict[int, List[int]] = defaultdict(list)  # step -> [prerequisite_steps]
        self.subassemblies: List[Dict[str, Any]] = []
    
    def add_step(self, step_number: int, step_info: Dict[str, Any]):
        """Add a step to the graph."""
        self.nodes[step_number] = step_info
        logger.debug(f"Added step {step_number} to dependency graph")
    
    def add_dependency(self, prerequisite: int, dependent: int):
        """Add a dependency edge (prerequisite must be completed before dependent)."""
        if prerequisite not in self.nodes or dependent not in self.nodes:
            logger.warning(f"Cannot add dependency: step {prerequisite} or {dependent} not in graph")
            return
        
        self.edges[prerequisite].append(dependent)
        self.reverse_edges[dependent].append(prerequisite)
        logger.debug(f"Added dependency: step {prerequisite} -> step {dependent}")
    
    def infer_dependencies(self, extracted_steps: List[Dict[str, Any]]):
        """
        Infer dependencies from extracted step information.
        Filters out errors and renumbers steps sequentially.

        Args:
            extracted_steps: List of extracted step dictionaries (may include errors)
        """
        logger.info("Inferring step dependencies...")

        # Filter out failed extractions (steps with "error" key)
        valid_steps = []
        for step_info in extracted_steps:
            if "error" not in step_info:
                valid_steps.append(step_info)
            else:
                old_num = step_info.get("step_number", "unknown")
                logger.warning(f"Skipping step {old_num} due to extraction error")

        logger.info(f"Valid steps: {len(valid_steps)}/{len(extracted_steps)}")

        if not valid_steps:
            logger.error("No valid steps to process!")
            return

        # ALWAYS renumber steps sequentially (1, 2, 3, ...)
        # This fixes issues where VLM reads incorrect step numbers from images
        step_number_mapping = {}  # old_number -> new_number

        for i, step_info in enumerate(valid_steps):
            old_step_number = step_info.get("step_number")
            new_step_number = i + 1

            # Track the mapping for dependency updates
            if old_step_number:
                step_number_mapping[old_step_number] = new_step_number

            # Update step_info with sequential number
            step_info["step_number"] = new_step_number

            if old_step_number != new_step_number:
                logger.debug(f"Renumbered step {old_step_number} → {new_step_number}")

            self.add_step(new_step_number, step_info)
        
        # Infer dependencies using valid_steps with new numbering
        for i, step_info in enumerate(valid_steps):
            step_number = i + 1  # Already renumbered above

            # Method 1: Explicit dependencies from VLM extraction
            explicit_deps = step_info.get("dependencies", "")
            if explicit_deps:
                old_dep_steps = self._parse_dependencies(explicit_deps)
                # Map old step numbers to new ones
                for old_dep in old_dep_steps:
                    new_dep = step_number_mapping.get(old_dep, old_dep)
                    if new_dep < step_number and new_dep in self.nodes:
                        self.add_dependency(new_dep, step_number)

            # Method 2: Sequential dependency (each step depends on previous)
            if step_number > 1 and (step_number - 1) in self.nodes:
                # Only add if no explicit dependencies were found
                if not self.reverse_edges[step_number]:
                    self.add_dependency(step_number - 1, step_number)

            # Method 3: Part-based dependency inference
            self._infer_part_dependencies(step_number, step_info, valid_steps[:i])

        logger.info(f"Inferred dependencies for {len(self.nodes)} steps")
    
    def _parse_dependencies(self, dep_string: str) -> List[int]:
        """Parse dependency string to extract step numbers."""
        import re
        
        # Extract numbers from strings like "Step 1, 2" or "步骤 1-3"
        numbers = re.findall(r'\d+', str(dep_string))
        return [int(n) for n in numbers]
    
    def _infer_part_dependencies(
        self, 
        current_step: int, 
        current_info: Dict[str, Any],
        previous_steps: List[Dict[str, Any]]
    ):
        """Infer dependencies based on part usage across steps."""
        current_parts = set(
            p.get("description", "") 
            for p in current_info.get("parts_required", [])
        )
        
        # Check which previous steps produced parts used in current step
        for i, prev_step_info in enumerate(previous_steps):
            prev_step_num = prev_step_info.get("step_number", i + 1)
            
            # Check if previous step's output is used in current step
            prev_assembly = prev_step_info.get("existing_assembly", "")
            new_parts = prev_step_info.get("new_parts_to_add", [])
            
            # Simple heuristic: if current step mentions previous assembly, add dependency
            if prev_assembly and prev_assembly in str(current_info):
                if prev_step_num not in self.reverse_edges[current_step]:
                    self.add_dependency(prev_step_num, current_step)
    
    
    def topological_sort(self) -> List[int]:
        """
        Perform topological sort to get valid build order.
        
        Returns:
            Ordered list of step numbers
        """
        in_degree = {step: len(self.reverse_edges[step]) for step in self.nodes}
        queue = deque([step for step in self.nodes if in_degree[step] == 0])
        result = []
        
        while queue:
            step = queue.popleft()
            result.append(step)
            
            for dependent in self.edges[step]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(self.nodes):
            logger.warning("Cycle detected in dependency graph!")
            return list(self.nodes.keys())
        
        return result
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the dependency graph.

        Returns:
            (is_valid, list of validation errors)
        """
        errors = []
        warnings = []

        # Check for cycles
        sorted_steps = self.topological_sort()
        if len(sorted_steps) != len(self.nodes):
            errors.append("Cycle detected in dependency graph")

        # Check for missing steps (steps should be consecutive after renumbering)
        if self.nodes:
            min_step = min(self.nodes.keys())
            max_step = max(self.nodes.keys())
            expected_steps = set(range(min_step, max_step + 1))
            actual_steps = set(self.nodes.keys())
            missing = expected_steps - actual_steps

            if missing:
                # This is only an error if steps were not properly renumbered
                errors.append(f"Missing steps: {sorted(missing)}")
                logger.error(f"Step sequence has gaps: expected {min_step}-{max_step}, got {sorted(actual_steps)}")

        # Check for isolated nodes (warn, not error - parallel subassemblies are valid)
        isolated = [
            step for step in self.nodes
            if not self.edges[step] and not self.reverse_edges[step] and step != 1
        ]
        if isolated:
            warnings.append(f"Isolated steps (parallel subassemblies?): {isolated}")
            logger.warning(f"Found isolated steps: {isolated}. This may indicate parallel subassemblies.")

        is_valid = len(errors) == 0

        if warnings:
            logger.info(f"Validation warnings: {'; '.join(warnings)}")

        return is_valid, errors
    
    def _compute_cumulative_parts(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Compute cumulative parts list for each step.

        For each step, aggregates all parts from step 1 to current step,
        handling quantity accumulation when the same part appears in multiple steps.

        Returns:
            Dict mapping step_number -> cumulative_parts_list
        """
        cumulative = {}
        parts_tracker = {}  # key: (color, shape, description) -> quantity

        # Use build order to process steps in dependency order
        build_order = self.topological_sort()

        for step_num in build_order:
            step_data = self.nodes[step_num]

            # Add parts from this step
            for part in step_data.get("parts_required", []):
                # Create unique key for part identification
                key = (
                    part.get("color", "").lower(),
                    part.get("shape", "").lower(),
                    part.get("description", "").lower()
                )
                # Accumulate quantity
                parts_tracker[key] = parts_tracker.get(key, 0) + part.get("quantity", 1)

            # Convert to list format for this step
            cumulative[step_num] = [
                {
                    "color": color,
                    "shape": shape,
                    "description": description,
                    "quantity": quantity
                }
                for (color, shape, description), quantity in sorted(parts_tracker.items())
            ]

            logger.debug(
                f"Step {step_num}: {len(cumulative[step_num])} unique parts, "
                f"{sum(p['quantity'] for p in cumulative[step_num])} total pieces"
            )

        return cumulative

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        # Compute cumulative parts for all steps
        cumulative_parts = self._compute_cumulative_parts()

        # Create a copy of nodes with cumulative_parts added
        nodes_with_cumulative = {}
        for step_num, node_data in self.nodes.items():
            nodes_with_cumulative[step_num] = {
                **node_data,
                "cumulative_parts": cumulative_parts.get(step_num, [])
            }

        return {
            "nodes": nodes_with_cumulative,
            "edges": dict(self.edges),
            "reverse_edges": dict(self.reverse_edges),
            "subassemblies": self.subassemblies,
            "build_order": self.topological_sort()
        }

