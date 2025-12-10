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
        
        Args:
            extracted_steps: List of extracted step dictionaries
        """
        logger.info("Inferring step dependencies...")
        
        # Add all steps to graph
        for i, step_info in enumerate(extracted_steps):
            step_number = step_info.get("step_number", i + 1)
            self.add_step(step_number, step_info)
        
        # Infer dependencies
        for i, step_info in enumerate(extracted_steps):
            step_number = step_info.get("step_number", i + 1)
            
            # Method 1: Explicit dependencies from VLM extraction
            explicit_deps = step_info.get("dependencies", "")
            if explicit_deps:
                dep_steps = self._parse_dependencies(explicit_deps)
                for dep in dep_steps:
                    if dep < step_number and dep in self.nodes:
                        self.add_dependency(dep, step_number)
            
            # Method 2: Sequential dependency (each step depends on previous)
            if step_number > 1 and (step_number - 1) in self.nodes:
                # Only add if no explicit dependencies were found
                if not self.reverse_edges[step_number]:
                    self.add_dependency(step_number - 1, step_number)
            
            # Method 3: Part-based dependency inference
            self._infer_part_dependencies(step_number, step_info, extracted_steps[:i])
        
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
    
    def detect_parallel_paths(self) -> List[List[int]]:
        """
        Detect independent subassembly paths that can be built in parallel.
        
        Returns:
            List of parallel step sequences
        """
        logger.info("Detecting parallel assembly paths...")
        
        # Find steps with no dependencies (starting points)
        start_steps = [
            step for step in self.nodes 
            if not self.reverse_edges[step]
        ]
        
        # Trace paths from each starting point
        parallel_paths = []
        visited = set()
        
        for start in start_steps:
            if start in visited:
                continue
            
            path = self._trace_path(start, visited)
            if path:
                parallel_paths.append(path)
        
        logger.info(f"Detected {len(parallel_paths)} parallel paths")
        return parallel_paths
    
    def _trace_path(self, start: int, visited: Set[int]) -> List[int]:
        """Trace a path from a starting step using DFS."""
        path = []
        stack = [start]
        
        while stack:
            step = stack.pop()
            if step in visited:
                continue
            
            visited.add(step)
            path.append(step)
            
            # Add dependent steps to stack
            for dependent in self.edges[step]:
                if dependent not in visited:
                    stack.append(dependent)
        
        return path
    
    def group_into_subassemblies(self, max_steps_per_subassembly: int = 5) -> List[Dict[str, Any]]:
        """
        Group steps into logical subassemblies.
        
        Args:
            max_steps_per_subassembly: Maximum steps per subassembly
        
        Returns:
            List of subassembly definitions
        """
        logger.info("Grouping steps into subassemblies...")
        
        parallel_paths = self.detect_parallel_paths()
        subassemblies = []
        
        for path_idx, path in enumerate(parallel_paths):
            # Split path into chunks
            for i in range(0, len(path), max_steps_per_subassembly):
                chunk = path[i:i + max_steps_per_subassembly]
                
                subassembly = {
                    "id": f"sub_{path_idx}_{i // max_steps_per_subassembly}",
                    "steps": chunk,
                    "prerequisites": self._get_prerequisites(chunk),
                    "parts": self._collect_parts(chunk)
                }
                subassemblies.append(subassembly)
        
        self.subassemblies = subassemblies
        logger.info(f"Created {len(subassemblies)} subassemblies")
        return subassemblies
    
    def _get_prerequisites(self, steps: List[int]) -> List[int]:
        """Get all prerequisite steps for a group of steps."""
        prerequisites = set()
        for step in steps:
            prerequisites.update(self.reverse_edges[step])
        
        # Remove steps that are within the group itself
        prerequisites -= set(steps)
        return list(prerequisites)
    
    def _collect_parts(self, steps: List[int]) -> List[Dict[str, Any]]:
        """Collect all parts used in a group of steps."""
        parts = []
        for step in steps:
            step_info = self.nodes.get(step, {})
            step_parts = step_info.get("parts_required", [])
            for part in step_parts:
                part_copy = part.copy()
                part_copy["step"] = step
                parts.append(part_copy)
        return parts
    
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
        
        # Check for cycles
        sorted_steps = self.topological_sort()
        if len(sorted_steps) != len(self.nodes):
            errors.append("Cycle detected in dependency graph")
        
        # Check for missing steps
        expected_steps = set(range(1, len(self.nodes) + 1))
        actual_steps = set(self.nodes.keys())
        missing = expected_steps - actual_steps
        if missing:
            errors.append(f"Missing steps: {sorted(missing)}")
        
        # Check for isolated nodes
        isolated = [
            step for step in self.nodes 
            if not self.edges[step] and not self.reverse_edges[step] and step != 1
        ]
        if isolated:
            errors.append(f"Isolated steps: {isolated}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": self.nodes,
            "edges": dict(self.edges),
            "reverse_edges": dict(self.reverse_edges),
            "subassemblies": self.subassemblies,
            "build_order": self.topological_sort()
        }

