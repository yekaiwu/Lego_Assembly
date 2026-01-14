"""
Graph Manager - Query interface for hierarchical assembly graphs.
Provides operations to query nodes, relationships, and assembly states.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from ..config import get_settings


class GraphManager:
    """Manages hierarchical assembly graph queries and operations."""
    
    def __init__(self):
        """Initialize graph manager."""
        self.settings = get_settings()
        self.graphs = {}  # Cache loaded graphs in memory
        logger.info("GraphManager initialized")
    
    def load_graph(self, manual_id: str) -> Optional[Dict[str, Any]]:
        """
        Load graph from storage. Caches in memory for performance.
        
        Args:
            manual_id: Manual identifier
        
        Returns:
            Graph structure or None if not found
        """
        # Check cache first
        if manual_id in self.graphs:
            logger.debug(f"Graph for {manual_id} loaded from cache")
            return self.graphs[manual_id]
        
        # Load from file
        graph_path = Path(self.settings.output_dir) / f"{manual_id}_graph.json"
        
        if not graph_path.exists():
            logger.warning(f"Graph file not found: {graph_path}")
            return None
        
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
            
            # Cache for future queries
            self.graphs[manual_id] = graph
            logger.info(f"Graph loaded for manual {manual_id}")
            return graph
            
        except Exception as e:
            logger.error(f"Error loading graph for {manual_id}: {e}")
            return None
    
    def get_node(
        self,
        manual_id: str,
        node_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get node by ID.
        
        Args:
            manual_id: Manual identifier
            node_id: Node identifier
        
        Returns:
            Node data or None
        """
        graph = self.load_graph(manual_id)
        if not graph:
            return None
        
        for node in graph.get("nodes", []):
            if node["node_id"] == node_id:
                return node
        
        return None
    
    def get_node_by_name(
        self,
        manual_id: str,
        name: str,
        fuzzy: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by name with optional fuzzy matching.
        
        Args:
            manual_id: Manual identifier
            name: Node name to search for
            fuzzy: Whether to use fuzzy matching
        
        Returns:
            List of matching nodes
        """
        graph = self.load_graph(manual_id)
        if not graph:
            return []
        
        name_lower = name.lower()
        matches = []
        
        for node in graph.get("nodes", []):
            node_name = node.get("name", "").lower()
            
            if fuzzy:
                # Fuzzy match: check if query is substring of node name
                if name_lower in node_name:
                    matches.append(node)
            else:
                # Exact match
                if name_lower == node_name:
                    matches.append(node)
        
        logger.debug(f"Found {len(matches)} nodes matching '{name}'")
        return matches
    
    def get_children(
        self,
        manual_id: str,
        node_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all children of a node.
        
        Args:
            manual_id: Manual identifier
            node_id: Parent node identifier
        
        Returns:
            List of child nodes
        """
        node = self.get_node(manual_id, node_id)
        if not node:
            return []
        
        child_ids = node.get("children", [])
        children = []
        
        for child_id in child_ids:
            child = self.get_node(manual_id, child_id)
            if child:
                children.append(child)
        
        return children
    
    def get_parents(
        self,
        manual_id: str,
        node_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all parents of a node.
        
        Args:
            manual_id: Manual identifier
            node_id: Child node identifier
        
        Returns:
            List of parent nodes
        """
        node = self.get_node(manual_id, node_id)
        if not node:
            return []
        
        parent_ids = node.get("parents", [])
        parents = []
        
        for parent_id in parent_ids:
            parent = self.get_node(manual_id, parent_id)
            if parent:
                parents.append(parent)
        
        return parents
    
    def get_subassembly_parts(
        self,
        manual_id: str,
        subassembly_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all parts that make up a subassembly.
        
        Args:
            manual_id: Manual identifier
            subassembly_id: Subassembly node identifier
        
        Returns:
            List of part nodes
        """
        children = self.get_children(manual_id, subassembly_id)
        parts = [child for child in children if child.get("type") == "part"]
        return parts
    
    def find_subassembly_containing_part(
        self,
        manual_id: str,
        part_id: str
    ) -> List[Dict[str, Any]]:
        """
        Find subassemblies that contain a specific part.
        
        Args:
            manual_id: Manual identifier
            part_id: Part node identifier
        
        Returns:
            List of subassembly nodes containing the part
        """
        parents = self.get_parents(manual_id, part_id)
        subassemblies = [
            parent for parent in parents 
            if parent.get("type") == "subassembly"
        ]
        return subassemblies
    
    def get_steps_for_node(
        self,
        manual_id: str,
        node_id: str
    ) -> List[int]:
        """
        Get all steps where a node is involved.
        
        Args:
            manual_id: Manual identifier
            node_id: Node identifier
        
        Returns:
            List of step numbers
        """
        node = self.get_node(manual_id, node_id)
        if not node:
            return []
        
        return node.get("steps_used", [])
    
    def get_step_state(
        self,
        manual_id: str,
        step_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete state at a specific step.
        
        Args:
            manual_id: Manual identifier
            step_number: Step number
        
        Returns:
            Step state snapshot or None
        """
        graph = self.load_graph(manual_id)
        if not graph:
            return None
        
        step_states = graph.get("step_states", [])
        
        for state in step_states:
            if state.get("step_number") == step_number:
                return state
        
        return None
    
    def find_equivalent_parts(
        self,
        manual_id: str,
        node_id: str
    ) -> List[Dict[str, Any]]:
        """
        Find all equivalent (identical) parts.
        
        Args:
            manual_id: Manual identifier
            node_id: Part node identifier
        
        Returns:
            List of equivalent part nodes
        """
        node = self.get_node(manual_id, node_id)
        if not node or node.get("type") != "part":
            return []
        
        # Get part signature
        color = node.get("color", "").lower()
        shape = node.get("shape", "").lower()
        
        if not color or not shape:
            return []
        
        # Find parts with same color and shape
        graph = self.load_graph(manual_id)
        if not graph:
            return []
        
        equivalents = []
        for n in graph.get("nodes", []):
            if n.get("type") == "part" and n["node_id"] != node_id:
                if (n.get("color", "").lower() == color and
                    n.get("shape", "").lower() == shape):
                    equivalents.append(n)
        
        return equivalents
    
    def get_assembly_path(
        self,
        manual_id: str,
        node_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get path from root to node (shows hierarchy).
        
        Args:
            manual_id: Manual identifier
            node_id: Target node identifier
        
        Returns:
            List of nodes from root to target
        """
        path = []
        current_id = node_id
        visited = set()
        
        while current_id:
            if current_id in visited:
                logger.warning(f"Cycle detected in graph for {manual_id}")
                break
            visited.add(current_id)
            
            node = self.get_node(manual_id, current_id)
            if not node:
                break
            
            path.insert(0, node)
            
            # Move to first parent
            parents = node.get("parents", [])
            if not parents:
                break
            
            current_id = parents[0]
        
        return path
    
    def estimate_step_from_subassemblies(
        self,
        manual_id: str,
        subassembly_names: List[str]
    ) -> Tuple[Optional[int], float]:
        """
        Estimate current step from detected subassemblies.
        
        Args:
            manual_id: Manual identifier
            subassembly_names: List of detected subassembly names
        
        Returns:
            Tuple of (estimated_step, confidence)
        """
        if not subassembly_names:
            return None, 0.0
        
        graph = self.load_graph(manual_id)
        if not graph:
            return None, 0.0
        
        # Find subassemblies by name
        matched_steps = []
        
        for name in subassembly_names:
            nodes = self.get_node_by_name(manual_id, name, fuzzy=True)
            for node in nodes:
                if node.get("type") == "subassembly":
                    steps = node.get("steps_used", [])
                    matched_steps.extend(steps)
        
        if not matched_steps:
            return None, 0.0
        
        # Use the maximum step (most recent)
        estimated_step = max(matched_steps)
        
        # Calculate confidence based on number of matches
        confidence = min(len(matched_steps) / len(subassembly_names), 1.0)
        
        logger.info(f"Estimated step {estimated_step} from subassemblies (confidence: {confidence:.2f})")
        return estimated_step, confidence
    
    def get_nodes_by_type(
        self,
        manual_id: str,
        node_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get all nodes of a specific type.
        
        Args:
            manual_id: Manual identifier
            node_type: Node type (part, subassembly, model)
        
        Returns:
            List of nodes of the specified type
        """
        graph = self.load_graph(manual_id)
        if not graph:
            return []
        
        return [
            node for node in graph.get("nodes", [])
            if node.get("type") == node_type
        ]
    
    def get_discovered_subassemblies(
        self,
        manual_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all subassemblies discovered by post-processing analysis.

        Args:
            manual_id: Manual identifier

        Returns:
            List of discovered subassembly nodes
        """
        graph = self.load_graph(manual_id)
        if not graph:
            return []

        discovered = [
            node for node in graph.get("nodes", [])
            if node.get("type") == "subassembly" and
               node.get("discovery_method") is not None
        ]

        logger.debug(f"Found {len(discovered)} discovered subassemblies for {manual_id}")
        return discovered

    def get_original_subassemblies(
        self,
        manual_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all subassemblies from original VLM-based detection.

        Args:
            manual_id: Manual identifier

        Returns:
            List of original subassembly nodes
        """
        graph = self.load_graph(manual_id)
        if not graph:
            return []

        original = [
            node for node in graph.get("nodes", [])
            if node.get("type") == "subassembly" and
               node.get("discovery_method") is None
        ]

        logger.debug(f"Found {len(original)} original subassemblies for {manual_id}")
        return original

    def get_subassemblies_by_method(
        self,
        manual_id: str,
        discovery_method: str
    ) -> List[Dict[str, Any]]:
        """
        Get subassemblies discovered by a specific method.

        Args:
            manual_id: Manual identifier
            discovery_method: Method name (e.g., 'part_similarity',
                            'dependency_analysis', 'spatial_temporal')

        Returns:
            List of subassembly nodes discovered by the method
        """
        graph = self.load_graph(manual_id)
        if not graph:
            return []

        filtered = [
            node for node in graph.get("nodes", [])
            if node.get("type") == "subassembly" and
               node.get("discovery_method") == discovery_method
        ]

        logger.debug(
            f"Found {len(filtered)} subassemblies for {manual_id} "
            f"using method '{discovery_method}'"
        )
        return filtered

    def get_graph_summary(
        self,
        manual_id: str
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a graph.

        Args:
            manual_id: Manual identifier

        Returns:
            Summary dictionary
        """
        graph = self.load_graph(manual_id)
        if not graph:
            return {}

        metadata = graph.get("metadata", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Count discovered vs original subassemblies
        subassemblies = [n for n in nodes if n.get("type") == "subassembly"]
        discovered = [s for s in subassemblies if s.get("discovery_method")]
        original = [s for s in subassemblies if not s.get("discovery_method")]

        # Count by discovery method
        discovery_methods = {}
        for subasm in discovered:
            method = subasm.get("discovery_method")
            if method:
                discovery_methods[method] = discovery_methods.get(method, 0) + 1

        return {
            "manual_id": manual_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_parts": metadata.get("total_parts", 0),
            "total_subassemblies": metadata.get("total_subassemblies", 0),
            "original_subassemblies": len(original),
            "discovered_subassemblies": metadata.get("discovered_subassemblies", 0),
            "discovery_methods": discovery_methods,
            "total_steps": metadata.get("total_steps", 0),
            "max_depth": metadata.get("max_depth", 0),
            "node_types": {
                "parts": len([n for n in nodes if n.get("type") == "part"]),
                "subassemblies": len(subassemblies),
                "model": len([n for n in nodes if n.get("type") == "model"])
            }
        }


# Singleton instance
_graph_manager_instance = None


def get_graph_manager() -> GraphManager:
    """Get GraphManager singleton instance."""
    global _graph_manager_instance
    if _graph_manager_instance is None:
        _graph_manager_instance = GraphManager()
    return _graph_manager_instance



