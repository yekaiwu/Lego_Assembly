#!/usr/bin/env python3
"""
Graph Visualization Tool
Generates visual representations of hierarchical assembly graphs.

Location: backend/app/scripts/visualize_graph.py
Usage: python3 backend/app/scripts/visualize_graph.py output/6454922_graph.json [options]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_graph(graph_path: Path) -> Optional[Dict[str, Any]]:
    """Load graph from JSON file."""
    try:
        with open(graph_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None


def print_tree(
    node: Dict[str, Any],
    node_map: Dict[str, Dict[str, Any]],
    prefix: str = "",
    is_last: bool = True,
    max_parts: int = 3
):
    """
    Print a node and its children as an ASCII tree.
    
    Args:
        node: Current node to print
        node_map: Map of all nodes by ID
        prefix: Current line prefix for indentation
        is_last: Whether this is the last child
        max_parts: Maximum number of parts to show before collapsing
    """
    # Choose connector based on position
    connector = "└── " if is_last else "├── "
    
    # Choose icon based on node type
    if node["type"] == "model":
        icon = "📦"
    elif node["type"] == "subassembly":
        icon = "🔧"
    else:  # part
        icon = "🧱"
    
    # Format node name
    node_name = node.get("name", "Unknown")
    node_type = node["type"]
    step_created = node.get("step_created", "?")
    
    if node_type != "model":
        node_label = f"{icon} {node_name} [{node_type}, step {step_created}]"
    else:
        node_label = f"{icon} {node_name} [root]"
    
    # Print current node
    print(f"{prefix}{connector}{node_label}")
    
    # Update prefix for children
    if is_last:
        new_prefix = prefix + "    "
    else:
        new_prefix = prefix + "│   "
    
    # Get children
    children = node.get("children", [])
    child_nodes = [node_map.get(cid) for cid in children if cid in node_map]
    
    # Separate subassemblies and parts
    subassemblies = [c for c in child_nodes if c["type"] == "subassembly"]
    parts = [c for c in child_nodes if c["type"] == "part"]
    
    # Print subassemblies (all of them)
    for i, child in enumerate(subassemblies):
        child_is_last = (i == len(subassemblies) - 1) and len(parts) == 0
        print_tree(child, node_map, new_prefix, child_is_last, max_parts)
    
    # Print parts (limited)
    parts_to_show = min(max_parts, len(parts))
    for i, child in enumerate(parts[:parts_to_show]):
        child_is_last = (i == parts_to_show - 1) and (len(parts) <= max_parts)
        print_tree(child, node_map, new_prefix, child_is_last, max_parts)
    
    # Show count if parts were collapsed
    if len(parts) > max_parts:
        connector = "└── " if True else "├── "
        print(f"{new_prefix}{connector}... ({len(parts) - max_parts} more parts)")


def print_graph_summary(graph: Dict[str, Any]):
    """Print a summary of the graph."""
    metadata = graph.get("metadata", {})
    
    print("=" * 80)
    print("HIERARCHICAL ASSEMBLY GRAPH VISUALIZATION")
    print("=" * 80)
    print(f"\nManual ID: {graph.get('manual_id', 'Unknown')}")
    print(f"Generated: {metadata.get('generated_at', 'Unknown')}")
    print(f"\n📊 Statistics:")
    print(f"   Parts: {metadata.get('total_parts', 0)}")
    print(f"   Subassemblies: {metadata.get('total_subassemblies', 0)}")
    print(f"   Steps: {metadata.get('total_steps', 0)}")
    print(f"   Max Depth: {metadata.get('max_depth', 0)} layers")
    
    # Part catalog info
    part_catalog = graph.get("part_catalog", {})
    if "parts_by_role" in part_catalog:
        print(f"\n📦 Parts by Role:")
        for role, count in part_catalog["parts_by_role"].items():
            print(f"   {role}: {count}")
    
    print("\n" + "=" * 80)
    print("🌳 HIERARCHY TREE")
    print("=" * 80)
    print()


def list_subassemblies(graph: Dict[str, Any]):
    """List all subassemblies with details."""
    nodes = graph.get("nodes", [])
    subassemblies = [n for n in nodes if n.get("type") == "subassembly"]
    
    if not subassemblies:
        print("\n❌ No subassemblies found in this graph.")
        return
    
    print("\n" + "=" * 80)
    print("🔧 SUBASSEMBLIES")
    print("=" * 80)
    
    for i, subasm in enumerate(subassemblies, 1):
        print(f"\n{i}. {subasm['name']} (ID: {subasm['node_id']})")
        print(f"   Step: {subasm.get('step_created', '?')}")
        print(f"   Layer: {subasm.get('layer', '?')}")
        print(f"   Children: {len(subasm.get('children', []))}")
        print(f"   Description: {subasm.get('description', 'N/A')}")
        
        # Completeness markers
        markers = subasm.get("completeness_markers", {})
        if markers:
            print(f"   Completeness:")
            print(f"     Required parts: {markers.get('required_parts', 'N/A')}")
            print(f"     Connections: {len(markers.get('required_connections', []))}")
            print(f"     Spatial: {markers.get('spatial_signature', 'N/A')}")


def show_step_progression(graph: Dict[str, Any], steps_to_show: int = 10):
    """Show step-by-step progression."""
    step_states = graph.get("step_states", [])
    
    if not step_states:
        print("\n❌ No step states found in this graph.")
        return
    
    print("\n" + "=" * 80)
    print("📈 STEP PROGRESSION")
    print("=" * 80)
    
    for state in step_states[:steps_to_show]:
        step_num = state.get("step_number", "?")
        new_nodes = len(state.get("new_nodes", []))
        total_nodes = len(state.get("existing_nodes", []))
        connections = len(state.get("new_connections", []))
        active_subasm = len(state.get("active_subassemblies", []))
        progress = state.get("completion_percentage", 0)
        
        print(f"\nStep {step_num}: {progress:.1f}% complete")
        print(f"   New: {new_nodes} nodes, {connections} connections")
        print(f"   Total: {total_nodes} nodes | Active subassemblies: {active_subasm}")
    
    if len(step_states) > steps_to_show:
        print(f"\n... ({len(step_states) - steps_to_show} more steps)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hierarchical LEGO assembly graphs"
    )
    parser.add_argument(
        "graph_path",
        type=str,
        help="Path to graph JSON file (e.g., output/6454922_graph.json)"
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=3,
        help="Maximum number of parts to show per subassembly (default: 3)"
    )
    parser.add_argument(
        "--show-subassemblies",
        action="store_true",
        help="Show detailed subassembly list"
    )
    parser.add_argument(
        "--show-steps",
        type=int,
        default=0,
        help="Show step progression (number of steps to display)"
    )
    
    args = parser.parse_args()
    
    # Load graph
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        print(f"❌ Error: Graph file not found: {graph_path}")
        return 1
    
    print(f"Loading graph from: {graph_path}")
    graph = load_graph(graph_path)
    
    if not graph:
        return 1
    
    # Print summary
    print_graph_summary(graph)
    
    # Build node map
    nodes = graph.get("nodes", [])
    node_map = {n["node_id"]: n for n in nodes}
    
    # Find root node
    root = next((n for n in nodes if n["type"] == "model"), None)
    
    if not root:
        print("❌ Error: No root model node found in graph")
        return 1
    
    # Print tree
    print_tree(root, node_map, "", True, args.max_parts)
    
    # Optional: Show subassemblies
    if args.show_subassemblies:
        list_subassemblies(graph)
    
    # Optional: Show step progression
    if args.show_steps > 0:
        show_step_progression(graph, args.show_steps)
    
    print("\n" + "=" * 80)
    print("✓ Visualization complete")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

