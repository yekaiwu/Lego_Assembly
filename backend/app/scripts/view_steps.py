#!/usr/bin/env python3
"""
Quick script to view extracted steps and graph information.
Usage: python view_steps.py <manual_id> [--summary|--steps|--graph]
"""

import json
import sys
from pathlib import Path

def view_summary(manual_id: str):
    """Show summary of processed manual."""
    # Path relative to project root (two levels up from scripts/)
    project_root = Path(__file__).parent.parent.parent.parent
    extracted_path = project_root / "output" / f"{manual_id}_extracted.json"
    graph_path = project_root / "output" / f"{manual_id}_graph.json"
    
    if extracted_path.exists():
        with open(extracted_path, 'r', encoding='utf-8') as f:
            steps = json.load(f)
        
        valid_steps = [s for s in steps if s.get("step_number") and s.get("step_number") > 0]
        print(f"\n{'='*60}")
        print(f"Manual ID: {manual_id}")
        print(f"{'='*60}")
        print(f"Total steps extracted: {len(valid_steps)}")
        
        if graph_path.exists():
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
            
            print(f"\nGraph Statistics:")
            print(f"  Nodes: {len(graph['nodes'])}")
            print(f"  Edges: {len(graph['edges'])}")
            print(f"  Parts: {graph['metadata']['total_parts']}")
            print(f"  Subassemblies: {graph['metadata']['total_subassemblies']}")
    else:
        print(f"Error: {extracted_path} not found")

def view_steps(manual_id: str, limit: int = 10):
    """Show detailed step information."""
    # Path relative to project root
    project_root = Path(__file__).parent.parent.parent.parent
    extracted_path = project_root / "output" / f"{manual_id}_extracted.json"
    
    if not extracted_path.exists():
        print(f"Error: {extracted_path} not found")
        return
    
    with open(extracted_path, 'r', encoding='utf-8') as f:
        steps = json.load(f)
    
    valid_steps = [s for s in steps if s.get("step_number") and s.get("step_number") > 0]
    
    print(f"\n{'='*60}")
    print(f"Showing first {min(limit, len(valid_steps))} of {len(valid_steps)} steps")
    print(f"{'='*60}\n")
    
    for step in valid_steps[:limit]:
        step_num = step.get("step_number", "?")
        parts = step.get("parts_required", [])
        actions = step.get("actions", [])
        
        print(f"Step {step_num}:")
        print(f"  Parts ({len(parts)}):")
        for part in parts[:5]:  # Show first 5 parts
            desc = part.get("description", "N/A")
            color = part.get("color", "")
            shape = part.get("shape", "")
            qty = part.get("quantity", 1)
            print(f"    - {qty}x {color} {shape} ({desc[:50]})")
        if len(parts) > 5:
            print(f"    ... and {len(parts) - 5} more parts")
        
        if actions:
            print(f"  Actions ({len(actions)}):")
            for action in actions[:3]:  # Show first 3 actions
                verb = action.get("action_verb", "")
                target = action.get("target", "")
                print(f"    - {verb} {target}")
            if len(actions) > 3:
                print(f"    ... and {len(actions) - 3} more actions")
        
        notes = step.get("notes", "")
        if notes:
            print(f"  Notes: {notes[:100]}...")
        
        print()

def view_graph(manual_id: str):
    """Show graph structure."""
    # Path relative to project root
    project_root = Path(__file__).parent.parent.parent.parent
    graph_path = project_root / "output" / f"{manual_id}_graph.json"
    
    if not graph_path.exists():
        print(f"Error: {graph_path} not found")
        return
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Graph Structure for {manual_id}")
    print(f"{'='*60}\n")
    
    print(f"Metadata:")
    metadata = graph['metadata']
    print(f"  Total steps: {metadata['total_steps']}")
    print(f"  Total parts: {metadata['total_parts']}")
    print(f"  Total subassemblies: {metadata['total_subassemblies']}")
    
    # Show subassemblies
    subassemblies = [n for n in graph['nodes'] if n['type'] == 'subassembly']
    if subassemblies:
        print(f"\nSubassemblies ({len(subassemblies)}):")
        for sa in subassemblies[:10]:
            print(f"  - {sa['name']} (Step {sa['step_created']}, {len(sa['children'])} parts)")
        if len(subassemblies) > 10:
            print(f"  ... and {len(subassemblies) - 10} more")
    
    # Show sample parts
    parts = [n for n in graph['nodes'] if n['type'] == 'part']
    print(f"\nParts ({len(parts)} total, showing first 10):")
    for part in parts[:10]:
        qty = part.get('quantity', 1)
        print(f"  - {qty}x {part.get('color', '')} {part.get('shape', '')} ({part.get('role', 'unknown')})")
    
    # Show step states
    step_states = graph.get('step_states', [])
    if step_states:
        print(f"\nStep States (showing first 5):")
        for state in step_states[:5]:
            step_num = state['step_number']
            parts_added = len(state['new_parts_added'])
            completion = state['final_state']['completion_percentage']
            print(f"  Step {step_num}: {parts_added} parts, {completion:.1f}% complete")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_steps.py <manual_id> [--summary|--steps|--graph]")
        print("Example: python view_steps.py 6454922 --steps")
        sys.exit(1)
    
    manual_id = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--summary"
    
    if mode == "--summary":
        view_summary(manual_id)
    elif mode == "--steps":
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        view_steps(manual_id, limit)
    elif mode == "--graph":
        view_graph(manual_id)
    else:
        print(f"Unknown mode: {mode}")
        print("Use --summary, --steps, or --graph")

