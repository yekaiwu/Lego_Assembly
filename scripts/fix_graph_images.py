#!/usr/bin/env python3
"""Quick fix: Add assembled_product image paths to subassembly nodes in graph."""

import json
from pathlib import Path

def main():
    assembly_id = "6262059"
    output_dir = Path("output")

    # Load extracted steps and graph
    extracted_path = output_dir / f"{assembly_id}_extracted.json"
    graph_path = output_dir / f"{assembly_id}_graph.json"

    with open(extracted_path, 'r') as f:
        extracted_steps = json.load(f)

    with open(graph_path, 'r') as f:
        graph = json.load(f)

    # Create mapping of step_number -> assembled_product data
    step_to_assembly = {}
    for step in extracted_steps:
        step_num = step.get('step_number')
        assembled_product = step.get('assembled_product', {})
        if step_num and isinstance(assembled_product, dict):
            step_to_assembly[step_num] = assembled_product

    print(f"Found assembled_product data for {len(step_to_assembly)} steps")

    # Update subassembly nodes with image paths
    updated = 0
    for node in graph['nodes']:
        if node['type'] == 'subassembly':
            step_created = node.get('step_created')
            if step_created in step_to_assembly:
                assembly_data = step_to_assembly[step_created]
                node['image_path'] = assembly_data.get('cropped_image_path')
                node['mask_path'] = assembly_data.get('mask_path')
                node['bounding_box'] = assembly_data.get('bounding_box')

                if node['image_path']:
                    print(f"  ✓ Updated {node['name']} (step {step_created}): {node['image_path']}")
                    updated += 1

    # Save updated graph
    backup_path = output_dir / f"{assembly_id}_graph_backup.json"
    graph_path.rename(backup_path)
    print(f"\nBacked up original to: {backup_path}")

    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Updated {updated} subassembly nodes with image paths")
    print(f"💾 Saved to: {graph_path}")

if __name__ == "__main__":
    main()
