#!/usr/bin/env python3
"""
Test script to regenerate hierarchical graph with enhanced implementation.
This validates all the improvements: PromptManager integration, enhanced logging, etc.
"""

import json
import sys
from pathlib import Path
from loguru import logger

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from src.plan_generation.graph_builder import GraphBuilder


def main():
    """Test graph regeneration with manual 6454922."""
    # Configuration
    assembly_id = "6454922"
    output_dir = project_root / "output"
    image_dir = output_dir / "temp_pages"
    
    logger.info("=" * 80)
    logger.info("Testing Enhanced Hierarchical Graph Builder")
    logger.info("=" * 80)
    
    # Check prerequisites
    extracted_path = output_dir / f"{assembly_id}_extracted.json"
    if not extracted_path.exists():
        logger.error(f"Extracted data not found: {extracted_path}")
        return 1
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return 1
    
    # Load extracted steps
    logger.info(f"Loading extracted steps from {extracted_path}")
    with open(extracted_path, 'r', encoding='utf-8') as f:
        extracted_steps = json.load(f)
    
    logger.info(f"Loaded {len(extracted_steps)} steps")
    
    # Build graph with new implementation
    logger.info("\nInitializing enhanced GraphBuilder...")
    graph_builder = GraphBuilder()
    
    logger.info("\nBuilding hierarchical graph with:")
    logger.info("  ✓ PromptManager integration")
    logger.info("  ✓ Enhanced subassembly detection")
    logger.info("  ✓ Improved logging and debugging")
    logger.info("  ✓ Page relevance verification")
    
    # Build graph
    hierarchical_graph = graph_builder.build_graph(
        extracted_steps=extracted_steps,
        assembly_id=assembly_id,
        image_dir=image_dir
    )
    
    # Save graph
    graph_path = output_dir / f"{assembly_id}_graph_test.json"
    logger.info(f"\nSaving graph to {graph_path}")
    graph_builder.save_graph(hierarchical_graph, graph_path)
    
    # Verify outputs
    summary_path = output_dir / f"{assembly_id}_graph_test_summary.txt"
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Graph Regeneration Complete!")
    logger.info("=" * 80)
    logger.info(f"\nGenerated files:")
    logger.info(f"  1. {graph_path.name} - Full hierarchical graph JSON")
    logger.info(f"  2. {summary_path.name} - Human-readable summary")
    
    logger.info(f"\n📊 Graph Statistics:")
    logger.info(f"   Parts: {hierarchical_graph['metadata']['total_parts']}")
    logger.info(f"   Subassemblies: {hierarchical_graph['metadata']['total_subassemblies']}")
    logger.info(f"   Steps: {hierarchical_graph['metadata']['total_steps']}")
    logger.info(f"   Max Depth: {hierarchical_graph['metadata']['max_depth']} layers")
    
    # Compare with old graph if it exists
    old_graph_path = output_dir / f"{assembly_id}_graph.json"
    if old_graph_path.exists():
        logger.info("\n📈 Comparison with old graph:")
        with open(old_graph_path, 'r', encoding='utf-8') as f:
            old_graph = json.load(f)
        
        old_metadata = old_graph.get("metadata", {})
        new_metadata = hierarchical_graph["metadata"]
        
        logger.info(f"   Parts: {old_metadata.get('total_parts', '?')} → {new_metadata['total_parts']}")
        logger.info(f"   Subassemblies: {old_metadata.get('total_subassemblies', '?')} → {new_metadata['total_subassemblies']}")
        logger.info(f"   Max Depth: {old_metadata.get('max_depth', '?')} → {new_metadata['max_depth']}")
    
    logger.info("\n💡 Next Steps:")
    logger.info("  1. Review the summary file:")
    logger.info(f"     cat {summary_path}")
    logger.info("  2. Visualize the hierarchy:")
    logger.info(f"     python3 visualize_graph.py {graph_path} --show-subassemblies")
    logger.info("  3. Compare with manual PDF to verify accuracy")
    
    return 0


if __name__ == "__main__":
    exit(main())
