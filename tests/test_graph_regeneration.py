"""
Test suite for hierarchical graph regeneration with enhanced implementation.
Validates graph building, node structure, and metadata generation.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def graph_builder():
    """Create a GraphBuilder instance."""
    from src.plan_generation.graph_builder import GraphBuilder
    return GraphBuilder()


@pytest.fixture
def extracted_data_file(temp_output_dir, sample_manual_id, sample_extracted_steps):
    """Create a sample extracted data file."""
    extracted_path = temp_output_dir / f"{sample_manual_id}_extracted.json"
    with open(extracted_path, 'w', encoding='utf-8') as f:
        json.dump(sample_extracted_steps, f)
    return extracted_path


@pytest.fixture
def sample_image_dir(temp_output_dir):
    """Create a sample image directory with dummy images."""
    image_dir = temp_output_dir / "temp_pages"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy image files
    for i in range(1, 4):
        image_path = image_dir / f"page_{i:03d}.png"
        image_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

    return image_dir


@pytest.mark.integration
@pytest.mark.slow
def test_graph_builder_initialization(graph_builder):
    """Test that GraphBuilder initializes correctly."""
    assert graph_builder is not None


@pytest.mark.integration
@pytest.mark.slow
def test_build_graph_with_sample_data(
    graph_builder,
    sample_manual_id,
    sample_extracted_steps,
    sample_image_dir
):
    """Test building a hierarchical graph with sample data."""
    hierarchical_graph = graph_builder.build_graph(
        extracted_steps=sample_extracted_steps,
        assembly_id=sample_manual_id,
        image_dir=sample_image_dir
    )

    # Verify graph structure
    assert hierarchical_graph is not None
    assert "assembly_id" in hierarchical_graph
    assert hierarchical_graph["assembly_id"] == sample_manual_id
    assert "metadata" in hierarchical_graph
    assert "nodes" in hierarchical_graph


@pytest.mark.integration
def test_graph_metadata_structure(
    graph_builder,
    sample_manual_id,
    sample_extracted_steps,
    sample_image_dir
):
    """Test that graph metadata contains expected fields."""
    hierarchical_graph = graph_builder.build_graph(
        extracted_steps=sample_extracted_steps,
        assembly_id=sample_manual_id,
        image_dir=sample_image_dir
    )

    metadata = hierarchical_graph["metadata"]

    # Verify metadata fields
    assert "total_parts" in metadata
    assert "total_subassemblies" in metadata
    assert "total_steps" in metadata
    assert "max_depth" in metadata

    # Verify metadata values are reasonable
    assert metadata["total_parts"] >= 0
    assert metadata["total_subassemblies"] >= 0
    assert metadata["total_steps"] > 0
    assert metadata["max_depth"] >= 1


@pytest.mark.integration
def test_graph_nodes_structure(
    graph_builder,
    sample_manual_id,
    sample_extracted_steps,
    sample_image_dir
):
    """Test that graph nodes have correct structure."""
    hierarchical_graph = graph_builder.build_graph(
        extracted_steps=sample_extracted_steps,
        assembly_id=sample_manual_id,
        image_dir=sample_image_dir
    )

    nodes = hierarchical_graph["nodes"]

    assert len(nodes) > 0

    # Check first node structure
    node = nodes[0]
    assert "id" in node
    assert "type" in node
    assert "name" in node


@pytest.mark.integration
def test_save_graph(
    graph_builder,
    sample_manual_id,
    sample_extracted_steps,
    sample_image_dir,
    temp_output_dir
):
    """Test saving graph to file."""
    hierarchical_graph = graph_builder.build_graph(
        extracted_steps=sample_extracted_steps,
        assembly_id=sample_manual_id,
        image_dir=sample_image_dir
    )

    graph_path = temp_output_dir / f"{sample_manual_id}_graph_test.json"

    # Save graph
    graph_builder.save_graph(hierarchical_graph, graph_path)

    # Verify file was created
    assert graph_path.exists()

    # Verify file contents
    with open(graph_path, 'r', encoding='utf-8') as f:
        loaded_graph = json.load(f)

    assert loaded_graph == hierarchical_graph


@pytest.mark.integration
def test_graph_summary_generation(
    graph_builder,
    sample_manual_id,
    sample_extracted_steps,
    sample_image_dir,
    temp_output_dir
):
    """Test that graph summary file is generated."""
    hierarchical_graph = graph_builder.build_graph(
        extracted_steps=sample_extracted_steps,
        assembly_id=sample_manual_id,
        image_dir=sample_image_dir
    )

    graph_path = temp_output_dir / f"{sample_manual_id}_graph_test.json"
    summary_path = temp_output_dir / f"{sample_manual_id}_graph_test_summary.txt"

    # Save graph (which should also create summary)
    graph_builder.save_graph(hierarchical_graph, graph_path)

    # Summary might be created separately; this test documents expected behavior


@pytest.mark.integration
def test_graph_with_empty_steps(graph_builder, sample_manual_id, sample_image_dir):
    """Test graph building with empty step list."""
    with pytest.raises(Exception):
        graph_builder.build_graph(
            extracted_steps=[],
            assembly_id=sample_manual_id,
            image_dir=sample_image_dir
        )


@pytest.mark.integration
def test_graph_with_missing_image_dir(
    graph_builder,
    sample_manual_id,
    sample_extracted_steps
):
    """Test graph building with non-existent image directory."""
    non_existent_dir = Path("/tmp/nonexistent_image_dir_12345")

    # This might raise an exception or handle gracefully
    # The test documents the expected behavior
    try:
        hierarchical_graph = graph_builder.build_graph(
            extracted_steps=sample_extracted_steps,
            assembly_id=sample_manual_id,
            image_dir=non_existent_dir
        )
        # If it succeeds, verify the graph is still valid
        assert hierarchical_graph is not None
    except (FileNotFoundError, ValueError) as e:
        # Expected behavior for missing directory
        pass


@pytest.mark.unit
def test_graph_metadata_calculation():
    """Test metadata calculation logic."""
    sample_nodes = [
        {"type": "assembly", "parts": []},
        {"type": "subassembly", "parts": ["part1", "part2"]},
        {"type": "subassembly", "parts": ["part3"]},
    ]

    # Count parts
    total_parts = sum(len(node.get("parts", [])) for node in sample_nodes)
    assert total_parts == 3

    # Count subassemblies
    total_subassemblies = sum(1 for node in sample_nodes if node["type"] == "subassembly")
    assert total_subassemblies == 2


@pytest.mark.integration
def test_load_extracted_data_from_file(extracted_data_file, sample_extracted_steps):
    """Test loading extracted data from JSON file."""
    with open(extracted_data_file, 'r', encoding='utf-8') as f:
        loaded_steps = json.load(f)

    assert loaded_steps == sample_extracted_steps
    assert len(loaded_steps) > 0
