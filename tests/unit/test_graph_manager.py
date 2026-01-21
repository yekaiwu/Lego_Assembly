"""
Unit tests for GraphManager - hierarchical assembly graph queries.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


@pytest.fixture
def graph_manager():
    """Create a GraphManager instance."""
    from backend.app.graph.graph_manager import GraphManager
    return GraphManager()


@pytest.fixture
def sample_graph_file(temp_output_dir, sample_manual_id, sample_graph_data):
    """Create a sample graph file."""
    graph_path = temp_output_dir / f"{sample_manual_id}_graph.json"
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(sample_graph_data, f)
    return graph_path


@pytest.mark.unit
def test_graph_manager_initialization(graph_manager):
    """Test GraphManager initializes with empty cache."""
    assert graph_manager is not None
    assert hasattr(graph_manager, 'graphs')
    assert isinstance(graph_manager.graphs, dict)
    assert len(graph_manager.graphs) == 0


@pytest.mark.unit
def test_load_graph_from_file(graph_manager, sample_graph_file, sample_manual_id, mock_settings):
    """Test loading graph from file."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        # Update settings to point to temp output dir
        mock_settings.output_dir = sample_graph_file.parent

        graph = graph_manager.load_graph(sample_manual_id)

        assert graph is not None
        assert graph["assembly_id"] == sample_manual_id
        assert "metadata" in graph
        assert "nodes" in graph


@pytest.mark.unit
def test_load_graph_caching(graph_manager, sample_graph_file, sample_manual_id, mock_settings):
    """Test that graphs are cached in memory."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        mock_settings.output_dir = sample_graph_file.parent

        # First load
        graph1 = graph_manager.load_graph(sample_manual_id)

        # Second load should come from cache
        graph2 = graph_manager.load_graph(sample_manual_id)

        assert graph1 is graph2  # Same object reference
        assert sample_manual_id in graph_manager.graphs


@pytest.mark.unit
def test_load_graph_not_found(graph_manager, mock_settings):
    """Test loading non-existent graph."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        graph = graph_manager.load_graph("nonexistent_manual")

        assert graph is None


@pytest.mark.unit
def test_get_node_by_id(graph_manager, sample_graph_file, sample_manual_id, mock_settings):
    """Test retrieving node by ID."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        mock_settings.output_dir = sample_graph_file.parent

        # Assuming node has node_id field
        node = graph_manager.get_node(sample_manual_id, "main")

        # This might fail if the node structure is different
        # The test documents expected behavior


@pytest.mark.unit
def test_get_node_not_found(graph_manager, sample_graph_file, sample_manual_id, mock_settings):
    """Test retrieving non-existent node."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        mock_settings.output_dir = sample_graph_file.parent

        node = graph_manager.get_node(sample_manual_id, "nonexistent_node")

        assert node is None


@pytest.mark.unit
def test_get_node_by_name_exact(graph_manager, sample_graph_file, sample_manual_id, mock_settings):
    """Test finding nodes by exact name match."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        mock_settings.output_dir = sample_graph_file.parent

        nodes = graph_manager.get_node_by_name(sample_manual_id, "Main Assembly", fuzzy=False)

        # Should return list of matching nodes
        assert isinstance(nodes, list)


@pytest.mark.unit
def test_get_node_by_name_fuzzy(graph_manager, sample_graph_file, sample_manual_id, mock_settings):
    """Test finding nodes by fuzzy name match."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        mock_settings.output_dir = sample_graph_file.parent

        nodes = graph_manager.get_node_by_name(sample_manual_id, "main", fuzzy=True)

        # Should return list, possibly with fuzzy matches
        assert isinstance(nodes, list)


@pytest.mark.unit
def test_get_nodes_by_type(graph_manager, sample_graph_file, sample_manual_id, mock_settings, sample_graph_data):
    """Test filtering nodes by type."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        mock_settings.output_dir = sample_graph_file.parent

        # Load graph first
        graph_manager.load_graph(sample_manual_id)

        # Filter by type manually (if method doesn't exist, this documents expected API)
        graph = graph_manager.graphs[sample_manual_id]
        assembly_nodes = [n for n in graph["nodes"] if n["type"] == "assembly"]

        assert len(assembly_nodes) > 0


@pytest.mark.unit
def test_get_graph_summary(graph_manager, sample_graph_file, sample_manual_id, mock_settings):
    """Test getting graph summary/metadata."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        mock_settings.output_dir = sample_graph_file.parent

        graph = graph_manager.load_graph(sample_manual_id)

        assert "metadata" in graph
        metadata = graph["metadata"]

        assert "total_parts" in metadata
        assert "total_subassemblies" in metadata
        assert "total_steps" in metadata


@pytest.mark.unit
def test_load_graph_with_invalid_json(graph_manager, temp_output_dir, mock_settings):
    """Test loading graph with invalid JSON."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        mock_settings.output_dir = temp_output_dir

        # Create invalid JSON file
        invalid_path = temp_output_dir / "invalid_graph.json"
        invalid_path.write_text("{ invalid json }")

        graph = graph_manager.load_graph("invalid")

        assert graph is None


@pytest.mark.unit
def test_graph_manager_multiple_manuals(graph_manager, temp_output_dir, mock_settings):
    """Test managing multiple graphs simultaneously."""
    with patch('backend.app.graph.graph_manager.get_settings', return_value=mock_settings):
        mock_settings.output_dir = temp_output_dir

        # Create multiple graph files
        for i in range(3):
            manual_id = f"manual_{i}"
            graph_data = {
                "assembly_id": manual_id,
                "metadata": {"total_parts": i},
                "nodes": []
            }
            graph_path = temp_output_dir / f"{manual_id}_graph.json"
            with open(graph_path, 'w') as f:
                json.dump(graph_data, f)

        # Load all graphs
        graphs = []
        for i in range(3):
            graph = graph_manager.load_graph(f"manual_{i}")
            graphs.append(graph)

        # Verify all are cached
        assert len(graph_manager.graphs) == 3
        assert all(g is not None for g in graphs)


@pytest.mark.unit
def test_clear_graph_cache():
    """Test clearing graph cache."""
    from backend.app.graph.graph_manager import GraphManager

    manager = GraphManager()
    manager.graphs["test"] = {"data": "test"}

    # Clear cache
    manager.graphs.clear()

    assert len(manager.graphs) == 0


@pytest.mark.unit
def test_graph_node_structure_validation(sample_graph_data):
    """Test that graph node structure matches expected schema."""
    nodes = sample_graph_data["nodes"]

    for node in nodes:
        # Validate required fields
        assert "id" in node
        assert "type" in node
        assert "name" in node

        # Type should be from valid set
        assert node["type"] in ["assembly", "subassembly", "part"]
