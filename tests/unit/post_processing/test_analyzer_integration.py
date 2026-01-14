"""
Integration tests for PostProcessingSubassemblyAnalyzer.

Tests the full workflow of detecting patterns and integrating them into the graph.
"""

import pytest
from src.plan_generation.post_processing.analyzer import (
    PostProcessingSubassemblyAnalyzer,
    PostProcessingConfig
)


class TestPostProcessingAnalyzerIntegration:
    """Integration test suite for PostProcessingSubassemblyAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        config = PostProcessingConfig(
            enabled=True,
            jaccard_threshold=0.7,
            min_pattern_steps=2,
            min_confidence=0.6
        )
        return PostProcessingSubassemblyAnalyzer(
            vlm_client=None,
            config=config
        )

    @pytest.fixture
    def sample_graph(self):
        """Sample graph structure."""
        return {
            "manual_id": "test_123",
            "metadata": {
                "total_parts": 6,
                "total_subassemblies": 0,
                "total_steps": 6,
                "max_depth": 2
            },
            "nodes": [
                {
                    "node_id": "model_test_123",
                    "type": "model",
                    "name": "Test Model",
                    "children": [
                        "part_0", "part_1", "part_2",
                        "part_3", "part_4", "part_5"
                    ],
                    "parents": [],
                    "layer": 0
                },
                {
                    "node_id": "part_0",
                    "type": "part",
                    "name": "Red Brick",
                    "color": "red",
                    "shape": "2x4_brick",
                    "children": [],
                    "parents": ["model_test_123"],
                    "step_created": 1,
                    "layer": 1
                },
                {
                    "node_id": "part_1",
                    "type": "part",
                    "name": "Red Brick",
                    "color": "red",
                    "shape": "2x4_brick",
                    "children": [],
                    "parents": ["model_test_123"],
                    "step_created": 2,
                    "layer": 1
                },
                {
                    "node_id": "part_2",
                    "type": "part",
                    "name": "Red Brick",
                    "color": "red",
                    "shape": "2x4_brick",
                    "children": [],
                    "parents": ["model_test_123"],
                    "step_created": 3,
                    "layer": 1
                },
                {
                    "node_id": "part_3",
                    "type": "part",
                    "name": "Blue Axle",
                    "color": "blue",
                    "shape": "axle",
                    "children": [],
                    "parents": ["model_test_123"],
                    "step_created": 4,
                    "layer": 1
                },
                {
                    "node_id": "part_4",
                    "type": "part",
                    "name": "Green Plate",
                    "color": "green",
                    "shape": "plate",
                    "children": [],
                    "parents": ["model_test_123"],
                    "step_created": 5,
                    "layer": 1
                },
                {
                    "node_id": "part_5",
                    "type": "part",
                    "name": "Green Plate",
                    "color": "green",
                    "shape": "plate",
                    "children": [],
                    "parents": ["model_test_123"],
                    "step_created": 6,
                    "layer": 1
                }
            ],
            "edges": [
                {
                    "from": "part_0",
                    "to": "model_test_123",
                    "type": "component",
                    "created_step": 1
                },
                {
                    "from": "part_1",
                    "to": "model_test_123",
                    "type": "component",
                    "created_step": 2
                },
                {
                    "from": "part_2",
                    "to": "model_test_123",
                    "type": "component",
                    "created_step": 3
                },
                {
                    "from": "part_3",
                    "to": "model_test_123",
                    "type": "component",
                    "created_step": 4
                },
                {
                    "from": "part_4",
                    "to": "model_test_123",
                    "type": "component",
                    "created_step": 5
                },
                {
                    "from": "part_5",
                    "to": "model_test_123",
                    "type": "component",
                    "created_step": 6
                }
            ]
        }

    @pytest.fixture
    def sample_steps_with_patterns(self):
        """Sample steps with repeated patterns."""
        return [
            {
                "step_number": 1,
                "parts_required": [
                    {"color": "red", "shape": "2x4_brick"}
                ],
                "notes": "Add first leg brick",
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 2,
                "parts_required": [
                    {"color": "red", "shape": "2x4_brick"}
                ],
                "notes": "Add second leg brick",
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 3,
                "parts_required": [
                    {"color": "red", "shape": "2x4_brick"}
                ],
                "notes": "Add third leg brick",
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 4,
                "parts_required": [
                    {"color": "blue", "shape": "axle"}
                ],
                "notes": "Add axle piece",
                "existing_assembly": "legs",
                "actions": []
            },
            {
                "step_number": 5,
                "parts_required": [
                    {"color": "green", "shape": "plate"}
                ],
                "notes": "Stack plate on top",
                "existing_assembly": "base",
                "actions": [{"destination": "above"}]
            },
            {
                "step_number": 6,
                "parts_required": [
                    {"color": "green", "shape": "plate"}
                ],
                "notes": "Add another layer on top",
                "existing_assembly": "base",
                "actions": [{"destination": "top"}]
            }
        ]

    def test_analyze_and_augment_graph(
        self,
        analyzer,
        sample_graph,
        sample_steps_with_patterns
    ):
        """Test full analysis and augmentation workflow."""
        augmented_graph = analyzer.analyze_and_augment_graph(
            graph=sample_graph,
            extracted_steps=sample_steps_with_patterns
        )

        # Check that graph was returned
        assert augmented_graph is not None
        assert "nodes" in augmented_graph
        assert "edges" in augmented_graph
        assert "metadata" in augmented_graph

        # Check if any patterns were discovered
        nodes = augmented_graph["nodes"]
        discovered_subassemblies = [
            n for n in nodes
            if n.get("type") == "subassembly" and
               n.get("discovery_method") is not None
        ]

        # Should discover at least one pattern
        # (repeated red bricks in steps 1-3)
        assert len(discovered_subassemblies) > 0

    def test_pattern_integration_into_graph(
        self,
        analyzer,
        sample_graph,
        sample_steps_with_patterns
    ):
        """Test that patterns are properly integrated into graph."""
        augmented_graph = analyzer.analyze_and_augment_graph(
            graph=sample_graph,
            extracted_steps=sample_steps_with_patterns
        )

        nodes = augmented_graph["nodes"]
        edges = augmented_graph["edges"]

        # Find discovered subassemblies
        discovered = [
            n for n in nodes
            if n.get("type") == "subassembly" and
               n.get("discovery_method")
        ]

        if discovered:
            subasm = discovered[0]

            # Check subassembly has proper structure
            assert "node_id" in subasm
            assert "name" in subasm
            assert "description" in subasm
            assert "children" in subasm
            assert "parents" in subasm
            assert "discovery_method" in subasm
            assert "confidence" in subasm
            assert "step_range" in subasm

            # Check confidence is valid
            assert 0.0 <= subasm["confidence"] <= 1.0

            # Check subassembly has children
            assert len(subasm["children"]) > 0

            # Check parts are re-parented to subassembly
            for part_id in subasm["children"]:
                part = next(n for n in nodes if n["node_id"] == part_id)
                assert subasm["node_id"] in part["parents"]

            # Check edges exist for subassembly
            subasm_edges = [
                e for e in edges
                if e["from"] == subasm["node_id"] or e["to"] == subasm["node_id"]
            ]
            assert len(subasm_edges) > 0

    def test_metadata_update(
        self,
        analyzer,
        sample_graph,
        sample_steps_with_patterns
    ):
        """Test that metadata is updated correctly."""
        augmented_graph = analyzer.analyze_and_augment_graph(
            graph=sample_graph,
            extracted_steps=sample_steps_with_patterns
        )

        metadata = augmented_graph["metadata"]

        # Check metadata fields
        assert "total_subassemblies" in metadata
        assert "discovered_subassemblies" in metadata

        # If patterns were discovered, counts should be updated
        if metadata.get("discovered_subassemblies", 0) > 0:
            assert metadata["total_subassemblies"] >= metadata["discovered_subassemblies"]

    def test_layer_recalculation(
        self,
        analyzer,
        sample_graph,
        sample_steps_with_patterns
    ):
        """Test that layers are recalculated after augmentation."""
        augmented_graph = analyzer.analyze_and_augment_graph(
            graph=sample_graph,
            extracted_steps=sample_steps_with_patterns
        )

        nodes = augmented_graph["nodes"]

        # All nodes should have valid layer numbers
        for node in nodes:
            assert "layer" in node
            assert isinstance(node["layer"], int)
            assert node["layer"] >= 0

        # Model should be at layer 0
        model = next(n for n in nodes if n["type"] == "model")
        assert model["layer"] == 0

        # Children should be at higher layers than parents
        for node in nodes:
            if node["type"] != "model":
                for parent_id in node.get("parents", []):
                    parent = next(n for n in nodes if n["node_id"] == parent_id)
                    assert node["layer"] > parent["layer"]

    def test_disabled_post_processing(
        self,
        sample_graph,
        sample_steps_with_patterns
    ):
        """Test with post-processing disabled."""
        config = PostProcessingConfig(enabled=False)
        analyzer = PostProcessingSubassemblyAnalyzer(
            vlm_client=None,
            config=config
        )

        augmented_graph = analyzer.analyze_and_augment_graph(
            graph=sample_graph,
            extracted_steps=sample_steps_with_patterns
        )

        # Graph should be unchanged
        assert augmented_graph == sample_graph

    def test_validation_filters_low_confidence(self, analyzer, sample_graph):
        """Test that low-confidence patterns are filtered."""
        # Create analyzer with high confidence threshold
        config = PostProcessingConfig(
            enabled=True,
            min_confidence=0.95  # Very high threshold
        )
        analyzer = PostProcessingSubassemblyAnalyzer(
            vlm_client=None,
            config=config
        )

        steps = [
            {
                "step_number": 1,
                "parts_required": [{"color": "red", "shape": "brick"}],
                "notes": "",
                "actions": []
            },
            {
                "step_number": 2,
                "parts_required": [{"color": "red", "shape": "brick"}],
                "notes": "",
                "actions": []
            }
        ]

        augmented_graph = analyzer.analyze_and_augment_graph(
            graph=sample_graph,
            extracted_steps=steps
        )

        # Most patterns should be filtered out due to high threshold
        nodes = augmented_graph["nodes"]
        discovered = [
            n for n in nodes
            if n.get("type") == "subassembly" and n.get("discovery_method")
        ]

        # Should have fewer (or no) discovered patterns
        assert len(discovered) <= 1

    def test_max_patterns_limit(self, analyzer, sample_graph):
        """Test that max_discovered_patterns limit is enforced."""
        config = PostProcessingConfig(
            enabled=True,
            max_discovered_patterns=1  # Limit to 1 pattern
        )
        analyzer = PostProcessingSubassemblyAnalyzer(
            vlm_client=None,
            config=config
        )

        # Create steps with multiple potential patterns
        steps = []
        for i in range(1, 21):  # 20 steps
            steps.append({
                "step_number": i,
                "parts_required": [{"color": "red", "shape": "brick"}],
                "notes": "step " + str(i),
                "actions": []
            })

        augmented_graph = analyzer.analyze_and_augment_graph(
            graph=sample_graph,
            extracted_steps=steps
        )

        # Should have at most 1 discovered pattern
        nodes = augmented_graph["nodes"]
        discovered = [
            n for n in nodes
            if n.get("type") == "subassembly" and n.get("discovery_method")
        ]

        assert len(discovered) <= 1

    def test_empty_graph(self, analyzer):
        """Test with empty graph."""
        empty_graph = {
            "manual_id": "empty",
            "metadata": {},
            "nodes": [],
            "edges": []
        }

        augmented_graph = analyzer.analyze_and_augment_graph(
            graph=empty_graph,
            extracted_steps=[]
        )

        # Should handle gracefully
        assert augmented_graph is not None

    def test_graph_with_existing_subassemblies(self, analyzer):
        """Test with graph that already has subassemblies."""
        graph_with_subasm = {
            "manual_id": "test",
            "metadata": {
                "total_parts": 2,
                "total_subassemblies": 1,
                "total_steps": 3,
                "max_depth": 2
            },
            "nodes": [
                {
                    "node_id": "model_test",
                    "type": "model",
                    "children": ["subasm_0"],
                    "parents": [],
                    "layer": 0
                },
                {
                    "node_id": "subasm_0",
                    "type": "subassembly",
                    "name": "Existing Subassembly",
                    "children": ["part_0", "part_1"],
                    "parents": ["model_test"],
                    "step_created": 1,
                    "layer": 1
                },
                {
                    "node_id": "part_0",
                    "type": "part",
                    "name": "Part 0",
                    "children": [],
                    "parents": ["subasm_0"],
                    "step_created": 1,
                    "layer": 2
                },
                {
                    "node_id": "part_1",
                    "type": "part",
                    "name": "Part 1",
                    "children": [],
                    "parents": ["subasm_0"],
                    "step_created": 2,
                    "layer": 2
                }
            ],
            "edges": []
        }

        steps = [
            {
                "step_number": 1,
                "parts_required": [{"color": "red", "shape": "brick"}],
                "notes": "",
                "actions": []
            },
            {
                "step_number": 2,
                "parts_required": [{"color": "red", "shape": "brick"}],
                "notes": "",
                "actions": []
            }
        ]

        augmented_graph = analyzer.analyze_and_augment_graph(
            graph=graph_with_subasm,
            extracted_steps=steps
        )

        # Should not interfere with existing subassembly
        nodes = augmented_graph["nodes"]
        existing = next(n for n in nodes if n["node_id"] == "subasm_0")
        assert existing["name"] == "Existing Subassembly"
