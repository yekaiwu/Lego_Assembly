"""
Unit tests for PartSimilarityClusterer.
"""

import pytest
from src.plan_generation.post_processing.strategies.part_similarity import (
    PartSimilarityClusterer
)


class TestPartSimilarityClusterer:
    """Test suite for PartSimilarityClusterer."""

    @pytest.fixture
    def clusterer(self):
        """Create clusterer instance."""
        return PartSimilarityClusterer(
            jaccard_threshold=0.7,
            min_pattern_steps=2
        )

    @pytest.fixture
    def sample_steps_repeated(self):
        """Sample steps with repeated structure."""
        return [
            {
                "step_number": 1,
                "parts_required": [
                    {"color": "red", "shape": "2x4_brick"},
                    {"color": "blue", "shape": "axle"}
                ]
            },
            {
                "step_number": 2,
                "parts_required": [
                    {"color": "red", "shape": "2x4_brick"},
                    {"color": "blue", "shape": "axle"}
                ]
            },
            {
                "step_number": 3,
                "parts_required": [
                    {"color": "red", "shape": "2x4_brick"},
                    {"color": "blue", "shape": "axle"}
                ]
            },
            {
                "step_number": 4,
                "parts_required": [
                    {"color": "green", "shape": "plate"}
                ]
            }
        ]

    @pytest.fixture
    def sample_steps_no_pattern(self):
        """Sample steps without repeated pattern."""
        return [
            {
                "step_number": 1,
                "parts_required": [
                    {"color": "red", "shape": "2x4_brick"}
                ]
            },
            {
                "step_number": 2,
                "parts_required": [
                    {"color": "blue", "shape": "axle"}
                ]
            },
            {
                "step_number": 3,
                "parts_required": [
                    {"color": "green", "shape": "plate"}
                ]
            }
        ]

    def test_build_part_usage_matrix(self, clusterer, sample_steps_repeated):
        """Test building part usage matrix."""
        matrix = clusterer._build_part_usage_matrix(sample_steps_repeated)

        assert len(matrix) == 4
        assert 1 in matrix
        assert 2 in matrix
        assert "red:2x4_brick" in matrix[1]
        assert "blue:axle" in matrix[1]

    def test_jaccard_similarity(self, clusterer):
        """Test Jaccard similarity calculation."""
        set_a = {"red:brick", "blue:axle"}
        set_b = {"red:brick", "blue:axle"}
        set_c = {"green:plate"}

        # Identical sets
        assert clusterer._jaccard_similarity(set_a, set_b) == 1.0

        # No overlap
        assert clusterer._jaccard_similarity(set_a, set_c) == 0.0

        # Partial overlap
        set_d = {"red:brick", "yellow:plate"}
        similarity = clusterer._jaccard_similarity(set_a, set_d)
        assert 0.0 < similarity < 1.0

    def test_find_repeated_structures(
        self,
        clusterer,
        sample_steps_repeated
    ):
        """Test finding repeated structures."""
        graph = {"nodes": [], "edges": []}
        patterns = clusterer.find_repeated_structures(
            graph,
            sample_steps_repeated
        )

        # Should find pattern in steps 1-3
        assert len(patterns) > 0

        pattern = patterns[0]
        assert pattern["type"] == "repeated_structure"
        assert len(pattern["steps"]) == 3
        assert pattern["steps"] == [1, 2, 3]
        assert pattern["confidence"] > 0.8

    def test_no_pattern_found(self, clusterer, sample_steps_no_pattern):
        """Test when no patterns exist."""
        graph = {"nodes": [], "edges": []}
        patterns = clusterer.find_repeated_structures(
            graph,
            sample_steps_no_pattern
        )

        # No repeated patterns
        assert len(patterns) == 0

    def test_min_pattern_steps_threshold(self, sample_steps_repeated):
        """Test min_pattern_steps threshold."""
        # Set threshold to 4 (higher than pattern length)
        clusterer = PartSimilarityClusterer(
            jaccard_threshold=0.7,
            min_pattern_steps=4
        )

        graph = {"nodes": [], "edges": []}
        patterns = clusterer.find_repeated_structures(
            graph,
            sample_steps_repeated
        )

        # Should find no patterns (only 3 consecutive steps)
        assert len(patterns) == 0

    def test_confidence_calculation(self, clusterer):
        """Test confidence score calculation."""
        # More steps = higher confidence
        steps_short = [1, 2]
        steps_long = [1, 2, 3, 4, 5]
        parts = {"red:brick", "blue:axle"}

        confidence_short = clusterer._calculate_confidence(steps_short, parts)
        confidence_long = clusterer._calculate_confidence(steps_long, parts)

        assert confidence_long > confidence_short
        assert 0.0 <= confidence_short <= 1.0
        assert 0.0 <= confidence_long <= 1.0

    def test_empty_steps(self, clusterer):
        """Test with empty steps list."""
        graph = {"nodes": [], "edges": []}
        patterns = clusterer.find_repeated_structures(graph, [])

        assert len(patterns) == 0

    def test_steps_without_parts(self, clusterer):
        """Test steps without parts_required field."""
        steps = [
            {"step_number": 1},
            {"step_number": 2, "parts_required": []}
        ]

        graph = {"nodes": [], "edges": []}
        patterns = clusterer.find_repeated_structures(graph, steps)

        assert len(patterns) == 0
