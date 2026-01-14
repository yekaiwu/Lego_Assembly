"""
Unit tests for SpatialTemporalPatternMiner.
"""

import pytest
from src.plan_generation.post_processing.strategies.spatial_temporal import (
    SpatialTemporalPatternMiner
)


class TestSpatialTemporalPatternMiner:
    """Test suite for SpatialTemporalPatternMiner."""

    @pytest.fixture
    def miner(self):
        """Create miner instance."""
        return SpatialTemporalPatternMiner(
            min_sequence_steps=3,
            similarity_threshold=0.6
        )

    @pytest.fixture
    def sample_steps_vertical(self):
        """Sample steps with vertical progression."""
        return [
            {
                "step_number": 1,
                "notes": "Place base plate on the bottom",
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 2,
                "notes": "Stack brick on top of base",
                "existing_assembly": "base",
                "actions": [{"action_verb": "attach", "destination": "above"}]
            },
            {
                "step_number": 3,
                "notes": "Add another layer on top",
                "existing_assembly": "existing structure",
                "actions": [{"action_verb": "place", "destination": "top"}]
            },
            {
                "step_number": 4,
                "notes": "Place roof piece above",
                "existing_assembly": "walls",
                "actions": []
            }
        ]

    @pytest.fixture
    def sample_steps_horizontal(self):
        """Sample steps with horizontal progression."""
        return [
            {
                "step_number": 1,
                "notes": "Place left wall",
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 2,
                "notes": "Attach right wall to the side",
                "existing_assembly": "left wall",
                "actions": [{"destination": "beside"}]
            },
            {
                "step_number": 3,
                "notes": "Connect side panels",
                "existing_assembly": "walls",
                "actions": [{"destination": "adjacent"}]
            }
        ]

    @pytest.fixture
    def sample_steps_enclosure(self):
        """Sample steps with enclosure build."""
        return [
            {
                "step_number": 1,
                "notes": "Build front wall",
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 2,
                "notes": "Add back wall to enclose",
                "existing_assembly": "front",
                "actions": []
            },
            {
                "step_number": 3,
                "notes": "Place door in wall",
                "existing_assembly": "walls",
                "actions": []
            }
        ]

    def test_detect_direction_vertical_up(self, miner):
        """Test detecting vertical upward direction."""
        text = "place brick on top of the base"
        direction = miner._detect_direction(text)

        assert direction == "vertical_up"

    def test_detect_direction_vertical_down(self, miner):
        """Test detecting vertical downward direction."""
        text = "place foundation at the bottom"
        direction = miner._detect_direction(text)

        assert direction == "vertical_down"

    def test_detect_direction_horizontal(self, miner):
        """Test detecting horizontal direction."""
        text = "attach piece to the side"
        direction = miner._detect_direction(text)

        assert direction == "horizontal"

    def test_detect_direction_enclosure(self, miner):
        """Test detecting enclosure direction."""
        text = "build wall to enclose the structure"
        direction = miner._detect_direction(text)

        assert direction == "enclosure"

    def test_detect_direction_expansion(self, miner):
        """Test detecting expansion direction."""
        text = "extend the structure outward"
        direction = miner._detect_direction(text)

        assert direction == "expansion"

    def test_detect_direction_none(self, miner):
        """Test when no direction is detected."""
        text = "add random piece"
        direction = miner._detect_direction(text)

        assert direction is None

    def test_analyze_spatial_directions(self, miner, sample_steps_vertical):
        """Test analyzing spatial directions."""
        sequence = miner._analyze_spatial_directions(sample_steps_vertical)

        assert len(sequence) > 0

        # Should detect vertical directions
        directions = [s["direction"] for s in sequence]
        assert "vertical_up" in directions or "vertical_down" in directions

    def test_find_progressive_builds_vertical(
        self,
        miner,
        sample_steps_vertical
    ):
        """Test finding vertical progression."""
        graph = {"nodes": [], "edges": []}
        patterns = miner.find_progressive_builds(
            graph,
            sample_steps_vertical
        )

        # Should find at least one pattern
        assert len(patterns) > 0

        pattern = patterns[0]
        assert pattern["type"] == "progressive_build"
        assert len(pattern["steps"]) >= 3
        assert pattern["confidence"] > 0.0

    def test_find_progressive_builds_horizontal(
        self,
        miner,
        sample_steps_horizontal
    ):
        """Test finding horizontal progression."""
        graph = {"nodes": [], "edges": []}
        patterns = miner.find_progressive_builds(
            graph,
            sample_steps_horizontal
        )

        # Should find pattern
        assert len(patterns) > 0

        pattern = patterns[0]
        assert "horizontal" in pattern["metadata"]["direction"]

    def test_find_progressive_builds_enclosure(
        self,
        miner,
        sample_steps_enclosure
    ):
        """Test finding enclosure progression."""
        graph = {"nodes": [], "edges": []}
        patterns = miner.find_progressive_builds(
            graph,
            sample_steps_enclosure
        )

        # Should find enclosure pattern
        assert len(patterns) > 0

        pattern = patterns[0]
        assert pattern["metadata"]["direction"] == "enclosure"

    def test_confidence_calculation(self, miner):
        """Test confidence score calculation."""
        steps_short = [1, 2, 3]
        steps_long = [1, 2, 3, 4, 5, 6]

        confidence_short = miner._calculate_confidence(
            steps_short,
            "vertical_up"
        )
        confidence_long = miner._calculate_confidence(
            steps_long,
            "vertical_up"
        )

        assert 0.0 <= confidence_short <= 1.0
        assert 0.0 <= confidence_long <= 1.0
        assert confidence_long >= confidence_short

    def test_confidence_reliable_direction(self, miner):
        """Test confidence boost for reliable directions."""
        steps = [1, 2, 3]

        confidence_reliable = miner._calculate_confidence(
            steps,
            "vertical_up"
        )
        confidence_normal = miner._calculate_confidence(
            steps,
            "horizontal"
        )

        # Reliable directions should have higher confidence
        assert confidence_reliable >= confidence_normal

    def test_empty_steps(self, miner):
        """Test with empty steps list."""
        graph = {"nodes": [], "edges": []}
        patterns = miner.find_progressive_builds(graph, [])

        assert len(patterns) == 0

    def test_min_sequence_threshold(self, sample_steps_vertical):
        """Test min_sequence_steps threshold."""
        # Set threshold higher than pattern length
        miner = SpatialTemporalPatternMiner(
            min_sequence_steps=10,
            similarity_threshold=0.6
        )

        graph = {"nodes": [], "edges": []}
        patterns = miner.find_progressive_builds(
            graph,
            sample_steps_vertical
        )

        # Should find no patterns (threshold too high)
        assert len(patterns) == 0

    def test_steps_without_directions(self, miner):
        """Test steps with no spatial keywords."""
        steps = [
            {
                "step_number": 1,
                "notes": "add piece",
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 2,
                "notes": "connect another piece",
                "existing_assembly": "",
                "actions": []
            }
        ]

        graph = {"nodes": [], "edges": []}
        patterns = miner.find_progressive_builds(graph, steps)

        # Should find no patterns (no spatial keywords)
        assert len(patterns) == 0
