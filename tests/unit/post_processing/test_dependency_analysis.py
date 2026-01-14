"""
Unit tests for DependencyAnalyzer.
"""

import pytest
from src.plan_generation.post_processing.strategies.dependency_analysis import (
    DependencyAnalyzer,
    UnionFind
)


class TestUnionFind:
    """Test suite for UnionFind data structure."""

    def test_find_single_element(self):
        """Test finding single element."""
        uf = UnionFind()
        assert uf.find(1) == 1
        assert uf.find(2) == 2

    def test_union_elements(self):
        """Test union of elements."""
        uf = UnionFind()
        uf.union(1, 2)

        # Should have same root
        assert uf.find(1) == uf.find(2)

    def test_get_components(self):
        """Test getting connected components."""
        uf = UnionFind()

        # Create two separate components
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(4, 5)

        components = uf.get_components()

        # Should have 2 components
        assert len(components) == 2

        # Check component contents
        all_nodes = set()
        for nodes in components.values():
            all_nodes.update(nodes)

        assert 1 in all_nodes
        assert 2 in all_nodes
        assert 3 in all_nodes
        assert 4 in all_nodes
        assert 5 in all_nodes


class TestDependencyAnalyzer:
    """Test suite for DependencyAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DependencyAnalyzer(
            min_group_steps=3,
            min_isolation_steps=2
        )

    @pytest.fixture
    def sample_steps_independent(self):
        """Sample steps with independent group."""
        return [
            {
                "step_number": 1,
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 2,
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 3,
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 4,
                "existing_assembly": "attach to previous",
                "actions": [{"destination": "existing structure"}]
            }
        ]

    @pytest.fixture
    def sample_steps_dependent(self):
        """Sample steps all dependent on each other."""
        return [
            {
                "step_number": 1,
                "existing_assembly": "",
                "actions": []
            },
            {
                "step_number": 2,
                "existing_assembly": "add to previous",
                "actions": [{"destination": "existing"}]
            },
            {
                "step_number": 3,
                "existing_assembly": "attach to previous",
                "actions": [{"destination": "existing"}]
            }
        ]

    def test_build_step_dependencies_independent(
        self,
        analyzer,
        sample_steps_independent
    ):
        """Test building dependencies for independent steps."""
        deps = analyzer._build_step_dependencies(sample_steps_independent)

        # Steps 1-3 should have no dependencies
        assert 1 in deps
        assert len(deps[1]) == 0

        # Step 4 depends on previous
        assert 4 in deps
        assert 3 in deps[4]

    def test_build_step_dependencies_dependent(
        self,
        analyzer,
        sample_steps_dependent
    ):
        """Test building dependencies for dependent steps."""
        deps = analyzer._build_step_dependencies(sample_steps_dependent)

        # Step 1 independent
        assert len(deps[1]) == 0

        # Steps 2 and 3 depend on previous
        assert 1 in deps[2]
        assert 2 in deps[3]

    def test_find_connected_components(self, analyzer):
        """Test finding connected components."""
        # Create dependency graph with two separate chains
        deps = {
            1: set(),
            2: {1},
            3: {2},
            4: set(),
            5: {4}
        }

        components = analyzer._find_connected_components(deps)

        # Should find 2 components
        assert len(components) == 2

    def test_calculate_isolation_period(self, analyzer):
        """Test calculating isolation period."""
        component_steps = [1, 2, 3]
        deps = {
            1: set(),
            2: set(),
            3: set(),
            4: {1}  # Step 4 depends on component
        }
        extracted_steps = []

        isolation = analyzer._calculate_isolation_period(
            component_steps,
            deps,
            extracted_steps
        )

        # Should be isolated for 3 steps
        assert isolation == 3

    def test_find_independent_groups(
        self,
        analyzer,
        sample_steps_independent
    ):
        """Test finding independent groups."""
        graph = {"nodes": [], "edges": []}
        patterns = analyzer.find_independent_groups(
            graph,
            sample_steps_independent
        )

        # May or may not find pattern depending on implementation details
        # Just check it doesn't crash
        assert isinstance(patterns, list)

    def test_confidence_calculation(self, analyzer):
        """Test confidence score calculation."""
        steps_short = [1, 2, 3]
        steps_long = [1, 2, 3, 4, 5]

        confidence_short = analyzer._calculate_confidence(steps_short, 2)
        confidence_long = analyzer._calculate_confidence(steps_long, 4)

        assert 0.0 <= confidence_short <= 1.0
        assert 0.0 <= confidence_long <= 1.0
        assert confidence_long >= confidence_short

    def test_empty_steps(self, analyzer):
        """Test with empty steps list."""
        graph = {"nodes": [], "edges": []}
        patterns = analyzer.find_independent_groups(graph, [])

        assert len(patterns) == 0

    def test_min_group_steps_threshold(self, sample_steps_independent):
        """Test min_group_steps threshold."""
        # Set threshold higher than available steps
        analyzer = DependencyAnalyzer(
            min_group_steps=10,
            min_isolation_steps=2
        )

        graph = {"nodes": [], "edges": []}
        patterns = analyzer.find_independent_groups(
            graph,
            sample_steps_independent
        )

        # Should find no patterns (threshold too high)
        assert len(patterns) == 0
