"""
Detection Strategies for Post-Processing Analysis.

This package contains three complementary detection strategies:
1. Part Similarity Clustering - Identifies repeated structures
2. Dependency Analysis - Finds independent/parallel assembly groups
3. Spatial-Temporal Pattern Mining - Detects progressive builds
"""

from .part_similarity import PartSimilarityClusterer
from .dependency_analysis import DependencyAnalyzer
from .spatial_temporal import SpatialTemporalPatternMiner

__all__ = [
    "PartSimilarityClusterer",
    "DependencyAnalyzer",
    "SpatialTemporalPatternMiner"
]
