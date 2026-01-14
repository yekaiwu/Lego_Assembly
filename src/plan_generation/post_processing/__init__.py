"""
Post-Processing Subassembly Analysis Module.

This module provides post-processing analysis capabilities to discover implicit
multi-step assembly patterns that forward-only VLM detection misses.
"""

from .analyzer import PostProcessingSubassemblyAnalyzer

__all__ = ["PostProcessingSubassemblyAnalyzer"]
