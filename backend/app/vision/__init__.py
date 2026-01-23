"""
Vision module for LEGO assembly analysis.

Provides VLM-based state detection and guidance generation.

REFACTORED: Now uses simplified VLM-only approach with DirectStepAnalyzer.
Legacy StateAnalyzer kept for backward compatibility with RAG pipeline.
"""

from .direct_step_analyzer import DirectStepAnalyzer, get_direct_step_analyzer
from .guidance_generator import GuidanceGenerator, get_guidance_generator
from .state_analyzer import StateAnalyzer, get_state_analyzer  # Legacy

__all__ = [
    # New VLM-only approach (recommended)
    'DirectStepAnalyzer',
    'get_direct_step_analyzer',
    'GuidanceGenerator',
    'get_guidance_generator',
    # Legacy (for RAG pipeline compatibility)
    'StateAnalyzer',
    'get_state_analyzer',
]



