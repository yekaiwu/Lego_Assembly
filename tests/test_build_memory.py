"""
Unit tests for build_memory.py (Phase 1 - Context-Aware Extraction).
Tests sliding window memory, long-term memory, and build memory coordination.
"""

import pytest
from src.vision_processing.build_memory import (
    BuildMemory, SlidingWindowMemory, LongTermMemory, StepSummary, Subassembly
)


def test_sliding_window_basic():
    """Test sliding window maintains correct size."""
    window = SlidingWindowMemory(window_size=3)

    # Add 5 steps
    for i in range(1, 6):
        summary = StepSummary(
            step_number=i,
            action_summary=f"Action {i}",
            parts_added=[f"part_{i}"],
            subassembly=None
        )
        window.add_step(summary)

    # Should only have last 3
    assert len(window.window) == 3
    assert window.window[0].step_number == 3
    assert window.window[-1].step_number == 5


def test_sliding_window_context_string():
    """Test sliding window generates correct context string."""
    window = SlidingWindowMemory(window_size=5)

    # Add a few steps
    for i in range(1, 4):
        summary = StepSummary(
            step_number=i,
            action_summary=f"Attached brick {i}",
            parts_added=[f"red 2x4 brick"],
            subassembly=None
        )
        window.add_step(summary)

    context = window.get_context()

    # Check context contains expected information
    assert "Recent steps:" in context
    assert "Step 1:" in context
    assert "Step 3:" in context
    assert "Attached brick" in context


def test_sliding_window_empty():
    """Test sliding window with no steps."""
    window = SlidingWindowMemory(window_size=5)

    context = window.get_context()
    assert context == "This is the first step."


def test_long_term_memory_subassembly_tracking():
    """Test long-term memory tracks subassemblies correctly."""
    memory = LongTermMemory("Fire Truck")

    # Start first subassembly
    memory.start_subassembly("base", "Base structure", starting_step=1)
    memory.add_step_to_current_subassembly(2)
    memory.add_step_to_current_subassembly(3)

    # Complete and start next
    memory.complete_current_subassembly()
    memory.start_subassembly("wheels", "Wheel assembly", starting_step=4)

    # Check state
    assert len(memory.completed_subassemblies) == 1
    assert memory.completed_subassemblies[0].name == "base"
    assert memory.completed_subassemblies[0].steps == [1, 2, 3]
    assert memory.completed_subassemblies[0].status == "completed"
    assert memory.current_subassembly.name == "wheels"
    assert memory.current_subassembly.status == "in_progress"


def test_long_term_memory_context_string():
    """Test long-term memory generates correct context string."""
    memory = LongTermMemory("Fire Truck Set #6454922")

    # Add completed subassemblies
    memory.start_subassembly("base", "Red base with support columns", starting_step=1)
    memory.add_step_to_current_subassembly(2)
    memory.add_step_to_current_subassembly(3)
    memory.complete_current_subassembly()

    memory.start_subassembly("wheels", "4-wheel chassis", starting_step=4)
    memory.add_step_to_current_subassembly(5)

    context = memory.get_context()

    # Check context contains expected information
    assert "Building: Fire Truck Set #6454922" in context
    assert "Completed subassemblies:" in context
    assert "base" in context
    assert "Red base with support columns" in context
    assert "Current work:" in context
    assert "wheels" in context


def test_build_memory_integration():
    """Test full BuildMemory system with both sliding window and long-term memory."""
    memory = BuildMemory("Fire Truck", window_size=5)

    # Add several steps with subassembly hints
    for i in range(1, 8):
        step_data = {
            "step_number": i,
            "parts_required": [{"description": f"part_{i}"}],
            "actions": [{"action_verb": "attach", "target": f"part_{i}"}],
        }

        # Add subassembly hints for some steps
        if i == 1:
            step_data["subassembly_hint"] = {
                "is_new_subassembly": True,
                "name": "base",
                "description": "Base structure"
            }
        elif i == 5:
            step_data["subassembly_hint"] = {
                "is_new_subassembly": True,
                "name": "wheels",
                "description": "Wheel assembly"
            }

        memory.add_step(step_data)

    # Check context
    context = memory.get_full_context()
    assert "sliding_window" in context
    assert "long_term" in context
    assert "token_estimate" in context

    # Check sliding window has recent steps
    assert "Step 7" in context["sliding_window"] or "Step 6" in context["sliding_window"]

    # Check long-term memory has build info
    assert "Fire Truck" in context["long_term"]


def test_build_memory_without_subassembly_hints():
    """Test BuildMemory handles steps without subassembly hints."""
    memory = BuildMemory("Test Build", window_size=3)

    # Add steps without subassembly hints
    for i in range(1, 4):
        step_data = {
            "step_number": i,
            "parts_required": [{"description": f"part_{i}"}],
            "actions": [{"action_verb": "attach", "target": f"part_{i}"}],
            # No subassembly_hint
        }
        memory.add_step(step_data)

    # Should still work and create default subassembly
    context = memory.get_full_context()
    assert "sliding_window" in context
    assert "long_term" in context

    # Check that a default subassembly was created
    assert memory.long_term.current_subassembly is not None
    assert memory.long_term.current_subassembly.name == "main_assembly"


def test_step_summary_to_context_string():
    """Test StepSummary conversion to context string."""
    summary = StepSummary(
        step_number=5,
        action_summary="Attached wheel to axle",
        parts_added=["wheel", "axle", "connector"],
        subassembly="wheel_assembly"
    )

    context_str = summary.to_context_string()

    assert "Step 5:" in context_str
    assert "Attached wheel to axle" in context_str
    assert "wheel" in context_str


def test_step_summary_with_many_parts():
    """Test StepSummary limits parts in context string."""
    summary = StepSummary(
        step_number=10,
        action_summary="Complex assembly",
        parts_added=["part1", "part2", "part3", "part4", "part5"],
        subassembly=None
    )

    context_str = summary.to_context_string()

    # Should show first 3 parts plus indication of more
    assert "part1" in context_str
    assert "part2" in context_str
    assert "part3" in context_str
    assert "(+2 more)" in context_str


def test_subassembly_to_context_string():
    """Test Subassembly conversion to context string."""
    subassembly = Subassembly(
        name="wheel_assembly",
        description="4-wheel chassis with axles",
        steps=[9, 10, 11, 12, 13, 14, 15],
        status="completed"
    )

    context_str = subassembly.to_context_string()

    assert "wheel_assembly" in context_str
    assert "4-wheel chassis with axles" in context_str
    assert "steps 9-15" in context_str
    assert "✓" in context_str  # Completed marker


def test_token_budget_manager():
    """Test token budget management."""
    from src.vision_processing.token_budget import TokenBudgetManager

    budget = TokenBudgetManager(max_tokens=10000)

    # Check small request
    result = budget.check_budget(
        image_count=1,
        sliding_window_size=3,
        long_term_memory_size=500,
        prompt_size=500
    )
    assert result["fits"] == True
    assert result["estimated_usage"] < result["available"]

    # Check oversized request
    result = budget.check_budget(
        image_count=1,
        sliding_window_size=20,  # Too large
        long_term_memory_size=500,
        prompt_size=500
    )
    assert result["fits"] == False
    assert "reduce_window_to" in result["recommendations"]


def test_token_budget_auto_adjust():
    """Test automatic window size adjustment."""
    from src.vision_processing.token_budget import TokenBudgetManager

    budget = TokenBudgetManager(max_tokens=10000)

    # Current window is too large for budget
    current_window = 20
    estimated_tokens = 15000  # Exceeds budget

    new_window = budget.auto_adjust_window_size(current_window, estimated_tokens)

    # Should reduce window size
    assert new_window < current_window
    assert new_window >= 2  # Minimum window size


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
