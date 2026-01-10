"""
Build Memory System for Context-Aware Step Extraction (Phase 1).
Maintains sliding window and long-term memory to provide context to VLM during extraction.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
from loguru import logger


@dataclass
class StepSummary:
    """Compact summary of a single step."""
    step_number: int
    action_summary: str  # e.g., "Attached 2 yellow bricks to base"
    parts_added: List[str]  # e.g., ["yellow 2x4 brick", "red 1x1 brick"]
    subassembly: Optional[str]  # e.g., "base_structure", "wheel_assembly"

    def to_context_string(self) -> str:
        """Convert to compact string for VLM context."""
        parts = ", ".join(self.parts_added[:3])  # Limit to 3 parts
        if len(self.parts_added) > 3:
            parts += f" (+{len(self.parts_added)-3} more)"

        return f"Step {self.step_number}: {self.action_summary} ({parts})"


@dataclass
class Subassembly:
    """Represents a subassembly being built."""
    name: str  # e.g., "wheel_assembly"
    description: str  # e.g., "4-wheel chassis with axles"
    steps: List[int]  # e.g., [9, 10, 11, 12, 13, 14, 15]
    status: str  # "in_progress" or "completed"

    def to_context_string(self) -> str:
        """Convert to compact string for VLM context."""
        step_range = f"{self.steps[0]}-{self.steps[-1]}" if len(self.steps) > 1 else str(self.steps[0])
        status_icon = "✓" if self.status == "completed" else "→"
        return f"{status_icon} {self.name}: {self.description} (steps {step_range})"


class SlidingWindowMemory:
    """
    Option A: Maintains a sliding window of recent steps.
    Provides immediate context for current step.
    """

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Number of previous steps to remember (default: 5)
        """
        self.window_size = window_size
        self.window: deque[StepSummary] = deque(maxlen=window_size)

    def add_step(self, step_summary: StepSummary):
        """Add a new step summary (automatically evicts oldest if full)."""
        self.window.append(step_summary)

    def get_context(self) -> str:
        """
        Get context string for VLM prompt.

        Returns compact summary of recent steps:
        "Recent steps:
         - Step 3: Attached yellow brick to base (yellow 2x4 brick)
         - Step 4: Added red piece on top (red 1x1 brick)
         - Step 5: Started wheel assembly (wheel, axle)"
        """
        if not self.window:
            return "This is the first step."

        context_lines = ["Recent steps:"]
        for summary in self.window:
            context_lines.append(f" - {summary.to_context_string()}")

        return "\n".join(context_lines)

    def get_token_estimate(self) -> int:
        """Estimate tokens used by this context (~300 tokens per step summary)."""
        return len(self.window) * 300


class LongTermMemory:
    """
    Option C: Maintains long-term build state.
    Tracks overall progress, subassemblies, and structure.
    """

    def __init__(self, main_build: str):
        """
        Args:
            main_build: What is being built (e.g., "Fire Truck Set #6454922")
        """
        self.main_build = main_build
        self.completed_subassemblies: List[Subassembly] = []
        self.current_subassembly: Optional[Subassembly] = None
        self.total_steps_processed: int = 0

    def start_subassembly(self, name: str, description: str, starting_step: int):
        """Start tracking a new subassembly."""
        if self.current_subassembly:
            # Complete previous subassembly
            self.complete_current_subassembly()

        self.current_subassembly = Subassembly(
            name=name,
            description=description,
            steps=[starting_step],
            status="in_progress"
        )

    def add_step_to_current_subassembly(self, step_number: int):
        """Add a step to the current subassembly."""
        if self.current_subassembly:
            self.current_subassembly.steps.append(step_number)

    def complete_current_subassembly(self):
        """Mark current subassembly as completed."""
        if self.current_subassembly:
            self.current_subassembly.status = "completed"
            self.completed_subassemblies.append(self.current_subassembly)
            self.current_subassembly = None

    def get_context(self) -> str:
        """
        Get context string for VLM prompt.

        Returns high-level build state:
        "Building: Fire Truck Set #6454922
         Completed subassemblies:
         ✓ base_structure: Red base with support columns (steps 1-8)
         ✓ wheel_assembly: 4-wheel chassis (steps 9-15)

         Current work:
         → cabin: Building driver cabin (steps 16-20)"
        """
        lines = [f"Building: {self.main_build}"]

        if self.completed_subassemblies:
            lines.append("\nCompleted subassemblies:")
            for sub in self.completed_subassemblies[-3:]:  # Last 3 to save tokens
                lines.append(f" {sub.to_context_string()}")

            if len(self.completed_subassemblies) > 3:
                lines.append(f" ... and {len(self.completed_subassemblies)-3} more")

        if self.current_subassembly:
            lines.append("\nCurrent work:")
            lines.append(f" {self.current_subassembly.to_context_string()}")

        return "\n".join(lines)

    def get_token_estimate(self) -> int:
        """Estimate tokens used by this context (~500 tokens total)."""
        return 500


class BuildMemory:
    """
    Coordinator for all memory systems.
    Provides unified interface for context-aware extraction.
    """

    def __init__(self, main_build: str, window_size: int = 5):
        self.sliding_window = SlidingWindowMemory(window_size)
        self.long_term = LongTermMemory(main_build)

    def add_step(self, step_data: Dict[str, Any]):
        """
        Process a completed step and update all memory systems.

        Args:
            step_data: Extracted step information including:
                - step_number
                - parts_required
                - actions
                - subassembly_hint (from VLM)
        """
        step_number = step_data.get("step_number", self.long_term.total_steps_processed + 1)

        # Create summary for sliding window
        summary = StepSummary(
            step_number=step_number,
            action_summary=self._summarize_actions(step_data.get("actions", [])),
            parts_added=[p.get("description", "") for p in step_data.get("parts_required", [])],
            subassembly=step_data.get("subassembly_hint", {}).get("name") if isinstance(step_data.get("subassembly_hint"), dict) else None
        )
        self.sliding_window.add_step(summary)

        # Update long-term memory
        subassembly_hint = step_data.get("subassembly_hint")
        if subassembly_hint and isinstance(subassembly_hint, dict):
            if subassembly_hint.get("is_new_subassembly"):
                # Starting a new subassembly
                self.long_term.start_subassembly(
                    name=subassembly_hint.get("name", f"subassembly_{step_number}"),
                    description=subassembly_hint.get("description", ""),
                    starting_step=step_number
                )
            elif self.long_term.current_subassembly:
                # Continuing current subassembly
                self.long_term.add_step_to_current_subassembly(step_number)
        elif not self.long_term.current_subassembly:
            # No subassembly context, create a default one
            if self.long_term.total_steps_processed == 0:
                self.long_term.start_subassembly(
                    name="main_assembly",
                    description="Main assembly sequence",
                    starting_step=step_number
                )
            elif self.long_term.current_subassembly:
                self.long_term.add_step_to_current_subassembly(step_number)

        self.long_term.total_steps_processed += 1

    def get_full_context(self) -> Dict[str, Any]:
        """
        Get all context for next step extraction.

        Returns:
            {
                "sliding_window": "Recent steps: ...",
                "long_term": "Building: Fire Truck...",
                "token_estimate": 2000
            }
        """
        return {
            "sliding_window": self.sliding_window.get_context(),
            "long_term": self.long_term.get_context(),
            "token_estimate": (
                self.sliding_window.get_token_estimate() +
                self.long_term.get_token_estimate()
            )
        }

    def _summarize_actions(self, actions: List[Dict]) -> str:
        """Create a concise summary of actions."""
        if not actions:
            return "No actions specified"

        if len(actions) == 1:
            a = actions[0]
            action_verb = a.get('action_verb', 'modify')
            target = a.get('target', 'parts')
            return f"{action_verb.capitalize()} {target}"
        else:
            verbs = [a.get('action_verb', 'modify') for a in actions]
            unique_verbs = list(set(verbs))
            return f"{len(actions)} actions: {', '.join(unique_verbs)}"
