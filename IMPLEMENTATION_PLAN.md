# LEGO Assembly System - Context-Aware Processing Implementation Plan

**Version:** 2.0 (UPDATED)
**Date:** 2026-01-10 (Updated based on current implementation)
**Target:** Phase 0 & Phase 1 - Document Understanding and Context-Aware Extraction
**Estimated Remaining Effort:** 1-2 days

---

## ⚠️ CURRENT IMPLEMENTATION STATUS

### ✅ Already Implemented (Phase 2 - Hierarchical Graph)

You have **already implemented** the hierarchical dependency graph system! Here's what exists:

**File:** `src/plan_generation/graph_builder.py`
- ✅ **SubassemblyDetector**: Detects subassemblies using VLM + heuristics
- ✅ **StepStateTracker**: Tracks assembly state progression at each step
- ✅ **GraphBuilder**: Builds hierarchical graph with nodes (parts, subassemblies, model) and edges
- ✅ **Parent-child relationships**: Properly connects subassemblies to parents
- ✅ **Layer assignment**: Calculates depth from root
- ✅ **Metadata tracking**: Total parts, subassemblies, max depth

**File:** `backend/app/graph/graph_manager.py`
- ✅ **Query interface**: Find nodes, children, parents, paths
- ✅ **Subassembly operations**: Find subassemblies containing parts
- ✅ **Step estimation**: Estimate current step from detected subassemblies

**Integrated in:** `main.py` Step 4
- ✅ Builds hierarchical graph after step extraction
- ✅ Saves graph with summary for debugging

### ❌ What's Missing (Phases 0 & 1)

**Phase 0: Document Understanding**
- ❌ No filtering of irrelevant pages (cover, ads, inventory)
- ❌ No user confirmation before processing
- ❌ All pages processed, including non-instruction pages

**Phase 1: Context-Aware Extraction**
- ❌ No sliding window memory during extraction
- ❌ No long-term memory tracking during extraction
- ❌ No context passed to VLM (steps extracted in isolation)
- ❌ Subassemblies detected AFTER extraction (not during)

**Current extraction** (stateless):
```python
for step_image in images:
    result = vlm.extract(step_image)  # No context!
```

**Needed** (context-aware):
```python
memory = BuildMemory()
for step_image in images:
    context = memory.get_context()  # Get previous steps + build state
    result = vlm.extract(step_image, context)  # Extract with context
    memory.update(result)  # Update memory
```

### 🎯 Focus of This Plan

This updated plan focuses on implementing **Phases 0 & 1 only**, since Phase 2 is already done.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current Implementation Analysis](#current-implementation-analysis)
3. [Problem Statement](#problem-statement)
4. [Proposed Solution](#proposed-solution)
5. [Implementation Details](#implementation-details)
6. [Context Window Management](#context-window-management)
7. [Testing Strategy](#testing-strategy)
8. [Success Criteria](#success-criteria)

---

## Executive Summary

### Goal
Enhance the LEGO Assembly System with context-aware extraction, enabling:
- **Filter irrelevant pages** before processing (Phase 0)
- **Understand build progression** during extraction (Phase 1)
- ✅ **Hierarchical subassembly graph** (Phase 2 - ALREADY DONE!)

### What's Different from Original Plan

**Original plan assumed starting from scratch. Reality: Phase 2 already exists!**

### Updated Approach
Implement **2 missing phases** to work with existing hierarchical graph:

**Phase 0:** Document Understanding (NEW - Priority 1)
- Analyze entire PDF to identify relevant instruction pages
- Filter out covers, ads, parts inventory
- Get user confirmation before processing
- **Saves ~10-15% API calls**

**Phase 1:** Context-Aware Step Extraction (NEW - Priority 2)
- **Option A:** Sliding window (5 previous steps)
- **Option C:** Long-term memory (build state tracker)
- Pass context to VLM during extraction
- **Improves subassembly detection quality**

**Phase 2:** Hierarchical Graph (EXISTING - Already Done ✅)
- `SubassemblyDetector` detects subassemblies
- `GraphBuilder` creates hierarchical structure
- No changes needed, works with enhanced Phase 1 output

### Key Changes
- **Before:** 52 pages → 52 isolated extractions → hierarchical graph
- **After:** PDF → filter 45 pages → 45 context-aware extractions → enhanced hierarchical graph

### Integration Point

The existing `GraphBuilder` will benefit from context-aware extraction because:
1. Extracted steps will have better `subassembly_hint` fields
2. `context_references` will be more accurate
3. Subassembly detection heuristics will have richer data
4. **No changes to GraphBuilder needed!**

---

## Current Implementation Analysis

### Existing Pipeline (main.py)

```
Step 1: Manual Processing
  └─ Extract all 52 pages as images ❌ (includes covers, ads)

Step 2: VLM Step Extraction ❌ (stateless)
  └─ For each image: extract step info without context

Step 3: Old Dependency Graph
  └─ Flat dependency tracking (being replaced)

Step 4: Hierarchical Graph ✅ (WORKING!)
  ├─ SubassemblyDetector: Uses heuristics to find subassemblies
  ├─ StepStateTracker: Tracks progressive build state
  ├─ GraphBuilder: Creates nodes (parts, subassemblies, model)
  └─ Saves hierarchical graph with metadata

Step 5: 3D Plan Generation
  └─ Uses hierarchical graph
```

### What Works Well

✅ **Hierarchical graph structure** (src/plan_generation/graph_builder.py)
- Detects subassemblies with ~70% accuracy using heuristics
- Builds proper parent-child relationships
- Tracks completeness markers
- Assigns layers (depth)

✅ **Graph querying** (backend/app/graph/graph_manager.py)
- Rich query API
- Step estimation from subassemblies
- Path finding, parent/child traversal

### What Needs Improvement

❌ **Page filtering**: Processes all 52 pages, including non-instructions
❌ **Context-less extraction**: Each step sees only current image
❌ **Subassembly hints**: Detected after extraction, not during
❌ **Inter-step references**: Not captured during extraction

### Impact of Adding Context-Aware Extraction

Adding Phases 0 & 1 will **enhance** the existing hierarchical graph:

**Before (current):**
```
Step 5: VLM extracts "Add wheel"
  ↓ (no context)
GraphBuilder heuristic: "This might be a wheel assembly"
  ↓ (guess based on keywords)
Subassembly created with ~60% confidence
```

**After (with context):**
```
Step 5: VLM extracts "Add wheel" WITH CONTEXT
  Context: "Previous steps built axle support"
  ↓
VLM returns: subassembly_hint = {
    "is_new_subassembly": true,
    "name": "wheel_assembly",
    "continues_from": ["axle_support"]
  }
  ↓
GraphBuilder: "VLM says this is a wheel assembly!"
  ↓
Subassembly created with ~90% confidence
```

---

## Problem Statement

### Current System Limitations

#### 1. **Processes Irrelevant Pages**
```
Input: 52-page PDF
Problem:
- Pages 1-5: Cover, warnings, parts inventory (NOT instructions)
- Pages 51-52: Advertisements, alternate builds (NOT main build)
Reality: Only pages 6-50 (45 pages) are actual assembly steps

Impact: Wasted API calls, polluted data, confused downstream tasks
```

#### 2. **No Context Between Steps**
```python
# Current approach (stateless)
step_1 = extract("image_1.png")  # "Add red base plate"
step_2 = extract("image_2.png")  # "Add yellow brick" - doesn't know about step 1!
step_10 = extract("image_10.png") # "Attach wheel assembly" - what wheel assembly?
```

**Problems:**
- Cannot resolve references like "attach to the base from step 4"
- Cannot identify when a subassembly starts/ends
- Cannot understand build progression
- Dependencies are guessed, not understood

#### 3. **Missing Hierarchical Structure**
```
Steps 1-8: Building base structure (subassembly A)
Steps 9-15: Building wheel assembly (subassembly B)
Steps 16-20: Combining A + B

Current system: Sees 20 independent steps
Reality: Hierarchical structure with subassemblies
```

**Impact:**
- Dependency graph is flat, not hierarchical
- Cannot identify parallel buildable components
- Cannot optimize assembly order
- Cannot provide modular guidance

---

## Current System Architecture

### Existing Pipeline (main.py)

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Manual Processing                                  │
│ - Download PDF                                               │
│ - Extract all pages as images (52 images)                  │
│ - Detect "step boundaries" (naive: 1 page = 1 step)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: VLM Step Extraction                                │
│ - For each image (stateless):                               │
│   • Call VLM with single image                              │
│   • Extract: parts, actions, spatial relationships          │
│   • Save JSON (no context from previous steps)              │
│ - Batch all results into extracted_steps.json               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Plan Generation                                    │
│ - Build dependency graph (guesses dependencies)             │
│ - Generate assembly plan                                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Files
- `main.py`: Orchestrator
- `src/vision_processing/vlm_step_extractor.py`: VLM extraction logic
- `src/api/gemini_api.py`: Gemini API client
- `src/plan_generation/graph_builder.py`: Dependency graph builder

---

## Proposed Solution

### New 3-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 0: Document Understanding (NEW)                       │
│ - Analyze entire PDF with VLM                               │
│ - Identify: main build, instruction pages, irrelevant pages │
│ - User confirmation                                          │
│ - Filter to only relevant pages                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Context-Aware Step Extraction (ENHANCED)           │
│                                                              │
│ Memory System:                                               │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ Option A: Sliding Window (5-step context)              │ │
│ │ - Stores: Last 5 step summaries                        │ │
│ │ - Purpose: Immediate context for current step          │ │
│ └────────────────────────────────────────────────────────┘ │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ Option C: Long-term Memory (build state)               │ │
│ │ - Stores: Completed subassemblies, current work        │ │
│ │ - Purpose: Overall build understanding                 │ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
│ For each step N:                                             │
│ 1. Load sliding window context (steps N-5 to N-1)          │
│ 2. Load long-term memory (build state)                     │
│ 3. Extract step N with full context                        │
│ 4. Update both memories                                     │
│ 5. Detect subassembly transitions                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Subassembly Verification (NEW)                     │
│                                                              │
│ Two-Pass Analysis (Option B):                               │
│ Pass 1: Collected all context-aware step extractions        │
│ Pass 2: Analyze entire sequence:                            │
│   • Identify subassembly boundaries                         │
│   • Group steps into modules                                │
│   • Validate inter-step references                          │
│   • Build hierarchical dependency graph                     │
│   • Detect parallel vs. sequential builds                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Plan Generation (EXISTING - Enhanced Input)        │
│ - Receives hierarchical structure                           │
│ - Generates optimized assembly plan                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Phase 0: Document Understanding

**File:** `src/vision_processing/document_analyzer.py` (NEW)

#### Purpose
Filter irrelevant pages and understand document structure before detailed processing.

#### Implementation

```python
class DocumentAnalyzer:
    """
    Analyzes LEGO manual PDFs to identify:
    - Main build (what is being assembled)
    - Instruction pages vs. cover/ads/inventory
    - Total number of actual assembly steps
    - Presence of alternate builds
    """

    def __init__(self, vlm_client):
        self.vlm_client = vlm_client

    def analyze_pdf(self, pdf_path: Path) -> DocumentMetadata:
        """
        Perform high-level analysis of entire PDF.

        Process:
        1. Sample pages from PDF (first 5, middle 3, last 5)
        2. Send to VLM with document understanding prompt
        3. Identify document structure
        4. Classify page ranges

        Returns:
            DocumentMetadata with:
            - main_build: str (e.g., "Fire Truck Set #6454922")
            - instruction_page_ranges: List[Tuple[int, int]]
            - cover_pages: List[int]
            - inventory_pages: List[int]
            - ad_pages: List[int]
            - total_steps: int
            - has_alternate_builds: bool
        """
        pass

    def classify_pages(self, all_pages: List[Path]) -> PageClassification:
        """
        Classify each page as:
        - 'cover': Cover page, warnings, copyright
        - 'inventory': Parts list/inventory
        - 'instruction': Assembly instruction
        - 'advertisement': Other builds, ads
        - 'reference': Final product photos, troubleshooting

        Uses heuristics + VLM for ambiguous pages.
        """
        pass

    def extract_relevant_pages(
        self,
        all_pages: List[Path],
        classification: PageClassification
    ) -> List[Path]:
        """Filter to only instruction pages."""
        return [p for p in all_pages
                if classification[p] == 'instruction']

    def get_user_confirmation(
        self,
        metadata: DocumentMetadata
    ) -> bool:
        """
        Present findings to user for confirmation:

        Example:
        ┌─────────────────────────────────────────────┐
        │ Document Analysis                            │
        ├─────────────────────────────────────────────┤
        │ Build: Fire Truck Set #6454922              │
        │ Total Pages: 52                              │
        │ Instruction Pages: 45 (pages 6-50)          │
        │ Filtered Out:                                │
        │   - Cover/Intro: pages 1-5                  │
        │   - Advertisements: pages 51-52              │
        │                                              │
        │ Proceed with processing 45 steps? (y/n)     │
        └─────────────────────────────────────────────┘
        """
        pass
```

#### VLM Prompt for Document Understanding

```
You are analyzing a LEGO instruction manual PDF.

I will show you sample pages from this document. Please identify:

1. **Main Build**: What is being built? (e.g., "Fire Truck", "Star Wars X-Wing")
2. **Set Number**: If visible (e.g., "6454922")
3. **Page Types**: For each page shown, classify as:
   - COVER: Title page, copyright, warnings
   - INVENTORY: Parts list showing all pieces
   - INSTRUCTION: Step-by-step assembly instructions
   - ADVERTISEMENT: Other sets, promotional content
   - REFERENCE: Final product photos, alternate angles

4. **Instruction Page Range**: Which pages contain assembly steps?
5. **Total Steps**: Approximately how many assembly steps are there?

Return JSON format:
{
  "main_build": "Fire Truck",
  "set_number": "6454922",
  "page_classifications": {
    "1-5": "COVER/INVENTORY",
    "6-50": "INSTRUCTION",
    "51-52": "ADVERTISEMENT"
  },
  "instruction_pages": [6, 50],
  "estimated_steps": 45
}
```

#### Integration into main.py

```python
# After PDF extraction, before step extraction
def main(...):
    # ... existing PDF download/extraction code ...

    # NEW: Phase 0
    if not checkpoint.is_step_complete("document_analysis"):
        logger.info("Phase 0: Analyzing document structure...")

        doc_analyzer = DocumentAnalyzer(vlm_client)
        doc_metadata = doc_analyzer.analyze_pdf(pdf_path)

        logger.info(f"Identified: {doc_metadata.main_build}")
        logger.info(f"Instruction pages: {doc_metadata.instruction_page_ranges}")
        logger.info(f"Total steps: {doc_metadata.total_steps}")

        # Get user confirmation
        if not doc_analyzer.get_user_confirmation(doc_metadata):
            logger.info("User cancelled. Exiting.")
            return

        # Filter to only relevant pages
        relevant_pages = doc_analyzer.extract_relevant_pages(
            all_pages, doc_metadata.page_classification
        )

        logger.info(f"Filtered: {len(all_pages)} → {len(relevant_pages)} pages")

        checkpoint.save("document_analysis", {
            "metadata": doc_metadata.to_dict(),
            "relevant_pages": relevant_pages
        })
    else:
        # Load from checkpoint
        checkpoint_data = checkpoint.load()
        doc_metadata = DocumentMetadata.from_dict(
            checkpoint_data["metadata"]
        )
        relevant_pages = checkpoint_data["relevant_pages"]

    # Continue with Phase 1 using only relevant_pages
    # ... existing step extraction code ...
```

---

### Phase 1: Context-Aware Step Extraction

**Files to Modify:**
- `src/vision_processing/vlm_step_extractor.py` (ENHANCE)
- `src/vision_processing/build_memory.py` (NEW)
- `src/api/gemini_api.py` (ENHANCE prompts)

#### Memory Systems Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ BuildMemory (Coordinator)                                    │
│                                                              │
│ ┌──────────────────────┐  ┌──────────────────────────────┐ │
│ │ SlidingWindowMemory  │  │ LongTermMemory               │ │
│ │ (Option A)           │  │ (Option C)                   │ │
│ │                      │  │                              │ │
│ │ Stores:              │  │ Stores:                      │ │
│ │ - Last 5 steps       │  │ - Completed subassemblies    │ │
│ │ - Step summaries     │  │ - Current subassembly        │ │
│ │ - Recent actions     │  │ - Overall build state        │ │
│ │                      │  │ - Main build description     │ │
│ │ Purpose:             │  │                              │ │
│ │ - Immediate context  │  │ Purpose:                     │ │
│ │ - "What just         │  │ - Big picture understanding  │ │
│ │   happened?"         │  │ - "Where are we in build?"   │ │
│ │                      │  │                              │ │
│ │ Token usage:         │  │ Token usage:                 │ │
│ │ ~1,500 tokens        │  │ ~500 tokens                  │ │
│ └──────────────────────┘  └──────────────────────────────┘ │
│                                                              │
│ Total context per step: ~2,000 tokens                       │
│ (Leaves plenty of room in context window)                   │
└─────────────────────────────────────────────────────────────┘
```

#### Implementation: BuildMemory System

**File:** `src/vision_processing/build_memory.py` (NEW)

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

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
        step_number = step_data["step_number"]

        # Create summary for sliding window
        summary = StepSummary(
            step_number=step_number,
            action_summary=self._summarize_actions(step_data["actions"]),
            parts_added=[p["description"] for p in step_data.get("parts_required", [])],
            subassembly=step_data.get("subassembly_hint")
        )
        self.sliding_window.add_step(summary)

        # Update long-term memory
        subassembly_hint = step_data.get("subassembly_hint")
        if subassembly_hint:
            if subassembly_hint.get("is_new_subassembly"):
                # Starting a new subassembly
                self.long_term.start_subassembly(
                    name=subassembly_hint["name"],
                    description=subassembly_hint["description"],
                    starting_step=step_number
                )
            elif self.long_term.current_subassembly:
                # Continuing current subassembly
                self.long_term.add_step_to_current_subassembly(step_number)

        self.long_term.total_steps_processed += 1

    def get_full_context(self) -> Dict[str, str]:
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
            return f"{a['action_verb'].capitalize()} {a['target']}"
        else:
            verbs = [a['action_verb'] for a in actions]
            return f"{len(actions)} actions: {', '.join(set(verbs))}"
```

#### Enhanced VLM Step Extractor

**File:** `src/vision_processing/vlm_step_extractor.py` (MODIFY)

```python
class VLMStepExtractor:
    """Enhanced with context awareness."""

    def __init__(self):
        # ... existing initialization ...
        self.build_memory: Optional[BuildMemory] = None

    def initialize_memory(self, main_build: str, window_size: int = 5):
        """Initialize memory systems before extraction."""
        self.build_memory = BuildMemory(main_build, window_size)

    def extract_step(
        self,
        image_paths: List[str],
        step_number: Optional[int] = None,
        use_primary: bool = True,
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract with full context awareness.

        NEW: Includes context from sliding window and long-term memory.
        """
        if use_primary:
            # Get context from memory systems
            context = None
            if self.build_memory:
                context = self.build_memory.get_full_context()

            result = self._extract_with_vlm_and_context(
                self.primary_vlm,
                image_paths,
                step_number,
                context,
                cache_context
            )

            # Update memory with extraction result
            if self.build_memory and "error" not in result:
                self.build_memory.add_step(result)

            return result
        else:
            return self._extract_with_fallback(
                image_paths, step_number, cache_context
            )

    def _extract_with_vlm_and_context(
        self,
        vlm_name: str,
        image_paths: List[str],
        step_number: Optional[int],
        context: Optional[Dict[str, str]],
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract using a specific VLM with context.

        NEW: Passes context to VLM for context-aware extraction.
        """
        client = self.clients.get(vlm_name)

        if not client:
            raise ValueError(f"Unknown VLM: {vlm_name}")

        logger.info(f"Extracting step info using {vlm_name} with context")

        try:
            # Build enhanced prompt with context
            prompt = self._build_context_aware_prompt(step_number, context)

            # Call client with enhanced prompt
            result = client.extract_step_info_with_context(
                image_paths,
                step_number,
                prompt,
                cache_context=cache_context
            )

            if self._validate_extraction(result):
                logger.info(f"Successfully extracted step info using {vlm_name}")
                return result
            else:
                logger.warning(f"Extraction from {vlm_name} failed validation")
                return {"error": "Validation failed", "raw_result": result}

        except Exception as e:
            logger.error(f"Error extracting with {vlm_name}: {e}")
            return {"error": str(e)}

    def _build_context_aware_prompt(
        self,
        step_number: Optional[int],
        context: Optional[Dict[str, str]]
    ) -> str:
        """
        Build enhanced prompt with context.

        Includes:
        - Sliding window (recent steps)
        - Long-term memory (build state)
        - Instructions to relate to previous work
        """
        step_context = f"Step {step_number}: " if step_number else ""

        prompt_parts = []

        # Add long-term context (high-level overview)
        if context and context.get("long_term"):
            prompt_parts.append(f"""
BUILD CONTEXT:
{context['long_term']}
""")

        # Add sliding window context (recent steps)
        if context and context.get("sliding_window"):
            prompt_parts.append(f"""
{context['sliding_window']}
""")

        # Main extraction instructions
        prompt_parts.append(f"""
CURRENT TASK:
Analyze {step_context}this LEGO instruction image.

Extract detailed information and return ONLY valid JSON:

{{
  "step_number": <number or null>,
  "parts_required": [
    {{
      "description": "part description",
      "color": "color name",
      "shape": "brick type and dimensions",
      "part_id": "LEGO part ID if visible",
      "quantity": <number>
    }}
  ],
  "existing_assembly": "description of already assembled parts shown",
  "new_parts_to_add": [
    "description of each new part being added in this step"
  ],
  "actions": [
    {{
      "action_verb": "attach|connect|place|align|rotate",
      "target": "what is being attached",
      "destination": "where it's being attached",
      "orientation": "directional cues"
    }}
  ],
  "spatial_relationships": {{
    "position": "top|bottom|left|right|front|back|center",
    "rotation": "rotation description if any",
    "alignment": "alignment instructions"
  }},
  "dependencies": "which previous steps are prerequisites",
  "notes": "any special instructions or warnings",

  "subassembly_hint": {{
    "is_new_subassembly": true/false,
    "name": "descriptive name if new (e.g., 'wheel_assembly')",
    "description": "what is being built (e.g., '4-wheel chassis')",
    "continues_previous": true/false
  }},

  "context_references": {{
    "references_previous_steps": true/false,
    "which_steps": [list of step numbers referenced],
    "reference_description": "what is being referenced (e.g., 'the base from step 4')"
  }}
}}

IMPORTANT INSTRUCTIONS:
1. Consider the build context and recent steps when analyzing this image
2. If this step references previous work (e.g., "attach to base"), identify which step in "context_references"
3. Determine if this is starting a new subassembly or continuing the current one
4. If the image shows parts already assembled in previous steps, mention them in "existing_assembly"
5. Focus on what's NEW in this step, not what was already built

Be detailed and precise. If information is unclear, mark as null or "unclear".
""")

        return "\n".join(prompt_parts)
```

#### Enhanced Gemini Client

**File:** `src/api/gemini_api.py` (MODIFY)

```python
class GeminiVisionClient:
    """Enhanced to support context-aware extraction."""

    def extract_step_info_with_context(
        self,
        image_paths: List[str],
        step_number: Optional[int] = None,
        custom_prompt: Optional[str] = None,
        use_json_mode: bool = True,
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract step info with custom context-aware prompt.

        NEW: Accepts custom_prompt that includes context.
        """
        # Check cache first
        cache_suffix = f":{cache_context}" if cache_context else ""
        cache_key = f"{self.model}:{','.join(image_paths)}:{step_number}{cache_suffix}"
        cached = self.cache.get(self.model, cache_key, image_paths)
        if cached:
            return cached

        # Use custom prompt if provided, otherwise build default
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_extraction_prompt(step_number, use_json_mode)

        # Prepare content with images
        parts = []
        parts.append({"text": prompt})

        # Add images
        for img_path in image_paths:
            image_data, mime_type = self._encode_image_to_base64(img_path)
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            })

        # Make API call with retry logic
        response = self._call_api_with_retry(parts, use_json_mode)

        # Parse response
        result = self._parse_response(response, use_json_mode)

        # Cache result
        self.cache.set(self.model, cache_key, result, image_paths)

        return result
```

---

### Phase 2: Subassembly Verification

**File:** `src/vision_processing/subassembly_analyzer.py` (NEW)

#### Purpose
Perform two-pass analysis (Option B) to identify and validate subassemblies across all extracted steps.

#### Implementation

```python
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class SubassemblyGroup:
    """Represents an identified subassembly."""
    id: str  # e.g., "sub_1"
    name: str  # e.g., "wheel_assembly"
    description: str  # e.g., "4-wheel chassis with axles"
    steps: List[int]  # e.g., [9, 10, 11, 12, 13, 14, 15]
    parts_used: List[str]  # All parts in this subassembly
    dependencies: List[str]  # IDs of subassemblies this depends on
    can_build_parallel: bool  # Can be built independently

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "parts_used": self.parts_used,
            "dependencies": self.dependencies,
            "can_build_parallel": self.can_build_parallel
        }


class SubassemblyAnalyzer:
    """
    Two-pass analysis system for identifying subassemblies.

    Pass 1: Collect all context-aware step extractions (done in Phase 1)
    Pass 2: Analyze entire sequence to identify patterns and structure
    """

    def __init__(self, vlm_client):
        self.vlm_client = vlm_client

    def analyze_subassemblies(
        self,
        extracted_steps: List[Dict[str, Any]],
        main_build: str
    ) -> List[SubassemblyGroup]:
        """
        Identify subassemblies from extracted steps.

        Process:
        1. Cluster steps by subassembly_hint from Phase 1
        2. Validate clusters with VLM
        3. Identify dependencies between subassemblies
        4. Determine parallel vs. sequential structure

        Args:
            extracted_steps: All steps from Phase 1 (with context)
            main_build: Main build description

        Returns:
            List of identified subassemblies with dependencies
        """
        logger.info("Phase 2: Analyzing subassemblies...")

        # Step 1: Cluster by hints from Phase 1
        clusters = self._cluster_by_hints(extracted_steps)

        # Step 2: Validate and refine clusters with VLM
        validated_subassemblies = self._validate_clusters_with_vlm(
            clusters,
            extracted_steps,
            main_build
        )

        # Step 3: Identify dependencies
        subassemblies_with_deps = self._identify_dependencies(
            validated_subassemblies,
            extracted_steps
        )

        # Step 4: Determine parallel buildability
        final_subassemblies = self._analyze_parallelism(
            subassemblies_with_deps
        )

        logger.info(f"Identified {len(final_subassemblies)} subassemblies")

        return final_subassemblies

    def _cluster_by_hints(
        self,
        extracted_steps: List[Dict[str, Any]]
    ) -> Dict[str, List[int]]:
        """
        Cluster steps based on subassembly_hint from Phase 1.

        Returns:
            {
                "base_structure": [1, 2, 3, 4, 5, 6, 7, 8],
                "wheel_assembly": [9, 10, 11, 12, 13, 14, 15],
                ...
            }
        """
        clusters = {}
        current_subassembly = None

        for step in extracted_steps:
            hint = step.get("subassembly_hint", {})

            if hint.get("is_new_subassembly"):
                # Start new cluster
                current_subassembly = hint.get("name", f"subassembly_{step['step_number']}")
                clusters[current_subassembly] = []

            if current_subassembly:
                clusters[current_subassembly].append(step["step_number"])

        return clusters

    def _validate_clusters_with_vlm(
        self,
        clusters: Dict[str, List[int]],
        extracted_steps: List[Dict[str, Any]],
        main_build: str
    ) -> List[SubassemblyGroup]:
        """
        Use VLM to validate and refine cluster boundaries.

        For each cluster:
        1. Send summary of all steps in cluster to VLM
        2. Ask: "Do these steps form a logical subassembly?"
        3. Refine boundaries if needed
        4. Generate description
        """
        validated = []

        for idx, (name, step_numbers) in enumerate(clusters.items()):
            # Get step data for this cluster
            cluster_steps = [
                s for s in extracted_steps
                if s["step_number"] in step_numbers
            ]

            # Build summary for VLM
            summary = self._build_cluster_summary(cluster_steps)

            # Validate with VLM
            validation_result = self._call_vlm_for_cluster_validation(
                name, summary, main_build
            )

            # Create SubassemblyGroup
            subassembly = SubassemblyGroup(
                id=f"sub_{idx+1}",
                name=validation_result.get("refined_name", name),
                description=validation_result["description"],
                steps=step_numbers,
                parts_used=self._collect_parts(cluster_steps),
                dependencies=[],  # Will be filled in next step
                can_build_parallel=False  # Will be determined later
            )

            validated.append(subassembly)

        return validated

    def _identify_dependencies(
        self,
        subassemblies: List[SubassemblyGroup],
        extracted_steps: List[Dict[str, Any]]
    ) -> List[SubassemblyGroup]:
        """
        Identify dependencies between subassemblies.

        Process:
        1. For each subassembly, check if its steps reference earlier subassemblies
        2. Use context_references from Phase 1 extraction
        3. Build dependency graph
        """
        # Create mapping of step_number -> subassembly_id
        step_to_sub = {}
        for sub in subassemblies:
            for step_num in sub.steps:
                step_to_sub[step_num] = sub.id

        # Analyze dependencies
        for sub in subassemblies:
            dependencies = set()

            for step_num in sub.steps:
                # Find this step's data
                step_data = next(
                    (s for s in extracted_steps if s["step_number"] == step_num),
                    None
                )

                if step_data:
                    # Check context_references
                    refs = step_data.get("context_references", {})
                    if refs.get("references_previous_steps"):
                        referenced_steps = refs.get("which_steps", [])

                        for ref_step in referenced_steps:
                            if ref_step in step_to_sub:
                                dep_sub_id = step_to_sub[ref_step]
                                if dep_sub_id != sub.id:  # Don't depend on self
                                    dependencies.add(dep_sub_id)

            sub.dependencies = list(dependencies)

        return subassemblies

    def _analyze_parallelism(
        self,
        subassemblies: List[SubassemblyGroup]
    ) -> List[SubassemblyGroup]:
        """
        Determine which subassemblies can be built in parallel.

        Rules:
        - If subassembly A has no dependencies, it can be built in parallel with others
        - If subassembly B depends on A, it cannot be built until A is complete
        """
        for sub in subassemblies:
            # Can build in parallel if:
            # 1. No dependencies, OR
            # 2. All dependencies are to earlier subassemblies that will be done first
            sub.can_build_parallel = len(sub.dependencies) == 0

        return subassemblies

    def _call_vlm_for_cluster_validation(
        self,
        cluster_name: str,
        summary: str,
        main_build: str
    ) -> Dict[str, Any]:
        """
        Ask VLM to validate and describe a subassembly cluster.

        Prompt:
        "These steps were grouped as '{cluster_name}':
         {summary}

         For the main build '{main_build}':
         1. Is this a logical subassembly?
         2. What is a better name for it?
         3. Provide a concise description.

         Return JSON: {
           'is_valid': true/false,
           'refined_name': '...',
           'description': '...'
         }"
        """
        prompt = f"""
Analyzing potential subassembly for: {main_build}

Proposed subassembly: "{cluster_name}"

Steps in this group:
{summary}

Questions:
1. Do these steps form a logical, cohesive subassembly?
2. What is an appropriate name for this subassembly?
3. Provide a concise description (1 sentence) of what this subassembly is.

Return JSON:
{{
  "is_valid": true/false,
  "refined_name": "descriptive_name",
  "description": "Brief description of what is being built"
}}
"""

        # Call VLM (text-only, no images needed)
        response = self.vlm_client.generate_text(prompt, use_json_mode=True)

        return response

    def _build_cluster_summary(
        self,
        cluster_steps: List[Dict[str, Any]]
    ) -> str:
        """Build a compact summary of steps for VLM analysis."""
        lines = []
        for step in cluster_steps:
            step_num = step["step_number"]
            parts = ", ".join([p["description"] for p in step.get("parts_required", [])[:3]])
            actions = step.get("actions", [])
            action_summary = actions[0]["action_verb"] if actions else "modify"

            lines.append(f"Step {step_num}: {action_summary} ({parts})")

        return "\n".join(lines)

    def _collect_parts(self, cluster_steps: List[Dict[str, Any]]) -> List[str]:
        """Collect all unique parts used in this subassembly."""
        parts = set()
        for step in cluster_steps:
            for part in step.get("parts_required", []):
                parts.add(part["description"])
        return list(parts)
```

#### Integration into main.py

```python
def main(...):
    # ... Phase 0 and Phase 1 ...

    # Phase 2: Subassembly Verification (NEW)
    if not checkpoint.is_step_complete("subassembly_analysis"):
        logger.info("Phase 2: Analyzing subassemblies...")

        subassembly_analyzer = SubassemblyAnalyzer(vlm_client)
        subassemblies = subassembly_analyzer.analyze_subassemblies(
            extracted_steps,
            doc_metadata.main_build
        )

        # Save subassembly analysis
        subassembly_path = output_dir / f"{assembly_id}_subassemblies.json"
        with open(subassembly_path, 'w', encoding='utf-8') as f:
            json.dump(
                [sub.to_dict() for sub in subassemblies],
                f,
                indent=2,
                ensure_ascii=False
            )

        logger.info(f"Saved subassembly analysis to {subassembly_path}")

        # Display subassembly structure
        logger.info("\nIdentified subassemblies:")
        for sub in subassemblies:
            parallel_indicator = "||" if sub.can_build_parallel else "→"
            logger.info(f"  {parallel_indicator} {sub.name}: {sub.description} (steps {sub.steps[0]}-{sub.steps[-1]})")
            if sub.dependencies:
                logger.info(f"     Depends on: {', '.join(sub.dependencies)}")

        checkpoint.save("subassembly_analysis", {
            "subassemblies": [sub.to_dict() for sub in subassemblies]
        })

    # ... Continue to Phase 3 (Plan Generation) ...
```

---

## Context Window Management

### Problem Statement

**Context window limits:**
- **Gemini 2.5 Flash**: 1,048,576 tokens (1M tokens) input
- **GPT-4o-mini**: 128,000 tokens input

**Our usage per step:**
```
Per-step token breakdown:
┌──────────────────────────────────────────────┐
│ Component                   │ Tokens         │
├──────────────────────────────────────────────┤
│ Image (base64)              │ ~2,000-3,000  │
│ Base prompt                 │ ~500          │
│ Sliding window (5 steps)    │ ~1,500        │
│ Long-term memory            │ ~500          │
│ Model response              │ ~1,000        │
├──────────────────────────────────────────────┤
│ TOTAL per step              │ ~5,500-6,500  │
└──────────────────────────────────────────────┘

For 45 steps: ~6,000 × 45 = 270,000 tokens total
Well within 1M token limit ✓
```

### Strategy: Token Budget Management

**File:** `src/vision_processing/token_budget.py` (NEW)

```python
class TokenBudgetManager:
    """
    Manages token usage to stay within context window limits.
    Dynamically adjusts memory window size if approaching limits.
    """

    def __init__(self, max_tokens: int = 1_000_000):
        """
        Args:
            max_tokens: Maximum context window size (default: 1M for Gemini)
        """
        self.max_tokens = max_tokens
        self.safety_margin = 0.8  # Use only 80% of max to be safe
        self.available_tokens = int(max_tokens * self.safety_margin)

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens in text (rough: 1 token ≈ 4 characters)."""
        return len(text) // 4

    def estimate_image_tokens(self, image_path: str) -> int:
        """Estimate tokens for an image (rough: 2000-3000 per image)."""
        return 2500

    def check_budget(
        self,
        image_count: int,
        sliding_window_size: int,
        long_term_memory_size: int,
        prompt_size: int
    ) -> Dict[str, Any]:
        """
        Check if current configuration fits in token budget.

        Returns:
            {
                "fits": true/false,
                "estimated_usage": 6500,
                "available": 800000,
                "recommendations": {
                    "reduce_window_to": 3,
                    "compress_long_term": false
                }
            }
        """
        estimated_usage = (
            image_count * 2500 +
            sliding_window_size * 300 +
            long_term_memory_size +
            prompt_size +
            1000  # Response tokens
        )

        fits = estimated_usage <= self.available_tokens

        recommendations = {}
        if not fits:
            # Calculate how much to reduce
            excess = estimated_usage - self.available_tokens

            # Option 1: Reduce sliding window
            tokens_per_step = 300
            reduce_by = (excess // tokens_per_step) + 1
            recommendations["reduce_window_to"] = max(2, sliding_window_size - reduce_by)

            # Option 2: Compress long-term memory
            if long_term_memory_size > 1000:
                recommendations["compress_long_term"] = True

        return {
            "fits": fits,
            "estimated_usage": estimated_usage,
            "available": self.available_tokens,
            "utilization": estimated_usage / self.available_tokens,
            "recommendations": recommendations
        }

    def auto_adjust_window_size(
        self,
        current_window_size: int,
        token_estimate: int
    ) -> int:
        """
        Automatically adjust window size to fit budget.

        Returns new window size (may be smaller than current).
        """
        if token_estimate <= self.available_tokens:
            return current_window_size

        # Calculate reduction needed
        excess = token_estimate - self.available_tokens
        tokens_per_step = 300
        reduce_by = (excess // tokens_per_step) + 1

        new_size = max(2, current_window_size - reduce_by)
        logger.warning(
            f"Token budget exceeded. Reducing window size: {current_window_size} → {new_size}"
        )

        return new_size
```

### Integration

```python
class BuildMemory:
    """Enhanced with token budget management."""

    def __init__(
        self,
        main_build: str,
        window_size: int = 5,
        max_tokens: int = 1_000_000
    ):
        self.sliding_window = SlidingWindowMemory(window_size)
        self.long_term = LongTermMemory(main_build)
        self.token_budget = TokenBudgetManager(max_tokens)

    def get_full_context(self) -> Dict[str, str]:
        """Get context with automatic budget management."""
        # Get initial context
        sliding_context = self.sliding_window.get_context()
        long_term_context = self.long_term.get_context()

        # Check budget
        budget_check = self.token_budget.check_budget(
            image_count=1,
            sliding_window_size=len(self.sliding_window.window),
            long_term_memory_size=self.token_budget.estimate_tokens(long_term_context),
            prompt_size=500
        )

        # Auto-adjust if needed
        if not budget_check["fits"]:
            recommendations = budget_check["recommendations"]

            if "reduce_window_to" in recommendations:
                new_size = recommendations["reduce_window_to"]
                self.sliding_window.window = deque(
                    list(self.sliding_window.window)[-new_size:],
                    maxlen=new_size
                )
                sliding_context = self.sliding_window.get_context()

            if recommendations.get("compress_long_term"):
                # Keep only last 2 completed subassemblies
                self.long_term.completed_subassemblies = \
                    self.long_term.completed_subassemblies[-2:]
                long_term_context = self.long_term.get_context()

        return {
            "sliding_window": sliding_context,
            "long_term": long_term_context,
            "token_estimate": budget_check["estimated_usage"]
        }
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_build_memory.py` (NEW)

```python
import pytest
from src.vision_processing.build_memory import (
    BuildMemory, SlidingWindowMemory, LongTermMemory, StepSummary
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
    assert memory.current_subassembly.name == "wheels"

def test_build_memory_integration():
    """Test full BuildMemory system."""
    memory = BuildMemory("Fire Truck", window_size=5)

    # Add several steps
    for i in range(1, 8):
        step_data = {
            "step_number": i,
            "parts_required": [{"description": f"part_{i}"}],
            "actions": [{"action_verb": "attach", "target": f"part_{i}"}],
            "subassembly_hint": {
                "is_new_subassembly": i == 1 or i == 5,
                "name": "base" if i < 5 else "wheels"
            }
        }
        memory.add_step(step_data)

    # Check context
    context = memory.get_full_context()
    assert "sliding_window" in context
    assert "long_term" in context
    assert "Step 7" in context["sliding_window"]
    assert "Fire Truck" in context["long_term"]

def test_token_budget_management():
    """Test token budget prevents overflow."""
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

    # Check oversized request
    result = budget.check_budget(
        image_count=1,
        sliding_window_size=20,  # Too large
        long_term_memory_size=500,
        prompt_size=500
    )
    assert result["fits"] == False
    assert "reduce_window_to" in result["recommendations"]
```

### Integration Tests

**File:** `tests/test_context_aware_extraction.py` (NEW)

```python
import pytest
from pathlib import Path
from src.vision_processing.vlm_step_extractor import VLMStepExtractor

def test_context_aware_extraction_flow():
    """Test full context-aware extraction flow."""
    extractor = VLMStepExtractor()
    extractor.initialize_memory("Fire Truck", window_size=5)

    # Simulate extracting 3 steps
    test_images = [
        Path("tests/fixtures/step_1.png"),
        Path("tests/fixtures/step_2.png"),
        Path("tests/fixtures/step_3.png")
    ]

    results = []
    for i, img_path in enumerate(test_images, 1):
        result = extractor.extract_step([str(img_path)], step_number=i)
        results.append(result)

    # Check that later steps have context
    assert extractor.build_memory is not None
    assert len(extractor.build_memory.sliding_window.window) == 3

    # Check that subassembly hints are present
    for result in results:
        assert "subassembly_hint" in result
        assert "context_references" in result
```

### End-to-End Test

**File:** `tests/test_full_pipeline.py` (NEW)

```python
def test_full_pipeline_with_context():
    """Test entire pipeline from PDF to subassembly analysis."""
    pdf_path = "tests/fixtures/sample_manual.pdf"
    output_dir = Path("tests/output")

    # Phase 0: Document understanding
    from src.vision_processing.document_analyzer import DocumentAnalyzer
    analyzer = DocumentAnalyzer(vlm_client)
    metadata = analyzer.analyze_pdf(pdf_path)

    assert metadata.main_build is not None
    assert len(metadata.instruction_page_ranges) > 0

    # Phase 1: Context-aware extraction
    extractor = VLMStepExtractor()
    extractor.initialize_memory(metadata.main_build)

    relevant_pages = analyzer.extract_relevant_pages(all_pages, metadata.page_classification)
    results = extractor.batch_extract(relevant_pages, batch_size=1)

    assert len(results) > 0
    assert all("subassembly_hint" in r for r in results)

    # Phase 2: Subassembly analysis
    from src.vision_processing.subassembly_analyzer import SubassemblyAnalyzer
    sub_analyzer = SubassemblyAnalyzer(vlm_client)
    subassemblies = sub_analyzer.analyze_subassemblies(results, metadata.main_build)

    assert len(subassemblies) > 0
    assert all(hasattr(sub, 'dependencies') for sub in subassemblies)
```

---

## Success Criteria

### Phase 0: Document Understanding
- ✅ Correctly identifies main build name
- ✅ Filters out cover pages, ads, and inventory
- ✅ Processes only instruction pages (reduces API calls by ~10-15%)
- ✅ User can confirm/reject analysis before proceeding

### Phase 1: Context-Aware Extraction
- ✅ Each step extraction includes context from previous 5 steps
- ✅ Long-term memory tracks overall build progress
- ✅ Steps correctly identify when starting new subassembly
- ✅ Context references are captured (e.g., "references step 4")
- ✅ Token usage stays within budget (< 80% of context window)

### Phase 2: Subassembly Verification
- ✅ Identifies logical subassemblies (typically 3-8 per manual)
- ✅ Correctly groups related steps
- ✅ Builds accurate dependency graph
- ✅ Identifies parallel vs. sequential subassemblies
- ✅ Validates all inter-step references

### Overall System
- ✅ Processing time: < 6 minutes for 45-step manual
- ✅ Accuracy improvement: > 30% better dependency detection
- ✅ Subassembly identification: > 90% accuracy
- ✅ No context window overflows
- ✅ Graceful degradation if token limits approached

---

## Performance Expectations

### Time Breakdown (45-step manual)

```
Phase 0: Document Understanding
├─ PDF analysis: ~30 seconds (1 VLM call)
└─ User confirmation: ~10 seconds

Phase 1: Context-Aware Extraction
├─ Step extraction: 45 steps × 4 sec delay = 180 seconds (3 min)
└─ Memory updates: negligible

Phase 2: Subassembly Verification
├─ Clustering: < 1 second
├─ VLM validation: 5-8 subassemblies × 5 sec = 40 seconds
└─ Dependency analysis: < 5 seconds

TOTAL: ~5-6 minutes
```

### Token Usage (per step)

```
Input tokens:
- Image: 2,500
- Sliding window (5 steps): 1,500
- Long-term memory: 500
- Base prompt: 500
Total input: ~5,000 tokens

Output tokens: ~1,000

Per-step total: ~6,000 tokens
45 steps total: ~270,000 tokens (27% of 1M limit) ✓
```

### API Costs (Gemini Free Tier)

```
Limits: 1,500 requests/day, 15 requests/minute

Usage:
- Phase 0: 1 request
- Phase 1: 45 requests
- Phase 2: ~8 requests
Total: ~54 requests

Daily capacity: 1,500 / 54 = 27 manuals per day ✓
```

---

## Migration Path

### Step 1: Implement Phase 0 (Week 1)
1. Create `DocumentAnalyzer` class
2. Add document understanding prompt
3. Integrate into main.py
4. Test with 3-5 sample manuals

### Step 2: Implement Memory Systems (Week 1-2)
1. Create `BuildMemory`, `SlidingWindowMemory`, `LongTermMemory`
2. Add token budget management
3. Write unit tests
4. Validate token usage stays within limits

### Step 3: Enhance VLM Extraction (Week 2)
1. Modify `VLMStepExtractor` for context awareness
2. Update prompts to include context
3. Add subassembly hint extraction
4. Test context propagation

### Step 4: Integration & Testing (Week 2)
1. Integrate Phase 0 and Phase 1 with existing Phase 2
2. Test context-aware extraction improves subassembly detection
3. End-to-end integration tests
4. Performance benchmarking
5. User acceptance testing
6. Documentation

**Note:** Phase 2 (hierarchical graph) already exists and requires NO changes!

---

## Rollback Plan

If issues arise, system can fall back to Phase 1 (current system):

```python
# Graceful degradation
try:
    # Try context-aware extraction
    result = extract_with_context(...)
except Exception as e:
    logger.warning(f"Context-aware extraction failed: {e}")
    logger.info("Falling back to stateless extraction")
    result = extract_without_context(...)
```

---

## Conclusion

This **UPDATED** implementation plan focuses on the two missing phases that will enhance your existing hierarchical graph system:

### What You Have (Phase 2 - Hierarchical Graph) ✅
- Working subassembly detection (~70% accuracy with heuristics)
- Complete hierarchical graph with nodes, edges, layers
- Rich query API for graph operations
- Integration with 3D plan generation

### What You Need (Phases 0 & 1) ❌
1. **Phase 0 - Document Understanding**: Filter irrelevant pages, save API calls
2. **Phase 1 - Context-Aware Extraction**: Provide context to VLM, improve extraction quality

### How They Work Together

```
Phase 0 (NEW)          Phase 1 (NEW)              Phase 2 (EXISTING) ✅
     ↓                      ↓                            ↓
Filter pages    →   Extract with context   →   Enhanced hierarchical graph
(45 vs 52)          (sliding window +              (better subassemblies)
                     long-term memory)
```

### Benefits of Implementation

With Phases 0 & 1 added:
- ✅ **10-15% fewer API calls** (filtering)
- ✅ **20-30% better subassembly detection** (context awareness)
- ✅ **More accurate dependencies** (context references)
- ✅ **No changes to existing Phase 2 code** (seamless integration)

### Updated Timeline

**Estimated implementation time: 1-2 weeks** (down from 2-3 weeks)
- Week 1: Implement Phase 0 (document understanding) + Phase 1 memory systems
- Week 2: Integration testing with existing Phase 2

**Expected improvement:**
- **Subassembly detection**: 70% → 90% accuracy
- **Inter-step references**: 0% → 80% captured
- **API efficiency**: +10-15% savings
