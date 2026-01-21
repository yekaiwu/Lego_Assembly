"""
Prompt templates for LEGO assembly video analysis.

These prompts are designed to work with Gemini Robotics ER VLM and integrate
with the Lego_Assembly manual processing system.
"""

import textwrap
import json
from typing import Optional, Dict, List


def get_assembly_video_prompt(
    manual_steps: List[Dict],
    dependencies: Optional[Dict] = None,
    expected_colors: Optional[List[str]] = None
) -> str:
    """
    Get the prompt for analyzing LEGO assembly videos.

    This prompt instructs the VLM to detect assembly steps from a video,
    using the manual's extracted steps as context for better accuracy.

    Args:
        manual_steps: List of step dictionaries from extracted.json
        dependencies: Optional step dependency graph from dependencies.json
        expected_colors: Optional list of LEGO colors expected in this build

    Returns:
        Formatted prompt string
    """
    # Build step reference text
    step_reference = "ASSEMBLY MANUAL STEPS:\n"
    for step in manual_steps[:20]:  # Limit to first 20 for context window
        step_num = step.get('step_number', '?')
        parts = step.get('parts_required', [])
        actions = step.get('actions', [])

        parts_text = ", ".join([
            f"{p.get('quantity', 1)}x {p.get('color', '')} {p.get('description', '')}"
            for p in parts
        ])

        action_text = actions[0].get('action_verb', 'attach') if actions else 'attach'

        step_reference += f"Step {step_num}: {action_text} {parts_text}\n"

    if len(manual_steps) > 20:
        step_reference += f"... and {len(manual_steps) - 20} more steps\n"

    # Build color context
    color_context = ""
    if expected_colors:
        color_context = f"\nEXPECTED LEGO COLORS: {', '.join(expected_colors)}\n"

    base_prompt = textwrap.dedent(f"""
        Watch this LEGO assembly video carefully. You are analyzing a user building a LEGO model
        following the instruction manual provided below.

        {step_reference}
        {color_context}

        TASK:
        For each assembly step you observe in the video, identify:
        1. When the step starts and ends (start_seconds, end_seconds)
        2. Which step from the manual this corresponds to (step_id)
        3. The exact attachment point on the LEGO structure (target_box_2d)
        4. The entire visible LEGO assembly (assembly_box_2d)

        VISUAL GROUNDING RULES:
        - The LEGO assembly is typically in the CENTER of the frame
        - Hands may be visible holding pieces - focus on the BRICKS themselves
        - The target coordinates MUST point to visible LEGO studs or attachment points
        - Do NOT point to empty background, hands, or table surface

        OUTPUT FORMAT:
        Return a JSON array with this exact structure:
        [
          {{
            "step_id": <number matching manual step_number>,
            "start_seconds": <when user starts this step>,
            "end_seconds": <when user completes this step>,
            "anchor_timestamp": <a frame time where the target is clearly visible>,
            "instruction": "<brief description of what's being attached>",
            "action": "<place|attach|connect>",
            "target_box_2d": [ymin, xmin, ymax, xmax],
            "assembly_box_2d": [ymin, xmin, ymax, xmax],
            "confidence": <0.0 to 1.0>,
            "reasoning": "<optional: explain how you identified this step>"
          }}
        ]

        COORDINATE FORMAT (CRITICAL):
        ALL coordinates are normalized to 0-1000 scale:
        - ymin/ymax: 0 = top edge of frame, 1000 = bottom edge
        - xmin/xmax: 0 = left edge of frame, 1000 = right edge
        - Format: [ymin, xmin, ymax, xmax] - Y values come first!

        Box sizing:
        - target_box_2d: Small box around specific attachment stud (~20-50 units wide)
        - assembly_box_2d: Larger box around entire LEGO structure (~150-300 units wide)

        VALIDATION CHECKLIST:
        Before returning each event, verify:
        ✓ step_id matches a step from the manual above
        ✓ target_box_2d center point is INSIDE assembly_box_2d
        ✓ Coordinates are integers between 0 and 1000
        ✓ ymin < ymax and xmin < xmax
        ✓ anchor_timestamp is between start_seconds and end_seconds

        Return ONLY the JSON array, no other text.
    """).strip()

    # Add dependency information if provided
    if dependencies:
        dep_text = "\n\nSTEP DEPENDENCIES:\n"
        dep_text += json.dumps(dependencies, indent=2)
        base_prompt += dep_text

    return base_prompt


def get_step_verification_prompt(
    step_number: int,
    step_description: str,
    parts_required: List[Dict]
) -> str:
    """
    Get a prompt for verifying if a specific step was completed in an image.

    Args:
        step_number: The step number to verify
        step_description: Description of what this step involves
        parts_required: List of parts that should be visible

    Returns:
        Formatted prompt string
    """
    parts_text = ", ".join([
        f"{p.get('quantity', 1)}x {p.get('color', '')} {p.get('description', '')}"
        for p in parts_required
    ])

    return textwrap.dedent(f"""
        Verify if this LEGO assembly step has been completed:

        Step {step_number}: {step_description}
        Required parts: {parts_text}

        Look at the image and determine:
        1. Are all required parts present and attached?
        2. How confident are you this step is complete?

        OUTPUT FORMAT:
        {{
          "completed": <true|false>,
          "confidence": <0.0 to 1.0>,
          "detected_parts": [
            {{"description": "...", "color": "...", "present": true}}
          ],
          "current_state_box_2d": [ymin, xmin, ymax, xmax],
          "reasoning": "<explain what you see>"
        }}

        COORDINATE FORMAT:
        current_state_box_2d should be normalized to 0-1000 scale:
        - [ymin, xmin, ymax, xmax] where 0 = edge, 1000 = opposite edge

        Return ONLY valid JSON.
    """).strip()


def get_refinement_prompt(
    step_number: int,
    instruction: str,
    detected_box: List[int]
) -> str:
    """
    Get a prompt for refining detected coordinates.

    Args:
        step_number: The step number
        instruction: The assembly instruction
        detected_box: The initially detected box coordinates

    Returns:
        Formatted prompt string
    """
    return textwrap.dedent(f"""
        You previously detected coordinates for this assembly step:
        Step {step_number}: {instruction}

        Detected coordinates (with red crosshair): {detected_box}

        Review the image with the red crosshair marking the detected target.

        TASK:
        Is the crosshair pointing to the correct attachment point on the LEGO assembly?
        If not, provide corrected coordinates.

        OUTPUT FORMAT:
        {{
          "correct": <true|false>,
          "corrected_box_2d": [ymin, xmin, ymax, xmax],
          "confidence": <0.0 to 1.0>,
          "reasoning": "<explain the correction if needed>"
        }}

        COORDINATE FORMAT:
        Normalized to 0-1000 scale: [ymin, xmin, ymax, xmax]

        Return ONLY valid JSON.
    """).strip()


# Default prompt for backward compatibility
ASSEMBLY_VIDEO_ANALYSIS_PROMPT = """
Watch this LEGO assembly video and detect each assembly step with timing and coordinates.
Return JSON array with format: [{"step_id": N, "start_seconds": X, "end_seconds": Y, ...}]
All coordinates normalized to 0-1000 scale.
"""
