"""
Video overlay rendering for LEGO assembly guidance.

This module renders visual overlays on video frames including:
- Target markers showing attachment points
- HUD panel with step progression
- Instruction cards with current step details
"""

import cv2
import json
import logging
import textwrap
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass

from .coordinate_utils import (
    validate_coordinates,
    get_center_from_box,
    to_pixel_coordinates,
    box_to_pixel_coordinates,
)

logger = logging.getLogger(__name__)


@dataclass
class OverlayConfig:
    """Configuration for overlay rendering."""
    # Marker settings
    marker_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    marker_radius: int = 20
    marker_pulse_amplitude: int = 5
    marker_pulse_frequency: float = 2.0  # Hz

    # HUD panel settings
    panel_width: int = 250
    panel_bg_color: Tuple[int, int, int, int] = (20, 20, 20, 220)

    # Instruction card settings
    card_margin: int = 20
    card_height: int = 120
    card_radius: int = 10
    card_bg_color: Tuple[int, int, int, int] = (0, 0, 0, 200)

    # Font sizes
    font_large_size: int = 32
    font_medium_size: int = 24
    font_small_size: int = 18


class OverlayRenderer:
    """Renders assembly guidance overlays on video frames."""

    def __init__(self, config: Optional[OverlayConfig] = None):
        """
        Initialize overlay renderer.

        Args:
            config: Overlay configuration
        """
        self.config = config or OverlayConfig()
        self._load_fonts()

    def _load_fonts(self):
        """Load fonts with fallbacks."""
        font_paths = [
            # macOS
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            # Windows
            "C:\\Windows\\Fonts\\arial.ttf",
        ]

        # Default fonts
        self.font_large = ImageFont.load_default()
        self.font_medium = ImageFont.load_default()
        self.font_small = ImageFont.load_default()

        # Try to load truetype fonts
        for font_path in font_paths:
            try:
                path = Path(font_path)
                if path.exists():
                    self.font_large = ImageFont.truetype(font_path, self.config.font_large_size)
                    self.font_medium = ImageFont.truetype(font_path, self.config.font_medium_size)
                    self.font_small = ImageFont.truetype(font_path, self.config.font_small_size)
                    logger.info(f"Loaded fonts from {font_path}")
                    break
            except Exception as e:
                continue

    def render_target_marker(
        self,
        draw: ImageDraw.Draw,
        box_2d: List[int],
        img_width: int,
        img_height: int,
        time_sec: float = 0.0,
    ):
        """
        Render target marker at attachment point.

        Args:
            draw: PIL ImageDraw object
            box_2d: [ymin, xmin, ymax, xmax] normalized to 0-1000
            img_width: Image width in pixels
            img_height: Image height in pixels
            time_sec: Current time for animation
        """
        if not validate_coordinates(box_2d, strict=False):
            logger.warning(f"Invalid target coordinates: {box_2d}")
            return

        # Get center point
        y_center, x_center = get_center_from_box(box_2d)

        # Convert to pixel coordinates
        cx, cy = to_pixel_coordinates(y_center, x_center, img_width, img_height)

        # Clamp to valid range
        cx = max(20, min(img_width - 20, cx))
        cy = max(20, min(img_height - 20, cy))

        # Pulsing animation
        r_base = self.config.marker_radius
        r_pulse = r_base + self.config.marker_pulse_amplitude * np.sin(
            time_sec * self.config.marker_pulse_frequency * 2 * np.pi
        )

        color = self.config.marker_color

        # Outer pulsing ring
        draw.ellipse(
            (cx - r_pulse * 1.5, cy - r_pulse * 1.5,
             cx + r_pulse * 1.5, cy + r_pulse * 1.5),
            outline=(*color, 150),
            width=2
        )

        # Inner circle
        draw.ellipse(
            (cx - r_base, cy - r_base, cx + r_base, cy + r_base),
            fill=(*color, 100),
            outline=(*color, 255),
            width=3
        )

        # Center dot
        draw.ellipse(
            (cx - 4, cy - 4, cx + 4, cy + 4),
            fill=(*color, 255)
        )

        # Label
        draw.text(
            (cx + 25, cy - 25),
            "TARGET",
            font=self.font_small,
            fill=(*color, 255)
        )

    def render_hud_panel(
        self,
        draw: ImageDraw.Draw,
        img_width: int,
        img_height: int,
        current_step: int,
        total_steps: int,
        is_active: bool = True,
    ):
        """
        Render HUD panel showing step progression.

        Args:
            draw: PIL ImageDraw object
            img_width: Image width
            img_height: Image height
            current_step: Current step number (1-indexed)
            total_steps: Total number of steps
            is_active: Whether step is currently active
        """
        panel_width = self.config.panel_width
        panel_bg = self.config.panel_bg_color

        # Draw background panel
        draw.rectangle(
            [img_width - panel_width, 0, img_width, img_height],
            fill=panel_bg
        )

        base_x = img_width - panel_width + 20
        y_offset = 40

        # Title
        draw.text(
            (base_x, y_offset),
            "ASSEMBLY STEPS",
            font=self.font_medium,
            fill=(255, 255, 255, 255)
        )
        y_offset += 60

        # Progress bar
        progress = current_step / total_steps if total_steps > 0 else 0
        bar_width = panel_width - 40
        bar_height = 20

        # Background bar
        draw.rectangle(
            [base_x, y_offset, base_x + bar_width, y_offset + bar_height],
            fill=(50, 50, 50, 255),
            outline=(100, 100, 100, 255)
        )

        # Progress fill
        if progress > 0:
            draw.rectangle(
                [base_x, y_offset,
                 base_x + int(bar_width * progress), y_offset + bar_height],
                fill=(0, 255, 0, 255)
            )

        # Progress text
        y_offset += bar_height + 10
        draw.text(
            (base_x, y_offset),
            f"{current_step} / {total_steps}",
            font=self.font_small,
            fill=(200, 200, 200, 255)
        )
        y_offset += 50

        # Step list (show window around current step)
        start_display = max(1, current_step - 2)
        end_display = min(total_steps, current_step + 3)

        for i in range(start_display, end_display + 1):
            step_text = f"Step {i}"

            if i < current_step:
                # Completed
                icon = "✓ "
                color = (100, 255, 100, 255)
                text = icon + step_text
                font = self.font_small
            elif i == current_step:
                # Current
                icon = "▶ "
                if is_active:
                    color = (255, 215, 0, 255)  # Gold
                    text = icon + step_text + " (Active)"
                else:
                    color = (200, 200, 200, 255)
                    text = icon + step_text
                font = self.font_medium
            else:
                # Future
                icon = "  "
                color = (150, 150, 150, 255)
                text = icon + step_text
                font = self.font_small

            draw.text((base_x, y_offset), text, font=font, fill=color)
            y_offset += 35

    def render_instruction_card(
        self,
        draw: ImageDraw.Draw,
        img_width: int,
        img_height: int,
        instruction: str,
        step_number: int,
    ):
        """
        Render instruction card at bottom of frame.

        Args:
            draw: PIL ImageDraw object
            img_width: Image width
            img_height: Image height
            instruction: Instruction text to display
            step_number: Current step number
        """
        panel_width = self.config.panel_width
        card_margin = self.config.card_margin
        card_height = self.config.card_height
        card_radius = self.config.card_radius
        card_bg = self.config.card_bg_color

        # Wrap instruction text
        wrapped = textwrap.fill(instruction, width=50)

        # Card dimensions
        card_w = img_width - panel_width - (card_margin * 2)
        card_x = card_margin
        card_y = img_height - card_height - card_margin

        # Draw rounded rectangle
        draw.rounded_rectangle(
            [card_x, card_y, card_x + card_w, card_y + card_height],
            radius=card_radius,
            fill=card_bg,
            outline=(255, 255, 255, 100),
            width=2
        )

        # Step number badge
        badge_size = 40
        draw.rounded_rectangle(
            [card_x + 20, card_y + 20,
             card_x + 20 + badge_size, card_y + 20 + badge_size],
            radius=5,
            fill=(0, 150, 255, 255)
        )

        draw.text(
            (card_x + 30, card_y + 28),
            str(step_number),
            font=self.font_medium,
            fill=(255, 255, 255, 255)
        )

        # Instruction text
        text_y = card_y + 30
        text_x = card_x + 80

        for line in wrapped.split('\n'):
            draw.text(
                (text_x, text_y),
                line,
                font=self.font_small,
                fill=(255, 255, 255, 255)
            )
            text_y += 25


def create_overlay_video(
    video_path: str,
    analysis_results: Dict,
    output_path: str,
    config: Optional[OverlayConfig] = None,
    show_target: bool = True,
    show_hud: bool = True,
    show_instruction: bool = True,
) -> str:
    """
    Create overlay video with visual guidance.

    Args:
        video_path: Path to input video
        analysis_results: Analysis results dict with detected_events
        output_path: Path for output video
        config: Overlay configuration
        show_target: Whether to show target markers
        show_hud: Whether to show HUD panel
        show_instruction: Whether to show instruction cards

    Returns:
        Path to created overlay video
    """
    logger.info(f"Creating overlay video: {video_path} -> {output_path}")

    events = analysis_results.get('detected_events', [])
    events.sort(key=lambda x: x.get('start_seconds', 0))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize renderer
    renderer = OverlayRenderer(config)

    frame_count = 0
    total_steps = len(events)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_count / fps

        # Find active event
        active_event = None
        for event in events:
            if event.get('start_seconds', 0) <= current_time_sec <= event.get('end_seconds', 999999):
                active_event = event
                break

        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb).convert('RGBA')

        # Create overlay layer
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Render overlays if active event
        if active_event:
            step_num = active_event.get('step_id', 1)

            # Render target marker
            if show_target and 'target_box_2d' in active_event:
                renderer.render_target_marker(
                    draw,
                    active_event['target_box_2d'],
                    width,
                    height,
                    current_time_sec
                )

            # Render HUD panel
            if show_hud:
                renderer.render_hud_panel(
                    draw,
                    width,
                    height,
                    step_num,
                    total_steps,
                    is_active=True
                )

            # Render instruction card
            if show_instruction and 'instruction' in active_event:
                renderer.render_instruction_card(
                    draw,
                    width,
                    height,
                    active_event['instruction'],
                    step_num
                )

        # Composite overlay onto frame
        pil_image = Image.alpha_composite(pil_image, overlay)

        # Convert back to BGR for OpenCV
        frame_with_overlay = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)

        # Write frame
        out.write(frame_with_overlay)

        frame_count += 1

        # Progress logging
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            logger.info(f"Processing: {progress:.1f}% ({frame_count}/{total_frames} frames)")

    # Cleanup
    cap.release()
    out.release()

    logger.info(f"Overlay video created: {output_path}")
    return output_path
