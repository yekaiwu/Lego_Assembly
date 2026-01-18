"""
Visualization utilities for debugging VLM extraction and component detection.

This module provides utilities to visualize:
- Center points extracted by VLM
- Bounding boxes and detection results
- Component extraction results
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from loguru import logger


def visualize_center_points(
    image_path: str,
    step_data: Dict[str, Any],
    output_path: str
) -> str:
    """
    Visualize VLM-extracted center points on the instruction page.

    Args:
        image_path: Path to the original instruction page
        step_data: Step data containing center_point and assembled_result_center
        output_path: Path to save visualization

    Returns:
        Path to saved visualization
    """
    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Get image dimensions for coordinate conversion
    height, width = img_bgr.shape[:2]

    # Draw parts_required center points
    parts_required = step_data.get("parts_required", [])
    for i, part in enumerate(parts_required):
        if "center_point" not in part or not part["center_point"]:
            continue

        # Convert normalized coordinates (0-1000) to pixel coordinates
        norm_x, norm_y = part["center_point"]
        pixel_x = int((norm_x / 1000.0) * width)
        pixel_y = int((norm_y / 1000.0) * height)
        center = (pixel_x, pixel_y)

        color = part.get("color", "unknown")
        shape = part.get("shape", "unknown")

        # Draw center point
        cv2.circle(img_bgr, center, 8, (0, 255, 0), -1)  # Green dot
        cv2.circle(img_bgr, center, 12, (0, 255, 0), 2)   # Green circle

        # Draw label
        label = f"P{i}: {color} {shape}"
        cv2.putText(
            img_bgr, label, (center[0] + 15, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Draw assembled_result_center
    if "assembled_result_center" in step_data and step_data["assembled_result_center"]:
        # Convert normalized coordinates (0-1000) to pixel coordinates
        norm_x, norm_y = step_data["assembled_result_center"]
        pixel_x = int((norm_x / 1000.0) * width)
        pixel_y = int((norm_y / 1000.0) * height)
        center = (pixel_x, pixel_y)

        # Draw center point
        cv2.circle(img_bgr, center, 8, (255, 0, 0), -1)  # Blue dot
        cv2.circle(img_bgr, center, 12, (255, 0, 0), 2)   # Blue circle

        # Draw label
        label = "Assembled Result"
        cv2.putText(
            img_bgr, label, (center[0] + 15, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

    # Add legend
    legend_y = 30
    cv2.putText(img_bgr, "Green = Parts", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img_bgr, "Blue = Assembled", (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(output_path)

    return output_path


def visualize_all_center_points(
    extracted_steps: List[Dict[str, Any]],
    output_dir: str
) -> List[str]:
    """
    Visualize center points for all extracted steps.

    Args:
        extracted_steps: List of all extracted step data
        output_dir: Directory to save visualizations

    Returns:
        List of paths to saved visualizations
    """
    viz_dir = os.path.join(output_dir, "center_point_visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    visualization_paths = []

    for step_idx, step in enumerate(extracted_steps):
        step_number = step.get("step_number", step_idx + 1)

        # Get source page paths
        source_pages = step.get("_source_page_paths", [])
        if not source_pages:
            logger.warning(f"Step {step_number}: No source page paths found")
            continue

        # Use first source page
        page_path = source_pages[0]
        if not os.path.exists(page_path):
            logger.warning(f"Step {step_number}: Source page not found: {page_path}")
            continue

        # Generate visualization
        output_path = os.path.join(viz_dir, f"step_{step_number:03d}_center_points.png")

        try:
            viz_path = visualize_center_points(page_path, step, output_path)
            visualization_paths.append(viz_path)
            logger.debug(f"Step {step_number}: Center point visualization saved to {viz_path}")
        except Exception as e:
            logger.error(f"Step {step_number}: Failed to create visualization: {e}")

    logger.info(f"Created {len(visualization_paths)} center point visualizations in {viz_dir}")

    return visualization_paths


def create_extraction_summary_image(
    component_extractor,
    output_path: str
) -> Optional[str]:
    """
    Create a summary image showing all extracted components in a grid.

    Args:
        component_extractor: ComponentExtractor instance
        output_path: Path to save summary image

    Returns:
        Path to saved summary image, or None if no components extracted
    """
    summary = component_extractor.get_extraction_summary()

    total_components = summary['total_parts'] + summary['total_subassemblies']
    if total_components == 0:
        logger.warning("No components extracted, cannot create summary image")
        return None

    # Collect all component images
    component_images = []
    component_labels = []

    # Add part images
    for part_id in summary['extracted_parts']:
        rel_path = component_extractor.get_part_image_path(part_id)
        if rel_path:
            abs_path = os.path.join(component_extractor.output_dir, rel_path)
            if os.path.exists(abs_path):
                component_images.append(abs_path)
                component_labels.append(f"Part: {part_id}")

    # Add subassembly images
    for subasm_id in summary['extracted_subassemblies']:
        rel_path = component_extractor.get_subassembly_image_path(subasm_id)
        if rel_path:
            abs_path = os.path.join(component_extractor.output_dir, rel_path)
            if os.path.exists(abs_path):
                component_images.append(abs_path)
                component_labels.append(f"Subasm: {subasm_id}")

    if not component_images:
        logger.warning("No component image files found")
        return None

    # Create grid layout
    cols = min(4, len(component_images))
    rows = (len(component_images) + cols - 1) // cols

    # Load and resize images
    thumb_size = 200
    thumbnails = []

    for img_path in component_images:
        img = Image.open(img_path)
        # Resize maintaining aspect ratio
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        thumbnails.append(img)

    # Create grid
    grid_width = cols * thumb_size
    grid_height = rows * thumb_size
    grid_img = Image.new('RGB', (grid_width, grid_height), (240, 240, 240))

    for idx, (thumb, label) in enumerate(zip(thumbnails, component_labels)):
        row = idx // cols
        col = idx % cols
        x = col * thumb_size + (thumb_size - thumb.width) // 2
        y = row * thumb_size + (thumb_size - thumb.height) // 2
        grid_img.paste(thumb, (x, y))

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid_img.save(output_path)
    logger.info(f"Created component summary image with {len(component_images)} components: {output_path}")

    return output_path


def log_component_extraction_results(component_extractor, output_dir: str):
    """
    Log detailed information about extracted components.

    Args:
        component_extractor: ComponentExtractor instance
        output_dir: Output directory for the assembly
    """
    summary = component_extractor.get_extraction_summary()

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPONENT EXTRACTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Parts Extracted: {summary['total_parts']}")
    logger.info(f"Total Subassemblies Extracted: {summary['total_subassemblies']}")
    logger.info(f"Components Directory: {summary['components_dir']}")
    logger.info("")

    if summary['total_parts'] > 0:
        logger.info("Extracted Parts:")
        for part_id in summary['extracted_parts']:
            rel_path = component_extractor.get_part_image_path(part_id)
            abs_path = os.path.join(output_dir, rel_path) if rel_path else "N/A"
            logger.info(f"  • {part_id}: {abs_path}")

    if summary['total_subassemblies'] > 0:
        logger.info("")
        logger.info("Extracted Subassemblies:")
        for subasm_id in summary['extracted_subassemblies']:
            rel_path = component_extractor.get_subassembly_image_path(subasm_id)
            abs_path = os.path.join(output_dir, rel_path) if rel_path else "N/A"
            logger.info(f"  • {subasm_id}: {abs_path}")

    logger.info("=" * 80)
    logger.info("")
