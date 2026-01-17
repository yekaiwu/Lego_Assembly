"""
Bounding Box Visualization Utility

Visualizes bounding boxes from VLM extraction results.
Saves annotated images to bbox_visualisation/ directory.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


class BBoxVisualizer:
    """Visualizes bounding boxes from VLM extraction results."""

    def __init__(self, output_dir: str = "bbox_visualisation"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Color palette for different parts (RGB)
        self.colors = [
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 0, 255),    # Magenta
            (255, 165, 0),    # Orange
            (0, 255, 255),    # Cyan
            (255, 255, 0),    # Yellow
            (255, 0, 0),      # Red
            (128, 0, 128),    # Purple
        ]

        # Try to load system fonts
        try:
            self.font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            self.font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            self.font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except Exception:
            try:
                # Try common Linux fonts
                self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
                self.font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            except Exception:
                # Fallback to default font
                self.font_large = ImageFont.load_default()
                self.font_medium = ImageFont.load_default()
                self.font_small = ImageFont.load_default()

        logger.info(f"BBoxVisualizer initialized - output directory: {self.output_dir}")

    def visualize_step_extraction(
        self,
        image_path: str,
        extraction_result: Dict[str, Any],
        step_number: Optional[int] = None,
        save_name: Optional[str] = None
    ) -> str:
        """
        Visualize bounding boxes from a single step extraction result.

        Args:
            image_path: Path to the original instruction page image
            extraction_result: Single step extraction result dict
            step_number: Optional step number for labeling
            save_name: Optional custom name for saved file

        Returns:
            Path to saved visualization image
        """
        # Load image
        try:
            img = Image.open(image_path).convert("RGBA")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

        img_width, img_height = img.size

        # Create drawing layer
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw title
        step_label = f"Step {step_number}" if step_number else "VLM Extraction"
        draw.text((10, 10), f"BBOX VISUALIZATION - {step_label}", fill=(255, 255, 0, 255), font=self.font_large)

        # Extract parts and bboxes
        parts_required = extraction_result.get("parts_required", [])
        assembled_bbox = extraction_result.get("assembled_result_bbox", None)

        bbox_count = 0

        # Draw part bboxes
        for idx, part in enumerate(parts_required):
            bbox = part.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue

            # Get color for this part (cycle through palette)
            color = self.colors[idx % len(self.colors)]
            color_rgba = (*color, 200)  # Add alpha for transparency

            # Validate bbox (should be in pixel format [x1, y1, x2, y2])
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Ensure bbox is within image bounds
            x1 = max(0, min(x1, img_width))
            x2 = max(0, min(x2, img_width))
            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))

            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bbox for part {idx}: {bbox}")
                continue

            # Draw bbox rectangle (thick lines for visibility)
            for i in range(3):
                draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color_rgba, width=1)

            # Prepare label
            part_desc = part.get("description", "Unknown")[:30]  # Truncate long descriptions
            part_color = part.get("color", "")
            part_qty = part.get("quantity", 1)

            label = f"Part {idx+1}: {part_qty}x {part_color} {part_desc}"

            # Draw label above bbox (if space available)
            label_y = max(y1 - 40, 30)

            # Draw text background for readability
            text_bbox = draw.textbbox((x1, label_y), label, font=self.font_small)
            draw.rectangle(text_bbox, fill=(0, 0, 0, 180))

            # Draw text
            draw.text((x1, label_y), label, fill=color_rgba, font=self.font_small)

            # Draw coordinates below bbox
            coord_label = f"Pixel: [{x1}, {y1}, {x2}, {y2}]"
            coord_y = min(y2 + 5, img_height - 20)

            coord_bbox = draw.textbbox((x1, coord_y), coord_label, font=self.font_small)
            draw.rectangle(coord_bbox, fill=(0, 0, 0, 180))
            draw.text((x1, coord_y), coord_label, fill=color_rgba, font=self.font_small)

            bbox_count += 1

        # Draw assembled result bbox (if present)
        if assembled_bbox and len(assembled_bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in assembled_bbox]

            # Ensure within bounds
            x1 = max(0, min(x1, img_width))
            x2 = max(0, min(x2, img_width))
            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))

            if x2 > x1 and y2 > y1:
                # Draw in white with dashed effect
                assembled_color = (255, 255, 255, 220)

                # Draw thicker lines for assembled result
                for i in range(4):
                    draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=assembled_color, width=1)

                # Label
                label = "ASSEMBLED RESULT"
                label_y = min(y2 + 25, img_height - 45)

                text_bbox = draw.textbbox((x1, label_y), label, font=self.font_medium)
                draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
                draw.text((x1, label_y), label, fill=assembled_color, font=self.font_medium)

                coord_label = f"Pixel: [{x1}, {y1}, {x2}, {y2}]"
                coord_y = min(y2 + 45, img_height - 25)

                coord_bbox = draw.textbbox((x1, coord_y), coord_label, font=self.font_small)
                draw.rectangle(coord_bbox, fill=(0, 0, 0, 180))
                draw.text((x1, coord_y), coord_label, fill=assembled_color, font=self.font_small)

                bbox_count += 1

        # Composite overlay onto original image
        img = Image.alpha_composite(img, overlay)

        # Convert back to RGB for saving
        img = img.convert("RGB")

        # Generate save path
        if save_name:
            output_path = self.output_dir / f"{save_name}.png"
        else:
            base_name = Path(image_path).stem
            step_suffix = f"_step{step_number}" if step_number else ""
            output_path = self.output_dir / f"{base_name}{step_suffix}_bbox_viz.png"

        # Save
        img.save(output_path)

        logger.info(f"✓ Saved bbox visualization ({bbox_count} boxes) to: {output_path}")

        return str(output_path)

    def visualize_all_steps(
        self,
        image_path: str,
        extraction_results: List[Dict[str, Any]],
        base_name: Optional[str] = None
    ) -> List[str]:
        """
        Visualize bounding boxes for multiple steps from the same image.

        Args:
            image_path: Path to the original instruction page image
            extraction_results: List of step extraction results
            base_name: Optional base name for saved files

        Returns:
            List of paths to saved visualization images
        """
        saved_paths = []

        for idx, result in enumerate(extraction_results):
            step_number = result.get("step_number", idx + 1)

            save_name = None
            if base_name:
                save_name = f"{base_name}_step{step_number}"

            path = self.visualize_step_extraction(
                image_path=image_path,
                extraction_result=result,
                step_number=step_number,
                save_name=save_name
            )

            if path:
                saved_paths.append(path)

        logger.info(f"✓ Saved {len(saved_paths)} bbox visualizations")

        return saved_paths

    def create_summary_report(
        self,
        extraction_results: List[Dict[str, Any]],
        image_path: str
    ) -> str:
        """
        Create a text summary report of bbox extraction.

        Args:
            extraction_results: List of step extraction results
            image_path: Path to the image

        Returns:
            Summary report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BOUNDING BOX EXTRACTION SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Image: {image_path}")
        report_lines.append(f"Total steps extracted: {len(extraction_results)}")
        report_lines.append("")

        for idx, result in enumerate(extraction_results):
            step_num = result.get("step_number", idx + 1)
            parts = result.get("parts_required", [])
            assembled_bbox = result.get("assembled_result_bbox", None)

            report_lines.append(f"Step {step_num}:")
            report_lines.append(f"  Parts detected: {len(parts)}")

            # Count how many parts have bboxes
            parts_with_bbox = sum(1 for p in parts if p.get("bbox") is not None)
            parts_without_bbox = len(parts) - parts_with_bbox

            report_lines.append(f"  Parts with bboxes: {parts_with_bbox}/{len(parts)}")

            if parts_without_bbox > 0:
                report_lines.append(f"  ⚠ Parts missing bboxes: {parts_without_bbox}")

            if assembled_bbox:
                report_lines.append(f"  ✓ Assembled result bbox: {assembled_bbox}")
            else:
                report_lines.append(f"  ⚠ No assembled result bbox")

            # List each part
            for pidx, part in enumerate(parts):
                desc = part.get("description", "Unknown")[:40]
                color = part.get("color", "")
                qty = part.get("quantity", 1)
                bbox = part.get("bbox", None)

                bbox_str = f"{bbox}" if bbox else "MISSING"
                report_lines.append(f"    [{pidx+1}] {qty}x {color} {desc} - bbox: {bbox_str}")

            report_lines.append("")

        report_text = "\n".join(report_lines)

        # Save report to file
        report_path = self.output_dir / f"{Path(image_path).stem}_bbox_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        logger.info(f"✓ Saved bbox summary report to: {report_path}")

        return report_text
