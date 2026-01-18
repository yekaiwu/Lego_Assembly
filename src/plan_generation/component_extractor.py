"""
Component Extraction Service

This module coordinates the extraction of component images (parts and subassemblies)
during the hierarchical graph building process. It uses Computer Vision (CV) based
part detection with color segmentation and contour detection.

Features:
- Extract part images during part catalog building using CV detection
- Extract subassembly images during subassembly detection
- Color-based matching to link VLM semantic data with CV detections
- Manage component image storage directory structure
- Return image paths for inclusion in graph nodes
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from loguru import logger

from src.vision_processing.cv_part_detector import create_cv_detector_from_env, CVPartDetector


class ComponentExtractor:
    """
    Service for extracting component images during graph building.

    Coordinates between SAM segmentation and the graph building pipeline to extract
    and save individual part and subassembly images.
    """

    def __init__(
        self,
        output_dir: str,
        manual_id: str,
        cv_detector: Optional[CVPartDetector] = None
    ):
        """
        Initialize component extractor.

        Args:
            output_dir: Base output directory for the manual
            manual_id: Unique identifier for the instruction manual
            cv_detector: Optional CV detector instance (creates one if None)
        """
        self.output_dir = output_dir
        self.manual_id = manual_id
        self.cv_detector = cv_detector or create_cv_detector_from_env()

        # Create components directory
        self.components_dir = self._get_components_dir()
        os.makedirs(self.components_dir, exist_ok=True)

        # Track extracted components
        self.extracted_parts: Dict[str, str] = {}  # part_id -> image_path
        self.extracted_subassemblies: Dict[str, str] = {}  # subasm_id -> image_path

        logger.info(f"Initialized ComponentExtractor: components_dir={self.components_dir}")

    def _get_components_dir(self) -> str:
        """Get the components directory path."""
        import os
        from dotenv import load_dotenv

        load_dotenv()
        components_subdir = os.getenv("COMPONENTS_DIR", "components")

        return os.path.join(self.output_dir, components_subdir)

    def is_enabled(self) -> bool:
        """Check if component extraction is enabled and available."""
        return self.cv_detector is not None

    def extract_part_image(
        self,
        part_id: str,
        page_path: str,
        part_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Extract and save a part image using VLM center point hint + CV detection.

        Args:
            part_id: Unique identifier for the part
            page_path: Path to the instruction page containing the part
            part_data: Optional part data dictionary with center_point hint

        Returns:
            Relative path to the extracted part image, or None if extraction failed
        """
        if not self.is_enabled():
            logger.debug(f"CV detector not enabled, skipping part image extraction for {part_id}")
            return None

        # Check if already extracted
        if part_id in self.extracted_parts:
            logger.debug(f"Part {part_id} already extracted")
            return self.extracted_parts[part_id]

        # Verify page exists
        if not os.path.exists(page_path):
            logger.warning(f"Page not found for part {part_id}: {page_path}")
            return None

        try:
            # Get center point hint from part_data
            center_point = None
            if part_data and "center_point" in part_data:
                center_point = tuple(part_data["center_point"])
                logger.debug(f"Using VLM center point hint for {part_id}: {center_point}")

            if not center_point:
                logger.warning(f"No center_point provided for part {part_id}, cannot extract")
                return None

            # Detect part at center point using edge-based detection
            detection = self.cv_detector.detect_at_center_point(
                image_path=page_path,
                center_point=center_point,
                search_radius=150  # Search within 150px of the hint
            )

            if not detection:
                logger.warning(f"No part detected at center point {center_point} for {part_id}")
                return None

            # Extract using CV detector
            output_path = os.path.join(self.components_dir, f"part_{part_id}.png")
            part_image = self.cv_detector.extract_part_image(
                image_path=page_path,
                detection=detection,
                output_path=output_path
            )

            if part_image:
                # Convert to relative path for storage in graph
                rel_path = self._get_relative_path(output_path)
                self.extracted_parts[part_id] = rel_path
                logger.info(f"Extracted part image: {part_id} -> {rel_path}")
                return rel_path
            else:
                logger.warning(f"Failed to extract part image for {part_id}")
                return None

        except Exception as e:
            logger.error(f"Error extracting part image for {part_id}: {e}")
            return None

    def _match_part_to_detection(
        self,
        part_data: Optional[Dict[str, Any]],
        detections: List[Any]
    ) -> Optional[Any]:
        """
        Match a part to a CV detection based on color and region.

        Args:
            part_data: Part data dictionary with color/shape info
            detections: List of CV DetectedPart objects

        Returns:
            Best matching detection, or None
        """
        if not part_data:
            # No part data - return first parts_box detection
            for detection in detections:
                if detection.region_type == "parts_box":
                    return detection
            return None

        # Get part color from part_data
        part_color = part_data.get("color", "").lower()

        # Find detections in parts_box that match the color
        matches = []
        for detection in detections:
            if detection.region_type != "parts_box":
                continue

            # Check if colors match
            detection_color = detection.color_name.lower()
            if part_color in detection_color or detection_color in part_color:
                matches.append(detection)

        # Return the largest match (most likely to be the main part)
        if matches:
            return max(matches, key=lambda d: d.area)

        # Fallback: return first parts_box detection
        for detection in detections:
            if detection.region_type == "parts_box":
                return detection

        return None

    def extract_subassembly_image(
        self,
        subassembly_id: str,
        page_path: str,
        subassembly_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Extract and save a subassembly image using VLM center point hint + CV detection.

        Args:
            subassembly_id: Unique identifier for the subassembly
            page_path: Path to the instruction page containing the subassembly
            subassembly_data: Optional subassembly data dictionary with assembled_result_center

        Returns:
            Relative path to the extracted subassembly image, or None if extraction failed
        """
        if not self.is_enabled():
            logger.debug(f"CV detector not enabled, skipping subassembly image extraction for {subassembly_id}")
            return None

        # Check if already extracted
        if subassembly_id in self.extracted_subassemblies:
            logger.debug(f"Subassembly {subassembly_id} already extracted")
            return self.extracted_subassemblies[subassembly_id]

        # Verify page exists
        if not os.path.exists(page_path):
            logger.warning(f"Page not found for subassembly {subassembly_id}: {page_path}")
            return None

        try:
            # Get center point hint from subassembly_data
            center_point = None
            if subassembly_data and "assembled_result_center" in subassembly_data:
                center_point = tuple(subassembly_data["assembled_result_center"])
                logger.debug(f"Using VLM center point hint for subassembly {subassembly_id}: {center_point}")

            if not center_point:
                logger.warning(f"No assembled_result_center provided for subassembly {subassembly_id}, cannot extract")
                return None

            # Detect subassembly at center point using edge-based detection
            detection = self.cv_detector.detect_at_center_point(
                image_path=page_path,
                center_point=center_point,
                search_radius=200  # Larger radius for assembled results
            )

            if not detection:
                logger.warning(f"No subassembly detected at center point {center_point} for {subassembly_id}")
                return None

            # Extract using CV detector
            output_path = os.path.join(self.components_dir, f"subasm_{subassembly_id}.png")
            subasm_image = self.cv_detector.extract_part_image(
                image_path=page_path,
                detection=detection,
                output_path=output_path
            )

            if subasm_image:
                # Convert to relative path for storage in graph
                rel_path = self._get_relative_path(output_path)
                self.extracted_subassemblies[subassembly_id] = rel_path
                logger.info(f"Extracted subassembly image: {subassembly_id} -> {rel_path}")
                return rel_path
            else:
                logger.warning(f"Failed to extract subassembly image for {subassembly_id}")
                return None

        except Exception as e:
            logger.error(f"Error extracting subassembly image for {subassembly_id}: {e}")
            return None

    def extract_step_components(
        self,
        step_number: int,
        page_path: str,
        max_components: int = 20
    ) -> List[str]:
        """
        Extract all components from a step's instruction page using CV detection.

        Args:
            step_number: Step number
            page_path: Path to the instruction page
            max_components: Maximum number of components to extract

        Returns:
            List of relative paths to extracted component images
        """
        if not self.is_enabled():
            return []

        if not os.path.exists(page_path):
            logger.warning(f"Page not found for step {step_number}: {page_path}")
            return []

        try:
            # Detect all parts and assembled results
            detections = self.cv_detector.detect_parts(
                image_path=page_path,
                include_assembled=True
            )

            # Limit to max_components
            detections = detections[:max_components]

            component_paths = []
            for i, detection in enumerate(detections):
                output_path = os.path.join(
                    self.components_dir,
                    f"step_{step_number:03d}_component_{i:02d}.png"
                )

                part_image = self.cv_detector.extract_part_image(
                    image_path=page_path,
                    detection=detection,
                    output_path=output_path
                )

                if part_image:
                    component_paths.append(output_path)

            # Convert to relative paths
            rel_paths = [self._get_relative_path(path) for path in component_paths]

            logger.info(f"Extracted {len(rel_paths)} components from step {step_number}")
            return rel_paths

        except Exception as e:
            logger.error(f"Error extracting components from step {step_number}: {e}")
            return []

    def _get_relative_path(self, absolute_path: str) -> str:
        """
        Convert absolute path to relative path from output directory.

        Args:
            absolute_path: Absolute file path

        Returns:
            Relative path from output directory
        """
        try:
            abs_path = Path(absolute_path).resolve()
            output_path = Path(self.output_dir).resolve()
            return str(abs_path.relative_to(output_path))
        except ValueError:
            # If path is not relative to output_dir, return as-is
            logger.warning(f"Path {absolute_path} is not relative to {self.output_dir}")
            return absolute_path

    def get_part_image_path(self, part_id: str) -> Optional[str]:
        """
        Get the image path for a previously extracted part.

        Args:
            part_id: Part identifier

        Returns:
            Relative path to the part image, or None if not extracted
        """
        return self.extracted_parts.get(part_id)

    def get_subassembly_image_path(self, subassembly_id: str) -> Optional[str]:
        """
        Get the image path for a previously extracted subassembly.

        Args:
            subassembly_id: Subassembly identifier

        Returns:
            Relative path to the subassembly image, or None if not extracted
        """
        return self.extracted_subassemblies.get(subassembly_id)

    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of extracted components.

        Returns:
            Dictionary with extraction statistics
        """
        summary = {
            "total_parts": len(self.extracted_parts),
            "total_subassemblies": len(self.extracted_subassemblies),
            "components_dir": self.components_dir,
            "cv_enabled": self.is_enabled(),
            "extracted_parts": list(self.extracted_parts.keys()),
            "extracted_subassemblies": list(self.extracted_subassemblies.keys())
        }

        return summary


# Factory function for easy integration
def create_component_extractor(
    output_dir: str,
    manual_id: str
) -> ComponentExtractor:
    """
    Create a component extractor instance.

    Args:
        output_dir: Base output directory for the manual
        manual_id: Unique identifier for the instruction manual

    Returns:
        ComponentExtractor instance
    """
    return ComponentExtractor(
        output_dir=output_dir,
        manual_id=manual_id
    )


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python component_extractor.py <output_dir> <manual_id>")
        sys.exit(1)

    output_dir = sys.argv[1]
    manual_id = sys.argv[2]

    # Create extractor
    extractor = create_component_extractor(output_dir, manual_id)

    print(f"Component Extractor Status:")
    print(f"  SAM Enabled: {extractor.is_enabled()}")
    print(f"  Components Dir: {extractor.components_dir}")

    # Example: Extract a part image
    if extractor.is_enabled():
        page_path = os.path.join(output_dir, "temp_pages", "page_001.png")
        if os.path.exists(page_path):
            part_image = extractor.extract_part_image(
                part_id="part_001",
                page_path=page_path
            )
            print(f"  Extracted part: {part_image}")

    # Print summary
    summary = extractor.get_extraction_summary()
    print(f"\nExtraction Summary:")
    print(f"  Total Parts: {summary['total_parts']}")
    print(f"  Total Subassemblies: {summary['total_subassemblies']}")
