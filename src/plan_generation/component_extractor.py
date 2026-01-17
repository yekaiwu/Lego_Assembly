"""
Component Extraction Service

This module coordinates the extraction of component images (parts and subassemblies)
during the hierarchical graph building process. It integrates SAM segmentation with
the part association and subassembly detection modules.

Features:
- Extract part images during part catalog building
- Extract subassembly images during subassembly detection
- Manage component image storage directory structure
- Return image paths for inclusion in graph nodes
- Handle graceful degradation when SAM is unavailable
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from loguru import logger

from src.vision_processing.sam_segmenter import create_sam_segmenter_from_env, SAMSegmenter


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
        sam_segmenter: Optional[SAMSegmenter] = None
    ):
        """
        Initialize component extractor.

        Args:
            output_dir: Base output directory for the manual
            manual_id: Unique identifier for the instruction manual
            sam_segmenter: Optional SAM segmenter instance (creates one if None)
        """
        self.output_dir = output_dir
        self.manual_id = manual_id
        self.sam_segmenter = sam_segmenter or create_sam_segmenter_from_env()

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
        return self.sam_segmenter is not None and self.sam_segmenter.is_available()

    def extract_part_image(
        self,
        part_id: str,
        page_path: str,
        bbox_hint: Optional[tuple] = None,
        part_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Extract and save a part image.

        Args:
            part_id: Unique identifier for the part
            page_path: Path to the instruction page containing the part
            bbox_hint: Optional bounding box hint (x1, y1, x2, y2)
            part_data: Optional part data dictionary (for future enhancements)

        Returns:
            Relative path to the extracted part image, or None if extraction failed
        """
        if not self.is_enabled():
            logger.debug(f"SAM not enabled, skipping part image extraction for {part_id}")
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
            # Extract the part image
            image_path = self.sam_segmenter.extract_part_image(
                page_path=page_path,
                part_id=part_id,
                output_dir=self.components_dir,
                bbox_hint=bbox_hint
            )

            if image_path:
                # Convert to relative path for storage in graph
                rel_path = self._get_relative_path(image_path)
                self.extracted_parts[part_id] = rel_path
                logger.info(f"Extracted part image: {part_id} -> {rel_path}")
                return rel_path
            else:
                logger.warning(f"Failed to extract part image for {part_id}")
                return None

        except Exception as e:
            logger.error(f"Error extracting part image for {part_id}: {e}")
            return None

    def extract_subassembly_image(
        self,
        subassembly_id: str,
        page_path: str,
        bbox_hint: Optional[tuple] = None,
        subassembly_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Extract and save a subassembly image.

        Args:
            subassembly_id: Unique identifier for the subassembly
            page_path: Path to the instruction page containing the subassembly
            bbox_hint: Optional bounding box hint (x1, y1, x2, y2)
            subassembly_data: Optional subassembly data dictionary (for future enhancements)

        Returns:
            Relative path to the extracted subassembly image, or None if extraction failed
        """
        if not self.is_enabled():
            logger.debug(f"SAM not enabled, skipping subassembly image extraction for {subassembly_id}")
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
            # Extract the subassembly image
            image_path = self.sam_segmenter.extract_subassembly_image(
                page_path=page_path,
                subassembly_id=subassembly_id,
                output_dir=self.components_dir,
                bbox_hint=bbox_hint
            )

            if image_path:
                # Convert to relative path for storage in graph
                rel_path = self._get_relative_path(image_path)
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
        Extract all components from a step's instruction page.

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
            # Extract all components from the page
            component_paths = self.sam_segmenter.extract_multiple_components(
                image_path=page_path,
                output_dir=self.components_dir,
                prefix=f"step_{step_number:03d}_component",
                max_components=max_components
            )

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
            Dictionary with extraction statistics including cache performance
        """
        summary = {
            "total_parts": len(self.extracted_parts),
            "total_subassemblies": len(self.extracted_subassemblies),
            "components_dir": self.components_dir,
            "sam_enabled": self.is_enabled(),
            "extracted_parts": list(self.extracted_parts.keys()),
            "extracted_subassemblies": list(self.extracted_subassemblies.keys())
        }

        # Include cache statistics if SAM is enabled
        if self.is_enabled() and self.sam_segmenter:
            cache_stats = self.sam_segmenter.get_cache_stats()
            summary["cache_stats"] = cache_stats

            # Log cache performance
            self.sam_segmenter.log_cache_stats()

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
