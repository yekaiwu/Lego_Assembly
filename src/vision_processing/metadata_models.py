"""
Data models for user-provided manual metadata.
Replaces automatic VLM-based document classification with user input.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class UserProvidedMetadata:
    """
    Metadata provided by user during manual ingestion.
    Replaces automatic VLM document classification.
    """

    # Core information (required)
    main_build: str  # e.g., "Race Car Set #42123"
    total_pages: int  # Total pages in PDF

    # User-specified page classifications
    instruction_pages: List[int]  # e.g., [5, 6, 7, ..., 47]
    final_product_pages: List[int] = None  # e.g., [48, 50] (optional)
    parts_list_pages: List[int] = None  # e.g., [51] (optional)

    # Optional metadata
    set_number: Optional[str] = None  # e.g., "42123"
    notes: Optional[str] = None  # Any additional user notes

    def __post_init__(self):
        """Validate and initialize derived fields."""
        # Initialize empty lists if None
        if self.final_product_pages is None:
            self.final_product_pages = []
        if self.parts_list_pages is None:
            self.parts_list_pages = []

        # Validate page numbers
        self._validate_pages()

        # Sort instruction pages for consistency
        self.instruction_pages = sorted(self.instruction_pages)

    def _validate_pages(self):
        """Validate that page numbers are within total_pages."""
        all_pages = (
            self.instruction_pages +
            self.final_product_pages +
            self.parts_list_pages
        )

        for page_num in all_pages:
            if page_num < 1 or page_num > self.total_pages:
                raise ValueError(
                    f"Page number {page_num} is out of range (1-{self.total_pages})"
                )

        # Check for duplicates across categories
        instruction_set = set(self.instruction_pages)
        final_set = set(self.final_product_pages)
        parts_set = set(self.parts_list_pages)

        overlap = instruction_set & final_set
        if overlap:
            logger.warning(
                f"Pages {overlap} marked as both instruction and final product. "
                f"Using as instruction pages."
            )

        overlap = instruction_set & parts_set
        if overlap:
            logger.warning(
                f"Pages {overlap} marked as both instruction and parts list. "
                f"Using as instruction pages."
            )

    @property
    def instruction_page_ranges(self) -> List[Tuple[int, int]]:
        """
        Convert list of instruction pages to ranges for backward compatibility.

        Example: [5, 6, 7, 10, 11, 12] → [(5, 7), (10, 12)]
        """
        return self._list_to_ranges(self.instruction_pages)

    @property
    def total_instruction_pages(self) -> int:
        """Total number of instruction pages."""
        return len(self.instruction_pages)

    @property
    def estimated_steps(self) -> int:
        """
        Estimate number of assembly steps.
        Assumes roughly 1 step per instruction page (conservative estimate).
        """
        return len(self.instruction_pages)

    def _list_to_ranges(self, pages: List[int]) -> List[Tuple[int, int]]:
        """
        Convert list of page numbers to consecutive ranges.

        Args:
            pages: Sorted list of page numbers

        Returns:
            List of (start, end) tuples representing consecutive ranges
        """
        if not pages:
            return []

        sorted_pages = sorted(pages)
        ranges = []
        start = sorted_pages[0]
        end = sorted_pages[0]

        for page in sorted_pages[1:]:
            if page == end + 1:
                # Consecutive page, extend range
                end = page
            else:
                # Gap found, save current range and start new one
                ranges.append((start, end))
                start = page
                end = page

        # Add final range
        ranges.append((start, end))

        return ranges

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "main_build": self.main_build,
            "total_pages": self.total_pages,
            "instruction_pages": self.instruction_pages,
            "final_product_pages": self.final_product_pages,
            "parts_list_pages": self.parts_list_pages,
            "set_number": self.set_number,
            "notes": self.notes,
            # Derived fields
            "instruction_page_ranges": self.instruction_page_ranges,
            "total_instruction_pages": self.total_instruction_pages,
            "estimated_steps": self.estimated_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProvidedMetadata':
        """Create from dictionary."""
        # Only use fields that are constructor parameters
        return cls(
            main_build=data["main_build"],
            total_pages=data["total_pages"],
            instruction_pages=data["instruction_pages"],
            final_product_pages=data.get("final_product_pages", []),
            parts_list_pages=data.get("parts_list_pages", []),
            set_number=data.get("set_number"),
            notes=data.get("notes"),
        )

    def get_page_classification(self) -> Dict[int, str]:
        """
        Get page classification mapping for backward compatibility.

        Returns:
            Dictionary mapping page number to classification
            (instruction, final_product, parts_list, other)
        """
        classification = {}

        # Classify all pages
        for page_num in range(1, self.total_pages + 1):
            if page_num in self.instruction_pages:
                classification[page_num] = "instruction"
            elif page_num in self.final_product_pages:
                classification[page_num] = "final_product"
            elif page_num in self.parts_list_pages:
                classification[page_num] = "parts_list"
            else:
                classification[page_num] = "other"

        return classification

    def display_summary(self) -> str:
        """
        Generate human-readable summary of metadata.

        Returns:
            Formatted string for display
        """
        lines = []
        lines.append("=" * 80)
        lines.append("MANUAL METADATA")
        lines.append("=" * 80)
        lines.append(f"\nBuild: {self.main_build}")

        if self.set_number:
            lines.append(f"Set Number: {self.set_number}")

        lines.append(f"Total Pages: {self.total_pages}")
        lines.append(f"\nInstruction Pages: {self.total_instruction_pages}")

        # Show instruction page ranges
        if self.instruction_page_ranges:
            ranges_str = ", ".join(
                f"{start}-{end}" if start != end else str(start)
                for start, end in self.instruction_page_ranges
            )
            lines.append(f"  Page ranges: {ranges_str}")

        lines.append(f"Estimated Steps: {self.estimated_steps}")

        # Show optional pages
        if self.final_product_pages:
            pages_str = ", ".join(map(str, self.final_product_pages))
            lines.append(f"\nFinal Product Pages: {pages_str}")

        if self.parts_list_pages:
            pages_str = ", ".join(map(str, self.parts_list_pages))
            lines.append(f"Parts List Pages: {pages_str}")

        if self.notes:
            lines.append(f"\nNotes: {self.notes}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)
