"""
User Metadata Collector for Manual Processing.
Replaces automatic VLM classification with user-provided metadata.
"""

import re
from typing import List, Dict, Any, Optional
from loguru import logger

from .metadata_models import UserProvidedMetadata


class UserMetadataCollector:
    """
    Collects manual metadata from user input.
    Supports both interactive prompts and programmatic input.
    """

    def __init__(self, total_pages: int):
        """
        Initialize collector.

        Args:
            total_pages: Total number of pages in the PDF
        """
        self.total_pages = total_pages

    def collect_metadata_interactive(self) -> UserProvidedMetadata:
        """
        Collect metadata through interactive prompts.

        Returns:
            UserProvidedMetadata object
        """
        print("\n" + "=" * 80)
        print("MANUAL METADATA COLLECTION")
        print("=" * 80)
        print(f"\nTotal pages in PDF: {self.total_pages}")
        print("\nPlease provide information about the manual structure.")
        print("Use page ranges (e.g., '5-47'), lists (e.g., '1,3,5'), or combinations.")
        print("=" * 80 + "\n")

        # Collect main build name
        main_build = self._prompt_text(
            "What is being built? (e.g., 'Fire Truck Set #6454922')",
            required=True
        )

        # Extract set number from main_build if it looks like it contains one
        set_number = self._extract_set_number(main_build)

        # Collect instruction pages (required)
        instruction_pages = self._prompt_pages(
            "Assembly instruction pages (e.g., '5-47' or '5,7-12,15-47')",
            required=True
        )

        # Collect optional pages
        final_product_pages = self._prompt_pages(
            "Final product/completed build pages (optional, press Enter to skip)",
            required=False
        )

        parts_list_pages = self._prompt_pages(
            "Parts list/inventory pages (optional, press Enter to skip)",
            required=False
        )

        # Optional notes
        notes = self._prompt_text(
            "Any additional notes? (optional, press Enter to skip)",
            required=False
        )

        # Create metadata object
        metadata = UserProvidedMetadata(
            main_build=main_build,
            total_pages=self.total_pages,
            instruction_pages=instruction_pages,
            final_product_pages=final_product_pages or [],
            parts_list_pages=parts_list_pages or [],
            set_number=set_number,
            notes=notes or None
        )

        # Display summary and confirm
        print("\n" + metadata.display_summary())

        confirm = input("\nProceed with this metadata? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("\nMetadata collection cancelled. Please restart.")
            raise ValueError("User cancelled metadata collection")

        logger.info(f"Metadata collected: {metadata.main_build}, {metadata.total_instruction_pages} instruction pages")

        return metadata

    def collect_metadata_from_dict(self, data: Dict[str, Any]) -> UserProvidedMetadata:
        """
        Collect metadata from dictionary (programmatic use).

        Args:
            data: Dictionary with metadata fields:
                - main_build: str (required)
                - instruction_pages: str or List[int] (required)
                - final_product_pages: str or List[int] (optional)
                - parts_list_pages: str or List[int] (optional)
                - set_number: str (optional)
                - notes: str (optional)

        Returns:
            UserProvidedMetadata object
        """
        # Parse instruction pages
        instruction_pages_raw = data.get("instruction_pages")
        if isinstance(instruction_pages_raw, str):
            instruction_pages = self.parse_page_input(instruction_pages_raw)
        elif isinstance(instruction_pages_raw, list):
            instruction_pages = instruction_pages_raw
        else:
            raise ValueError("instruction_pages must be string or list")

        # Parse optional pages
        final_product_pages = self._parse_optional_pages(
            data.get("final_product_pages")
        )
        parts_list_pages = self._parse_optional_pages(
            data.get("parts_list_pages")
        )

        # Extract set number if not provided
        set_number = data.get("set_number")
        if not set_number:
            set_number = self._extract_set_number(data["main_build"])

        return UserProvidedMetadata(
            main_build=data["main_build"],
            total_pages=self.total_pages,
            instruction_pages=instruction_pages,
            final_product_pages=final_product_pages,
            parts_list_pages=parts_list_pages,
            set_number=set_number,
            notes=data.get("notes")
        )

    def _parse_optional_pages(self, pages_input: Any) -> List[int]:
        """Parse optional page input."""
        if not pages_input:
            return []
        if isinstance(pages_input, str):
            return self.parse_page_input(pages_input)
        elif isinstance(pages_input, list):
            return pages_input
        else:
            return []

    def _prompt_text(self, prompt: str, required: bool = False) -> Optional[str]:
        """
        Prompt user for text input.

        Args:
            prompt: Prompt message
            required: Whether input is required

        Returns:
            User input or None
        """
        while True:
            response = input(f"{prompt}: ").strip()

            if response:
                return response
            elif not required:
                return None
            else:
                print("  ⚠️  This field is required. Please provide a value.")

    def _prompt_pages(self, prompt: str, required: bool = False) -> Optional[List[int]]:
        """
        Prompt user for page numbers.

        Args:
            prompt: Prompt message
            required: Whether input is required

        Returns:
            List of page numbers or None
        """
        while True:
            response = input(f"{prompt}: ").strip()

            if not response:
                if not required:
                    return None
                else:
                    print("  ⚠️  This field is required. Please provide page numbers.")
                    continue

            try:
                pages = self.parse_page_input(response)

                # Validate pages
                invalid_pages = [p for p in pages if p < 1 or p > self.total_pages]
                if invalid_pages:
                    print(f"  ⚠️  Invalid page numbers: {invalid_pages}. "
                          f"Pages must be between 1 and {self.total_pages}.")
                    continue

                return pages

            except ValueError as e:
                print(f"  ⚠️  Invalid format: {e}")
                print("  Use ranges (e.g., '5-47'), lists (e.g., '1,3,5'), or combinations.")

    def parse_page_input(self, page_str: str) -> List[int]:
        """
        Parse page number input string.

        Supports:
        - Ranges: "5-47" → [5, 6, 7, ..., 47]
        - Lists: "1, 3, 5" → [1, 3, 5]
        - Combined: "1-5, 10, 15-20" → [1,2,3,4,5,10,15,16,17,18,19,20]

        Args:
            page_str: Input string

        Returns:
            List of page numbers

        Raises:
            ValueError: If input format is invalid
        """
        pages = []

        # Split by comma
        parts = page_str.split(',')

        for part in parts:
            part = part.strip()

            if '-' in part:
                # Range: "5-47"
                try:
                    start, end = part.split('-')
                    start = int(start.strip())
                    end = int(end.strip())

                    if start > end:
                        raise ValueError(f"Invalid range: {start}-{end} (start > end)")

                    pages.extend(range(start, end + 1))
                except ValueError as e:
                    raise ValueError(f"Invalid range format '{part}': {e}")

            else:
                # Single page: "5"
                try:
                    page_num = int(part.strip())
                    pages.append(page_num)
                except ValueError:
                    raise ValueError(f"Invalid page number: '{part}'")

        # Remove duplicates and sort
        pages = sorted(set(pages))

        if not pages:
            raise ValueError("No pages specified")

        return pages

    def _extract_set_number(self, text: str) -> Optional[str]:
        """
        Try to extract LEGO set number from text.

        Common patterns:
        - "Set #42123"
        - "Set 42123"
        - "#42123"
        - "42123"

        Args:
            text: Text to search

        Returns:
            Set number if found, else None
        """
        # Pattern: # followed by 4-6 digits
        match = re.search(r'#?(\d{4,6})', text)
        if match:
            return match.group(1)

        return None
