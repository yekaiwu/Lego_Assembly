"""
Document Analyzer for LEGO instruction manuals (Phase 0).
Analyzes entire PDF to identify relevant instruction pages and filter out
irrelevant content like covers, ads, and parts inventory.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class DocumentMetadata:
    """Metadata about the analyzed LEGO manual document."""
    main_build: str
    set_number: Optional[str]
    instruction_page_ranges: List[Tuple[int, int]]
    cover_pages: List[int]
    inventory_pages: List[int]
    ad_pages: List[int]
    total_pages: int
    total_steps: int
    has_alternate_builds: bool
    page_classification: Dict[int, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create from dictionary."""
        # Convert page_classification keys from strings to ints if needed
        if 'page_classification' in data:
            data['page_classification'] = {
                int(k): v for k, v in data['page_classification'].items()
            }
        # Convert instruction_page_ranges from lists to tuples if needed
        if 'instruction_page_ranges' in data:
            data['instruction_page_ranges'] = [
                tuple(r) for r in data['instruction_page_ranges']
            ]
        return cls(**data)


class DocumentAnalyzer:
    """
    Analyzes LEGO manual PDFs to identify:
    - Main build (what is being assembled)
    - Instruction pages vs. cover/ads/inventory
    - Total number of actual assembly steps
    - Presence of alternate builds
    """

    def __init__(self, vlm_client):
        """
        Initialize document analyzer.

        Args:
            vlm_client: VLM client for analysis (GeminiVisionClient or compatible)
        """
        self.vlm_client = vlm_client

    def analyze_pdf(self, page_paths: List[Path]) -> DocumentMetadata:
        """
        Perform high-level analysis of entire PDF.

        Process:
        1. Sample pages from PDF (first 5, middle 3, last 5)
        2. Send to VLM with document understanding prompt
        3. Identify document structure
        4. Classify page ranges

        Args:
            page_paths: List of paths to extracted page images

        Returns:
            DocumentMetadata with document structure information
        """
        logger.info(f"Analyzing document structure ({len(page_paths)} pages)...")

        # Sample pages for analysis
        sampled_pages = self._sample_pages(page_paths)
        logger.info(f"Sampled {len(sampled_pages)} pages for initial analysis")

        # Analyze with VLM
        analysis_result = self._analyze_with_vlm(sampled_pages)

        # Classify all pages
        page_classification = self._classify_pages(
            page_paths,
            analysis_result
        )

        # Build metadata
        metadata = self._build_metadata(
            page_paths,
            analysis_result,
            page_classification
        )

        logger.info(f"Document analysis complete:")
        logger.info(f"  Main build: {metadata.main_build}")
        logger.info(f"  Instruction pages: {sum(1 for c in page_classification.values() if c == 'instruction')}")
        logger.info(f"  Filtered pages: {sum(1 for c in page_classification.values() if c != 'instruction')}")

        return metadata

    def _sample_pages(self, page_paths: List[Path]) -> List[Path]:
        """
        Sample representative pages from the PDF.
        Takes first 5, middle 3, and last 5 pages.

        Args:
            page_paths: All page paths

        Returns:
            List of sampled page paths
        """
        total = len(page_paths)
        sampled = []

        # First 5 pages (cover, intro, inventory)
        sampled.extend(page_paths[:min(5, total)])

        # Middle 3 pages (instruction content)
        if total > 10:
            mid = total // 2
            sampled.extend(page_paths[mid-1:mid+2])

        # Last 5 pages (might be ads or alternate builds)
        if total > 5:
            sampled.extend(page_paths[-min(5, total):])

        # Remove duplicates while preserving order
        seen = set()
        unique_sampled = []
        for page in sampled:
            if page not in seen:
                seen.add(page)
                unique_sampled.append(page)

        return unique_sampled

    def _analyze_with_vlm(self, sampled_pages: List[Path]) -> Dict[str, Any]:
        """
        Analyze sampled pages with VLM to understand document structure.

        Args:
            sampled_pages: List of sampled page image paths

        Returns:
            Analysis result from VLM
        """
        prompt = self._build_document_understanding_prompt(len(sampled_pages))

        # Convert paths to strings
        image_paths = [str(p) for p in sampled_pages]

        try:
            # Call VLM with custom prompt
            if hasattr(self.vlm_client, 'extract_step_info_with_context'):
                result = self.vlm_client.extract_step_info_with_context(
                    image_paths=image_paths,
                    step_number=None,
                    custom_prompt=prompt,
                    use_json_mode=True
                )
            elif hasattr(self.vlm_client, 'generate_text'):
                result = self.vlm_client.generate_text(prompt, use_json_mode=True)
            else:
                # Fallback: use extract_step_info (not ideal but works)
                logger.warning("VLM client doesn't support custom prompts, using extract_step_info")
                result = self.vlm_client.extract_step_info(
                    image_paths=image_paths,
                    step_number=None,
                    use_json_mode=True
                )

            return result

        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            # Return minimal fallback result
            return {
                "main_build": "Unknown Build",
                "set_number": None,
                "page_classifications": {},
                "instruction_pages": [1, len(sampled_pages)],
                "estimated_steps": len(sampled_pages)
            }

    def _build_document_understanding_prompt(self, num_pages: int) -> str:
        """
        Build VLM prompt for document understanding.

        Args:
            num_pages: Number of pages being analyzed

        Returns:
            Prompt string
        """
        return f"""You are analyzing a LEGO instruction manual PDF.

I will show you {num_pages} sample pages from this document. Please identify:

1. **Main Build**: What is being built? (e.g., "Fire Truck", "Star Wars X-Wing")
2. **Set Number**: If visible (e.g., "6454922", "75301")
3. **Page Types**: For the pages shown, identify patterns to classify ALL pages as:
   - COVER: Title page, copyright, warnings, introduction
   - INVENTORY: Parts list showing all pieces with counts
   - INSTRUCTION: Step-by-step assembly instructions (numbered steps)
   - ADVERTISEMENT: Other sets, promotional content
   - REFERENCE: Final product photos, alternate angles

4. **Instruction Page Pattern**: What pattern identifies instruction pages?
   (e.g., "Pages with step numbers", "Pages showing building actions")

5. **Total Steps**: Approximately how many assembly steps are there?

Return JSON format:
{{
  "main_build": "Fire Truck",
  "set_number": "6454922",
  "instruction_pattern": "Pages with numbered steps and building actions",
  "cover_intro_pages": "1-5",
  "inventory_pages": "2-3",
  "instruction_pages_estimate": "6-50",
  "ad_pages": "51-52",
  "estimated_steps": 45,
  "has_alternate_builds": false
}}

Be specific about page number ranges. If you can't determine something, use null or "unknown"."""

    def _classify_pages(
        self,
        all_pages: List[Path],
        analysis_result: Dict[str, Any]
    ) -> Dict[int, str]:
        """
        Classify each page based on VLM analysis.

        Args:
            all_pages: All page paths
            analysis_result: VLM analysis result

        Returns:
            Dictionary mapping page number to classification
        """
        total_pages = len(all_pages)
        page_classification = {}

        # Parse instruction page range from analysis
        instruction_range_str = analysis_result.get("instruction_pages_estimate", "")
        instruction_start, instruction_end = self._parse_page_range(
            instruction_range_str,
            total_pages
        )

        # Parse cover/intro pages
        cover_range_str = analysis_result.get("cover_intro_pages", "1-5")
        cover_start, cover_end = self._parse_page_range(cover_range_str, total_pages)

        # Parse inventory pages
        inventory_range_str = analysis_result.get("inventory_pages", "")
        inventory_start, inventory_end = self._parse_page_range(inventory_range_str, total_pages)

        # Parse ad pages
        ad_range_str = analysis_result.get("ad_pages", "")
        ad_start, ad_end = self._parse_page_range(ad_range_str, total_pages)

        # Classify each page
        for page_idx, page_path in enumerate(all_pages):
            page_num = page_idx + 1

            # Default to instruction if in estimated instruction range
            if instruction_start and instruction_end and instruction_start <= page_num <= instruction_end:
                classification = "instruction"
            # Check other categories
            elif cover_start and cover_end and cover_start <= page_num <= cover_end:
                classification = "cover"
            elif inventory_start and inventory_end and inventory_start <= page_num <= inventory_end:
                classification = "inventory"
            elif ad_start and ad_end and ad_start <= page_num <= ad_end:
                classification = "advertisement"
            else:
                # Default to instruction for safety (better to include than exclude)
                classification = "instruction"

            page_classification[page_num] = classification

        return page_classification

    def _parse_page_range(
        self,
        range_str: str,
        total_pages: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse page range string like "1-5" or "6-50".

        Args:
            range_str: Range string (e.g., "1-5")
            total_pages: Total number of pages

        Returns:
            Tuple of (start, end) page numbers, or (None, None) if invalid
        """
        if not range_str or range_str == "unknown":
            return (None, None)

        try:
            # Handle formats like "1-5", "pages 1-5", etc.
            range_str = range_str.strip().lower()
            range_str = range_str.replace("pages", "").replace("page", "").strip()

            if "-" in range_str:
                parts = range_str.split("-")
                start = int(parts[0].strip())
                end = int(parts[1].strip())

                # Validate range
                start = max(1, min(start, total_pages))
                end = max(start, min(end, total_pages))

                return (start, end)
            else:
                # Single page number
                page_num = int(range_str)
                page_num = max(1, min(page_num, total_pages))
                return (page_num, page_num)

        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse page range '{range_str}': {e}")
            return (None, None)

    def _build_metadata(
        self,
        all_pages: List[Path],
        analysis_result: Dict[str, Any],
        page_classification: Dict[int, str]
    ) -> DocumentMetadata:
        """
        Build DocumentMetadata from analysis results.

        Args:
            all_pages: All page paths
            analysis_result: VLM analysis result
            page_classification: Page classification mapping

        Returns:
            DocumentMetadata object
        """
        # Extract instruction page ranges
        instruction_pages = [
            page_num for page_num, cls in page_classification.items()
            if cls == "instruction"
        ]

        # Group consecutive instruction pages into ranges
        instruction_ranges = []
        if instruction_pages:
            start = instruction_pages[0]
            end = instruction_pages[0]

            for page_num in instruction_pages[1:]:
                if page_num == end + 1:
                    end = page_num
                else:
                    instruction_ranges.append((start, end))
                    start = page_num
                    end = page_num

            instruction_ranges.append((start, end))

        # Extract other page types
        cover_pages = [p for p, c in page_classification.items() if c == "cover"]
        inventory_pages = [p for p, c in page_classification.items() if c == "inventory"]
        ad_pages = [p for p, c in page_classification.items() if c == "advertisement"]

        return DocumentMetadata(
            main_build=analysis_result.get("main_build", "Unknown Build"),
            set_number=analysis_result.get("set_number"),
            instruction_page_ranges=instruction_ranges,
            cover_pages=cover_pages,
            inventory_pages=inventory_pages,
            ad_pages=ad_pages,
            total_pages=len(all_pages),
            total_steps=analysis_result.get("estimated_steps", len(instruction_pages)),
            has_alternate_builds=analysis_result.get("has_alternate_builds", False),
            page_classification=page_classification
        )

    def extract_relevant_pages(
        self,
        all_pages: List[Path],
        metadata: DocumentMetadata
    ) -> List[Path]:
        """
        Filter to only instruction pages.

        Args:
            all_pages: All page paths
            metadata: Document metadata

        Returns:
            List of instruction page paths only
        """
        relevant_pages = []

        for page_idx, page_path in enumerate(all_pages):
            page_num = page_idx + 1
            classification = metadata.page_classification.get(page_num, "instruction")

            if classification == "instruction":
                relevant_pages.append(page_path)

        logger.info(f"Filtered {len(all_pages)} pages → {len(relevant_pages)} instruction pages")

        return relevant_pages

    def get_user_confirmation(
        self,
        metadata: DocumentMetadata
    ) -> bool:
        """
        Present findings to user for confirmation.

        Args:
            metadata: Document metadata

        Returns:
            True if user confirms, False otherwise
        """
        print("\n" + "=" * 80)
        print("DOCUMENT ANALYSIS RESULTS")
        print("=" * 80)
        print(f"\nBuild: {metadata.main_build}")
        if metadata.set_number:
            print(f"Set Number: {metadata.set_number}")
        print(f"Total Pages: {metadata.total_pages}")
        print(f"Instruction Pages: {sum(1 for c in metadata.page_classification.values() if c == 'instruction')}")

        if metadata.instruction_page_ranges:
            ranges_str = ", ".join(
                f"{start}-{end}" if start != end else str(start)
                for start, end in metadata.instruction_page_ranges
            )
            print(f"  Page ranges: {ranges_str}")

        print(f"Estimated Steps: {metadata.total_steps}")

        if metadata.cover_pages or metadata.inventory_pages or metadata.ad_pages:
            print(f"\nFiltered Out:")
            if metadata.cover_pages:
                print(f"  - Cover/Intro: pages {self._format_page_list(metadata.cover_pages)}")
            if metadata.inventory_pages:
                print(f"  - Parts Inventory: pages {self._format_page_list(metadata.inventory_pages)}")
            if metadata.ad_pages:
                print(f"  - Advertisements: pages {self._format_page_list(metadata.ad_pages)}")

        if metadata.has_alternate_builds:
            print(f"\n⚠️  This manual may contain alternate builds")

        print("\n" + "=" * 80)

        try:
            response = input("\nProceed with processing these instruction pages? (y/n): ").strip().lower()
            return response in ['y', 'yes']
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled by user.")
            return False

    def _format_page_list(self, pages: List[int]) -> str:
        """
        Format list of page numbers into readable string.

        Args:
            pages: List of page numbers

        Returns:
            Formatted string (e.g., "1-5, 7, 9-12")
        """
        if not pages:
            return ""

        sorted_pages = sorted(pages)
        ranges = []
        start = sorted_pages[0]
        end = sorted_pages[0]

        for page in sorted_pages[1:]:
            if page == end + 1:
                end = page
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = page
                end = page

        # Add last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ", ".join(ranges)
