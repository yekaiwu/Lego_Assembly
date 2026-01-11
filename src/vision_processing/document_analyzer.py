"""
Document Analyzer - Legacy compatibility module.

This module now only provides conversion functions between the new UserProvidedMetadata
and legacy DocumentMetadata format. The VLM-based automatic classification has been removed.

Use instead:
- metadata_models.UserProvidedMetadata for data models
- user_metadata_collector.UserMetadataCollector for collecting metadata
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

from .metadata_models import UserProvidedMetadata


@dataclass
class DocumentMetadata:
    """
    Legacy metadata format - kept only for checkpoint compatibility.
    New code should use UserProvidedMetadata instead.
    """
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


def convert_user_metadata_to_document_metadata(
    user_metadata: UserProvidedMetadata
) -> DocumentMetadata:
    """
    Convert UserProvidedMetadata to legacy DocumentMetadata format.
    Used for Phase 1 compatibility.

    Args:
        user_metadata: UserProvidedMetadata object

    Returns:
        DocumentMetadata object
    """
    page_classification = user_metadata.get_page_classification()

    # Map new classifications to old ones
    classification_map = {
        "instruction": "instruction",
        "final_product": "cover",  # Treat as non-instruction
        "parts_list": "inventory",
        "other": "cover"  # Treat as non-instruction
    }

    mapped_classification = {
        page: classification_map[cls]
        for page, cls in page_classification.items()
    }

    return DocumentMetadata(
        main_build=user_metadata.main_build,
        set_number=user_metadata.set_number,
        instruction_page_ranges=user_metadata.instruction_page_ranges,
        cover_pages=[p for p, c in mapped_classification.items() if c == "cover"],
        inventory_pages=[p for p, c in mapped_classification.items() if c == "inventory"],
        ad_pages=[],  # Not tracked in new system
        total_pages=user_metadata.total_pages,
        total_steps=user_metadata.estimated_steps,
        has_alternate_builds=False,  # Not tracked in new system
        page_classification=mapped_classification
    )


def extract_relevant_pages(
    all_pages: List[Path],
    user_metadata: UserProvidedMetadata
) -> List[Path]:
    """
    Filter to only instruction pages based on user-provided metadata.

    Args:
        all_pages: All page paths
        user_metadata: User-provided metadata

    Returns:
        List of instruction page paths only
    """
    relevant_pages = []
    instruction_page_nums = set(user_metadata.instruction_pages)

    for page_idx, page_path in enumerate(all_pages):
        page_num = page_idx + 1  # Pages are 1-indexed
        if page_num in instruction_page_nums:
            relevant_pages.append(page_path)

    logger.info(f"Filtered {len(all_pages)} pages → {len(relevant_pages)} instruction pages (user-provided)")

    return relevant_pages
