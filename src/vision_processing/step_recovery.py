"""
Missing Step Recovery Module
Detects and recovers missing steps from multi-step pages using targeted re-extraction.
"""

from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from loguru import logger
from pathlib import Path


class StepRecoveryModule:
    """Recovers missing steps by re-extracting pages where surrounding steps were found."""

    def __init__(self, vlm_extractor):
        """
        Initialize step recovery module.

        Args:
            vlm_extractor: VLMStepExtractor instance for re-extraction
        """
        self.vlm_extractor = vlm_extractor
        self.max_recovery_attempts = 2  # Max times to retry recovering each missing step

    def detect_missing_steps(self, extracted_steps: List[Dict[str, Any]]) -> List[int]:
        """
        Detect which step numbers are missing from the sequence.

        Args:
            extracted_steps: List of extracted step dictionaries

        Returns:
            List of missing step numbers (sorted)
        """
        if not extracted_steps:
            return []

        # Get all step numbers
        step_numbers = [step.get("step_number") for step in extracted_steps if step.get("step_number")]

        if not step_numbers:
            logger.warning("No valid step numbers found in extracted steps")
            return []

        # Find the range
        min_step = min(step_numbers)
        max_step = max(step_numbers)

        # Find missing steps in the range
        expected_steps = set(range(min_step, max_step + 1))
        actual_steps = set(step_numbers)
        missing_steps = sorted(expected_steps - actual_steps)

        if missing_steps:
            logger.warning(f"Missing {len(missing_steps)} steps: {missing_steps}")

        return missing_steps


    def recover_missing_steps(
        self,
        extracted_steps: List[Dict[str, Any]],
        step_groups: List[List[str]],
        assembly_id: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Attempt to recover missing steps by re-extracting relevant pages.

        Args:
            extracted_steps: List of extracted steps (may have gaps)
            step_groups: List of image path groups (one per page)
            assembly_id: Assembly ID for cache context

        Returns:
            Tuple of (updated_extracted_steps, num_recovered)
        """
        # First, we need to track which page each step came from
        # This requires modifying the extraction loop in main.py
        # For now, let's create a simpler approach

        missing_steps = self.detect_missing_steps(extracted_steps)

        if not missing_steps:
            logger.info("✓ No missing steps detected - all steps sequential")
            return extracted_steps, 0

        logger.info(f"Attempting to recover {len(missing_steps)} missing steps...")

        # Build mapping of step number to the steps list index
        step_num_to_idx = {
            step.get("step_number"): idx
            for idx, step in enumerate(extracted_steps)
            if step.get("step_number")
        }

        recovered_count = 0
        newly_extracted = []

        # For each missing step, find surrounding steps and their pages
        for missing_step_num in missing_steps:
            logger.info(f"  Attempting to recover step {missing_step_num}...")

            # Find the previous and next steps that exist
            prev_step_num = missing_step_num - 1
            next_step_num = missing_step_num + 1

            # Find pages to re-extract
            pages_to_retry = self._find_retry_pages(
                missing_step_num,
                prev_step_num,
                next_step_num,
                extracted_steps,
                step_groups
            )

            if not pages_to_retry:
                logger.warning(f"    ✗ Could not determine which pages to re-extract for step {missing_step_num}")
                continue

            # Re-extract with targeted prompt
            recovered_step = self._targeted_reextraction(
                missing_step_num,
                pages_to_retry,
                assembly_id
            )

            if recovered_step:
                logger.info(f"    ✓ Successfully recovered step {missing_step_num}")
                newly_extracted.append(recovered_step)
                recovered_count += 1
            else:
                logger.warning(f"    ✗ Failed to recover step {missing_step_num}")

        # Merge recovered steps back into the main list
        if newly_extracted:
            all_steps = extracted_steps + newly_extracted
            # Sort by step number
            all_steps.sort(key=lambda s: s.get("step_number", 999999))
            logger.info(f"✓ Recovered {recovered_count} missing steps")
            return all_steps, recovered_count

        return extracted_steps, 0

    def _find_retry_pages(
        self,
        missing_step_num: int,
        prev_step_num: int,
        next_step_num: int,
        extracted_steps: List[Dict[str, Any]],
        step_groups: List[List[str]]
    ) -> List[Tuple[int, List[str]]]:
        """
        Find which pages to re-extract to find the missing step.

        Returns:
            List of (page_index, image_paths) tuples to retry
        """
        pages_to_retry = set()

        # Find the previous step
        for step in extracted_steps:
            if step.get("step_number") == prev_step_num:
                page_idx = step.get("_source_page_idx")
                if page_idx is not None:
                    pages_to_retry.add(page_idx)
                    logger.debug(f"      Will retry page {page_idx+1} (has step {prev_step_num})")

        # Find the next step
        for step in extracted_steps:
            if step.get("step_number") == next_step_num:
                page_idx = step.get("_source_page_idx")
                if page_idx is not None:
                    pages_to_retry.add(page_idx)
                    logger.debug(f"      Will retry page {page_idx+1} (has step {next_step_num})")

        # Convert to list of (page_idx, image_paths) tuples
        retry_list = []
        for page_idx in sorted(pages_to_retry):
            if page_idx < len(step_groups):
                retry_list.append((page_idx, step_groups[page_idx]))

        return retry_list

    def _targeted_reextraction(
        self,
        missing_step_num: int,
        pages_to_retry: List[Tuple[int, List[str]]],
        assembly_id: str
    ) -> Dict[str, Any]:
        """
        Re-extract specific pages looking for a specific step number.

        Args:
            missing_step_num: The step number we're looking for
            pages_to_retry: List of (page_idx, image_paths) to re-extract
            assembly_id: Assembly ID for cache context

        Returns:
            Extracted step dict if found, None otherwise
        """
        for page_idx, image_paths in pages_to_retry:
            logger.debug(f"    Re-extracting page {page_idx+1} looking for step {missing_step_num}...")

            try:
                # Use targeted extraction with specific step number hint
                results = self.vlm_extractor.extract_step(
                    image_paths,
                    step_number=missing_step_num,  # Give VLM a hint
                    use_primary=True,
                    cache_context=f"{assembly_id}_recovery_{missing_step_num}"
                )

                # Look for the specific step in results
                for result in results:
                    if result.get("step_number") == missing_step_num:
                        return result

            except Exception as e:
                logger.warning(f"    Error re-extracting page {page_idx+1}: {e}")
                continue

        return None


def add_page_tracking_to_extraction(extracted_steps: List[Dict[str, Any]], page_idx: int):
    """
    Helper to add page tracking metadata to extracted steps.
    This should be called during the extraction loop in main.py

    Args:
        extracted_steps: Steps that were just extracted
        page_idx: The page index they came from
    """
    for step in extracted_steps:
        if "error" not in step:
            step["_source_page_idx"] = page_idx
