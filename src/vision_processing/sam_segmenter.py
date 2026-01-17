"""
SAM (Segment Anything Model) Integration for Component Extraction

This module provides functionality to segment and extract individual components
(parts and subassemblies) from LEGO instruction manual pages using SAM 2.0.

Features:
- Load and initialize SAM models
- Segment instruction pages to identify discrete components
- Extract and save individual part images
- Extract and save subassembly images
- Bounding box detection and cropping
- Integration with existing pipeline configuration
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from loguru import logger

try:
    from ultralytics import SAM
    SAM_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics SAM not available. Install with: pip install ultralytics")
    SAM_AVAILABLE = False


class SAMSegmenter:
    """
    Segment Anything Model (SAM) integration for LEGO component extraction.

    Uses SAM 2.0 via ultralytics to segment instruction manual pages and extract
    individual components (parts and subassemblies).
    """

    def __init__(
        self,
        model_name: str = "sam2_b.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        inference_timeout: int = 300
    ):
        """
        Initialize SAM segmenter.

        Args:
            model_name: SAM model variant (sam2_b, sam2_l, sam2_s, sam2_t)
            confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
            device: Device to run model on ('cpu', 'cuda', 'mps')
            inference_timeout: Maximum seconds to wait for inference (default: 300)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.inference_timeout = inference_timeout
        self.model = None

        # Page segmentation cache to avoid re-segmenting the same page
        self._page_segment_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Track MPS warnings
        self._mps_warning_shown = False

        if not SAM_AVAILABLE:
            logger.warning("SAM is not available. Component extraction will be disabled.")
            return

        # Warn about MPS potential issues
        if device == "mps":
            logger.warning(
                "SAM on Apple Silicon MPS may have compatibility issues. "
                "If you experience hangs, consider using device='cpu' instead."
            )
            self._mps_warning_shown = True

        logger.info(f"Initializing SAM with model: {model_name}, device: {device}, timeout: {inference_timeout}s")
        self._load_model()

    def _load_model(self):
        """Load the SAM model."""
        if not SAM_AVAILABLE:
            return

        try:
            # Ultralytics SAM will automatically download the model if not present
            self.model = SAM(self.model_name)

            # Set device
            if self.device != "cpu":
                self.model.to(self.device)

            logger.info(f"SAM model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if SAM is available and loaded."""
        return SAM_AVAILABLE and self.model is not None

    def _run_inference_with_timeout(
        self,
        image_path: str,
        points: Optional[List[Tuple[int, int]]] = None,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ):
        """
        Run SAM inference with timeout protection.

        Args:
            image_path: Path to image
            points: Optional point prompts
            boxes: Optional box prompts

        Returns:
            SAM results

        Raises:
            TimeoutError: If inference exceeds timeout
        """
        def _inference():
            if points is not None:
                return self.model(image_path, points=points, conf=self.confidence_threshold)
            elif boxes is not None:
                return self.model(image_path, bboxes=boxes, conf=self.confidence_threshold)
            else:
                return self.model(image_path, conf=self.confidence_threshold)

        # Run inference with timeout
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_inference)
        try:
            results = future.result(timeout=self.inference_timeout)
            executor.shutdown(wait=True)
            return results
        except FuturesTimeoutError:
            # Cancel the future and shutdown without waiting
            # Note: The background thread may still run, but we don't wait for it
            future.cancel()
            executor.shutdown(wait=False)
            logger.error(
                f"SAM inference timeout ({self.inference_timeout}s) for {image_path}. "
                f"Consider using device='cpu' or increasing timeout. "
                f"Background inference may still be running."
            )
            raise TimeoutError(f"SAM inference exceeded {self.inference_timeout}s timeout")

    def segment_page(
        self,
        image_path: str,
        points: Optional[List[Tuple[int, int]]] = None,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Segment an instruction page image.

        Args:
            image_path: Path to the instruction page image
            points: Optional list of (x, y) points to guide segmentation
            boxes: Optional list of (x1, y1, x2, y2) bounding boxes to guide segmentation

        Returns:
            List of segment dictionaries containing masks, boxes, and confidence scores
        """
        if not self.is_available():
            logger.warning("SAM not available, returning empty segments")
            return []

        # Check cache (only for auto-segmentation without hints)
        if points is None and boxes is None:
            if image_path in self._page_segment_cache:
                self._cache_hits += 1
                logger.debug(f"Cache HIT for {image_path} (hits: {self._cache_hits}, misses: {self._cache_misses})")
                return self._page_segment_cache[image_path]
            self._cache_misses += 1

        try:
            # Log inference start
            page_name = os.path.basename(image_path)
            if points is not None:
                logger.info(f"Running SAM inference (point-guided) on {page_name}...")
            elif boxes is not None:
                logger.info(f"Running SAM inference (box-guided) on {page_name}...")
            else:
                logger.info(f"Running SAM auto-segmentation on {page_name} (may take 1-5 seconds)...")

            # Run SAM inference with timeout protection
            start_time = time.time()
            results = self._run_inference_with_timeout(image_path, points, boxes)
            inference_time = time.time() - start_time

            logger.debug(f"SAM inference completed in {inference_time:.2f}s for {page_name}")

            segments = []
            for result in results:
                if result.masks is not None:
                    for i, mask in enumerate(result.masks.data):
                        # Extract bounding box
                        mask_np = mask.cpu().numpy()
                        y_indices, x_indices = np.where(mask_np > 0.5)

                        if len(x_indices) == 0 or len(y_indices) == 0:
                            continue

                        x1, y1 = int(x_indices.min()), int(y_indices.min())
                        x2, y2 = int(x_indices.max()), int(y_indices.max())

                        segments.append({
                            "mask": mask_np,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": float(result.boxes.conf[i]) if result.boxes is not None else 1.0,
                            "area": int((x2 - x1) * (y2 - y1))
                        })

            # Sort by area (largest first)
            segments.sort(key=lambda x: x["area"], reverse=True)

            # Store in cache (only for auto-segmentation without hints)
            if points is None and boxes is None:
                self._page_segment_cache[image_path] = segments
                logger.debug(f"Cached segmentation for {image_path}: {len(segments)} segments")

            logger.info(f"Segmented {len(segments)} components from {image_path}")
            return segments

        except Exception as e:
            logger.error(f"Error segmenting image {image_path}: {e}")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics including hits, misses, hit rate, and size
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "cached_pages": len(self._page_segment_cache)
        }

    def clear_cache(self):
        """Clear the page segmentation cache to free memory."""
        cache_size = len(self._page_segment_cache)
        self._page_segment_cache.clear()
        logger.info(f"Cleared segmentation cache ({cache_size} pages)")

    def log_cache_stats(self):
        """Log cache performance statistics."""
        stats = self.get_cache_stats()
        if stats["total_requests"] > 0:
            logger.info(
                f"SAM Cache Performance: {stats['cache_hits']} hits, "
                f"{stats['cache_misses']} misses, "
                f"{stats['hit_rate_percent']}% hit rate, "
                f"{stats['cached_pages']} pages cached"
            )

    def extract_component(
        self,
        image_path: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        segment_data: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        padding: int = 10
    ) -> Optional[Image.Image]:
        """
        Extract a component from an instruction page.

        Args:
            image_path: Path to the instruction page image
            bbox: Bounding box (x1, y1, x2, y2) to extract
            segment_data: Segment dictionary from segment_page()
            output_path: Optional path to save the extracted component
            padding: Pixels to add around the bounding box

        Returns:
            PIL Image of the extracted component, or None if extraction failed
        """
        try:
            # Load the source image
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            # Get bounding box
            if segment_data is not None:
                bbox = segment_data["bbox"]

            if bbox is None:
                logger.warning("No bounding box provided for component extraction")
                return None

            # Add padding and clip to image bounds
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            # Crop the component
            component_image = image.crop((x1, y1, x2, y2))

            # Apply mask if available
            if segment_data is not None and "mask" in segment_data:
                mask = segment_data["mask"]
                mask_crop = mask[y1:y2, x1:x2]

                # Convert mask to PIL format
                mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
                mask_pil = mask_pil.resize(component_image.size, Image.Resampling.LANCZOS)

                # Apply mask to create transparent background
                component_array = np.array(component_image)
                mask_array = np.array(mask_pil) / 255.0

                # Create RGBA image with transparency
                rgba = np.dstack([component_array, (mask_array * 255).astype(np.uint8)])
                component_image = Image.fromarray(rgba, mode="RGBA")

            # Save if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                component_image.save(output_path)
                logger.debug(f"Saved component to {output_path}")

            return component_image

        except Exception as e:
            logger.error(f"Error extracting component from {image_path}: {e}")
            return None

    def extract_multiple_components(
        self,
        image_path: str,
        output_dir: str,
        prefix: str = "component",
        max_components: int = 50
    ) -> List[str]:
        """
        Extract all components from an instruction page.

        Args:
            image_path: Path to the instruction page image
            output_dir: Directory to save extracted components
            prefix: Filename prefix for saved components
            max_components: Maximum number of components to extract

        Returns:
            List of paths to extracted component images
        """
        if not self.is_available():
            logger.warning("SAM not available, skipping component extraction")
            return []

        # Segment the page
        segments = self.segment_page(image_path)

        if not segments:
            logger.warning(f"No components found in {image_path}")
            return []

        # Limit to max_components
        segments = segments[:max_components]

        # Extract each component
        output_paths = []
        os.makedirs(output_dir, exist_ok=True)

        for i, segment in enumerate(segments):
            output_path = os.path.join(output_dir, f"{prefix}_{i:03d}.png")

            component_image = self.extract_component(
                image_path=image_path,
                segment_data=segment,
                output_path=output_path
            )

            if component_image:
                output_paths.append(output_path)

        logger.info(f"Extracted {len(output_paths)} components from {image_path}")
        return output_paths

    def extract_part_image(
        self,
        page_path: str,
        part_id: str,
        output_dir: str,
        bbox_hint: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[str]:
        """
        Extract a single part image from an instruction page.

        Args:
            page_path: Path to the instruction page image
            part_id: Unique identifier for the part
            output_dir: Directory to save the part image
            bbox_hint: Optional bounding box hint (x1, y1, x2, y2)

        Returns:
            Path to the extracted part image, or None if extraction failed
        """
        if not self.is_available():
            return None

        output_path = os.path.join(output_dir, f"part_{part_id}.png")

        # If bbox hint provided, use it directly (VLM-guided extraction)
        if bbox_hint:
            logger.info(f"Extracting part {part_id} using VLM bbox hint: {bbox_hint}")
            component = self.extract_component(
                image_path=page_path,
                bbox=bbox_hint,
                output_path=output_path
            )
            return output_path if component else None

        # Otherwise, segment the page and use the first appropriate segment (fallback)
        logger.info(f"No VLM bbox for part {part_id}, using auto-segmentation fallback")
        segments = self.segment_page(page_path)

        if not segments:
            logger.warning(f"No segments found for part {part_id} in {page_path}")
            return None

        # Use the largest segment as the part
        # (This is a heuristic; in practice, you'd want more sophisticated selection)
        component = self.extract_component(
            image_path=page_path,
            segment_data=segments[0],
            output_path=output_path
        )

        return output_path if component else None

    def extract_subassembly_image(
        self,
        page_path: str,
        subassembly_id: str,
        output_dir: str,
        bbox_hint: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[str]:
        """
        Extract a subassembly image from an instruction page.

        Args:
            page_path: Path to the instruction page image
            subassembly_id: Unique identifier for the subassembly
            output_dir: Directory to save the subassembly image
            bbox_hint: Optional bounding box hint (x1, y1, x2, y2)

        Returns:
            Path to the extracted subassembly image, or None if extraction failed
        """
        if not self.is_available():
            return None

        output_path = os.path.join(output_dir, f"subasm_{subassembly_id}.png")

        # If bbox hint provided, use it directly
        if bbox_hint:
            component = self.extract_component(
                image_path=page_path,
                bbox=bbox_hint,
                output_path=output_path
            )
            return output_path if component else None

        # Otherwise, segment the page and use the largest segment
        segments = self.segment_page(page_path)

        if not segments:
            logger.warning(f"No segments found for subassembly {subassembly_id} in {page_path}")
            return None

        # Use the largest segment as the subassembly
        # (Subassemblies are typically larger than individual parts)
        component = self.extract_component(
            image_path=page_path,
            segment_data=segments[0],
            output_path=output_path
        )

        return output_path if component else None


def create_sam_segmenter_from_env() -> Optional[SAMSegmenter]:
    """
    Create a SAM segmenter from environment configuration.

    Returns:
        SAMSegmenter instance if SAM is enabled, None otherwise
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Check if SAM is enabled
    enable_sam = os.getenv("ENABLE_SAM", "true").lower() == "true"

    if not enable_sam:
        logger.info("SAM is disabled in configuration")
        return None

    if not SAM_AVAILABLE:
        logger.warning("SAM is enabled but ultralytics is not installed")
        return None

    # Get configuration
    model_name = os.getenv("SAM_MODEL", "sam2_b.pt")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"

    confidence_threshold = float(os.getenv("SAM_CONFIDENCE_THRESHOLD", "0.5"))
    inference_timeout = int(os.getenv("SAM_INFERENCE_TIMEOUT", "300"))

    # Detect device
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"

    logger.info(f"Creating SAM segmenter: model={model_name}, device={device}, timeout={inference_timeout}s")

    return SAMSegmenter(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        device=device,
        inference_timeout=inference_timeout
    )


# Example usage
if __name__ == "__main__":
    # Initialize segmenter
    segmenter = create_sam_segmenter_from_env()

    if segmenter and segmenter.is_available():
        # Example: Extract components from a page
        page_path = "output/temp_pages/page_001.png"
        output_dir = "output/components"

        if os.path.exists(page_path):
            component_paths = segmenter.extract_multiple_components(
                image_path=page_path,
                output_dir=output_dir,
                prefix="step_001_component"
            )

            print(f"Extracted {len(component_paths)} components:")
            for path in component_paths:
                print(f"  - {path}")
        else:
            print(f"Page not found: {page_path}")
    else:
        print("SAM segmenter not available")
