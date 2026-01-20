"""
Visual Matcher - Uses SAM to segment user's assembled product
and compares it to graph node images using ORB feature matching.

This module provides visual similarity matching between user photos
and reference assembly images from the graph, tailored for LEGO structures.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vision_processing.roboflow_sam3_segmenter import RoboflowSAM3Segmenter


class VisualMatcher:
    """
    Visual similarity matcher for LEGO assemblies.

    Uses SAM3 (text-prompted segmentation) and ORB features optimized for LEGO structures.
    """

    def __init__(
        self,
        sam3_segmenter: Optional[RoboflowSAM3Segmenter] = None,
        n_features: int = 2000,
        similarity_threshold: float = 0.15,
        min_matches: int = 10
    ):
        """
        Initialize visual matcher.

        Args:
            sam3_segmenter: Roboflow SAM3 segmenter instance (creates new if None)
            n_features: Number of ORB features to detect (higher for blocky LEGO)
            similarity_threshold: Minimum similarity score (0.0-1.0)
            min_matches: Minimum number of feature matches required
        """
        # Use Roboflow SAM3 for text-prompted segmentation
        if sam3_segmenter is None:
            # Try to create from environment config
            import os
            enable_roboflow = os.getenv("ENABLE_ROBOFLOW_SAM3", "false").lower() == "true"
            if enable_roboflow:
                api_key = os.getenv("ROBOFLOW_API_KEY", "")
                if api_key:
                    self.sam3_segmenter = RoboflowSAM3Segmenter(
                        api_key=api_key,
                        confidence_threshold=float(os.getenv("ROBOFLOW_SAM3_CONFIDENCE_THRESHOLD", "0.7")),
                        output_dir=os.getenv("ROBOFLOW_SAM3_OUTPUT_DIR", "./output/segmented_parts"),
                        save_masks=os.getenv("ROBOFLOW_SAM3_SAVE_MASKS", "true").lower() == "true",
                        save_cropped_images=os.getenv("ROBOFLOW_SAM3_SAVE_CROPPED_IMAGES", "true").lower() == "true",
                        output_format=os.getenv("ROBOFLOW_SAM3_OUTPUT_FORMAT", "json")
                    )
                else:
                    self.sam3_segmenter = None
            else:
                self.sam3_segmenter = None
        else:
            self.sam3_segmenter = sam3_segmenter

        if self.sam3_segmenter is not None:
            logger.info("Visual matching initialized with Roboflow SAM3 (text-prompted segmentation)")
        else:
            logger.warning("SAM3 not available - visual matching disabled")
            logger.info("To enable SAM3, set ENABLE_ROBOFLOW_SAM3=true in .env")

        self.n_features = n_features
        self.similarity_threshold = similarity_threshold
        self.min_matches = min_matches

        # Initialize ORB detector optimized for LEGO
        # Use more features and different scale factors for blocky structures
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,  # Standard scale pyramid
            nlevels=8,  # Standard pyramid levels
            edgeThreshold=15,  # Reduced to detect more edge features (LEGO has lots of edges)
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10  # Lower threshold to detect more keypoints on flat surfaces
        )

        # Initialize FLANN matcher for fast matching
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        logger.info(
            f"VisualMatcher initialized: n_features={n_features}, "
            f"threshold={similarity_threshold}"
        )

    def is_available(self) -> bool:
        """Check if SAM3 segmenter is available."""
        return self.sam3_segmenter is not None

    def segment_user_assembly(
        self,
        image_path: str,
        prompt: str = "assembled LEGO structure"
    ) -> Optional[Dict[str, Any]]:
        """
        Segment user's assembled product using Roboflow SAM3 text-prompted segmentation.

        Args:
            image_path: Path to user's assembly photo
            prompt: Text prompt for segmentation (e.g., "assembled LEGO structure")

        Returns:
            Dictionary containing:
                - cropped_image: numpy array of segmented assembly
                - mask: binary mask
                - bbox: (x1, y1, x2, y2) bounding box
                - confidence: segmentation confidence
            Returns None if segmentation fails
        """
        if not self.is_available():
            logger.warning("SAM3 not available, cannot segment assembly")
            return None

        try:
            from PIL import Image as PILImage

            # Load image
            logger.debug(f"Segmenting user assembly with prompt: '{prompt}'")

            # Call Roboflow SAM3 API
            api_response = self.sam3_segmenter._call_roboflow_api(image_path, [prompt])

            if not api_response:
                logger.warning(f"No segments found in {image_path}")
                return None

            # Parse Roboflow API response
            # Response format: {"segmentations": [{"mask": [...], "bounding_box": {...}, "confidence": 0.X}]}
            segmentations = api_response.get("segmentations", [])
            if not segmentations:
                logger.warning(f"No segments found in {image_path}")
                return None

            # Get best result (highest confidence)
            best_seg = max(segmentations, key=lambda s: s.get("confidence", 0.0))
            confidence = best_seg.get("confidence", 0.0)

            # Extract bounding box
            bbox_dict = best_seg.get("bounding_box", {})
            if not bbox_dict:
                logger.error("No bounding box returned from Roboflow SAM3")
                return None

            x1 = int(bbox_dict.get("x", 0))
            y1 = int(bbox_dict.get("y", 0))
            x2 = int(bbox_dict.get("x", 0) + bbox_dict.get("width", 0))
            y2 = int(bbox_dict.get("y", 0) + bbox_dict.get("height", 0))

            # Load image and create mask from RLE
            pil_image = PILImage.open(image_path).convert("RGB")
            image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            h, w = image_cv.shape[:2]

            # Decode mask from RLE or polygon
            mask = self._decode_mask(best_seg, h, w)
            if mask is None:
                logger.error("Failed to decode mask from Roboflow response")
                return None

            # Crop to bounding box with padding
            padding = 20
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)

            cropped = image_cv[y1_pad:y2_pad, x1_pad:x2_pad]

            # Crop mask to same region
            mask_crop = mask[y1_pad:y2_pad, x1_pad:x2_pad]

            # Apply mask (set background to white for better feature detection)
            masked_image = cropped.copy()
            masked_image[mask_crop < 0.5] = [255, 255, 255]

            logger.debug(f"Segmentation successful (confidence: {confidence:.2f})")

            return {
                "cropped_image": masked_image,
                "mask": mask_crop,
                "bbox": (x1_pad, y1_pad, x2_pad, y2_pad),
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error segmenting user assembly with SAM3: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _decode_mask(self, segmentation: Dict[str, Any], height: int, width: int) -> Optional[np.ndarray]:
        """
        Decode mask from Roboflow API response (handles RLE or polygon formats).

        Args:
            segmentation: Segmentation dict from Roboflow API
            height: Image height
            width: Image width

        Returns:
            Binary mask as numpy array or None if failed
        """
        try:
            # Check if mask is in RLE format
            if "mask" in segmentation:
                mask_data = segmentation["mask"]
                if isinstance(mask_data, dict) and "counts" in mask_data:
                    # RLE format - decode using pycocotools if available
                    try:
                        from pycocotools import mask as mask_utils
                        rle = mask_data
                        rle["size"] = [height, width]
                        mask = mask_utils.decode(rle)
                        return mask.astype(np.float32)
                    except ImportError:
                        logger.warning("pycocotools not available, cannot decode RLE mask")
                        return None
                elif isinstance(mask_data, list):
                    # Polygon format - convert to mask
                    mask = np.zeros((height, width), dtype=np.uint8)
                    points = np.array(mask_data).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], 1)
                    return mask.astype(np.float32)

            # Fallback: create mask from bounding box
            bbox = segmentation.get("bounding_box", {})
            if bbox:
                mask = np.zeros((height, width), dtype=np.float32)
                x = int(bbox.get("x", 0))
                y = int(bbox.get("y", 0))
                w = int(bbox.get("width", 0))
                h = int(bbox.get("height", 0))
                mask[y:y+h, x:x+w] = 1.0
                return mask

            return None

        except Exception as e:
            logger.error(f"Error decoding mask: {e}")
            return None

    def extract_orb_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List, np.ndarray]:
        """
        Extract ORB features optimized for LEGO structures.

        Args:
            image: Input image (BGR format)
            mask: Optional binary mask to focus feature detection

        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This helps detect features on flat LEGO surfaces
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Detect keypoints and compute descriptors
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
            keypoints, descriptors = self.orb.detectAndCompute(enhanced, mask_uint8)
        else:
            keypoints, descriptors = self.orb.detectAndCompute(enhanced, None)

        return keypoints, descriptors

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        kp1: List,
        kp2: List,
        ratio_threshold: float = 0.75
    ) -> Tuple[List, float]:
        """
        Match ORB features using Lowe's ratio test and RANSAC.

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            ratio_threshold: Lowe's ratio test threshold

        Returns:
            Tuple of (good_matches, inlier_ratio)
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return [], 0.0

        try:
            # Match descriptors using FLANN
            matches = self.matcher.knnMatch(desc1, desc2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.min_matches:
                return good_matches, 0.0

            # Use RANSAC to filter outliers and find geometric consistency
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography using RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is None:
                return good_matches, 0.0

            # Calculate inlier ratio
            inliers = mask.ravel().tolist()
            inlier_count = sum(inliers)
            inlier_ratio = inlier_count / len(good_matches) if good_matches else 0.0

            # Filter to only inlier matches
            inlier_matches = [m for m, inlier in zip(good_matches, inliers) if inlier]

            return inlier_matches, inlier_ratio

        except Exception as e:
            logger.debug(f"Error during feature matching: {e}")
            return [], 0.0

    def compute_similarity_score(
        self,
        user_image: np.ndarray,
        reference_image: np.ndarray,
        user_mask: Optional[np.ndarray] = None,
        ref_mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute visual similarity score between two images.

        Args:
            user_image: User's assembly image
            reference_image: Reference assembly image from graph
            user_mask: Optional mask for user image
            ref_mask: Optional mask for reference image

        Returns:
            Similarity score (0.0-1.0), higher means more similar
        """
        # Extract features from both images
        kp1, desc1 = self.extract_orb_features(user_image, user_mask)
        kp2, desc2 = self.extract_orb_features(reference_image, ref_mask)

        if desc1 is None or desc2 is None:
            logger.debug("No ORB features detected in one or both images")
            return 0.0

        logger.debug(f"    ORB features detected: User={len(kp1)}, Reference={len(kp2)}")

        # Match features
        inlier_matches, inlier_ratio = self.match_features(desc1, desc2, kp1, kp2)

        logger.debug(f"    Feature matches: {len(inlier_matches)} inliers (ratio: {inlier_ratio:.2%})")

        if len(inlier_matches) < self.min_matches:
            logger.debug(f"    Too few matches ({len(inlier_matches)} < {self.min_matches} min)")
            return 0.0

        # Compute similarity score based on:
        # 1. Number of matches (normalized by average features detected)
        # 2. Inlier ratio (geometric consistency)
        avg_features = (len(kp1) + len(kp2)) / 2.0
        match_score = min(1.0, len(inlier_matches) / (avg_features * 0.3))  # 30% match is good

        # Combined score (weighted average)
        similarity = 0.6 * match_score + 0.4 * inlier_ratio

        logger.debug(
            f"    ORB Similarity: {similarity:.2%} "
            f"(match_score: {match_score:.2%}, inlier_ratio: {inlier_ratio:.2%})"
        )

        return similarity

    def match_user_assembly_to_graph(
        self,
        user_image_paths: List[str],
        manual_id: str,
        graph_manager,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Match user's assembly to graph node images using visual similarity.

        This is the main entry point for visual matching.

        Args:
            user_image_paths: Paths to user's assembly photos
            manual_id: Manual identifier
            graph_manager: GraphManager instance to access graph data
            top_k: Number of top matches to return

        Returns:
            List of matches with visual similarity scores:
            [
                {
                    "step_number": 3,
                    "visual_similarity": 0.78,
                    "match_reason": "Visual: 245 feature matches (78% similarity)",
                    "reference_image_path": "output/.../assembly_image.png",
                    "segmentation_confidence": 0.95
                },
                ...
            ]
        """
        if not self.is_available():
            logger.warning("Visual matching not available (SAM not loaded)")
            return []

        # Load graph
        graph = graph_manager.load_graph(manual_id)
        if not graph:
            logger.error(f"No graph found for manual {manual_id}")
            return []

        # Get step states with assembled product images
        step_states = graph.get("step_states", [])
        if not step_states:
            logger.error("No step_states found in graph")
            return []

        matches = []

        # Process each user image
        for user_image_path in user_image_paths:
            logger.info(f"Processing user image: {user_image_path}")

            # Segment user's assembly
            segmented = self.segment_user_assembly(user_image_path)
            if not segmented:
                logger.warning(f"Failed to segment {user_image_path}, skipping")
                continue

            user_img = segmented["cropped_image"]
            user_mask = segmented["mask"]
            seg_confidence = segmented["confidence"]

            logger.info(f"  SAM3 segmentation confidence: {seg_confidence:.2%}")
            logger.info(f"  Comparing against {len(step_states)} reference steps using ORB features...")

            # Compare against each step's assembled product image
            for step_state in step_states:
                step_number = step_state.get("step_number")
                if step_number is None:
                    continue

                # Get reference image path
                # Check for assembled_product field (from SAM3 segmentation)
                assembled_product = step_state.get("assembled_product", {})
                ref_image_path = assembled_product.get("cropped_image_path")

                # Fallback: look for assembly_image.png in step directory
                if not ref_image_path:
                    step_dir = f"output/segmented_parts/{manual_id}/step_{step_number:03d}"
                    ref_image_path = os.path.join(step_dir, "assembly_image.png")

                # Check if reference image exists
                if not os.path.exists(ref_image_path):
                    logger.debug(f"No reference image for step {step_number}")
                    continue

                # Load reference image
                ref_img = cv2.imread(ref_image_path)
                if ref_img is None:
                    logger.warning(f"Failed to load reference image: {ref_image_path}")
                    continue

                # Load reference mask if available
                ref_mask_path = assembled_product.get("mask_path")
                ref_mask = None
                if ref_mask_path and os.path.exists(ref_mask_path):
                    ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)

                # Compute similarity
                logger.debug(f"  Comparing with Step {step_number}...")
                similarity = self.compute_similarity_score(
                    user_img, ref_img, user_mask, ref_mask
                )

                if similarity >= self.similarity_threshold:
                    logger.info(f"  ✓ Match found: Step {step_number} - {similarity:.2%} ORB similarity")
                    matches.append({
                        "step_number": step_number,
                        "visual_similarity": similarity,
                        "match_reason": f"ORB Visual: {int(similarity * 100)}% similarity",
                        "reference_image_path": ref_image_path,
                        "segmentation_confidence": seg_confidence,
                        "user_image_path": user_image_path
                    })

        # Sort by similarity and return top-k
        matches.sort(key=lambda x: x["visual_similarity"], reverse=True)
        top_matches = matches[:top_k]

        logger.info(f"Visual matching complete: {len(matches)} matches found, returning top {len(top_matches)}")
        if top_matches:
            logger.info(
                f"Best visual match: Step {top_matches[0]['step_number']} "
                f"(similarity: {top_matches[0]['visual_similarity']:.2f})"
            )

        return top_matches

    def visualize_matches(
        self,
        user_image: np.ndarray,
        reference_image: np.ndarray,
        output_path: str,
        user_mask: Optional[np.ndarray] = None,
        ref_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Visualize feature matches between user and reference images.

        Useful for debugging and understanding match quality.

        Args:
            user_image: User's assembly image
            reference_image: Reference assembly image
            output_path: Path to save visualization
            user_mask: Optional mask for user image
            ref_mask: Optional mask for reference image
        """
        # Extract features
        kp1, desc1 = self.extract_orb_features(user_image, user_mask)
        kp2, desc2 = self.extract_orb_features(reference_image, ref_mask)

        if desc1 is None or desc2 is None:
            logger.warning("Cannot visualize: no features detected")
            return

        # Match features
        inlier_matches, inlier_ratio = self.match_features(desc1, desc2, kp1, kp2)

        # Draw matches
        match_img = cv2.drawMatches(
            user_image, kp1,
            reference_image, kp2,
            inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Save visualization
        cv2.imwrite(output_path, match_img)
        logger.info(
            f"Saved match visualization to {output_path} "
            f"({len(inlier_matches)} matches, {inlier_ratio:.2f} inlier ratio)"
        )


# Singleton instance
_visual_matcher_instance = None


def get_visual_matcher() -> Optional[VisualMatcher]:
    """Get VisualMatcher singleton instance."""
    global _visual_matcher_instance
    if _visual_matcher_instance is None:
        _visual_matcher_instance = VisualMatcher()
    return _visual_matcher_instance
