"""
Roboflow SAM3 Segmenter: Uses Roboflow's SAM3 Cloud API to segment LEGO parts
and assembled results from instruction step images using text prompts.

This implementation uses the Roboflow serverless API instead of local model inference.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger
from PIL import Image
import numpy as np
import cv2
import requests
import time
import base64
import io
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Reuse data structures from local sam3_segmenter
@dataclass
class PartSegmentation:
    """Represents a segmented LEGO part."""
    part_index: int
    description: str
    color: str
    shape: str
    mask: Optional[np.ndarray] = None
    bounding_box: Optional[List[int]] = None  # [x1, y1, x2, y2]
    cropped_image_path: Optional[str] = None
    mask_path: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization (excludes numpy arrays)."""
        return {
            "part_index": self.part_index,
            "description": self.description,
            "color": self.color,
            "shape": self.shape,
            "bounding_box": self.bounding_box,
            "cropped_image_path": self.cropped_image_path,
            "mask_path": self.mask_path,
            "confidence": self.confidence
        }

@dataclass
class AssemblySegmentation:
    """Represents a segmented assembled result."""
    step_number: int
    mask: Optional[np.ndarray] = None
    bounding_box: Optional[List[int]] = None  # [x1, y1, x2, y2]
    cropped_image_path: Optional[str] = None
    mask_path: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization (excludes numpy arrays)."""
        return {
            "step_number": self.step_number,
            "bounding_box": self.bounding_box,
            "cropped_image_path": self.cropped_image_path,
            "mask_path": self.mask_path,
            "confidence": self.confidence
        }


class RoboflowSAM3Segmenter:
    """
    Segments LEGO parts and assembled results using Roboflow's SAM3 Cloud API.

    Uses text prompts from VLM extraction to identify and segment:
    - Individual parts shown in instruction steps
    - Assembled results after adding parts

    API Documentation: https://inference.roboflow.com/foundation/sam3/
    """

    # Roboflow SAM3 API endpoints (serverless)
    EMBED_IMAGE_ENDPOINT = "https://serverless.roboflow.com/sam3/embed_image"
    CONCEPT_SEGMENT_ENDPOINT = "https://serverless.roboflow.com/sam3/concept_segment"

    def __init__(
        self,
        api_key: str,
        confidence_threshold: float = 0.7,
        output_dir: str = "./output/segmented_parts",
        save_masks: bool = True,
        save_cropped_images: bool = True,
        output_format: str = "json",
        max_retries: int = 3,
        timeout: int = 30,
        retry_vlm: Optional[str] = None,
        retry_enabled: bool = True
    ):
        """
        Initialize Roboflow SAM3 segmenter.

        Args:
            api_key: Roboflow API key (get from https://app.roboflow.com)
            confidence_threshold: Minimum confidence score for segmentation (0.0-1.0)
            output_dir: Directory to save segmented images and masks
            save_masks: Whether to save segmentation masks as images
            save_cropped_images: Whether to save cropped part images
            output_format: API output format ("json", "polygon", or "rle")
            max_retries: Maximum number of API retry attempts
            timeout: API request timeout in seconds
            retry_vlm: VLM model for box selection (e.g., "gemini/gemini-robotics-er-1.5-preview")
            retry_enabled: Enable retry logic for failed segmentations
        """
        if not api_key or api_key == "your_roboflow_api_key_here":
            raise ValueError(
                "Valid Roboflow API key required. Get one at https://app.roboflow.com"
            )

        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.save_masks = save_masks
        self.save_cropped_images = save_cropped_images
        self.output_format = output_format
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_enabled = retry_enabled

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Image embedding cache for efficiency
        self._embedding_cache: Dict[str, str] = {}

        # VLM client for retry logic (lazy loaded)
        self._retry_vlm = retry_vlm
        self._vlm_client = None

        logger.info(
            f"RoboflowSAM3Segmenter initialized "
            f"(threshold={confidence_threshold}, format={output_format}, retry={retry_enabled})"
        )

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string for API requests."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _image_to_base64_from_array(self, image: np.ndarray) -> str:
        """Convert numpy image array to base64 string."""
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def _embed_image(self, image_path: str) -> Optional[str]:
        """
        Embed image using Roboflow SAM3 API for faster subsequent segmentations.

        Returns:
            Image embedding ID or None if failed
        """
        # Check cache first
        if image_path in self._embedding_cache:
            logger.debug(f"Using cached embedding for {Path(image_path).name}")
            return self._embedding_cache[image_path]

        try:
            # Prepare request
            image_base64 = self._encode_image_to_base64(image_path)

            payload = {
                "image": {
                    "type": "base64",
                    "value": image_base64
                }
            }

            # API key goes in URL params, not JSON body
            logger.debug(f"Embedding image: {Path(image_path).name}")
            response = requests.post(
                f"{self.EMBED_IMAGE_ENDPOINT}?api_key={self.api_key}",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            embedding_id = result.get("image_id")

            if embedding_id:
                self._embedding_cache[image_path] = embedding_id
                logger.debug(f"✓ Image embedded: {embedding_id}")
                return embedding_id
            else:
                logger.warning("No embedding ID returned from API")
                return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Image embedding failed: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def _call_roboflow_api(
        self,
        image_path: str,
        text_prompts: List[str],
        output_prob_thresh: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Call Roboflow SAM3 concept_segment API with text prompts.

        Args:
            image_path: Path to image file
            text_prompts: List of text descriptions to segment
            output_prob_thresh: Confidence threshold override

        Returns:
            API response dict or None if failed
        """
        if not text_prompts:
            logger.warning("No text prompts provided for segmentation")
            return None

        try:
            # Prepare image input (use base64 directly - embed_image not available in serverless API)
            image_base64 = self._encode_image_to_base64(image_path)
            image_input = {
                "type": "base64",
                "value": image_base64
            }

            # Build prompts
            prompts = [{"type": "text", "text": prompt} for prompt in text_prompts]

            # Prepare API request (API key goes in URL, not JSON body)
            payload = {
                "image": image_input,
                "prompts": prompts,
                "output_prob_thresh": output_prob_thresh or self.confidence_threshold,
                "format": self.output_format
            }

            logger.debug(f"Segmenting with {len(text_prompts)} prompts: {text_prompts[:3]}...")
            response = requests.post(
                f"{self.CONCEPT_SEGMENT_ENDPOINT}?api_key={self.api_key}",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"  ✓ API call successful")
            logger.info(f"  API response keys: {list(result.keys())}")
            logger.info(f"  Number of prompt_results: {len(result.get('prompt_results', []))}")
            if result.get('prompt_results') and len(result['prompt_results']) > 0:
                logger.info(f"  First result keys: {list(result['prompt_results'][0].keys())}")
                logger.info(f"  First result sample: {str(result['prompt_results'][0])[:200]}")
            return result

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Invalid Roboflow API key. Get one at https://app.roboflow.com")
            elif e.response.status_code == 429:
                logger.warning("Roboflow API rate limit exceeded. Retrying...")
            else:
                logger.error(f"Roboflow API error: {e}")
            raise

        except requests.exceptions.Timeout as e:
            logger.warning(f"API request timed out after {self.timeout}s")
            raise

        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            return None

    def _polygon_to_mask(self, polygon_points: List, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert polygon coordinates to binary mask.

        Args:
            polygon_points: List of [x, y] coordinates (SAM3 format) or {x, y} dicts
            image_shape: (height, width) of image

        Returns:
            Binary mask as numpy array
        """
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert to OpenCV format: [[x, y], [x, y], ...]
        # Handle both [x, y] array format and {x:, y:} dict format
        if polygon_points and isinstance(polygon_points[0], list):
            # SAM3 format: [[x, y], [x, y], ...]
            points = np.array([[int(p[0]), int(p[1])] for p in polygon_points], dtype=np.int32)
        else:
            # Dict format: [{x:, y:}, {x:, y:}, ...]
            points = np.array([[int(p["x"]), int(p["y"])] for p in polygon_points], dtype=np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [points], 255)

        return mask

    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[List[int]]:
        """
        Extract bounding box [x1, y1, x2, y2] from binary mask.

        Args:
            mask: Binary mask array

        Returns:
            Bounding box as [x1, y1, x2, y2] or None if mask is empty
        """
        # Find non-zero pixels
        coords = np.column_stack(np.where(mask > 0))

        if len(coords) == 0:
            return None

        # Get bounding box (note: coords are in (y, x) format from np.where)
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)

        return [int(x1), int(y1), int(x2), int(y2)]

    def _crop_image_with_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: List[int]
    ) -> np.ndarray:
        """
        Crop image region using mask and bounding box.

        Args:
            image: Source image array
            mask: Binary mask array
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Cropped image with transparency where mask is 0
        """
        x1, y1, x2, y2 = bbox

        # Crop image and mask
        cropped_img = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2]

        # Convert to RGBA if needed
        if cropped_img.shape[2] == 3:  # RGB
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2RGBA)

        # Apply mask to alpha channel
        cropped_img[:, :, 3] = cropped_mask

        return cropped_img

    def _get_vlm_client(self):
        """Lazy load VLM client for box selection."""
        if self._vlm_client is None and self._retry_vlm:
            from ..api.litellm_vlm import UnifiedVLMClient
            logger.debug(f"Initializing VLM client for retry: {self._retry_vlm}")
            self._vlm_client = UnifiedVLMClient(self._retry_vlm)
        return self._vlm_client

    def _select_box_with_vlm(
        self,
        image: np.ndarray,
        boxes: List[List[int]],
        target_description: str
    ) -> Optional[Tuple[List[int], int]]:
        """
        Use VLM to select the correct bounding box from multiple candidates.

        Args:
            image: Source image (RGB numpy array)
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            target_description: Description of what to find (e.g., "black sloped LEGO piece")

        Returns:
            Tuple of (selected_box, box_index) or None if no match
        """
        if not boxes:
            return None

        vlm_client = self._get_vlm_client()
        if not vlm_client:
            logger.warning("VLM client not available for box selection")
            return None

        logger.info(f"   🤖 Using VLM to select from {len(boxes)} candidates...")
        logger.debug(f"      Target: '{target_description}'")

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)

        for i, box in enumerate(boxes):
            try:
                # Crop the bounding box
                x1, y1, x2, y2 = box
                cropped = pil_image.crop((x1, y1, x2, y2))

                # Save to temporary bytes buffer
                buffer = io.BytesIO()
                cropped.save(buffer, format="PNG")
                buffer.seek(0)

                # Create temporary file path for VLM
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    cropped.save(tmp, format="PNG")
                    tmp_path = tmp.name

                # Ask VLM
                prompt = f"Does this image show a {target_description}? Answer ONLY 'yes' or 'no'."

                logger.debug(f"      [{i}] Checking box at {box[:2]}...")

                # Query VLM
                response = vlm_client.generate_text_description(tmp_path, prompt)

                # Clean up temp file
                import os
                os.unlink(tmp_path)

                # Check response
                response_lower = response.lower().strip()
                if "yes" in response_lower and "no" not in response_lower:
                    logger.info(f"      ✅ [{i}] VLM confirmed match: '{response_lower}'")
                    return box, i
                else:
                    logger.debug(f"      ❌ [{i}] VLM rejected: '{response_lower}'")

            except Exception as e:
                logger.warning(f"      ⚠️  [{i}] VLM query failed: {e}")
                continue

        logger.warning("   ❌ VLM found no matching box")
        return None

    def _build_part_prompt(self, part_dict: Dict[str, Any]) -> str:
        """
        Build text prompt from part description, color, and shape.

        Args:
            part_dict: Part data from VLM extraction

        Returns:
            Text prompt for SAM3
        """
        parts = []

        # Add color if available
        color = part_dict.get("color", "").strip()
        if color and color.lower() not in ["unknown", "n/a", "none"]:
            parts.append(color)

        # Add shape/type
        shape = part_dict.get("shape", "").strip()
        if shape:
            parts.append(shape)

        # Add description as fallback
        description = part_dict.get("description", "").strip()
        if not parts and description:
            return f"{description} LEGO part"
        elif parts:
            prompt = " ".join(parts)
            # Add context
            return f"{prompt} LEGO brick"
        else:
            return "LEGO brick"

    def _build_assembly_prompt(self, step_dict: Dict[str, Any]) -> str:
        """
        Build text prompt for assembly segmentation using enhanced VLM description.

        Priority: assembled_product_visual_description > existing_assembly > subassembly_hint

        Args:
            step_dict: Extracted step data from VLM

        Returns:
            Text prompt for SAM3
        """
        # Priority 1: Enhanced visual description (NEW field from VLM)
        visual_desc = step_dict.get("assembled_product_visual_description", {})
        if isinstance(visual_desc, dict):
            overall = visual_desc.get("overall_appearance", "").strip()
            if overall:
                return overall

            # Fallback to distinctive features
            features = visual_desc.get("distinctive_features", "").strip()
            if features:
                return features

        # Priority 2: Existing assembly description
        existing_assembly = step_dict.get("existing_assembly", "").strip()
        if existing_assembly and existing_assembly.lower() not in ["none", "n/a", "nothing"]:
            return existing_assembly

        # Priority 3: Subassembly hint
        subassembly_hint = step_dict.get("subassembly_hint", {})
        if isinstance(subassembly_hint, dict):
            desc = subassembly_hint.get("description", "").strip()
            if desc:
                return f"LEGO {desc}"

        # Default fallback
        return "assembled LEGO structure"

    def _save_part_outputs(
        self,
        image: np.ndarray,
        part_seg: PartSegmentation,
        output_dir: Path
    ) -> PartSegmentation:
        """
        Save part mask and cropped image to disk.

        Args:
            image: Source image array
            part_seg: PartSegmentation object with mask and bbox
            output_dir: Output directory path

        Returns:
            Updated PartSegmentation with file paths
        """
        if part_seg.mask is None or part_seg.bounding_box is None:
            return part_seg

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build descriptive filename: {color}_{shape}_{index}
        filename_parts = []

        # Add color if available
        if part_seg.color and part_seg.color.lower() not in ["unknown", "n/a", "none", ""]:
            filename_parts.append(part_seg.color.lower().replace(" ", "_"))

        # Add shape if available
        if part_seg.shape and part_seg.shape.lower() not in ["unknown", "n/a", "none", ""]:
            # Clean up shape (remove "brick", "plate" prefixes that are redundant)
            shape_clean = part_seg.shape.lower().replace(" ", "_").replace("/", "_")
            filename_parts.append(shape_clean)
        elif part_seg.description:
            # Fallback to description
            desc_clean = part_seg.description.lower().replace(" ", "_").replace("/", "_")
            filename_parts.append(desc_clean)

        # Build base filename
        if filename_parts:
            part_name = "_".join(filename_parts)
            # Add index to handle duplicates
            part_name = f"{part_name}_{part_seg.part_index}"
        else:
            # Fallback to generic name
            part_name = f"part_{part_seg.part_index}"

        # Sanitize filename (remove special characters)
        import re
        part_name = re.sub(r'[^\w\-_]', '', part_name)
        part_name = part_name[:100]  # Limit length

        # Save mask
        if self.save_masks:
            mask_path = output_dir / f"{part_name}_mask.png"
            cv2.imwrite(str(mask_path), part_seg.mask)
            part_seg.mask_path = str(mask_path)
            logger.debug(f"  Saved mask: {mask_path.name}")

        # Save cropped image
        if self.save_cropped_images:
            cropped_img = self._crop_image_with_mask(
                image, part_seg.mask, part_seg.bounding_box
            )
            crop_path = output_dir / f"{part_name}.png"
            cv2.imwrite(str(crop_path), cv2.cvtColor(cropped_img, cv2.COLOR_RGBA2BGRA))
            part_seg.cropped_image_path = str(crop_path)
            logger.debug(f"  Saved crop: {crop_path.name}")

        return part_seg

    def _save_assembly_outputs(
        self,
        image: np.ndarray,
        assembly_seg: AssemblySegmentation,
        output_dir: Path
    ) -> AssemblySegmentation:
        """
        Save assembly mask and cropped image to disk.

        Args:
            image: Source image array
            assembly_seg: AssemblySegmentation object with mask and bbox
            output_dir: Output directory path

        Returns:
            Updated AssemblySegmentation with file paths
        """
        if assembly_seg.mask is None or assembly_seg.bounding_box is None:
            return assembly_seg

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use step number in filename for better organization
        step_num = assembly_seg.step_number
        assembly_name = f"assembly_step{step_num:03d}"

        # Save mask
        if self.save_masks:
            mask_path = output_dir / f"{assembly_name}_mask.png"
            cv2.imwrite(str(mask_path), assembly_seg.mask)
            assembly_seg.mask_path = str(mask_path)
            logger.debug(f"  Saved mask: {mask_path.name}")

        # Save cropped image
        if self.save_cropped_images:
            cropped_img = self._crop_image_with_mask(
                image, assembly_seg.mask, assembly_seg.bounding_box
            )
            crop_path = output_dir / f"{assembly_name}.png"
            cv2.imwrite(str(crop_path), cv2.cvtColor(cropped_img, cv2.COLOR_RGBA2BGRA))
            assembly_seg.cropped_image_path = str(crop_path)
            logger.debug(f"  Saved crop: {crop_path.name}")

        return assembly_seg

    def segment_step_parts(
        self,
        image_path: str,
        parts_list: List[Dict[str, Any]],
        step_number: int,
        assembly_id: str
    ) -> List[PartSegmentation]:
        """
        Segment individual LEGO parts using VLM-provided SAM3 prompts.

        Args:
            image_path: Path to instruction step image
            parts_list: List of part dicts from VLM extraction (with sam3_prompt field)
            step_number: Step number for output organization
            assembly_id: Assembly ID for output directory

        Returns:
            List of PartSegmentation objects with masks, crops, and metadata
        """
        if not parts_list:
            logger.debug(f"No parts to segment for step {step_number}")
            return []

        logger.info(f"🔍 Segmenting {len(parts_list)} parts for step {step_number}")
        logger.info(f"   Image: {Path(image_path).name}")

        # Extract sam3_prompt from VLM (fallback to building if not present)
        text_prompts = []
        for part in parts_list:
            if "sam3_prompt" in part and part["sam3_prompt"]:
                text_prompts.append(part["sam3_prompt"])
            else:
                # Fallback if VLM didn't provide sam3_prompt
                logger.debug(f"   Part missing sam3_prompt, using fallback")
                text_prompts.append(self._build_part_prompt(part))

        logger.info(f"   SAM3 prompts from VLM:")
        for i, prompt in enumerate(text_prompts):
            logger.info(f"     [{i}] \"{prompt}\"")

        # Call Roboflow API
        logger.debug(f"   Calling Roboflow SAM3 API...")
        api_response = self._call_roboflow_api(image_path, text_prompts)

        if not api_response:
            logger.warning(f"❌ Failed to segment parts for step {step_number} - API returned no response")
            return []

        # Load image for cropping
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape[:2]  # (height, width)

        # Parse API results
        segmentations = []
        prompt_results = api_response.get("prompt_results", [])

        logger.info(f"   Processing {len(prompt_results)} SAM3 results...")

        for idx, (part_dict, result) in enumerate(zip(parts_list, prompt_results)):
            part_desc = f"{part_dict.get('color', '')} {part_dict.get('shape', part_dict.get('description', 'part'))}".strip()

            # SAM3 returns predictions inside each prompt_result
            predictions = result.get("predictions", []) if result else []

            if not predictions:
                logger.warning(f"     [{idx}] ❌ No segmentation: {part_desc}")
                continue

            # Use the first (highest confidence) prediction
            prediction = predictions[0]

            # SAM3 uses "masks" field, not "polygon"
            if "masks" not in prediction or not prediction["masks"]:
                logger.warning(f"     [{idx}] ❌ No masks: {part_desc}")
                continue

            # masks[0] contains the polygon points
            polygon_points = prediction["masks"][0]
            mask = self._polygon_to_mask(polygon_points, image_shape)

            # Extract bounding box
            bbox = self._mask_to_bbox(mask)
            if bbox is None:
                logger.warning(f"     [{idx}] ❌ Empty mask: {part_desc}")
                continue

            # Get confidence (if available)
            confidence = prediction.get("confidence", self.confidence_threshold)

            # Create PartSegmentation
            part_seg = PartSegmentation(
                part_index=idx,
                description=part_dict.get("description", ""),
                color=part_dict.get("color", ""),
                shape=part_dict.get("shape", ""),
                mask=mask,
                bounding_box=bbox,
                confidence=confidence
            )

            # Save outputs
            output_dir = self.output_dir / assembly_id / f"step_{step_number:03d}"
            part_seg = self._save_part_outputs(image, part_seg, output_dir)

            # Log success with details
            saved_name = Path(part_seg.cropped_image_path).stem if part_seg.cropped_image_path else "N/A"
            logger.info(f"     [{idx}] ✅ {part_desc}")
            logger.info(f"          Confidence: {confidence:.3f} | BBox: {bbox} | Saved as: {saved_name}")

            segmentations.append(part_seg)

        # Summary
        success_rate = len(segmentations) / len(parts_list) * 100 if parts_list else 0
        if len(segmentations) == len(parts_list):
            logger.info(f"   ✅ Success: Segmented all {len(segmentations)} parts ({success_rate:.0f}%)")
        elif len(segmentations) > 0:
            logger.warning(f"   ⚠️  Partial: Segmented {len(segmentations)}/{len(parts_list)} parts ({success_rate:.0f}%)")
        else:
            logger.error(f"   ❌ Failed: No parts segmented (0/{len(parts_list)})")

        return segmentations

    def segment_assembled_result(
        self,
        image_path: str,
        step_number: int,
        assembly_id: str,
        text_hint: Optional[str] = None
    ) -> Optional[AssemblySegmentation]:
        """
        Segment the assembled result from instruction step image.

        Args:
            image_path: Path to instruction step image
            step_number: Step number
            assembly_id: Assembly ID for output directory
            text_hint: Optional text description of assembly (from VLM)

        Returns:
            AssemblySegmentation object or None if segmentation failed
        """
        # Use provided hint or generic prompt
        text_prompt = text_hint if text_hint else "assembled LEGO structure"

        logger.info(f"🔍 Segmenting assembled result for step {step_number}")
        logger.info(f"   SAM3 prompt: \"{text_prompt}\"")

        # Call Roboflow API
        logger.debug(f"   Calling Roboflow SAM3 API for assembly...")
        api_response = self._call_roboflow_api(image_path, [text_prompt])

        if not api_response:
            logger.warning(f"❌ Failed to segment assembly for step {step_number} - API returned no response")
            return None

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape[:2]

        # Parse API result (SAM3 format: predictions inside prompt_results)
        prompt_results = api_response.get("prompt_results", [])
        if not prompt_results:
            logger.warning(f"   ❌ No assembly segmentation found in API response")
            return None

        result = prompt_results[0]
        predictions = result.get("predictions", [])

        if not predictions:
            logger.warning(f"   ❌ No predictions for assembly")
            return None

        # Use the first (highest confidence) prediction
        prediction = predictions[0]

        # SAM3 uses "masks" field, not "polygon"
        if "masks" not in prediction or not prediction["masks"]:
            logger.warning(f"   ❌ No masks in assembly prediction")
            return None

        # masks[0] contains the polygon points
        polygon_points = prediction["masks"][0]
        mask = self._polygon_to_mask(polygon_points, image_shape)

        # Extract bounding box
        bbox = self._mask_to_bbox(mask)
        if bbox is None:
            logger.warning(f"   ❌ Empty assembly mask")
            return None

        # Get confidence
        confidence = prediction.get("confidence", self.confidence_threshold)

        # Create AssemblySegmentation
        assembly_seg = AssemblySegmentation(
            step_number=step_number,
            mask=mask,
            bounding_box=bbox,
            confidence=confidence
        )

        # Save outputs
        output_dir = self.output_dir / assembly_id / f"step_{step_number:03d}"
        assembly_seg = self._save_assembly_outputs(image, assembly_seg, output_dir)

        # Log success
        saved_name = Path(assembly_seg.cropped_image_path).stem if assembly_seg.cropped_image_path else "N/A"
        logger.info(f"   ✅ Assembly segmented successfully")
        logger.info(f"      Confidence: {confidence:.3f} | BBox: {bbox} | Saved as: {saved_name}")

        return assembly_seg

    def _retry_with_generic_prompt(
        self,
        image_path: str,
        target_description: str,
        part_dict: Dict[str, Any],
        part_index: int,
        step_number: int,
        assembly_id: str,
        is_assembly: bool = False
    ) -> Optional[PartSegmentation]:
        """
        Retry segmentation using generic "lego subassembly" prompt + VLM selection.

        Args:
            image_path: Path to instruction step image
            target_description: What we're looking for (e.g., "black sloped LEGO piece")
            part_dict: Original part dictionary from VLM
            part_index: Index of part in parts_required array
            step_number: Step number
            assembly_id: Assembly ID
            is_assembly: Whether this is for assembled_product (vs part)

        Returns:
            PartSegmentation if successful, None otherwise
        """
        logger.info(f"   🔄 Retrying with generic 'lego subassembly' prompt...")

        # Use generic prompt
        generic_prompt = "lego subassembly"
        api_response = self._call_roboflow_api(image_path, [generic_prompt])

        if not api_response:
            logger.warning("   ❌ Generic prompt API call failed")
            return None

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape[:2]

        # Parse API results
        prompt_results = api_response.get("prompt_results", [])
        if not prompt_results:
            logger.warning("   ❌ No results from generic prompt")
            return None

        result = prompt_results[0]
        predictions = result.get("predictions", [])

        if not predictions:
            logger.warning("   ❌ No predictions from generic prompt")
            return None

        logger.info(f"   ✓ Generic prompt returned {len(predictions)} candidates")

        # Extract all bounding boxes from predictions
        boxes = []
        masks_list = []
        confidences = []

        for pred in predictions:
            if "masks" not in pred or not pred["masks"]:
                continue

            polygon_points = pred["masks"][0]
            mask = self._polygon_to_mask(polygon_points, image_shape)
            bbox = self._mask_to_bbox(mask)

            if bbox:
                boxes.append(bbox)
                masks_list.append(mask)
                confidences.append(pred.get("confidence", 0.5))

        if not boxes:
            logger.warning("   ❌ No valid bounding boxes extracted")
            return None

        # Use VLM to select correct box
        result = self._select_box_with_vlm(image, boxes, target_description)

        if not result:
            logger.warning("   ❌ VLM could not select correct box")
            return None

        selected_box, box_idx = result
        selected_mask = masks_list[box_idx]
        selected_confidence = confidences[box_idx]

        logger.info(f"   ✅ Retry successful! Selected box {box_idx}/{len(boxes)}")

        # Create PartSegmentation (even for assembly, we use this structure temporarily)
        part_seg = PartSegmentation(
            part_index=part_index,
            description=part_dict.get("description", ""),
            color=part_dict.get("color", ""),
            shape=part_dict.get("shape", ""),
            mask=selected_mask,
            bounding_box=selected_box,
            confidence=selected_confidence
        )

        # Save outputs
        output_dir = self.output_dir / assembly_id / f"step_{step_number:03d}"
        part_seg = self._save_part_outputs(image, part_seg, output_dir)

        return part_seg

    def process_step(
        self,
        image_path: str,
        extracted_step: Dict[str, Any],
        assembly_id: str
    ) -> Dict[str, Any]:
        """
        Orchestrate complete SAM3 segmentation for a single step.

        Segments both individual parts and assembled result, then adds
        segmentation data to the extracted_step dict.

        Args:
            image_path: Path to instruction step image
            extracted_step: Step data from VLM extraction
            assembly_id: Assembly ID for output organization

        Returns:
            Updated extracted_step dict with segmentation data
        """
        step_number = extracted_step.get("step_number", 0)

        logger.info(f"Processing SAM3 segmentation for step {step_number}")

        # Segment individual parts
        parts_list = extracted_step.get("parts_required", [])
        part_segmentations = self.segment_step_parts(
            image_path, parts_list, step_number, assembly_id
        )

        # CRITICAL: Add delay between API calls on same image to avoid server-side race conditions
        # SAM3 API may have caching/state issues when processing same image rapidly
        if part_segmentations:
            logger.debug("   Waiting 2s before assembly segmentation (avoid API race condition)...")
            time.sleep(2)

        # Extract sam3_prompt from VLM's assembled_product (fallback to building)
        assembled_product = extracted_step.get("assembled_product", {})
        if isinstance(assembled_product, dict) and "sam3_prompt" in assembled_product:
            assembly_prompt = assembled_product["sam3_prompt"]
        else:
            # Fallback: build prompt from existing_assembly or generic
            logger.debug("   Assembly missing sam3_prompt, using fallback")
            assembly_prompt = self._build_assembly_prompt(extracted_step)

        # Segment assembled result
        assembly_segmentation = self.segment_assembled_result(
            image_path, step_number, assembly_id, text_hint=assembly_prompt
        )

        # Update parts_required with segmentation data (add to existing part dicts)
        for i, part_seg in enumerate(part_segmentations):
            if part_seg.part_index < len(parts_list):
                parts_list[part_seg.part_index].update({
                    "bounding_box": part_seg.bounding_box,
                    "cropped_image_path": part_seg.cropped_image_path,
                    "mask_path": part_seg.mask_path,
                    "confidence": part_seg.confidence
                })

        # RETRY LOGIC: Handle failed part segmentations
        if self.retry_enabled and self._retry_vlm:
            failed_parts = [
                (idx, part) for idx, part in enumerate(parts_list)
                if not part.get("bounding_box")
            ]

            if failed_parts:
                logger.info(f"📝 Retry queue: {len(failed_parts)} failed part(s)")

                for idx, part in failed_parts:
                    # Build description for VLM
                    sam3_prompt = part.get("sam3_prompt", "")
                    if not sam3_prompt:
                        # Fallback to building description
                        color = part.get("color", "")
                        shape = part.get("shape", "")
                        desc = part.get("description", "")
                        sam3_prompt = f"{color} {shape}".strip() or desc

                    logger.info(f"   Retrying part [{idx}]: {sam3_prompt}")

                    # Small delay before retry to avoid API issues
                    time.sleep(1)

                    retry_result = self._retry_with_generic_prompt(
                        image_path=image_path,
                        target_description=sam3_prompt,
                        part_dict=part,
                        part_index=idx,
                        step_number=step_number,
                        assembly_id=assembly_id,
                        is_assembly=False
                    )

                    if retry_result:
                        # Update part with retry results
                        parts_list[idx].update({
                            "bounding_box": retry_result.bounding_box,
                            "cropped_image_path": retry_result.cropped_image_path,
                            "mask_path": retry_result.mask_path,
                            "confidence": retry_result.confidence
                        })
                        logger.info(f"   ✅ Retry successful for part [{idx}]")
                    else:
                        logger.warning(f"   ❌ Retry failed for part [{idx}]")

        # Update assembled_product with segmentation data
        if assembly_segmentation:
            if not isinstance(extracted_step.get("assembled_product"), dict):
                extracted_step["assembled_product"] = {}
            extracted_step["assembled_product"].update({
                "bounding_box": assembly_segmentation.bounding_box,
                "cropped_image_path": assembly_segmentation.cropped_image_path,
                "mask_path": assembly_segmentation.mask_path,
                "confidence": assembly_segmentation.confidence
            })
        else:
            # Keep sam3_prompt even if segmentation failed
            if not isinstance(extracted_step.get("assembled_product"), dict):
                extracted_step["assembled_product"] = {}

            # RETRY LOGIC: Handle failed assembly segmentation
            if self.retry_enabled and self._retry_vlm:
                assembled_product = extracted_step.get("assembled_product", {})
                if isinstance(assembled_product, dict) and not assembled_product.get("bounding_box"):
                    # Get assembly description
                    assembly_desc = assembled_product.get("sam3_prompt", "") or assembly_prompt

                    logger.info(f"📝 Retrying assembly: {assembly_desc}")

                    # Small delay before retry
                    time.sleep(1)

                    # Create fake part_dict for assembly retry
                    assembly_part_dict = {
                        "description": assembly_desc,
                        "color": "",
                        "shape": ""
                    }

                    retry_result = self._retry_with_generic_prompt(
                        image_path=image_path,
                        target_description=assembly_desc,
                        part_dict=assembly_part_dict,
                        part_index=0,  # Dummy index
                        step_number=step_number,
                        assembly_id=assembly_id,
                        is_assembly=True
                    )

                    if retry_result:
                        extracted_step["assembled_product"].update({
                            "bounding_box": retry_result.bounding_box,
                            "cropped_image_path": retry_result.cropped_image_path,
                            "mask_path": retry_result.mask_path,
                            "confidence": retry_result.confidence
                        })
                        logger.info(f"   ✅ Assembly retry successful")
                    else:
                        logger.warning(f"   ❌ Assembly retry failed")

        # Final summary logging
        parts_with_bbox = sum(1 for p in parts_list if p.get("bounding_box"))
        assembly_ok = isinstance(extracted_step.get("assembled_product"), dict) and extracted_step["assembled_product"].get("bounding_box")

        logger.info(
            f"✓ Step {step_number}: Parts={parts_with_bbox}/{len(parts_list)}, "
            f"Assembly={'✓' if assembly_ok else '✗'}"
        )

        return extracted_step


def create_roboflow_sam3_segmenter_from_config() -> Optional[RoboflowSAM3Segmenter]:
    """
    Factory function to create RoboflowSAM3Segmenter from configuration.

    Returns:
        RoboflowSAM3Segmenter instance or None if disabled/not configured
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Check if enabled
    enabled = os.getenv("ENABLE_ROBOFLOW_SAM3", "false").lower() == "true"
    if not enabled:
        logger.info("Roboflow SAM3 segmentation disabled (ENABLE_ROBOFLOW_SAM3=false)")
        return None

    # Get configuration
    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key or api_key == "your_roboflow_api_key_here":
        logger.error(
            "Roboflow SAM3 enabled but no API key configured. "
            "Get one at https://app.roboflow.com"
        )
        return None

    confidence_threshold = float(os.getenv("ROBOFLOW_SAM3_CONFIDENCE_THRESHOLD", "0.7"))
    output_dir = os.getenv("ROBOFLOW_SAM3_OUTPUT_DIR", "./output/segmented_parts")
    save_masks = os.getenv("ROBOFLOW_SAM3_SAVE_MASKS", "true").lower() == "true"
    save_cropped_images = os.getenv("ROBOFLOW_SAM3_SAVE_CROPPED_IMAGES", "true").lower() == "true"
    output_format = os.getenv("ROBOFLOW_SAM3_OUTPUT_FORMAT", "json")

    try:
        segmenter = RoboflowSAM3Segmenter(
            api_key=api_key,
            confidence_threshold=confidence_threshold,
            output_dir=output_dir,
            save_masks=save_masks,
            save_cropped_images=save_cropped_images,
            output_format=output_format
        )
        logger.info("✓ Roboflow SAM3 segmenter initialized successfully")
        return segmenter
    except Exception as e:
        logger.error(f"Failed to initialize Roboflow SAM3 segmenter: {e}")
        return None
