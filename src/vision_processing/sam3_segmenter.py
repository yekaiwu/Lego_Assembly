"""
SAM3 Segmenter: Uses SAM3 (Segment Anything Model 3) to extract and segment
LEGO parts and assembled results from instruction step images.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from loguru import logger
from PIL import Image
import numpy as np
import os

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

@dataclass
class AssemblySegmentation:
    """Represents a segmented assembled result."""
    step_number: int
    mask: Optional[np.ndarray] = None
    bounding_box: Optional[List[int]] = None  # [x1, y1, x2, y2]
    cropped_image_path: Optional[str] = None
    mask_path: Optional[str] = None
    confidence: float = 0.0

class SAM3Segmenter:
    """
    Segments LEGO parts and assembled results using SAM3 model.

    Uses text prompts from VLM extraction to identify and segment:
    - Individual parts shown in instruction steps
    - Assembled results after adding parts
    """

    def __init__(
        self,
        model_type: str = "sam3",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.7,
        output_dir: str = "./output/segmented_parts",
        save_masks: bool = True,
        save_cropped_images: bool = True
    ):
        """
        Initialize SAM3 segmenter.

        Args:
            model_type: SAM3 model type (default: "sam3")
            checkpoint_path: Path to model checkpoint (None = auto-download from HuggingFace)
            device: Device to run model on ("cuda" or "cpu")
            confidence_threshold: Minimum confidence score for segmentation
            output_dir: Directory to save segmented images and masks
            save_masks: Whether to save segmentation masks
            save_cropped_images: Whether to save cropped part images
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.save_masks = save_masks
        self.save_cropped_images = save_cropped_images

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SAM3 model (lazy loading)
        self._model = None
        self._processor = None

        logger.info(f"SAM3Segmenter initialized (device={device}, threshold={confidence_threshold})")

    def _initialize_model(self):
        """Lazy load SAM3 model and processor."""
        if self._model is not None:
            return

        try:
            logger.info("=" * 80)
            logger.info("🔧 INITIALIZING SAM3 MODEL")
            logger.info("=" * 80)
            logger.info(f"Device: {self.device}")
            logger.info(f"Checkpoint: {self.checkpoint_path or 'Auto-download from HuggingFace'}")
            logger.info(f"Confidence threshold: {self.confidence_threshold}")
            logger.info(f"Output directory: {self.output_dir}")

            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info("📦 Importing SAM3 modules...")

            # Build model (auto-downloads checkpoint if path not specified)
            if self.checkpoint_path:
                logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
                self._model = build_sam3_image_model(
                    checkpoint_path=self.checkpoint_path,
                    device=self.device
                )
            else:
                # Auto-download from HuggingFace (requires authentication)
                logger.info("Downloading SAM3 checkpoint from HuggingFace (this may take a while on first run)...")
                self._model = build_sam3_image_model(
                    device=self.device,
                    load_from_HF=True
                )

            # Move to device
            logger.info(f"Moving model to device: {self.device}")
            self._model.to(self.device)
            self._model.eval()

            # Initialize processor
            logger.info("Initializing SAM3 processor...")
            self._processor = Sam3Processor(self._model)

            logger.success("✅ SAM3 model loaded successfully!")
            logger.info("=" * 80)

        except ImportError as e:
            logger.error(f"❌ Failed to import SAM3: {e}")
            logger.error("Please install SAM3: pip install -e external/sam3")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize SAM3 model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def segment_step_parts(
        self,
        image_path: str,
        parts_list: List[Dict[str, Any]],
        step_number: int,
        assembly_id: str
    ) -> List[PartSegmentation]:
        """
        Segment individual LEGO parts from instruction step image.

        Args:
            image_path: Path to instruction step image
            parts_list: List of parts from VLM extraction (with description, color, shape)
            step_number: Step number for organizing output
            assembly_id: Assembly/manual ID for organizing output

        Returns:
            List of PartSegmentation objects with masks and cropped images
        """
        logger.info(f"🔍 Segmenting parts for step {step_number} (assembly: {assembly_id})")
        logger.info(f"  Image: {image_path}")
        logger.info(f"  Parts to segment: {len(parts_list)}")

        self._initialize_model()

        # Load image
        logger.debug(f"Loading image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        logger.debug(f"Image size: {image.size}")

        # Set image in processor
        inference_state = self._processor.set_image(image)

        segmentations = []

        # Segment each part using text prompts
        for idx, part in enumerate(parts_list):
            try:
                # Build text prompt from part description
                text_prompt = self._build_part_prompt(part)

                logger.debug(f"Segmenting part {idx}: {text_prompt}")

                # Run segmentation with text prompt
                output = self._processor.set_text_prompt(
                    state=inference_state,
                    prompt=text_prompt
                )

                masks = output.get("masks")
                boxes = output.get("boxes")
                scores = output.get("scores")

                if masks is None or len(masks) == 0:
                    logger.warning(f"No mask found for part {idx}: {text_prompt}")
                    continue

                # Get best result (highest confidence)
                best_idx = np.argmax(scores) if scores is not None else 0
                mask = masks[best_idx]
                bbox = boxes[best_idx].tolist() if boxes is not None else None
                confidence = float(scores[best_idx]) if scores is not None else 0.0

                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    logger.warning(f"Part {idx} confidence {confidence:.2f} below threshold {self.confidence_threshold}")
                    continue

                # Create segmentation object
                part_seg = PartSegmentation(
                    part_index=idx,
                    description=part.get("description", ""),
                    color=part.get("color", ""),
                    shape=part.get("shape", ""),
                    mask=mask,
                    bounding_box=bbox,
                    confidence=confidence
                )

                # Save outputs
                if self.save_masks or self.save_cropped_images:
                    self._save_part_outputs(
                        part_seg,
                        image,
                        assembly_id,
                        step_number
                    )

                segmentations.append(part_seg)

                logger.info(f"  ✅ Part {idx}: {text_prompt} (confidence: {confidence:.2f})")
                if part_seg.cropped_image_path:
                    logger.info(f"     Saved: {part_seg.cropped_image_path}")

            except Exception as e:
                logger.error(f"  ❌ Failed to segment part {idx}: {e}")
                continue

        logger.success(f"✅ Segmented {len(segmentations)}/{len(parts_list)} parts for step {step_number}")
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
            step_number: Step number for organizing output
            assembly_id: Assembly/manual ID for organizing output
            text_hint: Optional text description of assembly (from VLM)

        Returns:
            AssemblySegmentation object with mask and cropped image, or None if failed
        """
        self._initialize_model()

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Set image in processor
        inference_state = self._processor.set_image(image)

        try:
            # Build text prompt for assembled result
            text_prompt = text_hint or "assembled LEGO structure"

            logger.debug(f"Segmenting assembled result: {text_prompt}")

            # Run segmentation with text prompt
            output = self._processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt
            )

            masks = output.get("masks")
            boxes = output.get("boxes")
            scores = output.get("scores")

            if masks is None or len(masks) == 0:
                logger.warning(f"No mask found for assembled result in step {step_number}")
                return None

            # Get best result (highest confidence)
            best_idx = np.argmax(scores) if scores is not None else 0
            mask = masks[best_idx]
            bbox = boxes[best_idx].tolist() if boxes is not None else None
            confidence = float(scores[best_idx]) if scores is not None else 0.0

            # Filter by confidence threshold
            if confidence < self.confidence_threshold:
                logger.warning(f"Assembly confidence {confidence:.2f} below threshold {self.confidence_threshold}")
                return None

            # Create segmentation object
            assembly_seg = AssemblySegmentation(
                step_number=step_number,
                mask=mask,
                bounding_box=bbox,
                confidence=confidence
            )

            # Save outputs
            if self.save_masks or self.save_cropped_images:
                self._save_assembly_outputs(
                    assembly_seg,
                    image,
                    assembly_id,
                    step_number
                )

            logger.debug(f"Successfully segmented assembly (confidence: {confidence:.2f})")
            return assembly_seg

        except Exception as e:
            logger.error(f"Failed to segment assembled result: {e}")
            return None

    def _build_part_prompt(self, part: Dict[str, Any]) -> str:
        """
        Build text prompt for segmenting a LEGO part.

        Args:
            part: Part dictionary with description, color, shape

        Returns:
            Text prompt string
        """
        color = part.get("color", "")
        shape = part.get("shape", "")
        description = part.get("description", "")

        # Build prompt combining available information
        if color and shape:
            return f"{color} LEGO {shape}"
        elif description:
            return f"LEGO {description}"
        elif color:
            return f"{color} LEGO brick"
        elif shape:
            return f"LEGO {shape}"
        else:
            return "LEGO brick"

    def _save_part_outputs(
        self,
        part_seg: PartSegmentation,
        original_image: Image.Image,
        assembly_id: str,
        step_number: int
    ):
        """Save part segmentation outputs (mask and/or cropped image)."""
        # Create step directory
        step_dir = self.output_dir / assembly_id / f"step_{step_number:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        part_idx = part_seg.part_index

        # Save mask
        if self.save_masks and part_seg.mask is not None:
            mask_path = step_dir / f"part_{part_idx}_mask.png"
            mask_image = Image.fromarray((part_seg.mask * 255).astype(np.uint8))
            mask_image.save(mask_path)
            part_seg.mask_path = str(mask_path)
            logger.debug(f"Saved mask: {mask_path}")

        # Save cropped image
        if self.save_cropped_images and part_seg.bounding_box is not None:
            crop_path = step_dir / f"part_{part_idx}_image.png"
            bbox = part_seg.bounding_box
            cropped = original_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            cropped.save(crop_path)
            part_seg.cropped_image_path = str(crop_path)
            logger.debug(f"Saved cropped image: {crop_path}")

    def _save_assembly_outputs(
        self,
        assembly_seg: AssemblySegmentation,
        original_image: Image.Image,
        assembly_id: str,
        step_number: int
    ):
        """Save assembly segmentation outputs (mask and/or cropped image)."""
        # Create step directory
        step_dir = self.output_dir / assembly_id / f"step_{step_number:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save mask
        if self.save_masks and assembly_seg.mask is not None:
            mask_path = step_dir / "assembly_mask.png"
            mask_image = Image.fromarray((assembly_seg.mask * 255).astype(np.uint8))
            mask_image.save(mask_path)
            assembly_seg.mask_path = str(mask_path)
            logger.debug(f"Saved assembly mask: {mask_path}")

        # Save cropped image
        if self.save_cropped_images and assembly_seg.bounding_box is not None:
            crop_path = step_dir / "assembly_image.png"
            bbox = assembly_seg.bounding_box
            cropped = original_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            cropped.save(crop_path)
            assembly_seg.cropped_image_path = str(crop_path)
            logger.debug(f"Saved assembly image: {crop_path}")

    def process_step(
        self,
        image_path: str,
        extracted_step: Dict[str, Any],
        assembly_id: str
    ) -> Dict[str, Any]:
        """
        Process a complete step: segment parts and assembled result.

        Args:
            image_path: Path to instruction step image
            extracted_step: Step data from VLM extraction
            assembly_id: Assembly/manual ID

        Returns:
            Updated step dictionary with segmentation data
        """
        step_number = extracted_step.get("step_number", 0)
        logger.info(f"🎯 Processing SAM3 for step {step_number}")
        logger.info(f"   Assembly ID: {assembly_id}")
        logger.info(f"   Image: {image_path}")

        # Segment individual parts
        parts_list = extracted_step.get("parts_required", [])
        if parts_list:
            part_segmentations = self.segment_step_parts(
                image_path,
                parts_list,
                step_number,
                assembly_id
            )

            # Update parts with segmentation data
            for part_seg in part_segmentations:
                if part_seg.part_index < len(parts_list):
                    parts_list[part_seg.part_index].update({
                        "bounding_box": part_seg.bounding_box,
                        "cropped_image_path": part_seg.cropped_image_path,
                        "mask_path": part_seg.mask_path,
                        "segmentation_confidence": part_seg.confidence
                    })

        # Segment assembled result
        assembly_hint = extracted_step.get("existing_assembly") or extracted_step.get("subassembly_hint", {}).get("description")
        assembly_seg = self.segment_assembled_result(
            image_path,
            step_number,
            assembly_id,
            text_hint=assembly_hint
        )

        if assembly_seg:
            extracted_step["assembled_result"] = {
                "bounding_box": assembly_seg.bounding_box,
                "cropped_image_path": assembly_seg.cropped_image_path,
                "mask_path": assembly_seg.mask_path,
                "segmentation_confidence": assembly_seg.confidence
            }

        return extracted_step


def create_sam3_segmenter_from_config() -> Optional[SAM3Segmenter]:
    """
    Create SAM3Segmenter from environment configuration.

    Returns:
        SAM3Segmenter instance if enabled, None otherwise
    """
    from ..utils.config import get_config

    config = get_config()

    # Check if SAM3 is enabled
    enable_sam3 = os.getenv("ENABLE_SAM3", "false").lower() == "true"

    if not enable_sam3:
        logger.info("SAM3 segmentation disabled in configuration")
        return None

    # Get SAM3 configuration
    model_type = os.getenv("SAM3_MODEL_TYPE", "sam3")
    checkpoint_path = os.getenv("SAM3_CHECKPOINT_PATH", "")
    # Strip comments and whitespace from checkpoint path
    if checkpoint_path:
        checkpoint_path = checkpoint_path.split('#')[0].strip()
    if not checkpoint_path or checkpoint_path == "":
        checkpoint_path = None

    device = os.getenv("SAM3_DEVICE", "cuda")
    confidence_threshold = float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.7"))
    output_dir = os.getenv("SAM3_OUTPUT_DIR", "./output/segmented_parts")
    save_masks = os.getenv("SAM3_SAVE_MASKS", "true").lower() == "true"
    save_cropped_images = os.getenv("SAM3_SAVE_CROPPED_IMAGES", "true").lower() == "true"

    try:
        segmenter = SAM3Segmenter(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            device=device,
            confidence_threshold=confidence_threshold,
            output_dir=output_dir,
            save_masks=save_masks,
            save_cropped_images=save_cropped_images
        )
        logger.success("SAM3 segmenter created from configuration")
        return segmenter
    except Exception as e:
        logger.error(f"Failed to create SAM3 segmenter: {e}")
        return None
