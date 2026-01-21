"""
Video analysis using Gemini Robotics ER VLM.

This module handles analyzing LEGO assembly videos to detect steps,
timing, and coordinates using the Gemini Robotics ER vision model.
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("google-genativeai not installed. Video analysis will not be available.")

from .video_prompts import get_assembly_video_prompt
from .coordinate_utils import (
    validate_coordinates,
    fix_inverted_coordinates,
    clamp_coordinates
)
from .metadata_extractor import extract_video_metadata

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyzes LEGO assembly videos using Gemini Robotics ER VLM."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-robotics-er-1.5-preview"
    ):
        """
        Initialize video analyzer.

        Args:
            api_key: Google API key
            model: Gemini model to use (gemini-robotics-er-1.5-preview recommended)
        """
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-generativeai package not installed")

        self.api_key = api_key
        self.model_name = model

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

        logger.info(f"Initialized VideoAnalyzer with model {model}")

    def analyze_video(
        self,
        video_path: str,
        manual_id: str,
        extracted_data_path: str,
        dependencies_path: Optional[str] = None,
        reference_images: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze a LEGO assembly video to detect steps and timing.

        Args:
            video_path: Path to video file
            manual_id: Manual ID
            extracted_data_path: Path to extracted.json from manual processing
            dependencies_path: Optional path to dependencies.json
            reference_images: Optional list of paths to reference step images

        Returns:
            Analysis results dictionary with detected events
        """
        logger.info(f"Starting video analysis for {video_path}")

        # Load manual context
        manual_context = self._load_manual_context(
            extracted_data_path,
            dependencies_path
        )

        # Extract video metadata
        metadata = extract_video_metadata(video_path)

        # Upload video to Gemini
        logger.info("Uploading video to Gemini...")
        video_file = genai.upload_file(video_path)
        logger.info(f"Video uploaded: {video_file.name}")

        # Upload reference images if provided
        uploaded_images = []
        if reference_images:
            logger.info(f"Uploading {len(reference_images)} reference images...")
            for img_path in reference_images[:10]:  # Limit to first 10 steps
                if Path(img_path).exists():
                    img_file = genai.upload_file(img_path)
                    uploaded_images.append(img_file)

        # Build prompt with manual context
        prompt = get_assembly_video_prompt(
            manual_steps=manual_context['steps'],
            dependencies=manual_context.get('dependencies'),
            expected_colors=manual_context.get('colors')
        )

        # Call VLM
        logger.info("Analyzing video with Gemini VLM...")
        content = [video_file] + uploaded_images + [prompt]

        try:
            response = self.model.generate_content(
                content,
                request_options={"timeout": 600}  # 10 minute timeout
            )

            # Parse response
            response_text = response.text.strip()
            logger.debug(f"Raw VLM response: {response_text}")

            # Extract JSON from response
            events = self._parse_vlm_response(response_text)

            # Validate and clean events
            validated_events = self._validate_events(
                events,
                manual_context['steps'],
                metadata
            )

            # Build final results
            results = {
                "analysis_id": None,  # Will be set by caller
                "manual_id": manual_id,
                "video_path": video_path,
                "total_duration_sec": metadata['duration_sec'],
                "fps": metadata['fps'],
                "resolution": metadata['resolution'],
                "detected_events": validated_events,
                "total_steps_detected": len(validated_events),
                "expected_steps": len(manual_context['steps']),
                "coverage_percentage": round(
                    len(validated_events) / len(manual_context['steps']) * 100, 1
                ),
                "average_confidence": round(
                    sum(e.get('confidence', 0.8) for e in validated_events) / len(validated_events), 2
                ) if validated_events else 0.0,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
                "context_provided": {
                    "manual_steps": True,
                    "dependencies": dependencies_path is not None,
                    "reference_images": len(uploaded_images) > 0
                }
            }

            logger.info(
                f"Analysis complete: {len(validated_events)} events detected "
                f"({results['coverage_percentage']}% coverage)"
            )

            return results

        except Exception as e:
            logger.error(f"Error during video analysis: {e}")
            raise

    def _load_manual_context(
        self,
        extracted_path: str,
        dependencies_path: Optional[str] = None
    ) -> Dict:
        """
        Load manual context from extracted JSON files.

        Args:
            extracted_path: Path to extracted.json
            dependencies_path: Optional path to dependencies.json

        Returns:
            Dictionary with steps, dependencies, and colors
        """
        logger.info(f"Loading manual context from {extracted_path}")

        with open(extracted_path, 'r') as f:
            extracted_data = json.load(f)

        context = {
            'steps': extracted_data if isinstance(extracted_data, list) else []
        }

        # Extract colors from parts
        colors = set()
        for step in context['steps']:
            for part in step.get('parts_required', []):
                color = part.get('color')
                if color:
                    colors.add(color)

        context['colors'] = list(colors)

        # Load dependencies if provided
        if dependencies_path and Path(dependencies_path).exists():
            with open(dependencies_path, 'r') as f:
                context['dependencies'] = json.load(f)

        logger.info(
            f"Loaded context: {len(context['steps'])} steps, "
            f"{len(context['colors'])} colors"
        )

        return context

    def _parse_vlm_response(self, response_text: str) -> List[Dict]:
        """
        Parse VLM response text to extract JSON events.

        Args:
            response_text: Raw response from VLM

        Returns:
            List of event dictionaries
        """
        # Try to find JSON array in response
        # VLM might wrap it in markdown code blocks or add text before/after

        # Remove markdown code blocks
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Find first [ and last ]
        start_idx = text.find('[')
        end_idx = text.rfind(']')

        if start_idx == -1 or end_idx == -1:
            logger.error(f"No JSON array found in response: {text}")
            raise ValueError("No JSON array in VLM response")

        json_text = text[start_idx:end_idx + 1]

        try:
            events = json.loads(json_text)
            if not isinstance(events, list):
                raise ValueError("Expected JSON array")
            return events
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nText: {json_text}")
            raise ValueError(f"Invalid JSON in VLM response: {e}")

    def _validate_events(
        self,
        events: List[Dict],
        manual_steps: List[Dict],
        metadata: Dict
    ) -> List[Dict]:
        """
        Validate and clean detected events.

        Args:
            events: Raw events from VLM
            manual_steps: Steps from manual
            metadata: Video metadata

        Returns:
            List of validated events
        """
        validated = []
        manual_step_ids = {step.get('step_number') for step in manual_steps}

        for i, event in enumerate(events):
            try:
                # Validate required fields
                if 'step_id' not in event:
                    logger.warning(f"Event {i} missing step_id, skipping")
                    continue

                # Check step_id exists in manual
                step_id = event['step_id']
                if step_id not in manual_step_ids:
                    logger.warning(
                        f"Event {i} has invalid step_id {step_id}, "
                        f"not in manual: {sorted(manual_step_ids)}"
                    )
                    continue

                # Validate timing
                start_sec = event.get('start_seconds', 0)
                end_sec = event.get('end_seconds', metadata['duration_sec'])
                anchor_sec = event.get('anchor_timestamp', (start_sec + end_sec) / 2)

                if start_sec >= end_sec:
                    logger.warning(
                        f"Event {i} has invalid timing: start={start_sec}, end={end_sec}"
                    )
                    continue

                if not (start_sec <= anchor_sec <= end_sec):
                    logger.warning(
                        f"Event {i} anchor {anchor_sec} not between start {start_sec} and end {end_sec}"
                    )
                    # Fix it
                    anchor_sec = (start_sec + end_sec) / 2

                # Validate and fix coordinates
                target_box = event.get('target_box_2d')
                assembly_box = event.get('assembly_box_2d')

                if target_box and len(target_box) == 4:
                    # Fix inverted coordinates
                    target_box = fix_inverted_coordinates(target_box)
                    # Clamp to valid range
                    target_box = clamp_coordinates(target_box)

                    # Validate
                    if not validate_coordinates(target_box, strict=True):
                        logger.warning(f"Event {i} has invalid target_box, skipping")
                        continue

                    event['target_box_2d'] = target_box

                if assembly_box and len(assembly_box) == 4:
                    assembly_box = fix_inverted_coordinates(assembly_box)
                    assembly_box = clamp_coordinates(assembly_box)

                    if not validate_coordinates(assembly_box, strict=True):
                        logger.warning(f"Event {i} has invalid assembly_box, skipping")
                        continue

                    event['assembly_box_2d'] = assembly_box

                # Enrich with manual data
                manual_step = next(
                    (s for s in manual_steps if s.get('step_number') == step_id),
                    None
                )

                if manual_step:
                    event['parts_required'] = manual_step.get('parts_required', [])
                    event['reference_image'] = manual_step.get('_source_page_paths', [None])[0]

                # Set defaults
                event.setdefault('confidence', 0.8)
                event.setdefault('action', 'attach')

                validated.append(event)

            except Exception as e:
                logger.error(f"Error validating event {i}: {e}")
                continue

        logger.info(f"Validated {len(validated)}/{len(events)} events")
        return validated
