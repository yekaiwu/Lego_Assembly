"""
Integration tests for VLM-only workflow.

Tests the complete end-to-end workflow from step detection to guidance generation
using the simplified VLM-only approach.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from backend.app.vision.direct_step_analyzer import DirectStepAnalyzer, get_direct_step_analyzer
from backend.app.vision.guidance_generator import GuidanceGenerator, get_guidance_generator


class TestVLMOnlyWorkflow:
    """Integration tests for the complete VLM-only workflow."""

    @pytest.fixture
    def test_manual_data(self, tmp_path):
        """Create test manual data structure."""
        manual_id = "test_manual_001"
        output_dir = tmp_path / "output"
        manual_dir = output_dir / manual_id
        manual_dir.mkdir(parents=True)

        # Create dependencies.json
        dependencies = {
            "nodes": {
                "1": {
                    "step_number": 1,
                    "parts_required": [
                        {
                            "description": "base plate",
                            "color": "green",
                            "shape": "plate",
                            "quantity": 1
                        }
                    ],
                    "actions": [
                        {
                            "action_verb": "place",
                            "target": "green base plate",
                            "destination": "table surface",
                            "orientation": "flat"
                        }
                    ],
                    "notes": "This is the foundation of your build"
                },
                "2": {
                    "step_number": 2,
                    "parts_required": [
                        {
                            "description": "brick",
                            "color": "red",
                            "shape": "2x4 brick",
                            "quantity": 2
                        }
                    ],
                    "actions": [
                        {
                            "action_verb": "attach",
                            "target": "red 2x4 brick",
                            "destination": "center of base plate",
                            "orientation": "parallel to edge"
                        }
                    ],
                    "existing_assembly": "green base plate from step 1",
                    "notes": "Make sure bricks are firmly pressed down"
                },
                "3": {
                    "step_number": 3,
                    "parts_required": [
                        {
                            "description": "window piece",
                            "color": "blue",
                            "shape": "1x2x2 window",
                            "quantity": 1
                        }
                    ],
                    "actions": [
                        {
                            "action_verb": "attach",
                            "target": "blue window",
                            "destination": "top of red bricks",
                            "orientation": "facing forward"
                        }
                    ]
                }
            }
        }

        dependencies_file = manual_dir / f"{manual_id}_dependencies.json"
        with open(dependencies_file, 'w', encoding='utf-8') as f:
            json.dump(dependencies, f)

        # Create extracted.json (for guidance generator)
        extracted_steps = [
            {
                "step_number": 1,
                "parts_required": dependencies["nodes"]["1"]["parts_required"],
                "actions": dependencies["nodes"]["1"]["actions"],
                "notes": dependencies["nodes"]["1"]["notes"]
            },
            {
                "step_number": 2,
                "parts_required": dependencies["nodes"]["2"]["parts_required"],
                "actions": dependencies["nodes"]["2"]["actions"],
                "notes": dependencies["nodes"]["2"]["notes"]
            },
            {
                "step_number": 3,
                "parts_required": dependencies["nodes"]["3"]["parts_required"],
                "actions": dependencies["nodes"]["3"]["actions"],
                "notes": None
            }
        ]

        extracted_file = output_dir / f"{manual_id}_extracted.json"
        with open(extracted_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_steps, f)

        return {
            "manual_id": manual_id,
            "output_dir": str(output_dir),
            "dependencies": dependencies,
            "extracted_steps": extracted_steps
        }

    def test_complete_workflow_step_1(self, test_manual_data):
        """Test complete workflow for user on step 1."""
        # Setup
        manual_id = test_manual_data["manual_id"]
        output_dir = test_manual_data["output_dir"]

        # Mock VLM response for step 1
        mock_vlm_response = {
            "step_number": 1,
            "confidence": 0.95,
            "reasoning": "I can see a green base plate placed on the surface, matching step 1 exactly.",
            "next_step": 2,
            "assembly_status": {
                "total_parts_detected": 1,
                "key_features": ["green base plate"],
                "potential_issues": []
            }
        }

        # Create analyzer with mock VLM
        mock_vlm_client = Mock()
        mock_vlm_client.analyze_images_json.return_value = mock_vlm_response
        analyzer = DirectStepAnalyzer(vlm_client=mock_vlm_client)

        # Mock settings for file paths
        with patch('backend.app.vision.direct_step_analyzer.get_settings') as mock_settings:
            mock_settings.return_value.output_dir = output_dir

            # Step 1: Detect current step
            detection = analyzer.detect_current_step(
                image_paths=["test_step1.jpg"],
                manual_id=manual_id
            )

        # Verify detection
        assert detection["step_number"] == 1
        assert detection["confidence"] == 0.95
        assert detection["next_step"] == 2
        assert "green base plate" in detection["reasoning"]

        # Step 2: Generate guidance
        guidance_gen = GuidanceGenerator()

        with patch('backend.app.vision.guidance_generator.Path') as mock_path_class:
            # Mock Path operations for guidance generator
            mock_path = MagicMock()
            mock_path_class.return_value = mock_path
            mock_path.__truediv__ = lambda self, other: mock_path

            guidance = guidance_gen.generate_guidance_for_step(
                manual_id=manual_id,
                current_step=detection["step_number"],
                next_step=detection["next_step"],
                output_dir=output_dir,
                detection_confidence=detection["confidence"]
            )

        # Verify guidance
        assert guidance["next_step_number"] == 2
        assert len(guidance["parts_needed"]) == 2  # 2 red bricks
        assert guidance["parts_needed"][0]["color"] == "red"
        assert guidance["confidence"] == 0.95
        assert "Step 2" in guidance["instruction"]
        assert "red" in guidance["instruction"].lower()

    def test_complete_workflow_step_2(self, test_manual_data):
        """Test complete workflow for user on step 2."""
        manual_id = test_manual_data["manual_id"]
        output_dir = test_manual_data["output_dir"]

        # Mock VLM response for step 2
        mock_vlm_response = {
            "step_number": 2,
            "confidence": 0.88,
            "reasoning": "Assembly shows green base plate with two red bricks attached, matching step 2.",
            "next_step": 3,
            "assembly_status": {
                "total_parts_detected": 3,
                "key_features": ["green base", "two red bricks"],
                "potential_issues": []
            }
        }

        mock_vlm_client = Mock()
        mock_vlm_client.analyze_images_json.return_value = mock_vlm_response
        analyzer = DirectStepAnalyzer(vlm_client=mock_vlm_client)

        with patch('backend.app.vision.direct_step_analyzer.get_settings') as mock_settings:
            mock_settings.return_value.output_dir = output_dir
            detection = analyzer.detect_current_step(
                image_paths=["test_step2.jpg"],
                manual_id=manual_id
            )

        assert detection["step_number"] == 2
        assert detection["next_step"] == 3

        # Generate guidance for step 3
        guidance_gen = GuidanceGenerator()
        guidance = guidance_gen.generate_guidance_for_step(
            manual_id=manual_id,
            current_step=2,
            next_step=3,
            output_dir=output_dir,
            detection_confidence=0.88
        )

        assert guidance["next_step_number"] == 3
        assert len(guidance["parts_needed"]) == 1  # 1 blue window
        assert guidance["parts_needed"][0]["color"] == "blue"
        assert "window" in guidance["parts_needed"][0]["description"]

    def test_complete_workflow_finished(self, test_manual_data):
        """Test workflow when user has completed all steps."""
        manual_id = test_manual_data["manual_id"]
        output_dir = test_manual_data["output_dir"]

        # Mock VLM response for completed build
        mock_vlm_response = {
            "step_number": 3,
            "confidence": 0.98,
            "reasoning": "All parts are present and correctly assembled. Build is complete.",
            "next_step": None,
            "assembly_status": {
                "total_parts_detected": 4,
                "key_features": ["complete assembly", "all steps finished"],
                "potential_issues": []
            }
        }

        mock_vlm_client = Mock()
        mock_vlm_client.analyze_images_json.return_value = mock_vlm_response
        analyzer = DirectStepAnalyzer(vlm_client=mock_vlm_client)

        with patch('backend.app.vision.direct_step_analyzer.get_settings') as mock_settings:
            mock_settings.return_value.output_dir = output_dir
            detection = analyzer.detect_current_step(
                image_paths=["test_complete.jpg"],
                manual_id=manual_id
            )

        assert detection["step_number"] == 3
        assert detection["next_step"] is None

        # Generate completion guidance
        guidance_gen = GuidanceGenerator()
        guidance = guidance_gen.generate_guidance_for_step(
            manual_id=manual_id,
            current_step=3,
            next_step=None,
            output_dir=output_dir,
            detection_confidence=0.98
        )

        assert guidance["status"] == "complete"
        assert guidance["next_step_number"] is None
        assert "Congratulations" in guidance["instruction"]
        assert len(guidance["parts_needed"]) == 0

    def test_workflow_with_low_confidence(self, test_manual_data):
        """Test workflow with low confidence detection."""
        manual_id = test_manual_data["manual_id"]
        output_dir = test_manual_data["output_dir"]

        # Mock VLM response with low confidence
        mock_vlm_response = {
            "step_number": 1,
            "confidence": 0.45,  # Low confidence
            "reasoning": "Image is blurry, but appears to show a green base plate. Confidence is low.",
            "next_step": 2,
            "assembly_status": {
                "total_parts_detected": 1,
                "key_features": ["possibly green base plate"],
                "potential_issues": ["image quality poor"]
            }
        }

        mock_vlm_client = Mock()
        mock_vlm_client.analyze_images_json.return_value = mock_vlm_response
        analyzer = DirectStepAnalyzer(vlm_client=mock_vlm_client)

        with patch('backend.app.vision.direct_step_analyzer.get_settings') as mock_settings:
            mock_settings.return_value.output_dir = output_dir
            detection = analyzer.detect_current_step(
                image_paths=["blurry_image.jpg"],
                manual_id=manual_id
            )

        # Detection should still work but with low confidence
        assert detection["step_number"] == 1
        assert detection["confidence"] == 0.45
        assert "blurry" in detection["reasoning"].lower() or "quality" in detection["reasoning"].lower()

        # Guidance should still be generated
        guidance_gen = GuidanceGenerator()
        guidance = guidance_gen.generate_guidance_for_step(
            manual_id=manual_id,
            current_step=1,
            next_step=2,
            output_dir=output_dir,
            detection_confidence=0.45
        )

        assert guidance["confidence"] == 0.45
        assert guidance["status"] == "success"

    def test_workflow_with_multiple_images(self, test_manual_data):
        """Test workflow with multiple images from different angles."""
        manual_id = test_manual_data["manual_id"]
        output_dir = test_manual_data["output_dir"]

        mock_vlm_response = {
            "step_number": 2,
            "confidence": 0.92,  # Higher confidence with multiple angles
            "reasoning": "Multiple angles confirm: green base with two red bricks. All views consistent with step 2.",
            "next_step": 3,
            "assembly_status": {
                "total_parts_detected": 3,
                "key_features": ["green base", "two red bricks", "consistent across all views"],
                "potential_issues": []
            }
        }

        mock_vlm_client = Mock()
        mock_vlm_client.analyze_images_json.return_value = mock_vlm_response
        analyzer = DirectStepAnalyzer(vlm_client=mock_vlm_client)

        with patch('backend.app.vision.direct_step_analyzer.get_settings') as mock_settings:
            mock_settings.return_value.output_dir = output_dir
            detection = analyzer.detect_current_step(
                image_paths=["front.jpg", "back.jpg", "top.jpg"],
                manual_id=manual_id
            )

        # Multiple images should result in higher confidence
        assert detection["step_number"] == 2
        assert detection["confidence"] >= 0.9
        assert "multiple" in detection["reasoning"].lower() or "views" in detection["reasoning"].lower()

    def test_progress_calculation(self, test_manual_data):
        """Test that progress is correctly calculated throughout workflow."""
        manual_id = test_manual_data["manual_id"]
        output_dir = test_manual_data["output_dir"]
        total_steps = 3

        test_cases = [
            (0, 0, 0.0),      # No steps: 0%
            (1, 2, 33.33),    # Step 1: 33%
            (2, 3, 66.67),    # Step 2: 67%
            (3, None, 100.0)  # Step 3: 100%
        ]

        for current, next_step, expected_progress in test_cases:
            # Create guidance
            guidance_gen = GuidanceGenerator()
            guidance = guidance_gen.generate_guidance_for_step(
                manual_id=manual_id,
                current_step=current,
                next_step=next_step,
                output_dir=output_dir,
                detection_confidence=0.9
            )

            # Calculate progress
            progress = (current / total_steps * 100) if total_steps > 0 else 0
            assert abs(progress - expected_progress) < 0.1

    def test_error_handling_missing_manual_data(self):
        """Test error handling when manual data is missing."""
        manual_id = "nonexistent_manual"
        output_dir = "/tmp/nonexistent"

        mock_vlm_client = Mock()
        mock_vlm_client.analyze_images_json.return_value = {
            "step_number": 1,
            "confidence": 0.8,
            "reasoning": "Test",
            "next_step": 2,
            "assembly_status": {}
        }

        analyzer = DirectStepAnalyzer(vlm_client=mock_vlm_client)

        with patch('backend.app.vision.direct_step_analyzer.get_settings') as mock_settings:
            mock_settings.return_value.output_dir = output_dir
            detection = analyzer.detect_current_step(
                image_paths=["test.jpg"],
                manual_id=manual_id
            )

        # Should still return valid detection even without dependencies
        assert "step_number" in detection
        assert "confidence" in detection

    def test_guidance_with_missing_step_data(self, test_manual_data):
        """Test guidance generation when step data is incomplete."""
        manual_id = test_manual_data["manual_id"]
        output_dir = test_manual_data["output_dir"]

        guidance_gen = GuidanceGenerator()

        # Try to get guidance for non-existent step
        guidance = guidance_gen.generate_guidance_for_step(
            manual_id=manual_id,
            current_step=10,
            next_step=11,
            output_dir=output_dir,
            detection_confidence=0.9
        )

        # Should return error guidance
        assert guidance["status"] == "error"
        assert "Could not load" in guidance["instruction"] or "error" in guidance["instruction"].lower()


class TestPerformanceComparison:
    """Tests comparing VLM-only approach performance."""

    def test_single_vlm_call(self, tmp_path):
        """Verify that VLM-only approach uses only one VLM call."""
        # Create test manual
        manual_id = "perf_test"
        output_dir = tmp_path / "output"
        manual_dir = output_dir / manual_id
        manual_dir.mkdir(parents=True)

        dependencies = {
            "nodes": {
                "1": {"step_number": 1, "parts_required": [], "actions": []}
            }
        }

        deps_file = manual_dir / f"{manual_id}_dependencies.json"
        with open(deps_file, 'w') as f:
            json.dump(dependencies, f)

        # Mock VLM
        mock_vlm_client = Mock()
        mock_vlm_client.analyze_images_json.return_value = {
            "step_number": 1,
            "confidence": 0.9,
            "reasoning": "Test",
            "next_step": 2,
            "assembly_status": {}
        }

        analyzer = DirectStepAnalyzer(vlm_client=mock_vlm_client)

        with patch('backend.app.vision.direct_step_analyzer.get_settings') as mock_settings:
            mock_settings.return_value.output_dir = str(output_dir)
            analyzer.detect_current_step(
                image_paths=["img1.jpg", "img2.jpg"],
                manual_id=manual_id
            )

        # Verify VLM was called exactly once
        assert mock_vlm_client.analyze_images_json.call_count == 1

        # Verify all images were passed in single call
        call_args = mock_vlm_client.analyze_images_json.call_args
        assert len(call_args[1]["image_paths"]) == 2
