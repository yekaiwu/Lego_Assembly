"""
Unit tests for DirectStepAnalyzer - Simplified VLM-only approach.

Tests the core functionality of direct step detection without requiring
actual VLM API calls (uses mocking).
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from backend.app.vision.direct_step_analyzer import DirectStepAnalyzer


class TestDirectStepAnalyzer:
    """Test suite for DirectStepAnalyzer."""

    @pytest.fixture
    def mock_vlm_client(self):
        """Create a mock VLM client."""
        client = Mock()
        client.analyze_images_json = Mock()
        return client

    @pytest.fixture
    def analyzer(self, mock_vlm_client):
        """Create DirectStepAnalyzer instance with mock VLM client."""
        return DirectStepAnalyzer(vlm_client=mock_vlm_client)

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies.json data."""
        return {
            "nodes": {
                "1": {
                    "step_number": 1,
                    "parts_required": [
                        {
                            "description": "rectangular brick",
                            "color": "tan",
                            "shape": "brick",
                            "quantity": 1
                        }
                    ],
                    "actions": [
                        {
                            "action_verb": "place",
                            "target": "tan rectangular brick",
                            "destination": "empty space"
                        }
                    ]
                },
                "2": {
                    "step_number": 2,
                    "parts_required": [
                        {
                            "description": "sloped brick",
                            "color": "brown",
                            "shape": "sloped brick",
                            "quantity": 2
                        }
                    ],
                    "actions": [
                        {
                            "action_verb": "place",
                            "target": "brown sloped brick",
                            "destination": "top of tan brick"
                        }
                    ],
                    "existing_assembly": "tan rectangular brick from step 1"
                },
                "3": {
                    "step_number": 3,
                    "parts_required": [
                        {
                            "description": "round brick",
                            "color": "red",
                            "shape": "round brick",
                            "quantity": 1
                        }
                    ],
                    "actions": [
                        {
                            "action_verb": "attach",
                            "target": "red round brick",
                            "destination": "front of assembly"
                        }
                    ]
                }
            }
        }

    def test_initialization(self, analyzer):
        """Test that DirectStepAnalyzer initializes correctly."""
        assert analyzer.vlm_client is not None
        assert analyzer.graph_manager is not None
        assert analyzer.prompt_manager is not None

    def test_detect_current_step_success(self, analyzer, mock_vlm_client, mock_dependencies):
        """Test successful step detection."""
        # Mock VLM response
        mock_vlm_response = {
            "step_number": 2,
            "confidence": 0.85,
            "reasoning": "The assembly shows a tan brick with two brown sloped pieces on top, matching step 2.",
            "next_step": 3,
            "assembly_status": {
                "total_parts_detected": 3,
                "key_features": ["tan base brick", "two brown sloped pieces"],
                "potential_issues": []
            }
        }
        mock_vlm_client.analyze_images_json.return_value = mock_vlm_response

        # Mock dependencies loading
        with patch.object(analyzer, '_load_dependencies', return_value=mock_dependencies):
            result = analyzer.detect_current_step(
                image_paths=["test_image1.jpg", "test_image2.jpg"],
                manual_id="6262059"
            )

        # Verify result
        assert result["step_number"] == 2
        assert result["confidence"] == 0.85
        assert result["next_step"] == 3
        assert "tan brick" in result["reasoning"].lower()
        assert result["assembly_status"]["total_parts_detected"] == 3

        # Verify VLM was called correctly
        mock_vlm_client.analyze_images_json.assert_called_once()
        call_args = mock_vlm_client.analyze_images_json.call_args
        assert call_args[1]["image_paths"] == ["test_image1.jpg", "test_image2.jpg"]
        assert "6262059" in call_args[1]["cache_context"]

    def test_detect_current_step_no_parts(self, analyzer, mock_vlm_client, mock_dependencies):
        """Test detection when no parts are assembled (step 0)."""
        mock_vlm_response = {
            "step_number": 0,
            "confidence": 1.0,
            "reasoning": "No LEGO parts are visible in the image.",
            "next_step": 1,
            "assembly_status": {
                "total_parts_detected": 0,
                "key_features": [],
                "potential_issues": []
            }
        }
        mock_vlm_client.analyze_images_json.return_value = mock_vlm_response

        with patch.object(analyzer, '_load_dependencies', return_value=mock_dependencies):
            result = analyzer.detect_current_step(
                image_paths=["empty_image.jpg"],
                manual_id="6262059"
            )

        assert result["step_number"] == 0
        assert result["confidence"] == 1.0
        assert result["next_step"] == 1

    def test_detect_current_step_complete(self, analyzer, mock_vlm_client, mock_dependencies):
        """Test detection when assembly is complete."""
        mock_vlm_response = {
            "step_number": 3,
            "confidence": 0.95,
            "reasoning": "All steps completed. Assembly shows all parts from steps 1-3.",
            "next_step": None,
            "assembly_status": {
                "total_parts_detected": 4,
                "key_features": ["complete assembly"],
                "potential_issues": []
            }
        }
        mock_vlm_client.analyze_images_json.return_value = mock_vlm_response

        with patch.object(analyzer, '_load_dependencies', return_value=mock_dependencies):
            result = analyzer.detect_current_step(
                image_paths=["complete_assembly.jpg"],
                manual_id="6262059"
            )

        assert result["step_number"] == 3
        assert result["next_step"] is None

    def test_detect_current_step_vlm_error(self, analyzer, mock_vlm_client, mock_dependencies):
        """Test handling of VLM errors."""
        mock_vlm_client.analyze_images_json.return_value = {
            "error": "VLM service unavailable"
        }

        with patch.object(analyzer, '_load_dependencies', return_value=mock_dependencies):
            result = analyzer.detect_current_step(
                image_paths=["test_image.jpg"],
                manual_id="6262059"
            )

        assert result["step_number"] == 0
        assert result["confidence"] == 0.0
        assert "error" in result
        assert "VLM" in result["reasoning"]

    def test_validate_step_detection_clamps_values(self, analyzer, mock_dependencies):
        """Test that validation properly clamps step_number and confidence."""
        # Test step number clamping
        with patch.object(analyzer, '_load_dependencies', return_value=mock_dependencies):
            # Step number too high
            result = analyzer._validate_step_detection(
                {"step_number": 999, "confidence": 0.5, "reasoning": "test"},
                "6262059"
            )
            assert result["step_number"] == 3  # Max step in mock_dependencies

            # Step number negative
            result = analyzer._validate_step_detection(
                {"step_number": -1, "confidence": 0.5, "reasoning": "test"},
                "6262059"
            )
            assert result["step_number"] == 0

        # Test confidence clamping
        result = analyzer._validate_step_detection(
            {"step_number": 1, "confidence": 1.5, "reasoning": "test"},
            "6262059"
        )
        assert result["confidence"] == 1.0

        result = analyzer._validate_step_detection(
            {"step_number": 1, "confidence": -0.2, "reasoning": "test"},
            "6262059"
        )
        assert result["confidence"] == 0.0

    def test_format_manual_steps(self, analyzer, mock_dependencies):
        """Test manual steps formatting for VLM prompt."""
        formatted = analyzer._format_manual_steps(mock_dependencies["nodes"])

        assert "Step 1:" in formatted
        assert "Step 2:" in formatted
        assert "Step 3:" in formatted
        assert "tan" in formatted
        assert "brown" in formatted
        assert "red" in formatted
        assert "place" in formatted
        assert "attach" in formatted

        # Check that existing assembly context is included
        assert "building on:" in formatted.lower()

    def test_load_dependencies_file_not_found(self, analyzer):
        """Test handling when dependencies file doesn't exist."""
        result = analyzer._load_dependencies("nonexistent_manual")
        assert result is None

    def test_load_dependencies_success(self, analyzer, mock_dependencies, tmp_path):
        """Test successful loading of dependencies."""
        # Create temporary dependencies file
        manual_id = "test_manual"
        output_dir = tmp_path / "output"
        manual_dir = output_dir / manual_id
        manual_dir.mkdir(parents=True)

        dependencies_file = manual_dir / f"{manual_id}_dependencies.json"
        with open(dependencies_file, 'w') as f:
            json.dump(mock_dependencies, f)

        # Mock settings
        with patch('backend.app.vision.direct_step_analyzer.get_settings') as mock_settings:
            mock_settings.return_value.output_dir = str(output_dir)
            result = analyzer._load_dependencies(manual_id)

        assert result is not None
        assert result["nodes"]["1"]["step_number"] == 1

    def test_get_step_info(self, analyzer, mock_dependencies):
        """Test retrieving specific step information."""
        with patch.object(analyzer, '_load_dependencies', return_value=mock_dependencies):
            # Get step 2
            step_info = analyzer.get_step_info("6262059", 2)
            assert step_info is not None
            assert step_info["step_number"] == 2
            assert "brown" in step_info["parts_required"][0]["color"]

            # Get non-existent step
            step_info = analyzer.get_step_info("6262059", 999)
            assert step_info is None

    def test_build_step_detection_prompt(self, analyzer, mock_dependencies):
        """Test building the VLM prompt with manual context."""
        with patch.object(analyzer, '_load_dependencies', return_value=mock_dependencies):
            prompt = analyzer._build_step_detection_prompt("6262059")

        # Verify prompt contains expected elements
        assert "6262059" in prompt
        assert "Step 1:" in prompt
        assert "Step 2:" in prompt
        assert "Step 3:" in prompt
        assert "tan" in prompt.lower()
        assert "JSON" in prompt  # Should ask for JSON response

    def test_detect_current_step_handles_exception(self, analyzer, mock_vlm_client):
        """Test that exceptions during detection are handled gracefully."""
        mock_vlm_client.analyze_images_json.side_effect = Exception("Unexpected error")

        result = analyzer.detect_current_step(
            image_paths=["test.jpg"],
            manual_id="6262059"
        )

        # Should return error response instead of raising
        assert result["step_number"] == 0
        assert result["confidence"] == 0.0
        assert "error" in result
        assert "Unexpected error" in result["error"]

    def test_validate_step_detection_invalid_types(self, analyzer):
        """Test validation with invalid data types."""
        # Invalid step_number type
        result = analyzer._validate_step_detection(
            {"step_number": "two", "confidence": 0.8, "reasoning": "test"},
            "6262059"
        )
        assert result["step_number"] == 0  # Should default to 0

        # Invalid confidence type
        result = analyzer._validate_step_detection(
            {"step_number": 2, "confidence": "high", "reasoning": "test"},
            "6262059"
        )
        assert result["confidence"] == 0.5  # Should default to 0.5

    def test_assembly_status_structure(self, analyzer):
        """Test that assembly_status is properly structured."""
        # Missing assembly_status
        result = analyzer._validate_step_detection(
            {"step_number": 1, "confidence": 0.8, "reasoning": "test"},
            "6262059"
        )
        assert "assembly_status" in result
        assert "total_parts_detected" in result["assembly_status"]
        assert "key_features" in result["assembly_status"]
        assert "potential_issues" in result["assembly_status"]

        # Invalid assembly_status type
        result = analyzer._validate_step_detection(
            {"step_number": 1, "confidence": 0.8, "reasoning": "test", "assembly_status": "invalid"},
            "6262059"
        )
        assert isinstance(result["assembly_status"], dict)


class TestDirectStepAnalyzerIntegration:
    """Integration tests that test multiple components together (still mocked VLM)."""

    @pytest.fixture
    def full_analyzer(self):
        """Create a DirectStepAnalyzer with all real components except VLM."""
        mock_vlm = Mock()
        return DirectStepAnalyzer(vlm_client=mock_vlm)

    def test_full_detection_workflow(self, full_analyzer, tmp_path):
        """Test complete detection workflow from images to result."""
        # Setup test data
        manual_id = "test_manual"
        output_dir = tmp_path / "output"
        manual_dir = output_dir / manual_id
        manual_dir.mkdir(parents=True)

        # Create dependencies file
        dependencies = {
            "nodes": {
                "1": {"step_number": 1, "parts_required": [], "actions": []},
                "2": {"step_number": 2, "parts_required": [], "actions": []}
            }
        }
        dependencies_file = manual_dir / f"{manual_id}_dependencies.json"
        with open(dependencies_file, 'w') as f:
            json.dump(dependencies, f)

        # Mock VLM response
        full_analyzer.vlm_client.analyze_images_json.return_value = {
            "step_number": 1,
            "confidence": 0.9,
            "reasoning": "Step 1 is complete",
            "next_step": 2,
            "assembly_status": {
                "total_parts_detected": 1,
                "key_features": ["base brick"],
                "potential_issues": []
            }
        }

        # Mock settings
        with patch('backend.app.vision.direct_step_analyzer.get_settings') as mock_settings:
            mock_settings.return_value.output_dir = str(output_dir)
            result = full_analyzer.detect_current_step(
                image_paths=["img1.jpg", "img2.jpg"],
                manual_id=manual_id
            )

        # Verify complete result
        assert result["step_number"] == 1
        assert result["confidence"] == 0.9
        assert result["next_step"] == 2
        assert isinstance(result["assembly_status"], dict)
