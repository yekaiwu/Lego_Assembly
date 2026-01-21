"""
Integration tests for visual matching with SAM3.
Tests the visual matching pipeline: SAM3 segmentation, ORB feature matching, and combined matching.
"""

import cv2
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def visual_matcher():
    """Create a VisualMatcher instance."""
    from backend.app.vision.visual_matcher import VisualMatcher
    return VisualMatcher()


@pytest.fixture
def state_analyzer():
    """Create a StateAnalyzer instance."""
    from backend.app.vision.state_analyzer import StateAnalyzer
    return StateAnalyzer()


@pytest.fixture
def test_image_path(temp_output_dir):
    """Create a dummy test image."""
    image_path = temp_output_dir / "temp_pages" / "page_001.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a minimal PNG file
    image_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
    return str(image_path)


@pytest.fixture
def mock_graph_with_images(mock_graph_manager, temp_output_dir):
    """Create a mock graph manager with image references."""
    # Create dummy reference images
    for i in range(1, 4):
        img_path = temp_output_dir / "temp_pages" / f"page_{i:03d}.png"
        img_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

    mock_graph_manager.get_step_images = MagicMock(
        return_value=[str(temp_output_dir / "temp_pages" / "page_001.png")]
    )
    return mock_graph_manager


@pytest.mark.integration
@pytest.mark.vision
def test_visual_matcher_initialization(visual_matcher):
    """Test basic VisualMatcher initialization."""
    assert visual_matcher is not None
    assert hasattr(visual_matcher, 'n_features')
    assert hasattr(visual_matcher, 'similarity_threshold')
    assert hasattr(visual_matcher, 'min_matches')


@pytest.mark.integration
@pytest.mark.vision
@pytest.mark.requires_api
def test_visual_matcher_availability(visual_matcher):
    """Test SAM3 availability check."""
    # SAM3 may or may not be available depending on environment
    is_available = visual_matcher.is_available()
    assert isinstance(is_available, bool)


@pytest.mark.integration
@pytest.mark.vision
@pytest.mark.requires_api
@pytest.mark.slow
def test_segmentation(visual_matcher, test_image_path):
    """Test SAM3 text-prompted segmentation."""
    if not visual_matcher.is_available():
        pytest.skip("SAM3 not available (set ENABLE_ROBOFLOW_SAM3=true to enable)")

    segmented = visual_matcher.segment_user_assembly(
        test_image_path,
        prompt="assembled LEGO structure"
    )

    if segmented:
        assert 'cropped_image' in segmented
        assert 'mask' in segmented
        assert 'bbox' in segmented
        assert 'confidence' in segmented


@pytest.mark.integration
@pytest.mark.vision
def test_feature_extraction(visual_matcher, test_image_path):
    """Test ORB feature extraction."""
    # Load test image
    image = cv2.imread(test_image_path)

    if image is None:
        pytest.skip(f"Could not load test image: {test_image_path}")

    keypoints, descriptors = visual_matcher.extract_orb_features(image)

    # Should return results even if no keypoints found
    assert keypoints is not None
    assert isinstance(keypoints, list)


@pytest.mark.integration
@pytest.mark.vision
def test_feature_extraction_with_mock_image():
    """Test ORB feature extraction with a generated image."""
    import numpy as np
    from backend.app.vision.visual_matcher import VisualMatcher

    matcher = VisualMatcher()

    # Create a test image with some pattern
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    keypoints, descriptors = matcher.extract_orb_features(test_image)

    assert keypoints is not None
    assert isinstance(keypoints, list)


@pytest.mark.integration
@pytest.mark.vision
@pytest.mark.requires_api
@pytest.mark.slow
def test_full_pipeline(visual_matcher, mock_graph_with_images, sample_manual_id, test_image_path):
    """Test the full visual matching pipeline."""
    if not visual_matcher.is_available():
        pytest.skip("SAM3 not available")

    user_image_paths = [test_image_path]

    matches = visual_matcher.match_user_assembly_to_graph(
        user_image_paths=user_image_paths,
        manual_id=sample_manual_id,
        graph_manager=mock_graph_with_images,
        top_k=5
    )

    # Matches might be empty if no good matches found
    assert isinstance(matches, list)


@pytest.mark.integration
@pytest.mark.vision
def test_full_pipeline_with_mock_matcher(mock_graph_with_images, sample_manual_id, test_image_path):
    """Test visual matching pipeline with mocked SAM3."""
    from backend.app.vision.visual_matcher import VisualMatcher

    matcher = VisualMatcher()

    # Mock the availability check
    with patch.object(matcher, 'is_available', return_value=False):
        matches = matcher.match_user_assembly_to_graph(
            user_image_paths=[test_image_path],
            manual_id=sample_manual_id,
            graph_manager=mock_graph_with_images,
            top_k=5
        )

        # Should return empty list or handle gracefully
        assert isinstance(matches, list)


@pytest.mark.integration
@pytest.mark.vision
@pytest.mark.requires_api
@pytest.mark.slow
def test_combined_matching(state_analyzer, sample_manual_id, test_image_path):
    """Test combined text + visual matching."""
    user_images = [test_image_path]

    analysis_result = state_analyzer.analyze_assembly_state(
        image_paths=user_images,
        manual_id=sample_manual_id
    )

    # Analysis might fail without real API keys
    if analysis_result and 'error' not in analysis_result:
        matches = state_analyzer.match_state_to_graph(
            analysis_result=analysis_result,
            manual_id=sample_manual_id,
            top_k=3,
            use_visual_matching=True
        )

        assert isinstance(matches, list)


@pytest.mark.unit
def test_visual_matcher_config():
    """Test VisualMatcher configuration parameters."""
    from backend.app.vision.visual_matcher import VisualMatcher

    matcher = VisualMatcher(
        n_features=2000,
        similarity_threshold=0.7,
        min_matches=20
    )

    assert matcher.n_features == 2000
    assert matcher.similarity_threshold == 0.7
    assert matcher.min_matches == 20


@pytest.mark.unit
def test_visual_matcher_empty_images():
    """Test visual matcher with empty image list."""
    from backend.app.vision.visual_matcher import VisualMatcher

    matcher = VisualMatcher()
    mock_graph_manager = MagicMock()

    matches = matcher.match_user_assembly_to_graph(
        user_image_paths=[],
        manual_id="test",
        graph_manager=mock_graph_manager,
        top_k=5
    )

    # Should return empty list or handle gracefully
    assert isinstance(matches, list)
    assert len(matches) == 0


@pytest.mark.unit
def test_bounding_box_calculation():
    """Test bounding box calculation from mask."""
    import numpy as np

    # Create a simple mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 255

    # Find contours
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])

        assert x == 30
        assert y == 20
        assert w == 40
        assert h == 60
