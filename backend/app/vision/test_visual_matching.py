"""
Test script for visual matching with SAM.

This script demonstrates and tests the visual matching pipeline:
1. SAM segmentation of user assembly
2. ORB feature matching against graph images
3. Combined text + visual matching

Usage:
    python -m backend.app.vision.test_visual_matching
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_visual_matcher_basic():
    """Test basic VisualMatcher functionality."""
    from backend.app.vision.visual_matcher import VisualMatcher

    logger.info("=" * 60)
    logger.info("Test 1: Basic VisualMatcher initialization")
    logger.info("=" * 60)

    # Initialize visual matcher
    matcher = VisualMatcher()

    if not matcher.is_available():
        logger.error("❌ SAM3 not available - visual matching cannot be tested")
        logger.info("To enable SAM3, ensure ENABLE_ROBOFLOW_SAM3=true in .env")
        return False

    logger.info("✓ VisualMatcher initialized successfully with SAM3")
    logger.info(f"  - ORB features: {matcher.n_features}")
    logger.info(f"  - Similarity threshold: {matcher.similarity_threshold}")
    logger.info(f"  - Min matches: {matcher.min_matches}")

    return True


def test_segmentation(image_path: str):
    """Test SAM3 text-prompted segmentation on a user image."""
    from backend.app.vision.visual_matcher import VisualMatcher

    logger.info("=" * 60)
    logger.info("Test 2: SAM3 text-prompted segmentation")
    logger.info("=" * 60)

    matcher = VisualMatcher()

    if not matcher.is_available():
        logger.error("❌ SAM3 not available")
        return False

    logger.info(f"Segmenting image: {image_path}")
    logger.info("Using prompt: 'assembled LEGO structure'")

    segmented = matcher.segment_user_assembly(
        image_path,
        prompt="assembled LEGO structure"
    )

    if not segmented:
        logger.error("❌ Segmentation failed")
        return False

    logger.info("✓ Segmentation successful")
    logger.info(f"  - Cropped image shape: {segmented['cropped_image'].shape}")
    logger.info(f"  - Mask shape: {segmented['mask'].shape}")
    logger.info(f"  - Bounding box: {segmented['bbox']}")
    logger.info(f"  - Confidence: {segmented['confidence']:.2f}")

    return True


def test_feature_extraction(image_path: str):
    """Test ORB feature extraction on an image."""
    import cv2
    from backend.app.vision.visual_matcher import VisualMatcher

    logger.info("=" * 60)
    logger.info("Test 3: ORB feature extraction (LEGO-optimized)")
    logger.info("=" * 60)

    matcher = VisualMatcher()

    if not matcher.is_available():
        logger.error("❌ SAM3 not available")
        return False

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"❌ Failed to load image: {image_path}")
        return False

    logger.info(f"Extracting features from: {image_path}")

    keypoints, descriptors = matcher.extract_orb_features(image)

    if descriptors is None:
        logger.error("❌ Feature extraction failed")
        return False

    logger.info("✓ Feature extraction successful")
    logger.info(f"  - Keypoints detected: {len(keypoints)}")
    logger.info(f"  - Descriptor shape: {descriptors.shape}")

    return True


def test_full_pipeline(manual_id: str, user_image_paths: list):
    """Test the full visual matching pipeline with SAM3."""
    from backend.app.graph.graph_manager import get_graph_manager
    from backend.app.vision.visual_matcher import VisualMatcher

    logger.info("=" * 60)
    logger.info("Test 4: Full visual matching pipeline (SAM3 + ORB)")
    logger.info("=" * 60)

    matcher = VisualMatcher()

    if not matcher.is_available():
        logger.error("❌ SAM3 not available")
        return False

    # Load graph
    graph_manager = get_graph_manager()
    graph = graph_manager.load_graph(manual_id)

    if not graph:
        logger.error(f"❌ Failed to load graph for manual {manual_id}")
        return False

    logger.info(f"Testing with manual: {manual_id}")
    logger.info(f"User images: {len(user_image_paths)}")

    # Run visual matching
    matches = matcher.match_user_assembly_to_graph(
        user_image_paths=user_image_paths,
        manual_id=manual_id,
        graph_manager=graph_manager,
        top_k=5
    )

    if not matches:
        logger.warning("⚠ No visual matches found")
        return True  # Not an error, just no matches

    logger.info("✓ Visual matching complete")
    logger.info(f"  - Found {len(matches)} matches")

    for i, match in enumerate(matches, 1):
        logger.info(f"\n  Match {i}:")
        logger.info(f"    - Step: {match['step_number']}")
        logger.info(f"    - Similarity: {match['visual_similarity']:.2f}")
        logger.info(f"    - Reason: {match['match_reason']}")
        logger.info(f"    - Reference: {match['reference_image_path']}")

    return True


def test_combined_matching(manual_id: str, user_image_paths: list):
    """Test combined text + visual matching."""
    from backend.app.vision.state_analyzer import StateAnalyzer

    logger.info("=" * 60)
    logger.info("Test 5: Combined text + visual matching")
    logger.info("=" * 60)

    analyzer = StateAnalyzer()

    logger.info(f"Analyzing assembly for manual: {manual_id}")
    logger.info(f"User images: {len(user_image_paths)}")

    # Analyze assembly state
    analysis_result = analyzer.analyze_assembly_state(
        image_paths=user_image_paths,
        manual_id=manual_id
    )

    if not analysis_result or 'error' in analysis_result:
        logger.error("❌ Assembly analysis failed")
        return False

    logger.info("✓ Assembly analysis complete")
    logger.info(f"  - Detected parts: {len(analysis_result.get('detected_parts', []))}")

    # Match to graph with visual matching
    matches = analyzer.match_state_to_graph(
        analysis_result=analysis_result,
        manual_id=manual_id,
        top_k=3,
        use_visual_matching=True
    )

    if not matches:
        logger.warning("⚠ No matches found")
        return True

    logger.info("✓ Combined matching complete")
    logger.info(f"  - Found {len(matches)} matches")

    for i, match in enumerate(matches, 1):
        logger.info(f"\n  Match {i}:")
        logger.info(f"    - Step: {match['step_number']}")

        if 'combined_confidence' in match:
            logger.info(f"    - Combined confidence: {match['combined_confidence']:.2f}")
            logger.info(f"    - Text confidence: {match.get('text_confidence', 0.0):.2f}")
            logger.info(f"    - Visual confidence: {match.get('visual_confidence', 0.0):.2f}")
        else:
            logger.info(f"    - Text-only confidence: {match.get('confidence', 0.0):.2f}")

        logger.info(f"    - Reason: {match['match_reason']}")

    return True


def main():
    """Run all tests."""
    import os

    logger.info("\n" + "=" * 60)
    logger.info("VISUAL MATCHING TEST SUITE")
    logger.info("=" * 60 + "\n")

    # Test 1: Basic initialization
    if not test_visual_matcher_basic():
        logger.error("❌ Test 1 failed - cannot continue")
        return

    # Check if we have test data
    # You can modify these paths to point to actual test images
    test_image_path = "output/temp_pages/page_001.png"
    manual_id = "6262059"
    user_images = ["output/temp_pages/page_001.png"]

    # Test 2: Segmentation (if test image exists)
    if os.path.exists(test_image_path):
        test_segmentation(test_image_path)
    else:
        logger.warning(f"⚠ Skipping segmentation test - image not found: {test_image_path}")

    # Test 3: Feature extraction (if test image exists)
    if os.path.exists(test_image_path):
        test_feature_extraction(test_image_path)
    else:
        logger.warning(f"⚠ Skipping feature extraction test - image not found: {test_image_path}")

    # Test 4: Full visual matching pipeline (requires graph)
    from backend.app.graph.graph_manager import get_graph_manager
    graph_manager = get_graph_manager()
    if graph_manager.load_graph(manual_id):
        if all(os.path.exists(img) for img in user_images):
            test_full_pipeline(manual_id, user_images)
        else:
            logger.warning("⚠ Skipping full pipeline test - user images not found")
    else:
        logger.warning(f"⚠ Skipping full pipeline test - graph not found for {manual_id}")

    # Test 5: Combined matching (requires graph and VLM)
    if graph_manager.load_graph(manual_id):
        if all(os.path.exists(img) for img in user_images):
            try:
                test_combined_matching(manual_id, user_images)
            except Exception as e:
                logger.error(f"❌ Combined matching test failed: {e}")
        else:
            logger.warning("⚠ Skipping combined matching test - user images not found")
    else:
        logger.warning(f"⚠ Skipping combined matching test - graph not found for {manual_id}")

    logger.info("\n" + "=" * 60)
    logger.info("TEST SUITE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
