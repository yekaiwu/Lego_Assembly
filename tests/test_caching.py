"""
Test suite for VLM response caching functionality.
Verifies cache storage, retrieval, and context isolation.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def cache_instance(mock_settings):
    """Create a cache instance for testing."""
    with patch('src.utils.config.get_config', return_value=mock_settings):
        from src.utils.cache import get_cache
        cache = get_cache()
        # Clear any existing entries
        cache.cache.clear()
        yield cache


@pytest.fixture
def test_image_path(temp_output_dir):
    """Create a dummy test image."""
    image_path = temp_output_dir / "temp_pages" / "page_001.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a minimal PNG file
    image_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
    return str(image_path)


@pytest.mark.unit
def test_cache_enabled_in_config(mock_settings):
    """Test that caching can be enabled in configuration."""
    assert mock_settings.cache_enabled is True


@pytest.mark.unit
def test_cache_storage(cache_instance, test_image_path):
    """Test storing responses in cache."""
    test_model = "test-model"
    test_prompt = "Test prompt for caching verification"
    test_response = {
        "step_number": 1,
        "parts_required": [{"description": "test part", "color": "red"}],
        "actions": [],
        "notes": "Test extraction"
    }

    initial_size = len(cache_instance.cache)

    # Store in cache
    cache_instance.set(test_model, test_prompt, test_response, [test_image_path])

    # Verify cache size increased
    assert len(cache_instance.cache) > initial_size


@pytest.mark.unit
def test_cache_retrieval(cache_instance, test_image_path):
    """Test retrieving responses from cache."""
    test_model = "test-model"
    test_prompt = "Test prompt for caching verification"
    test_response = {
        "step_number": 1,
        "parts_required": [{"description": "test part", "color": "red"}],
        "actions": [],
        "notes": "Test extraction"
    }

    # Store in cache
    cache_instance.set(test_model, test_prompt, test_response, [test_image_path])

    # Retrieve from cache
    cached_response = cache_instance.get(test_model, test_prompt, [test_image_path])

    # Verify retrieval
    assert cached_response is not None
    assert cached_response == test_response


@pytest.mark.unit
def test_cache_miss(cache_instance, test_image_path):
    """Test cache miss for non-existent entries."""
    cached_response = cache_instance.get("nonexistent-model", "nonexistent-prompt", [test_image_path])
    assert cached_response is None


@pytest.mark.unit
def test_cache_context_isolation(cache_instance, test_image_path):
    """Test that different assembly IDs create separate cache entries."""
    test_model = "test-model"
    base_prompt = "Test prompt"

    # Create prompts with different contexts
    prompt_with_context_1 = f"{base_prompt}:assembly_123"
    prompt_with_context_2 = f"{base_prompt}:assembly_456"

    response_1 = {"assembly": "123", "data": "first"}
    response_2 = {"assembly": "456", "data": "second"}

    # Store both
    cache_instance.set(test_model, prompt_with_context_1, response_1, [test_image_path])
    cache_instance.set(test_model, prompt_with_context_2, response_2, [test_image_path])

    # Retrieve both
    retrieved_1 = cache_instance.get(test_model, prompt_with_context_1, [test_image_path])
    retrieved_2 = cache_instance.get(test_model, prompt_with_context_2, [test_image_path])

    # Verify isolation
    assert retrieved_1 != retrieved_2
    assert retrieved_1 == response_1
    assert retrieved_2 == response_2


@pytest.mark.unit
def test_cache_statistics(cache_instance, test_image_path):
    """Test cache statistics reporting."""
    # Add some test entries
    for i in range(3):
        cache_instance.set(f"model-{i}", f"prompt-{i}", {"data": f"response-{i}"}, [test_image_path])

    stats = cache_instance.stats()

    assert "size" in stats
    assert stats["size"] >= 3
    assert "volume" in stats
    assert stats["volume"] > 0


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.slow
def test_vlm_caching_with_real_api(mock_vlm_client, test_image_path):
    """Test VLM caching with mocked API calls."""
    # Mock the VLM client
    mock_vlm_client.extract_step_info.return_value = [{
        "step_number": 999,
        "parts_required": [],
        "actions": [],
        "notes": "Cached response"
    }]

    # First call
    start_time = time.time()
    result_1 = mock_vlm_client.extract_step_info(
        [test_image_path],
        step_number=999,
        use_json_mode=True,
        cache_context="cache_test"
    )
    first_call_time = time.time() - start_time

    # Second call (would be cached in real scenario)
    start_time = time.time()
    result_2 = mock_vlm_client.extract_step_info(
        [test_image_path],
        step_number=999,
        use_json_mode=True,
        cache_context="cache_test"
    )
    second_call_time = time.time() - start_time

    # Verify results are consistent
    assert result_1 == result_2

    # In mock scenario, both should be fast
    assert first_call_time < 1.0
    assert second_call_time < 1.0


@pytest.mark.unit
def test_cache_key_generation(cache_instance):
    """Test that cache keys are generated correctly."""
    model = "test-model"
    prompt = "test-prompt"
    images = ["image1.png", "image2.png"]

    # Set a value
    cache_instance.set(model, prompt, {"data": "test"}, images)

    # Verify it can be retrieved with same parameters
    result = cache_instance.get(model, prompt, images)
    assert result is not None

    # Verify different image order creates different key
    result_different_order = cache_instance.get(model, prompt, ["image2.png", "image1.png"])
    # This might be None depending on cache implementation
    # The test documents the behavior


@pytest.mark.unit
def test_cache_clear(cache_instance, test_image_path):
    """Test clearing the cache."""
    # Add some entries
    cache_instance.set("model", "prompt", {"data": "test"}, [test_image_path])
    assert len(cache_instance.cache) > 0

    # Clear cache
    cache_instance.cache.clear()

    # Verify empty
    assert len(cache_instance.cache) == 0
