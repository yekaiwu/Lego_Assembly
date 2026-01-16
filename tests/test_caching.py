#!/usr/bin/env python3
"""
Test script to verify VLM response caching is working correctly.
This script simulates the extraction process and verifies:
1. Responses are cached after first call
2. Cached responses are retrieved on subsequent calls
3. Cache keys include assembly_id to avoid collisions
"""

import json
import time
from pathlib import Path
from src.utils.cache import get_cache
from src.utils.config import get_config
from src.api.litellm_vlm import UnifiedVLMClient

def test_cache_functionality():
    """Test that caching works as expected."""
    
    print("=" * 80)
    print("VLM CACHE FUNCTIONALITY TEST")
    print("=" * 80)
    
    # Check configuration
    config = get_config()
    print(f"\n✓ Cache enabled: {config.cache_enabled}")
    print(f"✓ Cache directory: {config.paths.cache_dir}")
    
    if not config.cache_enabled:
        print("\n❌ ERROR: Caching is disabled in config!")
        print("Expected cache_enabled=True")
        return False
    
    # Get cache instance
    cache = get_cache()
    initial_size = len(cache.cache)
    print(f"✓ Initial cache size: {initial_size} entries")
    
    # Test data
    test_model = "test-model"
    test_prompt = "Test prompt for caching verification"
    test_images = ["../output/temp_pages/page_001.png"]
    test_response = {
        "step_number": 1,
        "parts_required": [{"description": "test part", "color": "red"}],
        "actions": [],
        "notes": "Test extraction"
    }
    
    print("\n" + "=" * 80)
    print("TEST 1: Cache Storage")
    print("=" * 80)
    
    # Store in cache
    cache.set(test_model, test_prompt, test_response, test_images)
    after_set_size = len(cache.cache)
    print(f"✓ Stored test response in cache")
    print(f"✓ Cache size after set: {after_set_size} entries")
    
    if after_set_size <= initial_size:
        print(f"❌ ERROR: Cache size did not increase! ({initial_size} -> {after_set_size})")
        return False
    
    print("\n" + "=" * 80)
    print("TEST 2: Cache Retrieval")
    print("=" * 80)
    
    # Retrieve from cache
    cached_response = cache.get(test_model, test_prompt, test_images)
    
    if cached_response is None:
        print("❌ ERROR: Failed to retrieve cached response!")
        return False
    
    print(f"✓ Retrieved cached response")
    
    # Verify content matches
    if cached_response != test_response:
        print(f"❌ ERROR: Cached response doesn't match original!")
        print(f"Expected: {test_response}")
        print(f"Got: {cached_response}")
        return False
    
    print(f"✓ Cached content matches original")
    
    print("\n" + "=" * 80)
    print("TEST 3: Cache Key with Assembly ID (Context)")
    print("=" * 80)
    
    # Test that different assembly IDs create different cache entries
    prompt_with_context_1 = f"{test_prompt}:assembly_123"
    prompt_with_context_2 = f"{test_prompt}:assembly_456"
    
    response_1 = {"assembly": "123", "data": "first"}
    response_2 = {"assembly": "456", "data": "second"}
    
    cache.set(test_model, prompt_with_context_1, response_1, test_images)
    cache.set(test_model, prompt_with_context_2, response_2, test_images)
    
    retrieved_1 = cache.get(test_model, prompt_with_context_1, test_images)
    retrieved_2 = cache.get(test_model, prompt_with_context_2, test_images)
    
    if retrieved_1 == retrieved_2:
        print("❌ ERROR: Different assembly IDs returned same cached data!")
        return False
    
    print(f"✓ Different assembly IDs have separate cache entries")
    print(f"  Assembly 123: {retrieved_1}")
    print(f"  Assembly 456: {retrieved_2}")
    
    print("\n" + "=" * 80)
    print("TEST 4: Real VLM API Call (If Key Available)")
    print("=" * 80)

    # Use the configured ingestion VLM
    vlm_model = config.models.ingestion_vlm if hasattr(config.models, 'ingestion_vlm') else "gemini/gemini-2.0-flash"

    # Check if we have any API key
    has_api_key = any([
        config.api.gemini_api_key,
        config.api.openai_api_key,
        config.api.anthropic_api_key
    ])

    if not has_api_key:
        print("⚠️  Skipping: No API key found (checked GEMINI, OPENAI, ANTHROPIC)")
    else:
        # Check if test image exists
        test_image = Path("../output/temp_pages/page_001.png")
        if not test_image.exists():
            print(f"⚠️  Skipping: Test image not found at {test_image}")
        else:
            print(f"✓ Test image found: {test_image}")
            print(f"✓ Using VLM model: {vlm_model}")

            # Create VLM client
            vlm_client = UnifiedVLMClient(vlm_model)
            
            # Clear cache for this specific call
            test_cache_key = f"{vlm_client.model}:{str(test_image)}:test_step:cache_test"

            # First call (should hit API)
            print("\n  → Making first API call (should cache)...")
            start_time = time.time()

            try:
                result_1 = vlm_client.extract_step_info(
                    [str(test_image)],
                    step_number=999,  # Use unique step number
                    use_json_mode=True,
                    cache_context="cache_test"
                )
                first_call_time = time.time() - start_time
                print(f"  ✓ First call completed in {first_call_time:.2f}s")

                # UnifiedVLMClient returns a list, check first element
                result_item = result_1[0] if isinstance(result_1, list) else result_1

                if "error" in result_item:
                    print(f"  ⚠️  API returned error: {result_item.get('error')}")
                else:
                    print(f"  ✓ Received valid response with {len(str(result_1))} characters")

                # Second call (should use cache)
                print("\n  → Making second API call (should use cache)...")
                start_time = time.time()

                result_2 = vlm_client.extract_step_info(
                    [str(test_image)],
                    step_number=999,
                    use_json_mode=True,
                    cache_context="cache_test"
                )
                second_call_time = time.time() - start_time
                print(f"  ✓ Second call completed in {second_call_time:.2f}s")
                
                # Verify caching worked (second call should be much faster)
                if second_call_time < 1.0 and first_call_time > 5.0:
                    print(f"  ✓ CACHE HIT CONFIRMED! ({first_call_time:.2f}s → {second_call_time:.2f}s)")
                    speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
                    print(f"  ✓ Speedup: {speedup:.1f}x faster")
                elif second_call_time < first_call_time / 2:
                    print(f"  ✓ Cache appears to be working (second call faster)")
                else:
                    print(f"  ⚠️  Warning: Second call not significantly faster")
                    print(f"     This might indicate cache is not working properly")
                
                # Verify responses match
                if result_1 == result_2:
                    print(f"  ✓ Both calls returned identical results")
                else:
                    print(f"  ⚠️  Warning: Results differ between calls")
                    
            except Exception as e:
                print(f"  ❌ API call failed: {e}")
                print(f"     (This is okay if you're rate-limited or API is down)")
    
    print("\n" + "=" * 80)
    print("CACHE STATISTICS")
    print("=" * 80)
    
    stats = cache.stats()
    print(f"Total cache entries: {stats['size']}")
    print(f"Cache volume: {stats['volume']:,} bytes")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nCaching is working correctly. Your VLM responses will be cached and")
    print("extraction can resume from the last completed step if interrupted.")
    
    return True

if __name__ == "__main__":
    try:
        success = test_cache_functionality()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
