"""
Detailed Gemini API diagnostic tool.
Tests Gemini API key, model availability, and vision capabilities.
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_gemini_detailed():
    """Run detailed Gemini API tests."""

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")

    print("=" * 80)
    print("Gemini API Detailed Diagnostic")
    print("=" * 80)
    print()

    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        return

    print(f"✓ API Key found: {api_key[:15]}...{api_key[-4:]}")
    print(f"✓ Model to test: {model}")
    print()

    # Test 1: List available models
    print("-" * 80)
    print("Test 1: Listing Available Models")
    print("-" * 80)
    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✓ SUCCESS: API key is valid")
            models_data = response.json()
            models = models_data.get('models', [])

            print(f"✓ Total models available: {len(models)}")

            # List vision-capable models
            print("\nVision-capable models:")
            vision_models = []
            for m in models:
                model_name = m.get('name', '').replace('models/', '')
                if 'vision' in model_name.lower() or '1.5' in model_name or '2.0' in model_name:
                    vision_models.append(model_name)
                    print(f"  - {model_name}")

            # Check if our target model is available
            target_in_list = any(model in m.get('name', '') for m in models)
            if target_in_list:
                print(f"\n✓ Target model '{model}' is available")
            else:
                print(f"\n⚠ Target model '{model}' not found in list")
                print(f"  Available flash models:")
                for m in models:
                    name = m.get('name', '').replace('models/', '')
                    if 'flash' in name.lower():
                        print(f"    - {name}")

        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text}")
            return

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return

    print()

    # Test 2: Simple text generation
    print("-" * 80)
    print("Test 2: Simple Text Generation")
    print("-" * 80)

    test_models = [model, "gemini-1.5-flash", "gemini-1.5-pro"]

    for test_model in test_models:
        print(f"\nTrying model: {test_model}")
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{test_model}:generateContent?key={api_key}"

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Say 'Hello, Gemini is working!'"}
                        ]
                    }
                ]
            }

            response = requests.post(url, json=payload, timeout=30)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                print(f"  ✓ SUCCESS: {test_model} is working")
                result = response.json()
                try:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    print(f"  Response: {text}")
                    break  # Success! No need to test other models
                except:
                    print(f"  Response structure: {result}")
            else:
                print(f"  ❌ FAILED: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error: {error_data}")
                except:
                    print(f"  Raw response: {response.text}")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")

    print()

    # Test 3: Vision test with base64 image
    print("-" * 80)
    print("Test 3: Vision Request Test (with base64 image)")
    print("-" * 80)
    print(f"Testing {model} with a tiny test image...")
    print()

    # A minimal 1x1 red pixel PNG
    tiny_red_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Describe what you see in this image in one short sentence."},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": tiny_red_pixel
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 50
            }
        }

        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        print()

        if response.status_code == 200:
            result = response.json()
            try:
                text = result['candidates'][0]['content']['parts'][0]['text']
                print("✅ SUCCESS! Gemini vision is working!")
                print()
                print(f"Model Response: {text}")
                print()
                print("=" * 80)
                print("✓ Gemini is fully operational and ready to use!")
                print("✓ You can now run main.py with Gemini as primary VLM")
                print("=" * 80)
            except Exception as e:
                print(f"⚠ Response parsing error: {e}")
                print(f"Full response: {json.dumps(result, indent=2)}")

        else:
            print(f"❌ FAILED: {response.status_code}")
            print()

            try:
                error_data = response.json()
                print("Error Details:")
                print(json.dumps(error_data, indent=2))

                # Check for common errors
                error_msg = str(error_data)
                if "404" in str(response.status_code):
                    print("\n⚠ Model Not Found (404)")
                    print(f"  The model '{model}' might not exist or be available.")
                    print("  Try one of these instead:")
                    print("    - gemini-1.5-flash")
                    print("    - gemini-1.5-pro")
                    print("    - gemini-2.0-flash-exp")
                elif "API key not valid" in error_msg:
                    print("\n⚠ API Key Issue")
                    print("  Your Gemini API key may be invalid or restricted")
                elif "quota" in error_msg.lower():
                    print("\n⚠ Quota Issue")
                    print("  You may have exceeded your free tier quota")

            except:
                print(f"Raw response: {response.text}")

    except Exception as e:
        print(f"❌ ERROR: {e}")

    print()
    print("=" * 80)
    print("Diagnostic Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_gemini_detailed()
