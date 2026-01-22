#!/usr/bin/env python3
"""
Security Setup Verification Script

This script tests all security features to ensure they're working correctly.
Run this after setting up JWT_SECRET_KEY and before deploying.

Usage:
    python3 tests/test_security.py

    # Or with pytest
    pytest tests/test_security.py -v
"""

import os
import sys
import requests
from pathlib import Path
from datetime import timedelta

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

def test_jwt_secret():
    """Test JWT secret key is configured"""
    print("\n1️⃣  Testing JWT Secret Key...")

    jwt_secret = os.getenv("JWT_SECRET_KEY")
    if not jwt_secret:
        print("   ❌ JWT_SECRET_KEY not set in .env")
        return False

    if jwt_secret == "CHANGE_THIS_TO_A_SECURE_RANDOM_KEY":
        print("   ❌ JWT_SECRET_KEY is still the default value!")
        return False

    if len(jwt_secret) < 32:
        print("   ⚠️  JWT_SECRET_KEY is too short (should be 64+ characters)")
        return False

    print(f"   ✅ JWT_SECRET_KEY configured ({len(jwt_secret)} characters)")
    return True

def test_cors_config():
    """Test CORS configuration"""
    print("\n2️⃣  Testing CORS Configuration...")

    frontend_url = os.getenv("FRONTEND_URL")
    if not frontend_url:
        print("   ⚠️  FRONTEND_URL not set (will default to http://localhost:3000)")
    else:
        print(f"   ✅ FRONTEND_URL: {frontend_url}")

    prod_frontend = os.getenv("PRODUCTION_FRONTEND")
    if prod_frontend:
        print(f"   ✅ PRODUCTION_FRONTEND: {prod_frontend}")
    else:
        print("   ℹ️  PRODUCTION_FRONTEND not set (optional)")

    return True

def test_environment():
    """Test environment configuration"""
    print("\n3️⃣  Testing Environment Configuration...")

    env = os.getenv("ENVIRONMENT", "development")
    print(f"   ℹ️  ENVIRONMENT: {env}")

    if env == "production":
        print("   ⚠️  Running in PRODUCTION mode")
        print("      - HSTS headers enabled")
        print("      - Make sure HTTPS is configured")
    else:
        print("   ✅ Running in DEVELOPMENT mode")

    return True

def test_token_generation():
    """Test token generation"""
    print("\n4️⃣  Testing Token Generation...")

    try:
        from app.auth import create_access_token

        test_token = create_access_token(
            data={"sub": "test@example.com", "role": "admin"},
            expires_delta=timedelta(minutes=1)
        )

        if not test_token:
            print("   ❌ Failed to generate token")
            return False

        print(f"   ✅ Token generation working")
        print(f"   ℹ️  Sample token: {test_token[:50]}...")
        return True

    except Exception as e:
        print(f"   ❌ Error generating token: {e}")
        return False

def test_backend_running():
    """Test if backend is running"""
    print("\n5️⃣  Testing Backend Connection...")

    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Backend is running")
            print(f"   ℹ️  Status: {data.get('status')}")
            print(f"   ℹ️  Version: {data.get('version')}")
            return True
        else:
            print(f"   ❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ⚠️  Backend not running")
        print("   ℹ️  Start it with: cd backend && uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_authentication_required():
    """Test that admin endpoints require authentication"""
    print("\n6️⃣  Testing Authentication Protection...")

    backend_running = test_backend_running()
    if not backend_running:
        print("   ⏭️  Skipping (backend not running)")
        return None

    try:
        # Try to access public endpoint (should work)
        response = requests.get("http://localhost:8000/api/manuals", timeout=5)
        if response.status_code == 200:
            print("   ✅ Public endpoints accessible without auth")

        # Try to delete without auth (should fail with 401)
        response = requests.delete("http://localhost:8000/api/manual/test", timeout=5)

        if response.status_code == 401:
            print("   ✅ Admin endpoints require authentication")
            return True
        elif response.status_code == 403:
            print("   ✅ Admin endpoints require authorization")
            return True
        else:
            print(f"   ⚠️  Unexpected status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all security tests"""
    print("=" * 70)
    print("SECURITY SETUP VERIFICATION")
    print("=" * 70)
    print("\nThis script verifies your security configuration is correct.")
    print()

    # Load .env
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    tests = [
        ("JWT Secret Key", test_jwt_secret),
        ("CORS Configuration", test_cors_config),
        ("Environment", test_environment),
        ("Token Generation", test_token_generation),
        ("Backend Connection", test_backend_running),
        ("Authentication", test_authentication_required),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n   ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        if result is True:
            print(f"✅ {name}")
        elif result is False:
            print(f"❌ {name}")
        else:
            print(f"⏭️  {name} (skipped)")

    print()
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Skipped: {skipped}/{len(results)}")

    if failed > 0:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Add JWT_SECRET_KEY to .env")
        print("  2. Add FRONTEND_URL to .env")
        print("  3. Install dependencies: uv pip install -r requirements.txt")
        return 1
    elif skipped > 0:
        print("\n✅ Configuration looks good!")
        print("⚠️  Start the backend to test authentication:")
        print("    cd backend && uvicorn app.main:app --reload")
        return 0
    else:
        print("\n✅ All security features verified!")
        print("\nNext steps:")
        print("  1. Generate admin token: python3 generate_admin_token.py")
        print("  2. Test admin endpoints with token")
        print("  3. Deploy to production")
        return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(0)
