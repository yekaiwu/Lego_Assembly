#!/usr/bin/env python3
"""
Admin Token Generator for LEGO Assembly RAG System

This script generates a JWT token with admin privileges for accessing
admin-only endpoints like manual ingestion, deletion, and file uploads.

Usage:
    python3 generate_admin_token.py

The token will be printed to console and saved to .admin_token.txt (gitignored)
"""

import sys
from pathlib import Path
from datetime import timedelta

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.auth import create_access_token

def main():
    print("=" * 70)
    print("ADMIN TOKEN GENERATOR")
    print("=" * 70)
    print()

    # Get admin email
    email = input("Enter admin email (default: admin@example.com): ").strip()
    if not email:
        email = "admin@example.com"

    # Get token validity period
    print("\nToken validity:")
    print("  1. 1 day (recommended for testing)")
    print("  2. 30 days")
    print("  3. 90 days")
    print("  4. 1 year (for production deployment)")

    choice = input("\nSelect option (1-4, default: 4): ").strip()

    validity_map = {
        "1": timedelta(days=1),
        "2": timedelta(days=30),
        "3": timedelta(days=90),
        "4": timedelta(days=365),
    }

    expires_delta = validity_map.get(choice, timedelta(days=365))
    days = expires_delta.days

    # Create admin token
    admin_token = create_access_token(
        data={
            "sub": email,
            "role": "admin",
            "email": email
        },
        expires_delta=expires_delta
    )

    print()
    print("=" * 70)
    print("✅ ADMIN TOKEN GENERATED")
    print("=" * 70)
    print(f"\nEmail: {email}")
    print(f"Role: admin")
    print(f"Valid for: {days} days")
    print()
    print("Token:")
    print("-" * 70)
    print(admin_token)
    print("-" * 70)
    print()
    print("⚠️  KEEP THIS TOKEN SECRET! Anyone with this token has admin access.")
    print()

    # Save to file
    token_file = Path(__file__).parent / ".admin_token.txt"
    with open(token_file, "w") as f:
        f.write(f"# Admin Token for {email}\n")
        f.write(f"# Generated: {timedelta(days=365)}\n")
        f.write(f"# Valid for: {days} days\n")
        f.write(f"# Role: admin\n\n")
        f.write(f"export ADMIN_TOKEN='{admin_token}'\n\n")
        f.write(f"# Usage in curl:\n")
        f.write(f'# curl -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8000/api/manuals\n\n')
        f.write(f"# Raw token:\n")
        f.write(admin_token)

    print(f"✅ Token saved to: {token_file}")
    print("   (This file is gitignored for security)")
    print()

    print("=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    print()
    print("1. Test authentication:")
    print(f'   curl -H "Authorization: Bearer {admin_token[:50]}..." \\')
    print('        http://localhost:8000/api/manuals')
    print()
    print("2. Ingest a manual:")
    print(f'   curl -X POST \\')
    print(f'        -H "Authorization: Bearer {admin_token[:50]}..." \\')
    print('        http://localhost:8000/api/ingest/manual/6454922')
    print()
    print("3. Delete a manual:")
    print(f'   curl -X DELETE \\')
    print(f'        -H "Authorization: Bearer {admin_token[:50]}..." \\')
    print('        http://localhost:8000/api/manual/6454922')
    print()
    print("4. Use in your code:")
    print('   headers = {"Authorization": f"Bearer {admin_token}"}')
    print('   response = requests.get("http://localhost:8000/api/manuals", headers=headers)')
    print()
    print("=" * 70)
    print()
    print("To load the token in your shell:")
    print(f"  source {token_file}")
    print("  echo $ADMIN_TOKEN")
    print()
    print("Happy building! 🔒")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. JWT_SECRET_KEY is set in .env")
        print("  2. You're in the Lego_Assembly directory")
        print("  3. Security dependencies are installed")
        sys.exit(1)
