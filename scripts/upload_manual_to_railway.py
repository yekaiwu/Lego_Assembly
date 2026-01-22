#!/usr/bin/env python3
"""
Script to upload manual files from local output/ directory to Railway backend.

Usage:
    python scripts/upload_manual_to_railway.py <manual_id> <railway_url>
    
Example:
    python scripts/upload_manual_to_railway.py 6262059 https://legoassembly-production.up.railway.app
"""

import sys
import os
import requests
from pathlib import Path
from typing import Optional

def upload_manual(manual_id: str, railway_url: str, output_dir: Path = Path("./output")):
    """
    Upload all files for a manual to Railway backend.
    
    Args:
        manual_id: Manual identifier (e.g., "6262059")
        railway_url: Railway backend URL (e.g., "https://legoassembly-production.up.railway.app")
        output_dir: Local output directory path
    """
    # Remove trailing slash from URL
    railway_url = railway_url.rstrip('/')
    
    # Required files (now in manual_id subdirectory)
    manual_dir = output_dir / manual_id
    required_files = {
        'extracted_json': manual_dir / f"{manual_id}_extracted.json",
        'plan_json': manual_dir / f"{manual_id}_plan.json",
        'dependencies_json': manual_dir / f"{manual_id}_dependencies.json",
    }
    
    # Optional files
    optional_files = {
        'plan_txt': manual_dir / f"{manual_id}_plan.txt",
        'graph_json': manual_dir / f"{manual_id}_graph.json",
    }
    
    # Check required files exist
    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        print(f"❌ Error: Missing required files: {', '.join(missing)}")
        print(f"   Expected in: {output_dir}")
        sys.exit(1)
    
    # Prepare files for upload
    files = {}
    
    # Add required JSON files (single file per field)
    for key, path in required_files.items():
        files[key] = (path.name, open(path, 'rb'), 'application/json')
        print(f"✓ Found {path.name}")
    
    # Add optional JSON files (single file per field)
    for key, path in optional_files.items():
        if path.exists():
            files[key] = (path.name, open(path, 'rb'), 'application/json')
            print(f"✓ Found {path.name} (optional)")
    
    # Upload images from temp_pages/{manual_id}/
    # For multiple files with same field name, use a list of (filename, fileobj, content_type) tuples
    temp_pages_dir = output_dir / "temp_pages" / manual_id
    image_files = []
    if temp_pages_dir.exists():
        for img_path in sorted(temp_pages_dir.glob("page_*.png")):
            # Format: (filename, fileobj, content_type) - field name is the dict key
            image_files.append((img_path.name, open(img_path, 'rb'), 'image/png'))
            print(f"✓ Found image: {img_path.name}")
    
    # Add images list to files dict
    if image_files:
        files['images'] = image_files
    
    # Upload to Railway
    upload_url = f"{railway_url}/api/upload/manual/{manual_id}"
    print(f"\n📤 Uploading to: {upload_url}")
    
    # Prepare authentication headers
    headers = {}
    admin_token = os.getenv("ADMIN_TOKEN")
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"
    else:
        print("⚠️  Warning: ADMIN_TOKEN not set. Upload may fail if authentication is required.")
        print("   Set it with: export ADMIN_TOKEN='your-token-here'")
    
    try:
        response = requests.post(upload_url, files=files, headers=headers, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        print(f"\n✅ Upload successful!")
        print(f"   Uploaded {len(result['uploaded_files'])} files")
        print(f"   Manual ID: {result['manual_id']}")
        print(f"\n📝 Next step: Ingest the manual")
        if admin_token:
            print(f"   curl -X POST -H \"Authorization: Bearer $ADMIN_TOKEN\" {railway_url}/api/ingest/manual/{manual_id}")
        else:
            print(f"   curl -X POST -H \"Authorization: Bearer <your-token>\" {railway_url}/api/ingest/manual/{manual_id}")
        print(f"   Or use the frontend to trigger ingestion")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Upload failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        sys.exit(1)
    finally:
        # Close all file handles
        for file_data in files.values():
            if isinstance(file_data, list):
                # Multiple files: list of (filename, fileobj, content_type) tuples
                for file_tuple in file_data:
                    if isinstance(file_tuple, tuple) and len(file_tuple) >= 2:
                        file_tuple[1].close()  # Close the file object
            elif isinstance(file_data, tuple):
                # Single file: (filename, fileobj, content_type) tuple
                if len(file_data) == 3:
                    file_data[1].close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/upload_manual_to_railway.py <manual_id> <railway_url>")
        print("\nExample:")
        print("  python scripts/upload_manual_to_railway.py 6262059 https://legoassembly-production.up.railway.app")
        sys.exit(1)
    
    manual_id = sys.argv[1]
    railway_url = sys.argv[2]
    
    upload_manual(manual_id, railway_url)
