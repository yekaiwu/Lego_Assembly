#!/usr/bin/env python3
"""
Hugging Face Login Script
Authenticates with Hugging Face Hub using token from environment variables
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login

def huggingface_login():
    """Login to Hugging Face Hub using token from .env file"""
    # Load environment variables from .env file
    load_dotenv()

    # Get token from environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        print("✗ Error: HUGGINGFACE_TOKEN not found in .env file")
        print("  Please add: HUGGINGFACE_TOKEN=your_token_here to .env")
        return False

    try:
        login(token=hf_token, add_to_git_credential=True)
        print("✓ Successfully logged in to Hugging Face!")
        print("✓ Token has been saved to git credentials")
    except Exception as e:
        print(f"✗ Error logging in to Hugging Face: {e}")
        return False
    return True

if __name__ == "__main__":
    huggingface_login()
