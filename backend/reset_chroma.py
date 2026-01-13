#!/usr/bin/env python3
"""
Reset ChromaDB Collection

This script deletes and recreates the ChromaDB collection to fix dimension mismatches
when switching embedding models. All existing data will be lost.
"""

import sys
from pathlib import Path

# Add backend app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.vector_store.chroma_manager import get_chroma_manager
from app.config import get_settings
from loguru import logger


def main():
    """Reset the ChromaDB collection."""
    settings = get_settings()

    logger.info("=" * 60)
    logger.warning("WARNING: This will DELETE all data in ChromaDB!")
    logger.info(f"Collection: {settings.collection_name}")
    logger.info(f"Embedding model: {settings.rag_embedding_model}")
    logger.info("=" * 60)

    response = input("\nAre you sure you want to proceed? (yes/no): ")

    if response.lower() != "yes":
        logger.info("Aborted.")
        return

    try:
        logger.info("Resetting ChromaDB collection...")
        chroma_manager = get_chroma_manager()
        chroma_manager.reset()
        logger.info("✓ ChromaDB collection reset successfully!")
        logger.info(f"  New collection is ready for embeddings with model: {settings.rag_embedding_model}")
        logger.info(f"  You can now re-ingest your data.")

    except Exception as e:
        logger.error(f"✗ Failed to reset ChromaDB: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
