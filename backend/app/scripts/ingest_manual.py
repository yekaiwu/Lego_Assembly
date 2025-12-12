"""
CLI script to ingest a single manual into the vector store.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from app.ingestion.ingest_service import get_ingestion_service


def main():
    """Ingest a single manual."""
    if len(sys.argv) < 2:
        print("Usage: python -m app.scripts.ingest_manual <manual_id> [--no-multimodal]")
        print("Example: python -m app.scripts.ingest_manual 6454922")
        print("         python -m app.scripts.ingest_manual 6454922 --no-multimodal")
        sys.exit(1)
    
    manual_id = sys.argv[1]
    use_multimodal = True
    
    # Check for --no-multimodal flag
    if len(sys.argv) > 2 and sys.argv[2] == '--no-multimodal':
        use_multimodal = False
        logger.warning("Multimodal embeddings disabled - using text-only mode")
    
    logger.info(f"Ingesting manual {manual_id}...")
    
    try:
        from app.ingestion.ingest_service import IngestionService
        service = IngestionService(use_multimodal=use_multimodal)
        result = service.ingest_manual(manual_id)
        
        if result['status'] == 'success':
            logger.success(f"✓ Successfully ingested manual {manual_id}")
            logger.info(f"  Steps: {result['steps_ingested']}")
            logger.info(f"  Parts: {result['parts_ingested']}")
            logger.info(f"  Chunks: {result['chunks_created']}")
        else:
            logger.error(f"✗ Ingestion failed: {result['message']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


