"""
CLI script to ingest all available manuals into the vector store.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from app.ingestion.ingest_service import get_ingestion_service


def main():
    """Ingest all available manuals."""
    logger.info("Ingesting all available manuals...")
    
    try:
        service = get_ingestion_service()
        result = service.ingest_all_manuals()
        
        logger.success(f"✓ Ingestion complete!")
        logger.info(f"  Processed: {result['manuals_processed']} manuals")
        logger.info(f"  Successful: {result['successful']}")
        logger.info(f"  Failed: {result['failed']}")
        logger.info(f"  Total steps: {result['total_steps']}")
        logger.info(f"  Total chunks: {result['total_chunks']}")
        
        # Show details
        print("\nDetails:")
        for detail in result.get('details', []):
            status_icon = "✓" if detail['status'] == 'success' else "✗"
            print(f"  {status_icon} {detail['manual_id']}: {detail['steps_ingested']} steps")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


