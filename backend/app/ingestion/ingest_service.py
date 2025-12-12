"""
Ingestion Service for LEGO Manuals.
Orchestrates data processing and vector store population.
Supports multimodal embedding generation.
"""

from pathlib import Path
from typing import Dict, Any
from loguru import logger

from .data_processor import ManualDataProcessor
from ..vector_store.chroma_manager import get_chroma_manager
from ..llm.qwen_client import QwenClient
from ..config import get_settings


class IngestionService:
    """Service for ingesting LEGO manuals into vector store."""
    
    def __init__(self, use_multimodal: bool = True):
        """
        Initialize ingestion service.
        
        Args:
            use_multimodal: Whether to use multimodal processing (default: True)
        """
        self.settings = get_settings()
        self.processor = ManualDataProcessor(
            self.settings.output_dir,
            use_multimodal=use_multimodal
        )
        self.chroma = get_chroma_manager()
        
        # Initialize Qwen client for embeddings (needed for multimodal)
        if use_multimodal:
            api_key = self.settings.get_llm_api_key()
            self.qwen_client = QwenClient(api_key)
            logger.info("IngestionService initialized with multimodal support")
        else:
            self.qwen_client = None
            logger.info("IngestionService initialized (text-only mode)")
    
    def ingest_manual(self, manual_id: str) -> Dict[str, Any]:
        """
        Ingest a complete manual into the vector store.
        
        Args:
            manual_id: Manual identifier (e.g., "6454922")
        
        Returns:
            Dict with ingestion statistics
        """
        try:
            logger.info(f"Starting ingestion for manual {manual_id}")
            
            # Load manual data
            extracted_data, plan_data, dependencies_data, plan_text = \
                self.processor.load_manual_data(manual_id)
            
            # Create step chunks (with multimodal embeddings if enabled)
            step_chunks = self.processor.create_step_chunks(
                manual_id,
                extracted_data,
                plan_data,
                dependencies_data,
                qwen_client=self.qwen_client
            )
            
            # Create metadata chunk
            metadata_chunk = self.processor.create_manual_metadata_chunk(
                manual_id,
                plan_data
            )
            
            # Combine all chunks
            all_chunks = [metadata_chunk] + step_chunks
            
            # Prepare for ChromaDB
            documents = [chunk['content'] for chunk in all_chunks]
            metadatas = [chunk['metadata'] for chunk in all_chunks]
            ids = [chunk['id'] for chunk in all_chunks]
            
            # Check if we have pre-computed embeddings
            embeddings = []
            for chunk in all_chunks:
                if 'embedding' in chunk:
                    embeddings.append(chunk['embedding'])
            
            # Add to vector store
            if embeddings and len(embeddings) == len(all_chunks):
                # Use pre-computed multimodal embeddings
                logger.info(f"Using {len(embeddings)} pre-computed multimodal embeddings")
                self.chroma.add_documents_with_embeddings(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                # Fall back to automatic text-based embedding
                logger.info("Using automatic text-based embeddings")
                self.chroma.add_documents(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            # Calculate statistics
            total_parts = sum(
                len(chunk.get('step_data', {}).get('parts', []))
                for chunk in step_chunks
            )
            
            result = {
                "manual_id": manual_id,
                "status": "success",
                "message": f"Successfully ingested manual {manual_id}",
                "steps_ingested": len(step_chunks),
                "parts_ingested": total_parts,
                "chunks_created": len(all_chunks)
            }
            
            logger.info(f"Ingestion complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed for manual {manual_id}: {e}")
            return {
                "manual_id": manual_id,
                "status": "error",
                "message": str(e),
                "steps_ingested": 0,
                "parts_ingested": 0,
                "chunks_created": 0
            }
    
    def ingest_all_manuals(self) -> Dict[str, Any]:
        """
        Ingest all available manuals in the output directory.
        
        Returns:
            Dict with overall statistics
        """
        output_dir = Path(self.settings.output_dir)
        
        # Find all extracted.json files
        extracted_files = list(output_dir.glob("*_extracted.json"))
        
        results = []
        for extracted_file in extracted_files:
            manual_id = extracted_file.stem.replace("_extracted", "")
            result = self.ingest_manual(manual_id)
            results.append(result)
        
        # Aggregate statistics
        total_success = sum(1 for r in results if r['status'] == 'success')
        total_steps = sum(r['steps_ingested'] for r in results)
        total_chunks = sum(r['chunks_created'] for r in results)
        
        return {
            "status": "complete",
            "manuals_processed": len(results),
            "successful": total_success,
            "failed": len(results) - total_success,
            "total_steps": total_steps,
            "total_chunks": total_chunks,
            "details": results
        }
    
    def remove_manual(self, manual_id: str) -> Dict[str, Any]:
        """
        Remove a manual from the vector store.
        
        Args:
            manual_id: Manual identifier
        
        Returns:
            Dict with removal status
        """
        try:
            self.chroma.delete_manual(manual_id)
            return {
                "manual_id": manual_id,
                "status": "success",
                "message": f"Manual {manual_id} removed successfully"
            }
        except Exception as e:
            logger.error(f"Failed to remove manual {manual_id}: {e}")
            return {
                "manual_id": manual_id,
                "status": "error",
                "message": str(e)
            }


def get_ingestion_service() -> IngestionService:
    """Get IngestionService singleton."""
    return IngestionService()


