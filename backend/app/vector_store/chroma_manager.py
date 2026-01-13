"""
ChromaDB Vector Store Management.
Handles embedding storage, retrieval, and similarity search.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path

from ..config import get_settings
from ..llm.litellm_client import UnifiedLLMClient


class LiteLLMEmbeddingFunction(ChromaEmbeddingFunction):
    """Unified embedding function using LiteLLM."""

    def __init__(self, model: str, api_keys: Dict[str, str]):
        """
        Initialize with LiteLLM model and API keys.

        Args:
            model: LiteLLM model identifier (e.g., "gemini/text-embedding-004", "text-embedding-3-large")
            api_keys: Dictionary of API keys
        """
        self.client = UnifiedLLMClient(model=model, api_keys=api_keys)
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts (batch)."""
        return self.client.get_embeddings(input)

    def embed_query(self, input) -> List[float]:
        """Generate embedding for a single query text."""
        # ChromaDB passes input as a list containing the query text
        if isinstance(input, list):
            text = input[0] if input else ""
        elif isinstance(input, dict) and 'input' in input:
            text = input['input']
        else:
            text = str(input)

        logger.debug(f"embed_query ({self.model}): text='{text[:50]}'")

        # get_embeddings returns List[List[float]], we want the first embedding vector
        embeddings = self.client.get_embeddings([text])

        if embeddings and len(embeddings) > 0:
            result = embeddings[0]

            # Ensure it's actually a list
            if not isinstance(result, list):
                logger.error(f"embed_query returning wrong type: {type(result)}")
                raise TypeError(f"Expected list, got {type(result)}")

            return result
        else:
            logger.error("embed_query: No embeddings returned!")
            raise ValueError("No embeddings returned from get_embeddings")

    def name(self) -> str:
        """Return the name of this embedding function."""
        return self.model


class ChromaManager:
    """Manages ChromaDB vector store for LEGO manual embeddings."""
    
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        self.settings = get_settings()
        
        # Create persistent directory
        persist_dir = Path(self.settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Set up embedding function using LiteLLM
        api_keys = self.settings.get_api_keys_dict()
        self.embedding_function = LiteLLMEmbeddingFunction(
            model=self.settings.rag_embedding_model,
            api_keys=api_keys
        )
        logger.info(f"Using LiteLLM embeddings with model: {self.settings.rag_embedding_model}")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.settings.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "LEGO Assembly Manual Steps"}
        )
        
        logger.info(f"ChromaDB initialized: {self.collection.count()} documents")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """
        Add documents to the vector store.
        Embeddings will be generated automatically.
        
        Args:
            documents: List of text content to embed
            metadatas: List of metadata dicts for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def add_documents_with_embeddings(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: List[List[float]]
    ) -> None:
        """
        Add documents to the vector store with pre-computed embeddings.
        Useful for multimodal embeddings (text + image).
        
        Args:
            documents: List of text content
            metadatas: List of metadata dicts for each document
            ids: List of unique IDs for each document
            embeddings: List of pre-computed embedding vectors
        """
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"Added {len(documents)} documents with pre-computed embeddings")
        except Exception as e:
            logger.error(f"Error adding documents with embeddings: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store with semantic search.
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Metadata filters (e.g., {"manual_id": "6454922"})
        
        Returns:
            Dict with 'documents', 'metadatas', 'distances', 'ids'
        """
        try:
            # Generate embedding manually to avoid ChromaDB embedding function issues
            query_embedding = self.embedding_function.embed_query([query_text])
            logger.debug(f"Generated query embedding, type: {type(query_embedding)}, len: {len(query_embedding) if isinstance(query_embedding, list) else 'N/A'}")
            
            # Query using pre-computed embedding
            results = self.collection.query(
                query_embeddings=[query_embedding],  # Pass as list of embeddings
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise
    
    def get_by_manual(self, manual_id: str) -> List[Dict[str, Any]]:
        """
        Get all documents for a specific manual.
        
        Args:
            manual_id: Manual identifier
        
        Returns:
            List of document metadata
        """
        try:
            results = self.collection.get(
                where={"manual_id": manual_id}
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving manual documents: {e}")
            raise
    
    def delete_manual(self, manual_id: str) -> None:
        """
        Delete all documents for a specific manual.
        
        Args:
            manual_id: Manual identifier
        """
        try:
            self.collection.delete(
                where={"manual_id": manual_id}
            )
            logger.info(f"Deleted manual {manual_id} from vector store")
        except Exception as e:
            logger.error(f"Error deleting manual: {e}")
            raise
    
    def get_all_manuals(self) -> List[str]:
        """
        Get list of all ingested manual IDs.
        
        Returns:
            List of unique manual IDs
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            # Extract unique manual IDs
            manual_ids = set()
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if 'manual_id' in metadata:
                        manual_ids.add(metadata['manual_id'])
            
            return sorted(list(manual_ids))
        except Exception as e:
            logger.error(f"Error retrieving manual list: {e}")
            raise
    
    def count_documents(self, manual_id: Optional[str] = None) -> int:
        """
        Count documents in vector store.
        
        Args:
            manual_id: Optional manual ID to count specific manual's documents
        
        Returns:
            Document count
        """
        try:
            if manual_id:
                results = self.collection.get(where={"manual_id": manual_id})
                return len(results['ids']) if results else 0
            else:
                return self.collection.count()
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    def reset(self) -> None:
        """Reset the entire collection (use with caution!)."""
        try:
            self.client.delete_collection(self.settings.collection_name)
            self.collection = self.client.create_collection(
                name=self.settings.collection_name,
                embedding_function=self.embedding_function
            )
            logger.warning("Vector store collection reset")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise


# Global instance
_chroma_manager: Optional[ChromaManager] = None


def get_chroma_manager() -> ChromaManager:
    """Get or create ChromaManager singleton."""
    global _chroma_manager
    if _chroma_manager is None:
        _chroma_manager = ChromaManager()
    return _chroma_manager

