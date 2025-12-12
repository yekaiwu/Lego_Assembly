"""
Simple CLI script to inspect the vector store contents.
No external dependencies beyond what's already installed.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from app.vector_store.chroma_manager import get_chroma_manager


def main():
    """Inspect vector store."""
    try:
        print("\n" + "="*60)
        print("🔍 ChromaDB Vector Store Inspector")
        print("="*60 + "\n")
        
        # Initialize ChromaManager
        chroma = get_chroma_manager()
        collection = chroma.collection
        
        # Get collection info
        total_docs = collection.count()
        manual_ids = chroma.get_all_manuals()
        
        # Display summary
        print("📊 SUMMARY")
        print("-" * 60)
        print(f"Collection Name: {chroma.settings.collection_name}")
        print(f"Location: {chroma.settings.chroma_persist_dir}")
        print(f"Total Documents: {total_docs}")
        print(f"Total Manuals: {len(manual_ids)}")
        print()
        
        # Display per-manual stats
        print("📦 MANUALS IN DATABASE")
        print("-" * 60)
        print(f"{'Manual ID':<20} {'Documents':<15} {'Status'}")
        print("-" * 60)
        
        for manual_id in sorted(manual_ids):
            count = chroma.count_documents(manual_id)
            status = "✅ Active" if count > 0 else "⚠️  Empty"
            print(f"{manual_id:<20} {count:<15} {status}")
        
        print()
        
        # Display sample documents
        print("📄 SAMPLE DOCUMENTS (first 5)")
        print("-" * 60)
        
        all_data = collection.get(limit=5)
        
        for i in range(min(5, len(all_data['ids']))):
            doc_id = all_data['ids'][i]
            content = all_data['documents'][i]
            metadata = all_data['metadatas'][i]
            
            # Truncate content
            preview = content[:150] + "..." if len(content) > 150 else content
            
            print(f"\nDocument {i+1}:")
            print(f"  ID: {doc_id}")
            print(f"  Manual: {metadata.get('manual_id', 'N/A')}")
            print(f"  Step: {metadata.get('step_number', 'N/A')}")
            print(f"  Type: {metadata.get('chunk_type', 'N/A')}")
            print(f"  Has Diagram: {metadata.get('has_diagram', False)}")
            print(f"  Content Preview: {preview}")
        
        print("\n" + "-" * 60)
        print("\n💡 TIPS:")
        print("  • View in browser: http://localhost:8000/api/vector-store/inspect")
        print("  • Query specific manual: GET /api/manual/{manual_id}/steps")
        print("  • Test retrieval: Use the chat interface at http://localhost:3000")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Error inspecting vector store: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

