"""
CLI script to inspect the vector store contents.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from app.vector_store.chroma_manager import get_chroma_manager
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def main():
    """Inspect vector store."""
    try:
        console.print("\n[bold cyan]🔍 ChromaDB Vector Store Inspector[/bold cyan]\n")
        
        # Initialize ChromaManager
        chroma = get_chroma_manager()
        collection = chroma.collection
        
        # Get collection info
        total_docs = collection.count()
        manual_ids = chroma.get_all_manuals()
        
        # Display summary
        console.print(Panel(f"""
[bold]Collection:[/bold] {chroma.settings.collection_name}
[bold]Location:[/bold] {chroma.settings.chroma_persist_dir}
[bold]Total Documents:[/bold] {total_docs}
[bold]Total Manuals:[/bold] {len(manual_ids)}
        """.strip(), title="📊 Summary", border_style="cyan"))
        
        # Display per-manual stats
        console.print("\n[bold]📦 Manuals in Database:[/bold]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Manual ID", style="cyan", width=15)
        table.add_column("Documents", justify="right", style="green")
        table.add_column("Status", justify="center")
        
        for manual_id in sorted(manual_ids):
            count = chroma.count_documents(manual_id)
            status = "✅" if count > 0 else "⚠️"
            table.add_row(manual_id, str(count), status)
        
        console.print(table)
        
        # Display sample documents
        console.print("\n[bold]📄 Sample Documents (first 5):[/bold]\n")
        
        all_data = collection.get(limit=5)
        
        for i in range(min(5, len(all_data['ids']))):
            doc_id = all_data['ids'][i]
            content = all_data['documents'][i]
            metadata = all_data['metadatas'][i]
            
            # Truncate content
            preview = content[:150] + "..." if len(content) > 150 else content
            
            console.print(Panel(f"""
[bold]ID:[/bold] {doc_id}
[bold]Manual:[/bold] {metadata.get('manual_id', 'N/A')}
[bold]Step:[/bold] {metadata.get('step_number', 'N/A')}
[bold]Type:[/bold] {metadata.get('chunk_type', 'N/A')}
[bold]Has Diagram:[/bold] {metadata.get('has_diagram', False)}

[dim]{preview}[/dim]
            """.strip(), border_style="blue"))
        
        # Prompt for detailed view
        console.print("\n[bold yellow]💡 Tips:[/bold yellow]")
        console.print("  • View full document: Use the API endpoint /api/vector-store/inspect")
        console.print("  • Query specific manual: Check /api/manual/{manual_id}/steps")
        console.print("  • Test retrieval: Use the chat interface or /api/query/text")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        logger.error(f"Error inspecting vector store: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

