"""
Main workflow orchestrator for LEGO Assembly System.
Coordinates manual processing, VLM extraction, plan generation, and ingestion.
Supports checkpointing to resume interrupted processing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from src.vision_processing import ManualInputHandler, VLMStepExtractor, DependencyGraph
from src.plan_generation import PlanStructureGenerator, GraphBuilder
from src.utils import get_config, URLHandler

# Import backend services for Phase 2
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from app.ingestion.ingest_service import IngestionService

class ProcessingCheckpoint:
    """Manages processing checkpoints to enable resume functionality."""
    
    def __init__(self, output_dir: Path, assembly_id: str):
        self.checkpoint_path = output_dir / f".{assembly_id}_checkpoint.json"
        self.assembly_id = assembly_id
        self.output_dir = output_dir
    
    def load(self) -> Dict[str, Any]:
        """Load checkpoint data if it exists."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {
            "assembly_id": self.assembly_id,
            "completed_steps": [],
            "last_step": None,
            "status": "not_started"
        }
    
    def save(self, step: str, data: Optional[Dict[str, Any]] = None):
        """Save checkpoint after completing a step."""
        checkpoint = self.load()
        checkpoint["last_step"] = step
        if step not in checkpoint["completed_steps"]:
            checkpoint["completed_steps"].append(step)
        checkpoint["status"] = "in_progress"
        if data:
            checkpoint.update(data)
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.debug(f"Checkpoint saved: {step}")
    
    def mark_complete(self):
        """Mark processing as complete."""
        checkpoint = self.load()
        checkpoint["status"] = "complete"
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def is_step_complete(self, step: str) -> bool:
        """Check if a step was already completed."""
        checkpoint = self.load()
        return step in checkpoint.get("completed_steps", [])
    
    def clear(self):
        """Remove checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

def main(
    input_path: str,
    output_dir: str,
    assembly_id: Optional[str] = None,
    use_fallback: bool = False,
    display_output: bool = True,
    skip_ingestion: bool = False,
    use_multimodal: bool = True,
    resume: bool = True,
    batch_size: int = 10
):
    """
    Main workflow: Process LEGO manual, generate plan, and ingest into vector store.
    Supports checkpointing to resume interrupted processing.
    
    Args:
        input_path: Path to PDF manual, image directory, or URL
        output_dir: Directory for output files
        assembly_id: Unique assembly identifier
        use_fallback: Use fallback VLMs if primary fails
        display_output: Display JSON and text plans in console
        skip_ingestion: Skip Phase 2 (vector store ingestion)
        use_multimodal: Use multimodal embeddings in ingestion
        resume: Resume from checkpoint if available
        batch_size: Number of steps to process per API call (default: 10, reduces rate limiting)
    """
    logger.info("=" * 80)
    logger.info("LEGO Assembly System - Complete Workflow")
    logger.info("=" * 80)
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input is URL or local path
    is_url = input_path.startswith('http://') or input_path.startswith('https://')
    
    if is_url:
        logger.info(f"Input URL: {input_path}")
        # Download PDF from URL
        url_handler = URLHandler()
        try:
            input_path = url_handler.download_pdf(input_path)
            logger.info(f"Downloaded to: {input_path}")
        except Exception as e:
            logger.error(f"Failed to download PDF from URL: {e}")
            raise
    else:
        input_path = Path(input_path)
    
    if not assembly_id:
        assembly_id = input_path.stem
    
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Assembly ID: {assembly_id}")
    logger.info("")
    
    # Initialize checkpoint manager
    checkpoint = ProcessingCheckpoint(output_dir, assembly_id)
    
    if resume:
        checkpoint_data = checkpoint.load()
        if checkpoint_data["status"] == "complete":
            logger.info("✓ Processing already complete for this manual")
            logger.info("  Use --no-resume to reprocess from scratch")
            return
        elif checkpoint_data["completed_steps"]:
            logger.info(f"↻ Resuming from checkpoint: last completed step = {checkpoint_data['last_step']}")
            logger.info(f"  Completed: {', '.join(checkpoint_data['completed_steps'])}")
    else:
        # Clear existing checkpoint if not resuming
        checkpoint.clear()
    
    # Step 1: Manual Input Processing
    if not checkpoint.is_step_complete("page_extraction"):
        logger.info("Step 1/6: Processing manual input...")
        manual_handler = ManualInputHandler(output_dir=output_dir / "temp_pages")
        
        page_paths = manual_handler.process_manual(input_path)
        logger.info(f"Extracted {len(page_paths)} pages")
        
        # Detect step boundaries
        step_groups = manual_handler.detect_step_boundaries(page_paths)
        logger.info(f"Detected {len(step_groups)} steps")
        logger.info("")
        
        checkpoint.save("page_extraction", {"page_count": len(page_paths), "step_count": len(step_groups)})
    else:
        logger.info("Step 1/6: ✓ Page extraction already complete (skipping)")
        # Load step groups from checkpoint or re-detect
        manual_handler = ManualInputHandler(output_dir=output_dir / "temp_pages")
        page_paths = list((output_dir / "temp_pages").glob("page_*.png"))
        step_groups = manual_handler.detect_step_boundaries(page_paths)
    
    # Step 2: VLM-based Step Extraction
    if not checkpoint.is_step_complete("step_extraction"):
        logger.info("Step 2/6: Extracting step information using VLM...")
        vlm_extractor = VLMStepExtractor()
        
        # Use assembly_id as cache context to prevent cache collisions between different manuals
        extracted_steps = vlm_extractor.batch_extract(
            step_groups, 
            use_primary=not use_fallback,
            cache_context=assembly_id,
            batch_size=batch_size
        )
        logger.info(f"Extracted information from {len(extracted_steps)} steps")
        
        # Save extracted data
        extracted_path = output_dir / f"{assembly_id}_extracted.json"
        with open(extracted_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_steps, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved extracted data to {extracted_path}")
        logger.info("")
        
        checkpoint.save("step_extraction")
    else:
        logger.info("Step 2/6: ✓ Step extraction already complete (skipping)")
        # Load extracted data
        extracted_path = output_dir / f"{assembly_id}_extracted.json"
        with open(extracted_path, 'r', encoding='utf-8') as f:
            extracted_steps = json.load(f)
    
    # Step 3: Dependency Graph Construction
    if not checkpoint.is_step_complete("dependency_graph"):
        logger.info("Step 3/6: Building dependency graph...")
        dep_graph = DependencyGraph()
        dep_graph.infer_dependencies(extracted_steps)
        
        # Validate graph
        is_valid, errors = dep_graph.validate()
        if not is_valid:
            logger.warning(f"Dependency graph validation issues: {errors}")
        else:
            logger.info("Dependency graph is valid")
        
        # Save dependency graph
        graph_path = output_dir / f"{assembly_id}_dependencies.json"
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(dep_graph.to_dict(), f, indent=2)
        logger.info(f"Saved dependency graph to {graph_path}")
        logger.info("")
        
        checkpoint.save("dependency_graph")
    else:
        logger.info("Step 3/6: ✓ Dependency graph already complete (skipping)")
        # Load dependency graph
        graph_path = output_dir / f"{assembly_id}_dependencies.json"
        with open(graph_path, 'r', encoding='utf-8') as f:
            dep_graph_dict = json.load(f)
        dep_graph = DependencyGraph()
        dep_graph.nodes = dep_graph_dict.get('nodes', {})
        dep_graph.edges = dep_graph_dict.get('edges', [])
    
    # Step 4: Hierarchical Assembly Graph Construction
    if not checkpoint.is_step_complete("hierarchical_graph"):
        logger.info("Step 4/6: Building hierarchical assembly graph...")
        graph_builder = GraphBuilder()
        
        hierarchical_graph = graph_builder.build_graph(
            extracted_steps=extracted_steps,
            assembly_id=assembly_id,
            image_dir=output_dir / "temp_pages"
        )
        
        # Save hierarchical graph
        hierarchical_graph_path = output_dir / f"{assembly_id}_graph.json"
        graph_builder.save_graph(hierarchical_graph, hierarchical_graph_path)
        
        logger.info(f"Hierarchical graph: {hierarchical_graph['metadata']['total_parts']} parts, "
                    f"{hierarchical_graph['metadata']['total_subassemblies']} subassemblies")
        logger.info("")
        
        checkpoint.save("hierarchical_graph")
    else:
        logger.info("Step 4/6: ✓ Hierarchical graph already complete (skipping)")
    
    # Step 5: 3D Plan Generation
    if not checkpoint.is_step_complete("plan_generation"):
        logger.info("Step 5/6: Generating 3D assembly plan...")
        plan_generator = PlanStructureGenerator()
        
        metadata = {
            "source": str(input_path),
            "manual_pages": len(page_paths),
            "step_count": len(extracted_steps)
        }
        
        assembly_plan = plan_generator.generate_plan(
            extracted_steps=extracted_steps,
            dependency_graph=dep_graph,
            assembly_id=assembly_id,
            metadata=metadata
        )
        
        # Export plan
        json_plan_path = output_dir / f"{assembly_id}_plan.json"
        plan_generator.export_plan(assembly_plan, json_plan_path, format="json")
        
        text_plan_path = output_dir / f"{assembly_id}_plan.txt"
        plan_generator.export_plan(assembly_plan, text_plan_path, format="text")
        
        logger.info(f"Validation: {assembly_plan['validation']['summary']}")
        logger.info("")
        
        checkpoint.save("plan_generation")
        
        # Display plans in console if requested
        if display_output:
            logger.info("")
            logger.info("=" * 80)
            logger.info("STRUCTURED PLAN (JSON)")
            logger.info("=" * 80)
            print("\n" + json.dumps(assembly_plan, indent=2, ensure_ascii=False))
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("NATURAL LANGUAGE PLAN (TEXT)")
            logger.info("=" * 80)
            with open(text_plan_path, 'r', encoding='utf-8') as f:
                print("\n" + f.read())
    else:
        logger.info("Step 5/6: ✓ Plan generation already complete (skipping)")
        json_plan_path = output_dir / f"{assembly_id}_plan.json"
        text_plan_path = output_dir / f"{assembly_id}_plan.txt"
        with open(json_plan_path, 'r', encoding='utf-8') as f:
            assembly_plan = json.load(f)
    
    # Step 6: Vector Store Ingestion (Phase 2)
    if skip_ingestion:
        logger.info("Step 6/6: ✗ Vector store ingestion skipped (--skip-ingestion)")
    elif not checkpoint.is_step_complete("ingestion"):
        logger.info("Step 6/6: Ingesting into vector store...")
        logger.info("  (This creates searchable embeddings with multimodal fusion)")
        
        try:
            ingestion_service = IngestionService(use_multimodal=use_multimodal)
            result = ingestion_service.ingest_manual(assembly_id)
            
            if result['status'] == 'success':
                logger.info(f"✓ Successfully ingested manual {assembly_id}")
                logger.info(f"  Steps: {result['steps_ingested']}")
                logger.info(f"  Parts: {result['parts_ingested']}")
                logger.info(f"  Chunks: {result['chunks_created']}")
                checkpoint.save("ingestion")
            else:
                logger.error(f"✗ Ingestion failed: {result['message']}")
                logger.warning("  Phase 1 outputs are still available, but manual won't be searchable")
        except Exception as e:
            logger.error(f"✗ Ingestion error: {e}")
            logger.warning("  Phase 1 outputs are still available, but manual won't be searchable")
    else:
        logger.info("Step 6/6: ✓ Vector store ingestion already complete (skipping)")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ Processing complete!")
    logger.info(f"  Manual ID: {assembly_id}")
    logger.info(f"  JSON Plan: {json_plan_path}")
    logger.info(f"  Text Plan: {text_plan_path}")
    if not skip_ingestion:
        logger.info(f"  Vector Store: Ready for queries")
    logger.info("=" * 80)
    
    # Mark checkpoint as complete
    checkpoint.mark_complete()
    
    return assembly_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEGO Assembly System - Complete Workflow (Extraction + Ingestion)"
    )
    
    parser.add_argument(
        "input",
        type=str,
        nargs='?',  # Make input optional
        default=None,
        help="Path to LEGO instruction manual (PDF), image directory, or URL. If not provided, will prompt for URL."
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory for generated plans (default: ./output)"
    )
    
    parser.add_argument(
        "-id", "--assembly-id",
        type=str,
        default=None,
        help="Assembly identifier (default: input filename)"
    )
    
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use fallback VLMs if primary fails"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display plans in console (only save to files)"
    )
    
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip vector store ingestion (Phase 2) - only generate files"
    )
    
    parser.add_argument(
        "--no-multimodal",
        action="store_true",
        help="Use text-only embeddings instead of multimodal (faster but less accurate)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch instead of resuming from checkpoint"
    )
    
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable caching of VLM responses (disabled by default)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of steps to process per API call (default: 10, reduces rate limiting)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # Handle cache settings (disabled by default)
    if args.cache:
        import os
        os.environ["CACHE_ENABLED"] = "true"
        logger.info("Cache enabled - will reuse previous VLM responses")
    
    # Get input (from args or prompt)
    input_path = args.input
    
    if not input_path:
        # Interactive mode: prompt for URL
        print("\n" + "=" * 80)
        print("LEGO Assembly System - Interactive Mode")
        print("=" * 80)
        print("\nPlease enter the URL to the LEGO instruction manual PDF:")
        print("Example: https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6521147.pdf")
        print()
        
        try:
            input_path = input("URL: ").strip()
            
            if not input_path:
                print("Error: No URL provided.")
                exit(1)
            
            # Validate it looks like a URL
            if not (input_path.startswith('http://') or input_path.startswith('https://')):
                print(f"Warning: '{input_path}' doesn't look like a URL.")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Cancelled.")
                    exit(0)
        
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            exit(0)
        except EOFError:
            print("\n\nError: No input provided.")
            exit(1)
    
    # Run main workflow
    try:
        main(
            input_path=input_path,
            output_dir=args.output,
            assembly_id=args.assembly_id,
            use_fallback=args.use_fallback,
            display_output=not args.no_display,
            skip_ingestion=args.skip_ingestion,
            use_multimodal=not args.no_multimodal,
            resume=not args.no_resume,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        exit(1)

