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
from src.vision_processing.user_metadata_collector import UserMetadataCollector
from src.vision_processing.metadata_models import UserProvidedMetadata
from src.vision_processing.document_analyzer import (
    convert_user_metadata_to_document_metadata,
    extract_relevant_pages
)
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
    enable_spatial_relationships: bool = True,
    enable_spatial_temporal: bool = True
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
        enable_spatial_relationships: Enable spatial relationship extraction and processing
        enable_spatial_temporal: Enable spatial-temporal pattern analysis
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
    logger.info("Configuration:")
    logger.info(f"  Spatial Relationships: {'Enabled' if enable_spatial_relationships else 'DISABLED'}")
    logger.info(f"  Spatial-Temporal Patterns: {'Enabled' if enable_spatial_temporal else 'DISABLED'}")
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
        logger.info("Step 1/7: Processing manual input...")
        manual_handler = ManualInputHandler(output_dir=output_dir / "temp_pages")

        page_paths = manual_handler.process_manual(input_path)
        logger.info(f"Extracted {len(page_paths)} pages")

        checkpoint.save("page_extraction", {"page_count": len(page_paths)})
    else:
        logger.info("Step 1/7: ✓ Page extraction already complete (skipping)")
        # Load page paths from checkpoint
        manual_handler = ManualInputHandler(output_dir=output_dir / "temp_pages")
        page_paths = sorted(list((output_dir / "temp_pages").glob("page_*.png")))

    # Step 2 (UPDATED): User-Provided Metadata Collection
    if not checkpoint.is_step_complete("metadata_collection"):
        logger.info("Step 2/7: Collecting manual metadata from user...")
        logger.info("")

        # Create metadata collector
        collector = UserMetadataCollector(total_pages=len(page_paths))

        # Collect metadata interactively
        user_metadata = collector.collect_metadata_interactive()

        # Filter to only instruction pages
        relevant_page_paths = extract_relevant_pages(page_paths, user_metadata)

        logger.info(f"Filtered: {len(page_paths)} → {len(relevant_page_paths)} instruction pages")

        # Detect step boundaries on filtered pages
        step_groups = manual_handler.detect_step_boundaries(relevant_page_paths)
        logger.info(f"Detected {len(step_groups)} assembly steps")
        logger.info("")

        checkpoint.save("metadata_collection", {
            "metadata": user_metadata.to_dict(),
            "relevant_page_count": len(relevant_page_paths),
            "step_count": len(step_groups)
        })
    else:
        logger.info("Step 2/7: ✓ Metadata collection already complete (skipping)")
        # Load from checkpoint
        checkpoint_data = checkpoint.load()
        user_metadata = UserProvidedMetadata.from_dict(checkpoint_data["metadata"])

        # Re-extract relevant pages
        relevant_page_paths = extract_relevant_pages(page_paths, user_metadata)
        step_groups = manual_handler.detect_step_boundaries(relevant_page_paths)

    # Convert to legacy DocumentMetadata for compatibility with Phase 1
    doc_metadata = convert_user_metadata_to_document_metadata(user_metadata)
    
    # Step 3 (ENHANCED): Phase 1 - Context-Aware Step Extraction
    if not checkpoint.is_step_complete("step_extraction"):
        logger.info("Step 3/7: Extracting step information using VLM (Phase 1 - Context-Aware)...")
        vlm_extractor = VLMStepExtractor(enable_spatial_relationships=enable_spatial_relationships)

        # NEW: Initialize context-aware memory
        vlm_extractor.initialize_memory(
            main_build=doc_metadata.main_build,
            window_size=2,  # Remember last 2 steps (reduced from 5 for faster processing)
            max_tokens=1_000_000  # Gemini 2.5 Flash has 1M context window
        )
        logger.info(f"Initialized context-aware extraction for: {doc_metadata.main_build}")

        # Check for partially completed extraction (incremental save file)
        temp_extracted_path = output_dir / f"{assembly_id}_extracted_temp.json"
        extracted_steps = []
        start_step = 1
        
        if temp_extracted_path.exists():
            try:
                with open(temp_extracted_path, 'r', encoding='utf-8') as f:
                    extracted_steps = json.load(f)
                start_step = len(extracted_steps) + 1
                logger.info(f"Resuming from step {start_step} (found {len(extracted_steps)} cached steps)")
                
                # Restore context memory with previous steps
                for step_data in extracted_steps:
                    vlm_extractor.context_memory.add_step(step_data)
            except Exception as e:
                logger.warning(f"Failed to load temp extraction file: {e}. Starting from scratch.")
                extracted_steps = []
                start_step = 1

        # Process each page (pages may contain multiple steps)
        total_pages = len(step_groups)
        for i in range(start_step - 1, total_pages):
            page_num = i + 1
            image_paths = step_groups[i]
            logger.info(f"Processing page {page_num}/{total_pages}")

            try:
                # extract_step returns array of steps (1 or more per page)
                results = vlm_extractor.extract_step(
                    image_paths,
                    step_number=None,  # VLM detects step numbers from image
                    use_primary=not use_fallback,
                    cache_context=assembly_id
                )

                # Add each step to extracted_steps with page tracking
                for result in results:
                    step_num = result.get("step_number", "unknown")
                    logger.info(f"  └─ Extracted step {step_num}")
                    # Track which page this step came from
                    result["_source_page_idx"] = i
                    result["_source_page_paths"] = image_paths
                    extracted_steps.append(result)
                
                # INCREMENTAL SAVE: Save progress after each successful page
                with open(temp_extracted_path, 'w', encoding='utf-8') as f:
                    json.dump(extracted_steps, f, indent=2, ensure_ascii=False)
                logger.debug(f"Saved progress: {len(extracted_steps)} steps from {page_num}/{total_pages} pages")

            except Exception as e:
                logger.error(f"Error extracting page {page_num}: {e}")
                # Save progress even on error, then re-raise
                with open(temp_extracted_path, 'w', encoding='utf-8') as f:
                    json.dump(extracted_steps, f, indent=2, ensure_ascii=False)
                logger.info(f"Progress saved. {len(extracted_steps)} steps extracted from {page_num-1}/{total_pages} pages before error.")
                raise

        logger.info(f"Extracted information from {len(extracted_steps)} steps across {total_pages} pages (with context)")
        logger.info("")

        # NEW: Step Recovery - Detect and recover missing steps
        logger.info("Validating step sequence and recovering missing steps...")
        from src.vision_processing.step_recovery import StepRecoveryModule

        recovery_module = StepRecoveryModule(vlm_extractor)
        missing_steps = recovery_module.detect_missing_steps(extracted_steps)

        if missing_steps:
            logger.warning(f"Found {len(missing_steps)} missing steps: {missing_steps}")
            logger.info("Attempting recovery by re-extracting relevant pages...")

            # Recover missing steps
            extracted_steps, recovered_count = recovery_module.recover_missing_steps(
                extracted_steps,
                step_groups,
                assembly_id
            )

            if recovered_count > 0:
                logger.info(f"✓ Successfully recovered {recovered_count}/{len(missing_steps)} missing steps")
            else:
                logger.warning(f"✗ Could not recover any missing steps - proceeding with gaps")

            # Re-check for remaining gaps
            remaining_missing = recovery_module.detect_missing_steps(extracted_steps)
            if remaining_missing:
                logger.warning(f"Still missing {len(remaining_missing)} steps after recovery: {remaining_missing}")
        else:
            logger.info("✓ No missing steps detected - all steps sequential")

        logger.info("")

        # Save final extracted data
        extracted_path = output_dir / f"{assembly_id}_extracted.json"
        with open(extracted_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_steps, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved extracted data to {extracted_path}")
        
        # Clean up temp file after successful completion
        if temp_extracted_path.exists():
            temp_extracted_path.unlink()
            logger.debug("Removed temporary extraction file")
        logger.info("")

        checkpoint.save("step_extraction")
    else:
        logger.info("Step 3/7: ✓ Step extraction already complete (skipping)")
        # Load extracted data
        extracted_path = output_dir / f"{assembly_id}_extracted.json"
        with open(extracted_path, 'r', encoding='utf-8') as f:
            extracted_steps = json.load(f)
    
    # Step 4: Dependency Graph Construction
    if not checkpoint.is_step_complete("dependency_graph"):
        logger.info("Step 4/7: Building dependency graph...")
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
        logger.info("Step 4/7: ✓ Dependency graph already complete (skipping)")
        # Load dependency graph
        graph_path = output_dir / f"{assembly_id}_dependencies.json"
        with open(graph_path, 'r', encoding='utf-8') as f:
            dep_graph_dict = json.load(f)
        dep_graph = DependencyGraph()
        dep_graph.nodes = dep_graph_dict.get('nodes', {})
        dep_graph.edges = dep_graph_dict.get('edges', [])
    
    # Step 5: Hierarchical Assembly Graph Construction (Phase 2 - Already Implemented)
    if not checkpoint.is_step_complete("hierarchical_graph"):
        logger.info("Step 5/7: Building hierarchical assembly graph (Phase 2)...")
        graph_builder = GraphBuilder(
            enable_post_processing=enable_spatial_temporal,  # Controls spatial-temporal
            enable_spatial_relationships=enable_spatial_relationships,
            enable_spatial_temporal=enable_spatial_temporal
        )

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
        logger.info(f"  (Enhanced with context-aware extraction hints)")
        logger.info("")

        checkpoint.save("hierarchical_graph")
    else:
        logger.info("Step 5/7: ✓ Hierarchical graph already complete (skipping)")
    
    # Step 6: 3D Plan Generation
    if not checkpoint.is_step_complete("plan_generation"):
        logger.info("Step 6/7: Generating 3D assembly plan...")
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
        logger.info("Step 6/7: ✓ Plan generation already complete (skipping)")
        json_plan_path = output_dir / f"{assembly_id}_plan.json"
        text_plan_path = output_dir / f"{assembly_id}_plan.txt"
        with open(json_plan_path, 'r', encoding='utf-8') as f:
            assembly_plan = json.load(f)
    
    # Step 7: Vector Store Ingestion (Phase 2)
    if skip_ingestion:
        logger.info("Step 7/7: ✗ Vector store ingestion skipped (--skip-ingestion)")
    elif not checkpoint.is_step_complete("ingestion"):
        logger.info("Step 7/7: Ingesting into vector store...")
        logger.info("  (This creates searchable embeddings with multimodal fusion)")
        
        try:
            ingestion_service = IngestionService(
                use_multimodal=use_multimodal,
                enable_spatial_relationships=enable_spatial_relationships
            )
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
        logger.info("Step 7/7: ✓ Vector store ingestion already complete (skipping)")
    
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
        "--no-spatial",
        action="store_true",
        help="Disable all spatial features (relationships + temporal patterns) for comparison testing"
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
            enable_spatial_relationships=not args.no_spatial,
            enable_spatial_temporal=not args.no_spatial
        )
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        exit(1)

