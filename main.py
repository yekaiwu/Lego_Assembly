"""
Main workflow orchestrator for LEGO Assembly System Phase 1.
Coordinates manual processing, VLM extraction, and plan generation.
"""

import argparse
import json
from pathlib import Path
from typing import Optional
from loguru import logger

from src.vision_processing import ManualInputHandler, VLMStepExtractor, DependencyGraph
from src.plan_generation import PlanStructureGenerator
from src.utils import get_config, URLHandler

def main(
    input_path: str,
    output_dir: str,
    assembly_id: Optional[str] = None,
    use_fallback: bool = False,
    display_output: bool = True
):
    """
    Main workflow: Process LEGO manual and generate 3D assembly plan.
    
    Args:
        input_path: Path to PDF manual, image directory, or URL
        output_dir: Directory for output files
        assembly_id: Unique assembly identifier
        use_fallback: Use fallback VLMs if primary fails
        display_output: Display JSON and text plans in console
    """
    logger.info("=" * 80)
    logger.info("LEGO Assembly System - Phase 1: Manual Processing & Plan Generation")
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
    
    # Step 1: Manual Input Processing
    logger.info("Step 1/4: Processing manual input...")
    manual_handler = ManualInputHandler(output_dir=output_dir / "temp_pages")
    
    page_paths = manual_handler.process_manual(input_path)
    logger.info(f"Extracted {len(page_paths)} pages")
    
    # Detect step boundaries
    step_groups = manual_handler.detect_step_boundaries(page_paths)
    logger.info(f"Detected {len(step_groups)} steps")
    logger.info("")
    
    # Step 2: VLM-based Step Extraction
    logger.info("Step 2/4: Extracting step information using VLM...")
    vlm_extractor = VLMStepExtractor()
    
    # Use assembly_id as cache context to prevent cache collisions between different manuals
    extracted_steps = vlm_extractor.batch_extract(
        step_groups, 
        use_primary=not use_fallback,
        cache_context=assembly_id
    )
    logger.info(f"Extracted information from {len(extracted_steps)} steps")
    
    # Save extracted data
    extracted_path = output_dir / f"{assembly_id}_extracted.json"
    with open(extracted_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_steps, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved extracted data to {extracted_path}")
    logger.info("")
    
    # Step 3: Dependency Graph Construction
    logger.info("Step 3/4: Building dependency graph...")
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
    
    # Step 4: 3D Plan Generation
    logger.info("Step 4/4: Generating 3D assembly plan...")
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
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Plan generation complete!")
    logger.info(f"  JSON Plan: {json_plan_path}")
    logger.info(f"  Text Plan: {text_plan_path}")
    logger.info(f"  Validation: {assembly_plan['validation']['summary']}")
    logger.info("=" * 80)
    
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
    
    return assembly_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEGO Assembly System - Phase 1: Manual Processing & Plan Generation"
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
        "--cache",
        action="store_true",
        help="Enable caching of VLM responses (disabled by default)"
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
            display_output=not args.no_display
        )
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        exit(1)

