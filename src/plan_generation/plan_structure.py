"""
Plan Structure Generator: Creates hierarchical assembly plans from
extracted steps, dependencies, and spatial information.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger

from .part_database import PartDatabase
from .spatial_reasoning import SpatialReasoning
from ..vision_processing.dependency_graph import DependencyGraph

class PlanStructureGenerator:
    """Generates structured 3D assembly plans."""
    
    def __init__(self):
        self.part_db = PartDatabase()
        self.spatial_engine = SpatialReasoning()
        logger.info("Plan structure generator initialized")
    
    def generate_plan(
        self,
        extracted_steps: List[Dict[str, Any]],
        dependency_graph: DependencyGraph,
        assembly_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete assembly plan from extracted steps.
        
        Args:
            extracted_steps: List of VLM-extracted step information
            dependency_graph: Constructed dependency graph
            assembly_id: Unique identifier for this assembly
            metadata: Optional metadata (set name, source, etc.)
        
        Returns:
            Complete assembly plan structure
        """
        logger.info(f"Generating assembly plan for {assembly_id}...")
        
        # Build subassemblies
        subassemblies = dependency_graph.group_into_subassemblies()
        
        # Process each step and assign 3D positions
        processed_steps = self._process_steps(extracted_steps, dependency_graph)
        
        # Build hierarchical structure
        plan = {
            "assembly_id": assembly_id,
            "metadata": metadata or {},
            "generated_at": datetime.utcnow().isoformat(),
            "total_steps": len(extracted_steps),
            "subassemblies": subassemblies,
            "steps": processed_steps,
            "build_order": dependency_graph.topological_sort(),
            "dependencies": dependency_graph.to_dict(),
            "bounding_box": self._calculate_assembly_bounds(processed_steps)
        }
        
        # Validate plan
        validation_result = self._validate_plan(plan)
        plan["validation"] = validation_result
        
        logger.info(f"Plan generation complete. {len(processed_steps)} steps, {len(subassemblies)} subassemblies")
        return plan
    
    def _process_steps(
        self,
        extracted_steps: List[Dict[str, Any]],
        dependency_graph: DependencyGraph
    ) -> List[Dict[str, Any]]:
        """Process and enrich steps with spatial and part information."""
        processed_steps = []
        placed_parts = []  # Track all placed parts for spatial reference
        
        for i, step_info in enumerate(extracted_steps):
            step_number = step_info.get("step_number", i + 1)
            logger.debug(f"Processing step {step_number}...")
            
            # Extract and match parts
            parts_in_step = self._match_parts(step_info.get("parts_required", []))
            
            # Calculate spatial positions for new parts
            spatial_rel = step_info.get("spatial_relationships", {})
            positioned_parts = []
            
            for part in parts_in_step:
                # Calculate 3D position
                position = self.spatial_engine.calculate_position(
                    part, placed_parts, spatial_rel
                )
                
                # Calculate orientation
                orientation = self.spatial_engine.calculate_orientation(
                    part, spatial_rel
                )
                
                # Determine connection points
                connections = self.spatial_engine.determine_connection_points(
                    part, position
                )
                
                positioned_part = {
                    **part,
                    "position": position,
                    "rotation": orientation,
                    "connections": connections,
                    "step": step_number
                }
                
                positioned_parts.append(positioned_part)
                placed_parts.append(positioned_part)
            
            # Build processed step
            processed_step = {
                "step_number": step_number,
                "description": self._generate_step_description(step_info),
                "parts": positioned_parts,
                "actions": step_info.get("actions", []),
                "dependencies": dependency_graph.reverse_edges.get(step_number, []),
                "notes": step_info.get("notes", ""),
                "validation": self._validate_step(positioned_parts, placed_parts[:-len(positioned_parts)])
            }
            
            processed_steps.append(processed_step)
        
        return processed_steps
    
    def _match_parts(self, parts_required: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match extracted parts to database entries."""
        matched_parts = []
        
        for part_info in parts_required:
            description = part_info.get("description", "")
            color = part_info.get("color", "")
            shape = part_info.get("shape", "")
            
            # Try to match to database
            db_match = self.part_db.match_part_description(description, color, shape)
            
            if db_match:
                # Get dimensions
                dimensions = self.part_db.get_part_dimensions(db_match["part_num"])
                
                matched_part = {
                    "part_num": db_match["part_num"],
                    "part_name": db_match["name"],
                    "color": color,
                    "quantity": part_info.get("quantity", 1),
                    "dimensions": dimensions or self._estimate_dimensions(shape),
                    "matched": True
                }
            else:
                # Create temporary part entry
                matched_part = {
                    "part_num": f"unknown_{len(matched_parts)}",
                    "part_name": description,
                    "color": color,
                    "quantity": part_info.get("quantity", 1),
                    "dimensions": self._estimate_dimensions(shape),
                    "matched": False
                }
                logger.warning(f"Part not found in database: {description}")
            
            matched_parts.append(matched_part)
        
        return matched_parts
    
    def _estimate_dimensions(self, shape: str) -> Dict[str, float]:
        """Estimate part dimensions from shape description."""
        import re
        
        # Try to extract dimensions from shape (e.g., "2x4 brick")
        match = re.search(r'(\d+)\s*x\s*(\d+)', shape.lower())
        
        if match:
            width = int(match.group(1))
            depth = int(match.group(2))
            
            # Determine height based on type
            if "plate" in shape.lower():
                height = 0.33  # Plates are 1/3 height of bricks
            else:
                height = 1.0  # Standard brick height
            
            return {"width": float(width), "height": height, "depth": float(depth)}
        
        # Default dimensions if parsing fails
        return {"width": 1.0, "height": 1.0, "depth": 1.0}
    
    def _generate_step_description(self, step_info: Dict[str, Any]) -> str:
        """Generate natural language step description."""
        parts = step_info.get("parts_required", [])
        actions = step_info.get("actions", [])
        
        # Build description
        parts_desc = ", ".join([
            f"{p.get('color', '')} {p.get('shape', '')}"
            for p in parts
        ])
        
        action_desc = "; ".join([
            f"{a.get('action_verb', '')} {a.get('target', '')} to {a.get('destination', '')}"
            for a in actions
        ])
        
        description = f"Parts: {parts_desc}. Actions: {action_desc}"
        return description
    
    def _validate_step(
        self,
        new_parts: List[Dict[str, Any]],
        existing_parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate a single step for spatial consistency."""
        errors = []
        warnings = []
        
        # Check for collisions
        for new_part in new_parts:
            if self.spatial_engine.check_collision(new_part, existing_parts):
                errors.append(f"Collision detected for part {new_part.get('part_num')}")
        
        # Check for valid connections
        for new_part in new_parts:
            has_connection = False
            for existing in existing_parts:
                if self.spatial_engine.validate_connection(
                    new_part.get("connections", []),
                    existing.get("connections", [])
                ):
                    has_connection = True
                    break
            
            if not has_connection and existing_parts:
                warnings.append(f"Part {new_part.get('part_num')} may not be properly connected")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _calculate_assembly_bounds(self, steps: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate bounding box for entire assembly."""
        all_parts = []
        for step in steps:
            all_parts.extend(step.get("parts", []))
        
        return self.spatial_engine.calculate_bounding_box(all_parts)
    
    def _validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entire assembly plan."""
        errors = []
        warnings = []
        
        # Check for missing steps
        total_steps = plan["total_steps"]
        step_numbers = [s["step_number"] for s in plan["steps"]]
        missing = set(range(1, total_steps + 1)) - set(step_numbers)
        
        if missing:
            errors.append(f"Missing steps: {sorted(missing)}")
        
        # Check for unmatched parts
        unmatched_count = sum(
            1 for step in plan["steps"]
            for part in step["parts"]
            if not part.get("matched", True)
        )
        
        if unmatched_count > 0:
            warnings.append(f"{unmatched_count} parts could not be matched to database")
        
        # Check step validations
        invalid_steps = [
            s["step_number"] for s in plan["steps"]
            if not s.get("validation", {}).get("valid", True)
        ]
        
        if invalid_steps:
            warnings.append(f"Steps with validation issues: {invalid_steps}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "summary": f"{len(errors)} errors, {len(warnings)} warnings"
        }
    
    def export_plan(
        self,
        plan: Dict[str, Any],
        output_path: Path,
        format: str = "json"
    ):
        """
        Export plan to file.
        
        Args:
            plan: Assembly plan
            output_path: Output file path
            format: Export format (json, text)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
            logger.info(f"Plan exported to {output_path}")
        
        elif format == "text":
            text_plan = self._plan_to_text(plan)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_plan)
            logger.info(f"Text plan exported to {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _plan_to_text(self, plan: Dict[str, Any]) -> str:
        """Convert plan to human-readable text format."""
        lines = []
        
        lines.append(f"Assembly Plan: {plan['assembly_id']}")
        lines.append(f"Generated: {plan['generated_at']}")
        lines.append(f"Total Steps: {plan['total_steps']}")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
        
        for step in plan["steps"]:
            lines.append(f"Step {step['step_number']}:")
            lines.append(f"  {step['description']}")
            lines.append(f"  Parts:")
            
            for part in step["parts"]:
                pos = part["position"]
                lines.append(f"    - {part['part_name']} ({part['color']})")
                lines.append(f"      Position: ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f})")
            
            if step.get("notes"):
                lines.append(f"  Notes: {step['notes']}")
            
            lines.append("")
        
        return "\n".join(lines)

