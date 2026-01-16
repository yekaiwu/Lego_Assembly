"""
Hierarchical Graph Builder (Stage 2 of Manual2Skill approach).
Builds proper hierarchical assembly graph with subassembly detection.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from loguru import logger
from datetime import datetime

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.litellm_vlm import UnifiedVLMClient
from .part_association import PartAssociationModule, PartCatalog


class SubassemblyDetector:
    """Detects subassemblies using VLM-guided heuristics."""

    def __init__(self, vlm_client: UnifiedVLMClient, enable_spatial_relationships: bool = True):
        self.vlm_client = vlm_client
        self.enable_spatial_relationships = enable_spatial_relationships
    
    def detect_subassemblies(
        self,
        extracted_steps: List[Dict[str, Any]],
        part_catalog: Dict[str, Any],
        manual_pages: List[str],
        assembly_id: str
    ) -> List[Dict[str, Any]]:
        """
        Detect subassemblies across all steps.
        UPDATED: Every step now creates a new subassembly node for step-by-step progression.

        Returns:
            List of subassembly definitions
        """
        logger.info("Detecting subassemblies using VLM-guided analysis (every step creates node)")

        subassemblies = []
        subasm_id_counter = 0
        previous_step_subassembly = None  # Track previous step's subassembly for sequential linking

        for step_idx, step in enumerate(extracted_steps):
            step_number = step.get("step_number", step_idx + 1)

            # Skip non-build steps
            if not step.get("parts_required"):
                continue

            # Get parts from this step
            parts_in_step = step.get("parts_required", [])

            # Every step creates a new subassembly node
            subassembly = self._create_subassembly_for_step(
                step=step,
                step_number=step_number,
                parts=parts_in_step,
                previous_subassembly=previous_step_subassembly,
                manual_page=manual_pages[step_idx] if step_idx < len(manual_pages) else None,
                assembly_id=assembly_id
            )

            if subassembly:
                subassembly["subassembly_id"] = f"subasm_{subasm_id_counter}"
                subassemblies.append(subassembly)

                # Update previous step's subassembly for next iteration
                previous_step_subassembly = subassembly["subassembly_id"]

                logger.debug(f"Created subassembly for step {step_number}: {subassembly['name']}")
                subasm_id_counter += 1

        logger.info(f"Created {len(subassemblies)} subassemblies (one per step)")

        return subassemblies
    
    def _create_subassembly_for_step(
        self,
        step: Dict[str, Any],
        step_number: int,
        parts: List[Dict[str, Any]],
        previous_subassembly: Optional[str],
        manual_page: Optional[str],
        assembly_id: str
    ) -> Dict[str, Any]:
        """
        Create a subassembly node for this step.
        UPDATED: Every step creates a new subassembly, showing step-by-step progression.

        Two patterns for parent relationships:
        - Pattern 1 (continues_previous): Link to previous step's subassembly (sequential chain)
        - Pattern 2 (is_new_subassembly): Link to parent via context_references (cross-reference)
        """
        actions = step.get("actions", [])
        notes = (step.get("notes") or "").lower()

        # Extract component part signatures
        component_parts = [
            f"{p.get('color', 'unknown')}:{p.get('shape', 'unknown')}"
            for p in parts
        ]

        # Determine subassembly purpose/name
        subasm_name = self._infer_subassembly_name(parts, actions, notes)

        # Determine parent relationship using VLM hints
        parent_subassembly = None

        # Pattern 1: continues_previous → link to previous step's subassembly (sequential chain)
        if step.get("continues_previous", False) and previous_subassembly:
            parent_subassembly = previous_subassembly
            logger.debug(f"Step {step_number}: Sequential link to {previous_subassembly}")

        # Pattern 2: is_new_subassembly → link to parent via context_references (cross-reference)
        elif step.get("is_new_subassembly", False):
            # Try to find parent from context_references
            context_refs = step.get("context_references", [])
            if context_refs:
                # Find the most recent subassembly referenced
                # context_refs might contain step numbers or descriptions
                parent_subassembly = self._resolve_parent_from_context(context_refs, step_number)
                logger.debug(f"Step {step_number}: Cross-reference to {parent_subassembly}")
            else:
                # No context references, attach to model root
                parent_subassembly = None
                logger.debug(f"Step {step_number}: New subassembly with no parent (will attach to model)")

        # Default: if no pattern matched, treat as continues_previous
        else:
            if previous_subassembly:
                parent_subassembly = previous_subassembly
                logger.debug(f"Step {step_number}: Default sequential link to {previous_subassembly}")

        # Create subassembly definition
        subassembly = {
            "name": subasm_name,
            "created_in_step": step_number,
            "component_parts": component_parts,
            "parent_subassembly": parent_subassembly,
            "child_subassemblies": [],
            "purpose": self._infer_purpose(parts, actions, notes),
            "continues_previous": step.get("continues_previous", False),
            "is_new_subassembly": step.get("is_new_subassembly", False),
            "completeness_markers": {
                "required_parts": len(component_parts),
                "required_connections": self._extract_connections(actions),
                "spatial_signature": self._extract_spatial_signature(step)
            }
        }

        return subassembly

    def _resolve_parent_from_context(
        self,
        context_refs: List[Any],
        current_step: int
    ) -> Optional[str]:
        """
        Resolve parent subassembly from context_references.

        context_refs might contain:
        - Step numbers (e.g., [1, 2])
        - Descriptions (e.g., ["body assembly"])

        Returns the subassembly ID of the most recent reference.
        """
        # For now, use simple heuristic: find the highest step number less than current_step
        referenced_steps = []

        for ref in context_refs:
            if isinstance(ref, int) and ref < current_step:
                referenced_steps.append(ref)
            elif isinstance(ref, str) and ref.isdigit():
                step_num = int(ref)
                if step_num < current_step:
                    referenced_steps.append(step_num)

        if referenced_steps:
            # Return the subassembly ID for the most recent referenced step
            most_recent_step = max(referenced_steps)
            # Subassembly IDs are 0-indexed: step 1 → subasm_0, step 2 → subasm_1
            return f"subasm_{most_recent_step - 1}"

        return None

    def _infer_subassembly_name(
        self,
        parts: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        notes: str
    ) -> str:
        """Infer a descriptive name for the subassembly."""
        # Look for key functional parts
        functional_parts = []
        for part in parts:
            shape = (part.get("shape") or "").lower()
            if any(kw in shape for kw in ["wheel", "wing", "door", "window", "roof", "base"]):
                functional_parts.append(shape)
        
        if functional_parts:
            # Use first functional part as basis
            main_part = functional_parts[0].replace("_", " ").title()
            return f"{main_part} Assembly"
        
        # Fallback: use color + generic name
        colors = [p.get("color", "") for p in parts if p.get("color")]
        if colors:
            return f"{colors[0].title()} Subassembly"
        
        return "Subassembly"
    
    def _infer_purpose(
        self,
        parts: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        notes: str
    ) -> str:
        """Infer the purpose of the subassembly."""
        # Check notes for purpose clues
        if "wheel" in notes.lower():
            return "Provides mobility"
        elif any(kw in notes.lower() for kw in ["base", "foundation", "frame"]):
            return "Structural foundation"
        elif any(kw in notes.lower() for kw in ["roof", "wall", "door", "window"]):
            return "Enclosure element"
        elif any(kw in notes.lower() for kw in ["wing", "tail", "propeller"]):
            return "Aerodynamic component"
        
        # Check parts
        part_shapes = " ".join((p.get("shape") or "") for p in parts).lower()
        if "wheel" in part_shapes:
            return "Provides mobility"
        elif "plate" in part_shapes or "brick" in part_shapes:
            return "Structural support"
        
        return "Functional component"
    
    def _extract_connections(self, actions: List[Dict[str, Any]]) -> List[str]:
        """Extract connection descriptions from actions."""
        connections = []
        for action in actions:
            action_verb = action.get("action_verb", "")
            target = action.get("target", "")
            destination = action.get("destination", "")
            
            if action_verb in ["attach", "connect"] and target and destination:
                connections.append(f"{target}->{destination}")
        
        return connections
    
    def _extract_spatial_signature(self, step: Dict[str, Any]) -> str:
        """Extract spatial signature from step."""
        if not self.enable_spatial_relationships:
            return "spatial analysis disabled"

        spatial = step.get("spatial_relationships", {})
        position = spatial.get("position", "")
        orientation = spatial.get("rotation", "")

        if position and orientation:
            return f"{position}, {orientation}"
        elif position:
            return position
        elif orientation:
            return orientation

        return "standard arrangement"


class StepStateTracker:
    """Tracks assembly state at each step."""
    
    def build_step_states(
        self,
        extracted_steps: List[Dict[str, Any]],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build step-by-step state progression.
        
        Args:
            extracted_steps: Extracted step data
            nodes: Graph nodes (parts, subassemblies)
            edges: Graph edges (connections)
        
        Returns:
            List of step states
        """
        logger.info("Building step state progression")
        
        step_states = []
        existing_nodes = set()
        
        total_steps = len([s for s in extracted_steps if s.get("parts_required")])
        
        for step_idx, step in enumerate(extracted_steps):
            step_number = step.get("step_number", step_idx + 1)
            
            # Skip non-build steps
            if not step.get("parts_required"):
                continue
            
            # Find new nodes created in this step
            new_nodes = []
            for node in nodes:
                if node.get("step_created") == step_number:
                    new_nodes.append(node["node_id"])
                    existing_nodes.add(node["node_id"])
            
            # Find new connections made in this step
            new_connections = []
            for edge in edges:
                if edge.get("created_step") == step_number:
                    new_connections.append({
                        "from": edge["from"],
                        "to": edge["to"],
                        "type": edge.get("type", "connection")
                    })
            
            # Find active subassemblies (created but not yet attached to parent)
            active_subassemblies = [
                node["node_id"] for node in nodes
                if node.get("type") == "subassembly" and 
                   node.get("step_created", 0) <= step_number and
                   node["node_id"] in existing_nodes
            ]
            
            # Calculate completion percentage
            completion_pct = (len([s for s in extracted_steps[:step_idx+1] 
                                  if s.get("parts_required")]) / total_steps * 100) if total_steps > 0 else 0
            
            step_state = {
                "step_number": step_number,
                "existing_nodes": list(existing_nodes),
                "new_nodes": new_nodes,
                "new_connections": new_connections,
                "active_subassemblies": active_subassemblies,
                "completion_percentage": round(completion_pct, 1)
            }
            
            step_states.append(step_state)
        
        logger.info(f"Built {len(step_states)} step states")
        
        return step_states


class GraphBuilder:
    """
    Builds hierarchical assembly graph following Manual2Skill's 2-stage approach.
    """

    def __init__(self, vlm_client=None, enable_spatial_relationships: bool = True):
        # Use provided VLM client or get from config
        if vlm_client is None:
            from src.vision_processing.vlm_step_extractor import VLMStepExtractor
            from src.utils.config import get_config
            config = get_config()
            extractor = VLMStepExtractor()
            ingestion_vlm = config.models.ingestion_vlm
            self.vlm_client = extractor._get_client(ingestion_vlm)
            logger.info(f"GraphBuilder initialized with INGESTION_VLM: {ingestion_vlm}")
        else:
            self.vlm_client = vlm_client
            logger.info("GraphBuilder initialized with provided VLM client")

        self.part_association = PartAssociationModule(vlm_client=self.vlm_client)
        self.subassembly_detector = SubassemblyDetector(
            self.vlm_client,
            enable_spatial_relationships=enable_spatial_relationships
        )
        self.state_tracker = StepStateTracker()
        self.enable_spatial_relationships = enable_spatial_relationships

        logger.info("GraphBuilder initialized with 2-stage hierarchical construction")

        if enable_spatial_relationships:
            logger.info("Spatial relationships: ENABLED")
        else:
            logger.warning("⚠ Spatial relationships: DISABLED")
    
    def build_graph(
        self,
        extracted_steps: List[Dict[str, Any]],
        assembly_id: str,
        image_dir: Path
    ) -> Dict[str, Any]:
        """
        Build complete hierarchical graph.
        
        Args:
            extracted_steps: Previously extracted step data
            assembly_id: Assembly identifier
            image_dir: Directory with manual page images
        
        Returns:
            Complete hierarchical graph structure
        """
        logger.info("=" * 60)
        logger.info("Building Hierarchical Assembly Graph")
        logger.info(f"Assembly ID: {assembly_id}")
        logger.info("=" * 60)
        
        # Get manual pages
        manual_pages = sorted(list(image_dir.glob("page_*.png")))
        logger.info(f"Found {len(manual_pages)} manual pages")
        
        # Stage 1: Part Association
        logger.info("\n[Stage 1/2] Part Association & Role Assignment")
        part_catalog = self.part_association.build_part_catalog(
            extracted_steps=extracted_steps,
            manual_pages=[str(p) for p in manual_pages],
            assembly_id=assembly_id
        )
        
        # Stage 2: Subassembly Detection
        logger.info("\n[Stage 2/2] Subassembly Identification")
        subassemblies = self.subassembly_detector.detect_subassemblies(
            extracted_steps=extracted_steps,
            part_catalog=part_catalog,
            manual_pages=[str(p) for p in manual_pages],
            assembly_id=assembly_id
        )
        
        # Build graph structure
        logger.info("\n[Graph Construction] Building nodes and edges")
        nodes, edges = self._build_graph_structure(
            extracted_steps=extracted_steps,
            part_catalog=part_catalog,
            subassemblies=subassemblies,
            assembly_id=assembly_id
        )
        
        # Build step states
        logger.info("\n[State Tracking] Building step progression")
        step_states = self.state_tracker.build_step_states(
            extracted_steps=extracted_steps,
            nodes=nodes,
            edges=edges
        )
        
        # Calculate metadata
        metadata = self._calculate_metadata(nodes, subassemblies, extracted_steps)

        graph = {
            "manual_id": assembly_id,
            "metadata": metadata,
            "nodes": nodes,
            "edges": edges,
            "step_states": step_states,
            "part_catalog": part_catalog
        }

        logger.info("\n" + "=" * 60)
        logger.info("✓ Hierarchical Graph Construction Complete")
        logger.info(f"  Parts: {metadata['total_parts']}")
        logger.info(f"  Subassemblies: {metadata['total_subassemblies']}")
        logger.info(f"  Steps: {metadata['total_steps']}")
        logger.info(f"  Max Depth: {metadata['max_depth']} layers")
        logger.info("=" * 60)

        return graph
    
    def _build_graph_structure(
        self,
        extracted_steps: List[Dict[str, Any]],
        part_catalog: Dict[str, Any],
        subassemblies: List[Dict[str, Any]],
        assembly_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build nodes and edges."""
        nodes = []
        edges = []
        
        # Create root model node
        model_node = {
            "node_id": f"model_{assembly_id}",
            "type": "model",
            "name": f"LEGO Model {assembly_id}",
            "description": f"Complete LEGO model {assembly_id}",
            "children": [],
            "parents": [],
            "step_created": 0,
            "layer": 0
        }
        nodes.append(model_node)
        
        # Create part nodes
        parts_catalog = part_catalog.get("parts_catalog", {})
        for part_id, part_data in parts_catalog.items():
            part_node = {
                "node_id": part_id,
                "type": "part",
                "name": part_data.get("name", "Unknown part"),
                "description": part_data.get("appearance_description", ""),
                "color": part_data.get("color", "unknown"),
                "shape": part_data.get("shape", "unknown"),
                "role": part_data.get("role", "structural"),
                "children": [],
                "parents": [],
                "step_created": part_data.get("first_appears_step", 1),
                "layer": None  # Will be set later
            }
            nodes.append(part_node)
        
        # Create subassembly nodes
        for subasm in subassemblies:
            subasm_node = {
                "node_id": subasm["subassembly_id"],
                "type": "subassembly",
                "name": subasm["name"],
                "description": subasm.get("purpose", ""),
                "children": [],
                "parents": [],
                "step_created": subasm["created_in_step"],
                "layer": None,  # Will be set later
                "completeness_markers": subasm.get("completeness_markers", {})
            }
            nodes.append(subasm_node)
        
        # Build relationships
        self._build_relationships(nodes, subassemblies, part_catalog, extracted_steps, edges)
        
        # Assign layers
        self._assign_layers(nodes)
        
        return nodes, edges
    
    def _build_relationships(
        self,
        nodes: List[Dict[str, Any]],
        subassemblies: List[Dict[str, Any]],
        part_catalog: Dict[str, Any],
        extracted_steps: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ):
        """Build parent-child relationships and edges."""
        # Map for quick lookup
        node_map = {node["node_id"]: node for node in nodes}
        model_node = [n for n in nodes if n["type"] == "model"][0]
        
        # Connect subassemblies to their parents
        for subasm in subassemblies:
            subasm_id = subasm["subassembly_id"]
            parent_id = subasm.get("parent_subassembly")
            
            if parent_id and parent_id in node_map:
                # Has a parent subassembly
                node_map[subasm_id]["parents"].append(parent_id)
                node_map[parent_id]["children"].append(subasm_id)
                
                edges.append({
                    "from": subasm_id,
                    "to": parent_id,
                    "type": "attachment",
                    "created_step": subasm["created_in_step"]
                })
            else:
                # Top-level subassembly - attach to model
                node_map[subasm_id]["parents"].append(model_node["node_id"])
                model_node["children"].append(subasm_id)
                
                edges.append({
                    "from": subasm_id,
                    "to": model_node["node_id"],
                    "type": "attachment",
                    "created_step": subasm["created_in_step"]
                })
        
        # Connect parts to subassemblies or model
        parts_catalog = part_catalog.get("parts_catalog", {})
        for part_id, part_data in parts_catalog.items():
            # Find which subassembly this part belongs to
            parent_subasm = self._find_part_parent_subassembly(
                part_data, subassemblies, extracted_steps
            )
            
            if parent_subasm:
                node_map[part_id]["parents"].append(parent_subasm)
                node_map[parent_subasm]["children"].append(part_id)
                
                edges.append({
                    "from": part_id,
                    "to": parent_subasm,
                    "type": "component",
                    "created_step": part_data.get("first_appears_step", 1)
                })
            else:
                # Attach directly to model
                node_map[part_id]["parents"].append(model_node["node_id"])
                model_node["children"].append(part_id)
                
                edges.append({
                    "from": part_id,
                    "to": model_node["node_id"],
                    "type": "component",
                    "created_step": part_data.get("first_appears_step", 1)
                })
    
    def _find_part_parent_subassembly(
        self,
        part_data: Dict[str, Any],
        subassemblies: List[Dict[str, Any]],
        extracted_steps: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Find which subassembly a part belongs to.
        UPDATED: Simplified to match parts to their step's subassembly (every step has one).
        """
        part_step = part_data.get("first_appears_step", 1)

        # Find the subassembly created in the same step
        for subasm in subassemblies:
            if subasm["created_in_step"] == part_step:
                return subasm["subassembly_id"]

        return None
    
    def _assign_layers(self, nodes: List[Dict[str, Any]]):
        """Assign layer numbers (depth from root)."""
        node_map = {node["node_id"]: node for node in nodes}
        
        # BFS from root to assign layers
        queue = [(n["node_id"], 0) for n in nodes if n["type"] == "model"]
        visited = set()
        
        while queue:
            node_id, layer = queue.pop(0)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            node = node_map[node_id]
            node["layer"] = layer
            
            # Add children to queue
            for child_id in node.get("children", []):
                if child_id not in visited:
                    queue.append((child_id, layer + 1))
    
    def _calculate_metadata(
        self,
        nodes: List[Dict[str, Any]],
        subassemblies: List[Dict[str, Any]],
        extracted_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate graph metadata."""
        total_parts = len([n for n in nodes if n["type"] == "part"])
        total_subassemblies = len([n for n in nodes if n["type"] == "subassembly"])
        total_steps = len([s for s in extracted_steps if s.get("parts_required")])
        max_depth = max([n.get("layer", 0) for n in nodes])

        return {
            "total_parts": total_parts,
            "total_subassemblies": total_subassemblies,
            "total_steps": total_steps,
            "max_depth": max_depth,
            "generated_at": datetime.now().isoformat(),
            "configuration": {
                "spatial_relationships_enabled": self.enable_spatial_relationships
            }
        }
    
    def save_graph(self, graph: Dict[str, Any], output_path: Path):
        """Save graph to JSON file and generate human-readable summary."""
        # Save JSON graph
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Graph JSON saved to {output_path}")
        
        # Also save a human-readable summary for easy inspection
        summary_path = output_path.parent / f"{graph['manual_id']}_graph_summary.txt"
        self._save_graph_summary(graph, summary_path)
        logger.info(f"✓ Graph summary saved to {summary_path}")
    
    def _save_graph_summary(self, graph: Dict[str, Any], output_path: Path):
        """Save human-readable graph summary for easy verification and debugging."""
        lines = []
        lines.append("=" * 80)
        lines.append("HIERARCHICAL ASSEMBLY GRAPH SUMMARY")
        lines.append("=" * 80)
        lines.append(f"\nManual ID: {graph['manual_id']}")
        lines.append(f"Generated: {graph['metadata']['generated_at']}")

        # Configuration info
        config = graph['metadata'].get('configuration', {})
        lines.append(f"\n⚙️  CONFIGURATION:")
        lines.append(f"  Spatial Relationships: {'Enabled' if config.get('spatial_relationships_enabled', True) else 'DISABLED'}")
        lines.append(f"  Spatial-Temporal Patterns: {'Enabled' if config.get('spatial_temporal_enabled', True) else 'DISABLED'}")

        lines.append(f"\n📊 STATISTICS:")
        lines.append(f"  Total Parts: {graph['metadata']['total_parts']}")
        lines.append(f"  Total Subassemblies: {graph['metadata']['total_subassemblies']}")
        lines.append(f"  Total Steps: {graph['metadata']['total_steps']}")
        lines.append(f"  Max Hierarchy Depth: {graph['metadata']['max_depth']} layers")
        
        # Part catalog summary
        part_catalog = graph.get("part_catalog", {})
        if "parts_by_role" in part_catalog:
            lines.append(f"\n📦 PARTS BY ROLE:")
            for role, count in part_catalog["parts_by_role"].items():
                lines.append(f"  {role}: {count} parts")
        
        # Show hierarchy tree
        lines.append(f"\n" + "=" * 80)
        lines.append("🌳 HIERARCHY STRUCTURE (Tree View)")
        lines.append("=" * 80)
        
        # Get root node
        root = [n for n in graph["nodes"] if n["type"] == "model"][0]
        lines.append(f"\n📦 {root['name']} (root)")
        self._add_hierarchy_lines(root, graph["nodes"], lines, indent=1)
        
        # Show subassemblies detail
        lines.append(f"\n" + "=" * 80)
        lines.append("🔧 SUBASSEMBLIES DETAIL")
        lines.append("=" * 80)
        
        subassemblies = [n for n in graph["nodes"] if n["type"] == "subassembly"]
        if subassemblies:
            for i, subasm in enumerate(subassemblies, 1):
                lines.append(f"\n{i}. {subasm['node_id']}: {subasm['name']}")
                lines.append(f"   Step Created: {subasm['step_created']}")
                lines.append(f"   Layer: {subasm.get('layer', 'unknown')}")
                lines.append(f"   Description: {subasm.get('description', 'N/A')}")
                lines.append(f"   Children: {len(subasm.get('children', []))} parts/subassemblies")
                lines.append(f"   Parents: {', '.join(subasm.get('parents', [])) or 'none'}")
                
                # Show completeness markers if available
                markers = subasm.get('completeness_markers', {})
                if markers:
                    lines.append(f"   Completeness Markers:")
                    lines.append(f"     Required parts: {markers.get('required_parts', 'N/A')}")
                    lines.append(f"     Required connections: {len(markers.get('required_connections', []))}")
                    lines.append(f"     Spatial signature: {markers.get('spatial_signature', 'N/A')}")
        else:
            lines.append("\nNo subassemblies detected in this manual.")
        
        # Show step progression
        lines.append(f"\n" + "=" * 80)
        lines.append("📈 STEP PROGRESSION (First 5 steps)")
        lines.append("=" * 80)
        
        step_states = graph.get("step_states", [])
        for state in step_states[:5]:
            lines.append(f"\nStep {state['step_number']}:")
            lines.append(f"  New nodes: {len(state['new_nodes'])}")
            lines.append(f"  Total nodes: {len(state['existing_nodes'])}")
            lines.append(f"  New connections: {len(state['new_connections'])}")
            lines.append(f"  Active subassemblies: {len(state['active_subassemblies'])}")
            lines.append(f"  Progress: {state['completion_percentage']:.1f}%")
        
        if len(step_states) > 5:
            lines.append(f"\n  ... ({len(step_states) - 5} more steps)")
        
        # Usage instructions
        lines.append(f"\n" + "=" * 80)
        lines.append("💡 DEBUGGING GUIDE")
        lines.append("=" * 80)
        lines.append("\n1. Check if subassemblies make logical sense")
        lines.append("   - Do the names describe functional units? (e.g., 'Wheel Assembly')")
        lines.append("   - Are they created at the right steps?")
        lines.append("\n2. Verify hierarchy structure")
        lines.append("   - Are parts properly nested under subassemblies?")
        lines.append("   - Do subassemblies attach to correct parents?")
        lines.append("\n3. Review completeness markers")
        lines.append("   - Do required_parts counts match actual parts?")
        lines.append("   - Are spatial signatures descriptive enough?")
        lines.append("\n4. Compare with manual")
        lines.append("   - Open the PDF and check step numbers align")
        lines.append("   - Verify subassembly detection matches manual's structure")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # Also print key info to console
        logger.info(f"\n📋 GRAPH SUMMARY:")
        logger.info(f"   Parts: {graph['metadata']['total_parts']}")
        logger.info(f"   Subassemblies: {graph['metadata']['total_subassemblies']}")
        logger.info(f"   Hierarchy depth: {graph['metadata']['max_depth']} layers")
        logger.info(f"   Full summary saved to: {output_path.name}")
    
    def _add_hierarchy_lines(
        self,
        node: Dict[str, Any],
        all_nodes: List[Dict[str, Any]],
        lines: List[str],
        indent: int = 0,
        max_children: int = 10
    ):
        """Recursively add hierarchy lines with visual tree structure."""
        node_map = {n["node_id"]: n for n in all_nodes}
        
        # Get children, prioritizing subassemblies over parts
        children = node.get("children", [])
        child_nodes = [node_map.get(cid) for cid in children if node_map.get(cid)]
        
        # Sort: subassemblies first, then parts
        subassembly_children = [c for c in child_nodes if c["type"] == "subassembly"]
        part_children = [c for c in child_nodes if c["type"] == "part"]
        
        # Show subassemblies
        for child in subassembly_children[:max_children]:
            icon = "🔧" if child["type"] == "subassembly" else "🧱"
            prefix = "  " * indent + "├─ "
            step_info = f" [step {child.get('step_created', '?')}]"
            lines.append(f"{prefix}{icon} {child['name']}{step_info}")
            
            # Recurse for subassemblies
            self._add_hierarchy_lines(child, all_nodes, lines, indent + 1, max_children=5)
        
        # Show first few parts
        parts_to_show = min(3, len(part_children))
        for child in part_children[:parts_to_show]:
            prefix = "  " * indent + "├─ "
            lines.append(f"{prefix}🧱 {child['name']}")
        
        # Show count of remaining parts
        remaining_parts = len(part_children) - parts_to_show
        if remaining_parts > 0:
            prefix = "  " * indent + "└─ "
            lines.append(f"{prefix}... ({remaining_parts} more parts)")
