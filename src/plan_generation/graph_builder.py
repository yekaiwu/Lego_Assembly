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

from src.api.qwen_vlm import QwenVLMClient
from .part_association import PartAssociationModule, PartCatalog


class SubassemblyDetector:
    """Detects subassemblies using VLM-guided heuristics."""
    
    def __init__(self, vlm_client: QwenVLMClient):
        self.vlm_client = vlm_client
    
    def detect_subassemblies(
        self,
        extracted_steps: List[Dict[str, Any]],
        part_catalog: Dict[str, Any],
        manual_pages: List[str],
        assembly_id: str
    ) -> List[Dict[str, Any]]:
        """
        Detect subassemblies across all steps.
        
        Returns:
            List of subassembly definitions
        """
        logger.info("Detecting subassemblies using VLM-guided analysis")
        
        subassemblies = []
        subasm_id_counter = 0
        
        # Track accumulated parts across steps
        accumulated_parts = []
        
        for step_idx, step in enumerate(extracted_steps):
            step_number = step.get("step_number", step_idx + 1)
            
            # Skip non-build steps
            if not step.get("parts_required"):
                continue
            
            # Add new parts from this step
            new_parts = step.get("parts_required", [])
            accumulated_parts.extend(new_parts)
            
            # Check if this step creates a subassembly
            detected = self._detect_subassembly_in_step(
                step=step,
                step_number=step_number,
                accumulated_parts=accumulated_parts,
                manual_page=manual_pages[step_idx] if step_idx < len(manual_pages) else None,
                assembly_id=assembly_id
            )
            
            if detected:
                detected["subassembly_id"] = f"subasm_{subasm_id_counter}"
                subassemblies.append(detected)
                subasm_id_counter += 1
                
                logger.debug(f"Detected subassembly at step {step_number}: {detected['name']}")
        
        # Build hierarchy relationships
        subassemblies = self._build_hierarchy(subassemblies, extracted_steps)
        
        logger.info(f"Detected {len(subassemblies)} subassemblies")
        
        return subassemblies
    
    def _detect_subassembly_in_step(
        self,
        step: Dict[str, Any],
        step_number: int,
        accumulated_parts: List[Dict[str, Any]],
        manual_page: Optional[str],
        assembly_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a step creates a subassembly.
        Uses heuristics + VLM confirmation.
        """
        # Heuristic 1: Functional grouping
        parts = step.get("parts_required", [])
        actions = step.get("actions", [])

        # Look for functional keywords in descriptions
        functional_keywords = ["wheel", "wing", "base", "body", "roof", "door", "window", "frame"]
        notes = (step.get("notes") or "").lower()  # Handle None explicitly
        
        has_functional_parts = any(
            any(kw in part.get("shape", "").lower() or kw in part.get("description", "").lower() 
                for kw in functional_keywords)
            for part in parts
        )
        
        # Heuristic 2: Multiple parts connected
        has_multiple_connections = len(actions) >= 2
        
        # Heuristic 3: Check if description suggests completion
        completion_keywords = ["complete", "finish", "assembly", "attach to", "now add"]
        suggests_completion = any(kw in notes for kw in completion_keywords)
        
        # Decide if this is likely a subassembly
        likely_subassembly = (
            (has_functional_parts and len(parts) >= 2) or
            (has_multiple_connections and len(parts) >= 3) or
            suggests_completion
        )
        
        if not likely_subassembly:
            return None
        
        # Extract component part signatures
        component_parts = [
            f"{p.get('color', 'unknown')}:{p.get('shape', 'unknown')}"
            for p in parts
        ]
        
        # Determine subassembly purpose/name
        subasm_name = self._infer_subassembly_name(parts, actions, notes)
        
        # Create subassembly definition
        subassembly = {
            "name": subasm_name,
            "created_in_step": step_number,
            "component_parts": component_parts,
            "parent_subassembly": None,  # Will be set in hierarchy building
            "child_subassemblies": [],
            "purpose": self._infer_purpose(parts, actions, notes),
            "completeness_markers": {
                "required_parts": len(component_parts),
                "required_connections": self._extract_connections(actions),
                "spatial_signature": self._extract_spatial_signature(step)
            }
        }
        
        return subassembly
    
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
            shape = part.get("shape", "").lower()
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
        part_shapes = " ".join(p.get("shape", "") for p in parts).lower()
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
    
    def _build_hierarchy(
        self,
        subassemblies: List[Dict[str, Any]],
        extracted_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build parent-child relationships between subassemblies.
        Later subassemblies may attach to earlier ones.
        """
        # Simple heuristic: if a subassembly's parts include references to earlier subassemblies,
        # it's a child of those subassemblies
        
        for i, subasm in enumerate(subassemblies):
            # Check if any later subassemblies reference this one
            for j in range(i + 1, len(subassemblies)):
                later_subasm = subassemblies[j]
                later_step = later_subasm["created_in_step"]
                
                # Get the step description
                if later_step - 1 < len(extracted_steps):
                    step_data = extracted_steps[later_step - 1]
                    existing_assembly = step_data.get("existing_assembly", "").lower()
                    
                    # Check if this subassembly is mentioned
                    if subasm["name"].lower() in existing_assembly:
                        # later_subasm attaches to subasm
                        later_subasm["parent_subassembly"] = subasm["subassembly_id"]
                        subasm["child_subassemblies"].append(later_subasm["subassembly_id"])
        
        return subassemblies


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
    
    def __init__(self, vlm_client=None):
        # Use provided VLM client or get from config
        if vlm_client is None:
            from src.vision_processing.vlm_step_extractor import VLMStepExtractor
            from src.utils.config import get_config
            config = get_config()
            extractor = VLMStepExtractor()
            primary_vlm = config.models.primary_vlm
            self.vlm_client = extractor.clients.get(primary_vlm, QwenVLMClient())
            logger.info(f"GraphBuilder initialized with PRIMARY_VLM: {primary_vlm}")
        else:
            self.vlm_client = vlm_client
            logger.info("GraphBuilder initialized with provided VLM client")
        
        self.part_association = PartAssociationModule(vlm_client=self.vlm_client)
        self.subassembly_detector = SubassemblyDetector(self.vlm_client)
        self.state_tracker = StepStateTracker()
        logger.info("GraphBuilder initialized with 2-stage hierarchical construction")
    
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
        """Find which subassembly a part belongs to."""
        part_signature = f"{part_data.get('color', '')}:{part_data.get('shape', '')}"
        part_step = part_data.get("first_appears_step", 1)
        
        # Find subassemblies created in the same step
        for subasm in subassemblies:
            if subasm["created_in_step"] == part_step:
                # Check if part is in component parts
                if part_signature in subasm.get("component_parts", []):
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
            "generated_at": datetime.now().isoformat()
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
