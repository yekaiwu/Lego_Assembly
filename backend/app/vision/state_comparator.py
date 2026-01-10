"""
State Comparator - Compares detected assembly state with expected plan.
Determines progress, identifies errors, and maps current position in build sequence.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger


class StateComparator:
    """Compares detected state with expected assembly plan."""
    
    def __init__(self):
        """Initialize state comparator."""
        logger.info("StateComparator initialized")
    
    def compare_with_plan(
        self,
        detected_state: Dict[str, Any],
        manual_id: str,
        output_dir: str = "./output"
    ) -> Dict[str, Any]:
        """
        Compare detected state against Phase 1 plan data.
        
        Args:
            detected_state: Result from StateAnalyzer
            manual_id: Manual identifier
            output_dir: Directory containing Phase 1 outputs
        
        Returns:
            Dictionary containing:
                - completed_steps: List of step numbers completed
                - current_step: Current step number (last completed + 1)
                - progress_percentage: Overall progress (0-100)
                - next_steps: Recommended next steps
                - errors: List of detected errors
                - missing_parts: Parts needed but not detected
                - extra_parts: Parts detected but not expected yet
                - confidence: Confidence in the comparison
        """
        try:
            logger.info(f"Comparing state with plan for manual {manual_id}")
            
            # Load Phase 1 plan data
            plan_data = self._load_plan_data(manual_id, output_dir)
            
            if not plan_data:
                logger.error(f"Could not load plan data for manual {manual_id}")
                return self._create_error_result("Plan data not found")
            
            # Extract detected parts
            detected_parts = detected_state.get("detected_parts", [])
            
            # Try graph-aware matching first if graph is available
            graph_path = Path(output_dir) / f"{manual_id}_graph.json"
            if graph_path.exists():
                logger.info("Using graph-aware step estimation")
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                graph_match = self._match_steps_graph_aware(
                    detected_state, manual_id, graph_data
                )
                
                if graph_match and graph_match["confidence"] > 0.5:
                    # Use graph-aware result
                    current_step = graph_match["estimated_step"]
                    completed_steps = list(range(1, current_step))
                    logger.info(f"Graph-aware estimation: step {current_step} (confidence: {graph_match['confidence']})")
                else:
                    # Fallback to part-based matching
                    logger.info("Graph-aware estimation uncertain, falling back to part-based")
                    matched_steps = self._match_steps(detected_parts, plan_data["extracted_steps"])
                    completed_steps = matched_steps["completed_steps"]
                    current_step = matched_steps["current_step"]
            else:
                # No graph available, use traditional matching
                logger.info("No graph found, using part-based step estimation")
                matched_steps = self._match_steps(detected_parts, plan_data["extracted_steps"])
                completed_steps = matched_steps["completed_steps"]
                current_step = matched_steps["current_step"]
            
            total_steps = len(plan_data["extracted_steps"])
            progress_percentage = (len(completed_steps) / total_steps * 100) if total_steps > 0 else 0
            
            # Identify errors (enhanced with graph-aware detection)
            if graph_path.exists():
                errors = self._detect_errors_graph_aware(
                    detected_state,
                    plan_data["extracted_steps"],
                    current_step,
                    graph_data
                )
            else:
                errors = self._detect_errors(
                    detected_parts,
                    plan_data["extracted_steps"],
                    current_step
                )
            
            # Determine next steps using dependency graph
            next_steps = self._determine_next_steps(
                completed_steps,
                plan_data["dependencies"],
                total_steps
            )
            
            # Identify missing and extra parts
            missing_parts, extra_parts = self._identify_part_discrepancies(
                detected_parts,
                plan_data["extracted_steps"],
                current_step
            )
            
            # Calculate overall confidence
            confidence = self._calculate_comparison_confidence(
                detected_state,
                matched_steps,
                errors
            )
            
            result = {
                "completed_steps": completed_steps,
                "current_step": current_step,
                "progress_percentage": round(progress_percentage, 1),
                "total_steps": total_steps,
                "next_steps": next_steps,
                "errors": errors,
                "missing_parts": missing_parts,
                "extra_parts": extra_parts,
                "confidence": confidence,
                "status": "success"
            }
            
            logger.info(f"Comparison complete: {len(completed_steps)}/{total_steps} steps completed ({progress_percentage:.1f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error comparing with plan: {e}")
            return self._create_error_result(str(e))
    
    def _load_plan_data(
        self,
        manual_id: str,
        output_dir: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load Phase 1 plan data (extracted.json, dependencies.json).
        
        Args:
            manual_id: Manual identifier
            output_dir: Output directory path
        
        Returns:
            Dictionary with extracted steps and dependencies, or None if error
        """
        try:
            output_path = Path(output_dir)
            
            # Load extracted step data
            extracted_path = output_path / f"{manual_id}_extracted.json"
            if not extracted_path.exists():
                logger.error(f"Extracted data not found: {extracted_path}")
                return None
            
            with open(extracted_path, 'r', encoding='utf-8') as f:
                extracted_steps = json.load(f)
            
            # Load dependencies
            dependencies_path = output_path / f"{manual_id}_dependencies.json"
            dependencies = {}
            if dependencies_path.exists():
                with open(dependencies_path, 'r', encoding='utf-8') as f:
                    dependencies = json.load(f)
            else:
                logger.warning(f"Dependencies file not found: {dependencies_path}")
            
            logger.info(f"Loaded plan data: {len(extracted_steps)} steps")
            return {
                "extracted_steps": extracted_steps,
                "dependencies": dependencies
            }
            
        except Exception as e:
            logger.error(f"Error loading plan data: {e}")
            return None
    
    def _match_steps(
        self,
        detected_parts: List[Dict[str, Any]],
        expected_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Match detected parts to expected steps to determine progress.
        Uses BOTH simple part matching AND graph-aware subassembly matching.
        
        Args:
            detected_parts: Parts detected in user's assembly
            expected_steps: Steps from Phase 1 extracted data
        
        Returns:
            Dictionary with completed_steps and current_step
        """
        completed_steps = []
        
        # Create a simplified representation of detected parts for matching
        detected_summary = self._create_part_summary(detected_parts)
        
        # Check each step to see if it's completed
        for i, step in enumerate(expected_steps):
            step_number = step.get("step_number") or i + 1
            
            # Skip non-build steps (warnings, info pages, etc.)
            if not step.get("parts_required"):
                continue
            
            # Check if this step's parts are present in detected state
            expected_parts = step.get("parts_required", [])
            
            if self._step_parts_present(expected_parts, detected_summary):
                completed_steps.append(step_number)
        
        # Current step is the next step after the last completed
        current_step = max(completed_steps) + 1 if completed_steps else 1
        
        return {
            "completed_steps": completed_steps,
            "current_step": current_step,
            "match_confidence": len(completed_steps) / len(expected_steps) if expected_steps else 0
        }
    
    def _match_steps_graph_aware(
        self,
        detected_state: Dict[str, Any],
        manual_id: str,
        graph_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Graph-aware step estimation using subassembly detection.
        
        Args:
            detected_state: Full detected state from StateAnalyzer
            manual_id: Manual identifier
            graph_data: Hierarchical graph data
        
        Returns:
            Dictionary with estimated_step, confidence, matched_subassemblies
        """
        # Extract detected subassemblies from state
        assembled_structures = detected_state.get("assembled_structures", [])
        
        if not assembled_structures:
            # Fallback to part-based estimation
            logger.debug("No subassemblies detected, using part-based estimation")
            return None
        
        # Match detected structures to graph subassemblies
        matched_subassemblies = []
        for structure in assembled_structures:
            match = self._match_structure_to_subassembly(structure, graph_data)
            if match:
                matched_subassemblies.append(match)
        
        if not matched_subassemblies:
            return None
        
        # Find the furthest complete subassembly
        max_step = 0
        for match in matched_subassemblies:
            step_created = match.get("step_created", 0)
            completeness = match.get("completeness", 0.0)
            
            # Only count if reasonably complete
            if completeness > 0.6:
                max_step = max(max_step, step_created)
        
        # Estimate next step
        estimated_step = max_step + 1 if max_step > 0 else 1
        
        # Calculate confidence
        confidence = self._calculate_step_estimation_confidence(
            matched_subassemblies,
            assembled_structures
        )
        
        return {
            "estimated_step": estimated_step,
            "confidence": confidence,
            "matched_subassemblies": matched_subassemblies,
            "method": "graph_aware"
        }
    
    def _match_structure_to_subassembly(
        self,
        structure: Dict[str, Any],
        graph_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Match a detected structure to a graph subassembly.
        
        Args:
            structure: Detected assembled structure
            graph_data: Graph data with subassemblies
        
        Returns:
            Matched subassembly with similarity score, or None
        """
        structure_desc = structure.get("description", "").lower()
        structure_completeness = structure.get("completeness", "unknown").lower()
        
        # Get graph subassemblies
        nodes = graph_data.get("nodes", [])
        subassemblies = [n for n in nodes if n.get("type") == "subassembly"]
        
        if not subassemblies:
            return None
        
        # Find best match
        best_match = None
        best_score = 0.0
        
        for subasm in subassemblies:
            subasm_name = subasm.get("name", "").lower()
            subasm_desc = subasm.get("description", "").lower()
            
            # Calculate similarity score
            score = 0.0
            
            # Name overlap
            name_words = set(subasm_name.split())
            desc_words = set(structure_desc.split())
            overlap = len(name_words & desc_words)
            score += overlap * 0.3
            
            # Description overlap
            if subasm_desc and structure_desc:
                desc_overlap = len(set(subasm_desc.split()) & desc_words)
                score += desc_overlap * 0.2
            
            # Completeness match
            if "complete" in structure_completeness:
                score += 0.5
            elif "partial" in structure_completeness:
                score += 0.3
            
            if score > best_score and score > 0.5:  # Threshold
                best_score = score
                best_match = {
                    "subassembly_id": subasm["node_id"],
                    "name": subasm["name"],
                    "step_created": subasm.get("step_created", 1),
                    "similarity": round(score, 2),
                    "completeness": 1.0 if "complete" in structure_completeness else 0.6
                }
        
        return best_match
    
    def _calculate_step_estimation_confidence(
        self,
        matched_subassemblies: List[Dict[str, Any]],
        detected_structures: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence for step estimation."""
        if not matched_subassemblies:
            return 0.0
        
        # Factors:
        # - Number of matches (more is better)
        # - Average similarity scores
        # - Detection coverage
        
        avg_similarity = sum(m.get("similarity", 0) for m in matched_subassemblies) / len(matched_subassemblies)
        match_ratio = len(matched_subassemblies) / max(len(detected_structures), 1)
        
        confidence = (avg_similarity * 0.6 + match_ratio * 0.4)
        
        # Categorize
        if confidence >= 0.8:
            return round(confidence, 2)  # High confidence
        elif confidence >= 0.5:
            return round(confidence, 2)  # Medium confidence
        else:
            return round(confidence, 2)  # Low confidence
        
        return round(confidence, 2)
    
    def _create_part_summary(self, parts: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Create a summary of parts by color and shape for matching.
        
        Args:
            parts: List of detected parts
        
        Returns:
            Dictionary mapping part signature to quantity
        """
        summary = {}
        
        for part in parts:
            color = part.get("color", "unknown").lower()
            shape = part.get("shape", "unknown").lower()
            quantity = part.get("quantity", 1)
            
            # Create signature: "color:shape"
            signature = f"{color}:{shape}"
            summary[signature] = summary.get(signature, 0) + quantity
        
        return summary
    
    def _step_parts_present(
        self,
        expected_parts: List[Dict[str, Any]],
        detected_summary: Dict[str, int]
    ) -> bool:
        """
        Check if expected parts for a step are present in detected parts.
        
        Args:
            expected_parts: Parts required for step
            detected_summary: Summary of detected parts
        
        Returns:
            True if step appears completed
        """
        if not expected_parts:
            return False
        
        # Check if at least 50% of expected parts are detected
        matched = 0
        
        for expected_part in expected_parts:
            color = expected_part.get("color", "unknown").lower()
            shape = expected_part.get("shape", "unknown").lower()
            expected_qty = expected_part.get("quantity", 1)
            
            signature = f"{color}:{shape}"
            detected_qty = detected_summary.get(signature, 0)
            
            # If we have at least some of this part, count as partial match
            if detected_qty > 0:
                matched += min(detected_qty, expected_qty)
        
        # Consider step completed if 50%+ of parts are present
        total_expected = sum(p.get("quantity", 1) for p in expected_parts)
        match_ratio = matched / total_expected if total_expected > 0 else 0
        
        return match_ratio >= 0.5
    
    def _detect_errors(
        self,
        detected_parts: List[Dict[str, Any]],
        expected_steps: List[Dict[str, Any]],
        current_step: int
    ) -> List[Dict[str, str]]:
        """
        Detect assembly errors by comparing detected vs expected state.
        
        Args:
            detected_parts: Detected parts
            expected_steps: Expected steps
            current_step: Current step number
        
        Returns:
            List of error descriptions
        """
        errors = []
        
        # For now, implement basic error detection
        # Can be enhanced with more sophisticated analysis
        
        if not detected_parts:
            errors.append({
                "type": "no_parts_detected",
                "severity": "warning",
                "message": "No parts detected in images. Ensure photos show the assembly clearly."
            })
        
        # Check for parts that shouldn't appear yet
        # (This would require more sophisticated matching logic)
        
        return errors
    
    def _detect_errors_graph_aware(
        self,
        detected_state: Dict[str, Any],
        expected_steps: List[Dict[str, Any]],
        current_step: int,
        graph_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Enhanced error detection using hierarchical graph.
        
        Detects:
        - Wrong part at current step (part sequencing errors)
        - Missing connections
        - Wrong subassembly orientation
        - Incomplete subassemblies
        
        Args:
            detected_state: Full detected state from analyzer
            expected_steps: Expected steps
            current_step: Current step number
            graph_data: Hierarchical graph data
        
        Returns:
            List of error dictionaries with type, severity, message, suggested_fix
        """
        errors = []
        
        detected_parts = detected_state.get("detected_parts", [])
        assembled_structures = detected_state.get("assembled_structures", [])
        connections = detected_state.get("connections", [])
        
        # Priority Check 1: No parts detected
        if not detected_parts and not assembled_structures:
            errors.append({
                "type": "no_parts_detected",
                "severity": "error",
                "message": "No LEGO parts or assemblies detected in the images.",
                "suggested_fix": "Take clearer photos showing your current assembly from multiple angles. Ensure good lighting and the entire assembly is visible."
            })
            return errors  # Stop further checks
        
        # Get expected state for current step
        if current_step <= len(expected_steps):
            expected_step = expected_steps[current_step - 1]
            expected_parts = expected_step.get("parts_required", [])
            
            # Priority Check 2: Part sequencing error (parts from future steps)
            sequencing_errors = self._check_part_sequencing(
                detected_parts, expected_steps, current_step
            )
            errors.extend(sequencing_errors)
            
            # Priority Check 3: Incomplete step (missing required parts)
            missing_parts_errors = self._check_missing_parts(
                detected_parts, expected_parts, expected_step
            )
            errors.extend(missing_parts_errors)
            
            # Priority Check 4: Missing connections
            connection_errors = self._check_missing_connections(
                connections, expected_step, detected_state
            )
            errors.extend(connection_errors)
        
        # Priority Check 5: Incomplete subassemblies
        subassembly_errors = self._check_incomplete_subassemblies(
            assembled_structures, graph_data
        )
        errors.extend(subassembly_errors)
        
        # Priority Check 6: Orientation errors
        orientation_errors = self._check_orientation_errors(
            assembled_structures, graph_data
        )
        errors.extend(orientation_errors)
        
        # Sort by severity and limit to top 3
        severity_order = {"error": 0, "warning": 1, "info": 2}
        errors.sort(key=lambda e: severity_order.get(e["severity"], 3))
        errors = errors[:3]
        
        return errors
    
    def _check_part_sequencing(
        self,
        detected_parts: List[Dict[str, Any]],
        expected_steps: List[Dict[str, Any]],
        current_step: int
    ) -> List[Dict[str, str]]:
        """Check if parts from future steps are present."""
        errors = []
        
        # Get parts that should be present up to current step
        allowed_parts = set()
        future_parts = {}  # part_signature -> step_number
        
        for i, step in enumerate(expected_steps):
            step_num = step.get("step_number", i + 1)
            parts = step.get("parts_required", [])
            
            for part in parts:
                color = part.get("color", "").lower()
                shape = part.get("shape", "").lower()
                signature = f"{color}:{shape}"
                
                if step_num <= current_step:
                    allowed_parts.add(signature)
                else:
                    if signature not in future_parts:
                        future_parts[signature] = step_num
        
        # Check detected parts
        for part in detected_parts:
            color = part.get("color", "").lower()
            shape = part.get("shape", "").lower()
            signature = f"{color}:{shape}"
            
            if signature in future_parts and signature not in allowed_parts:
                future_step = future_parts[signature]
                errors.append({
                    "type": "part_sequencing_error",
                    "severity": "error",
                    "message": f"{part.get('description', part.get('color', '') + ' ' + part.get('shape', ''))} detected, but this part isn't used until step {future_step}. Currently at step {current_step}.",
                    "suggested_fix": f"Remove this part for now. It will be used in step {future_step}."
                })
                break  # Only report first sequencing error
        
        return errors
    
    def _check_missing_parts(
        self,
        detected_parts: List[Dict[str, Any]],
        expected_parts: List[Dict[str, Any]],
        expected_step: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Check if required parts for current step are missing."""
        errors = []
        
        # Create signature map of detected parts
        detected_signatures = {}
        for part in detected_parts:
            color = part.get("color", "").lower()
            shape = part.get("shape", "").lower()
            signature = f"{color}:{shape}"
            detected_signatures[signature] = detected_signatures.get(signature, 0) + part.get("quantity", 1)
        
        # Check expected parts
        missing = []
        for part in expected_parts:
            color = part.get("color", "").lower()
            shape = part.get("shape", "").lower()
            signature = f"{color}:{shape}"
            expected_qty = part.get("quantity", 1)
            detected_qty = detected_signatures.get(signature, 0)
            
            if detected_qty < expected_qty:
                missing.append(part)
        
        if missing and len(missing) <= 3:  # Only report if reasonable number
            part_names = ", ".join([p.get("description", f"{p.get('color', '')} {p.get('shape', '')}") for p in missing])
            errors.append({
                "type": "missing_parts",
                "severity": "warning",
                "message": f"Step appears incomplete. Missing parts: {part_names}",
                "suggested_fix": "Add the missing parts to complete this step before proceeding."
            })
        
        return errors
    
    def _check_missing_connections(
        self,
        detected_connections: List[Dict[str, Any]],
        expected_step: Dict[str, Any],
        detected_state: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Check if required connections are missing."""
        errors = []
        
        # Get expected actions/connections
        expected_actions = expected_step.get("actions", [])
        
        if not expected_actions or not detected_connections:
            return errors  # Can't verify
        
        # Look for connection actions
        expected_connection_count = sum(
            1 for action in expected_actions 
            if action.get("action_verb") in ["attach", "connect"]
        )
        
        detected_connection_count = len(detected_connections)
        
        # If significantly fewer connections detected
        if expected_connection_count > 0 and detected_connection_count < expected_connection_count * 0.5:
            errors.append({
                "type": "missing_connection",
                "severity": "warning",
                "message": f"Some parts may not be properly connected. Expected {expected_connection_count} connections, detected {detected_connection_count}.",
                "suggested_fix": "Ensure all parts are firmly pressed together. Check that studs are properly aligned and connected."
            })
        
        return errors
    
    def _check_incomplete_subassemblies(
        self,
        detected_structures: List[Dict[str, Any]],
        graph_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Check for incomplete subassemblies."""
        errors = []
        
        # Match structures to subassemblies
        for structure in detected_structures:
            completeness = structure.get("completeness", "").lower()
            
            if any(kw in completeness for kw in ["partial", "incomplete"]):
                structure_desc = structure.get("description", "unknown structure")
                errors.append({
                    "type": "incomplete_subassembly",
                    "severity": "warning",
                    "message": f"{structure_desc} appears incomplete.",
                    "suggested_fix": "Complete this subassembly before attaching it or moving to the next step."
                })
                break  # Only report first incomplete subassembly
        
        return errors
    
    def _check_orientation_errors(
        self,
        detected_structures: List[Dict[str, Any]],
        graph_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Check for orientation errors in subassemblies."""
        errors = []
        
        # This is difficult without 6D pose, so we look for clues in descriptions
        for structure in detected_structures:
            desc = structure.get("description", "").lower()
            
            # Look for orientation keywords that suggest issues
            if any(kw in desc for kw in ["upside down", "backwards", "rotated", "inverted"]):
                structure_name = structure.get("description", "subassembly")
                errors.append({
                    "type": "orientation_error",
                    "severity": "warning",
                    "message": f"{structure_name} may be oriented incorrectly.",
                    "suggested_fix": "Check the manual diagram to verify the correct orientation of this subassembly."
                })
                break  # Only report first orientation issue
        
        return errors
    
    def _determine_next_steps(
        self,
        completed_steps: List[int],
        dependencies: Dict[str, Any],
        total_steps: int
    ) -> List[int]:
        """
        Determine which steps should be done next based on dependencies.
        
        Args:
            completed_steps: Steps already completed
            dependencies: Dependency graph
            total_steps: Total number of steps
        
        Returns:
            List of next step numbers
        """
        next_steps = []
        
        # Simple approach: next step is current_step
        if completed_steps:
            next_step = max(completed_steps) + 1
            if next_step <= total_steps:
                next_steps.append(next_step)
        else:
            # Start from step 1
            next_steps.append(1)
        
        # Could enhance with dependency graph analysis for parallel steps
        
        return next_steps
    
    def _identify_part_discrepancies(
        self,
        detected_parts: List[Dict[str, Any]],
        expected_steps: List[Dict[str, Any]],
        current_step: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Identify missing and extra parts.
        
        Args:
            detected_parts: Detected parts
            expected_steps: Expected steps
            current_step: Current step number
        
        Returns:
            Tuple of (missing_parts, extra_parts)
        """
        missing_parts = []
        extra_parts = []
        
        # Get expected parts up to current step
        if current_step <= len(expected_steps):
            current_step_data = expected_steps[current_step - 1]
            expected_parts = current_step_data.get("parts_required", [])
            
            # Simple check: are current step parts detected?
            detected_summary = self._create_part_summary(detected_parts)
            
            for part in expected_parts:
                color = part.get("color", "unknown").lower()
                shape = part.get("shape", "unknown").lower()
                signature = f"{color}:{shape}"
                
                if signature not in detected_summary:
                    missing_parts.append({
                        "description": part.get("description", "Unknown part"),
                        "color": part.get("color", "unknown"),
                        "shape": part.get("shape", "unknown"),
                        "quantity": part.get("quantity", 1)
                    })
        
        return missing_parts, extra_parts
    
    def _calculate_comparison_confidence(
        self,
        detected_state: Dict[str, Any],
        matched_steps: Dict[str, Any],
        errors: List[Dict[str, str]]
    ) -> float:
        """
        Calculate confidence score for the comparison.
        
        Args:
            detected_state: Detected state from analyzer
            matched_steps: Step matching results
            errors: Detected errors
        
        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence from detection
        detection_confidence = detected_state.get("confidence", 0.5)
        
        # Matching confidence
        matching_confidence = matched_steps.get("match_confidence", 0.5)
        
        # Error penalty
        error_penalty = len(errors) * 0.1
        
        # Combined confidence
        confidence = (detection_confidence + matching_confidence) / 2
        confidence = max(0.0, min(1.0, confidence - error_penalty))
        
        return round(confidence, 2)
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            "completed_steps": [],
            "current_step": 1,
            "progress_percentage": 0.0,
            "total_steps": 0,
            "next_steps": [1],
            "errors": [{
                "type": "comparison_error",
                "severity": "error",
                "message": error_message
            }],
            "missing_parts": [],
            "extra_parts": [],
            "confidence": 0.0,
            "status": "error"
        }


# Singleton instance
_state_comparator_instance = None


def get_state_comparator() -> StateComparator:
    """Get StateComparator singleton instance."""
    global _state_comparator_instance
    if _state_comparator_instance is None:
        _state_comparator_instance = StateComparator()
    return _state_comparator_instance




