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
            
            # Match detected state to expected states
            matched_steps = self._match_steps(detected_parts, plan_data["extracted_steps"])
            
            # Determine progress
            completed_steps = matched_steps["completed_steps"]
            current_step = matched_steps["current_step"]
            total_steps = len(plan_data["extracted_steps"])
            progress_percentage = (len(completed_steps) / total_steps * 100) if total_steps > 0 else 0
            
            # Identify errors
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


