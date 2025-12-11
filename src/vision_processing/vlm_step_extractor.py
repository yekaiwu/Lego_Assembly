"""
VLM Step Extractor: Uses Vision-Language Models to extract structured information
from LEGO instruction steps. Manages multiple VLM providers with fallback logic.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..api.qwen_vlm import QwenVLMClient
from ..api.deepseek_api import DeepSeekClient
from ..api.kimi_api import KimiClient
from ..utils.config import get_config

class VLMStepExtractor:
    """Extracts structured step information using VLMs with fallback support."""
    
    def __init__(self):
        config = get_config()
        self.primary_vlm = config.models.primary_vlm
        self.secondary_vlm = config.models.secondary_vlm
        self.fallback_vlm = config.models.fallback_vlm
        
        # Initialize VLM clients
        self.clients = {
            "qwen-vl-max": QwenVLMClient(),
            "qwen-vl-plus": QwenVLMClient(),
            "deepseek-v2": DeepSeekClient(),
            "kimi-vision": KimiClient()
        }
        
        logger.info(f"VLM Step Extractor initialized with primary: {self.primary_vlm}")
    
    def extract_step(
        self, 
        image_paths: List[str], 
        step_number: Optional[int] = None,
        use_primary: bool = True,
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured information from a single step.
        
        Args:
            image_paths: List of paths to step images
            step_number: Optional step number
            use_primary: Whether to use primary VLM (True) or try all (False)
            cache_context: Optional context to differentiate cache entries between manuals
        
        Returns:
            Extracted step information
        """
        if use_primary:
            return self._extract_with_vlm(self.primary_vlm, image_paths, step_number, cache_context)
        else:
            return self._extract_with_fallback(image_paths, step_number, cache_context)
    
    def _extract_with_vlm(
        self, 
        vlm_name: str, 
        image_paths: List[str], 
        step_number: Optional[int],
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract using a specific VLM."""
        client = self.clients.get(vlm_name)
        
        if not client:
            raise ValueError(f"Unknown VLM: {vlm_name}")
        
        logger.info(f"Extracting step info using {vlm_name}")
        
        try:
            # Pass cache_context to the client if supported
            if hasattr(client, 'extract_step_info'):
                # Check if method accepts cache_context parameter
                import inspect
                sig = inspect.signature(client.extract_step_info)
                if 'cache_context' in sig.parameters:
                    result = client.extract_step_info(image_paths, step_number, cache_context=cache_context)
                else:
                    result = client.extract_step_info(image_paths, step_number)
            else:
                result = client.extract_step_info(image_paths, step_number)
            
            # Validate result
            if self._validate_extraction(result):
                logger.info(f"Successfully extracted step info using {vlm_name}")
                return result
            else:
                logger.warning(f"Extraction from {vlm_name} failed validation")
                return {"error": "Validation failed", "raw_result": result}
        
        except Exception as e:
            logger.error(f"Error extracting with {vlm_name}: {e}")
            return {"error": str(e)}
    
    def _extract_with_fallback(
        self, 
        image_paths: List[str], 
        step_number: Optional[int],
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract with fallback logic through multiple VLMs."""
        vlm_sequence = [self.primary_vlm, self.secondary_vlm, self.fallback_vlm]
        
        for vlm_name in vlm_sequence:
            logger.info(f"Trying VLM: {vlm_name}")
            
            try:
                result = self._extract_with_vlm(vlm_name, image_paths, step_number, cache_context)
                
                # If extraction succeeded, return
                if "error" not in result:
                    return result
                
                logger.warning(f"{vlm_name} failed, trying next VLM...")
            
            except Exception as e:
                logger.error(f"{vlm_name} raised exception: {e}")
                continue
        
        # All VLMs failed
        logger.error("All VLMs failed to extract step information")
        return {
            "error": "All VLMs failed",
            "step_number": step_number,
            "image_paths": image_paths
        }
    
    def _validate_extraction(self, result: Dict[str, Any]) -> bool:
        """Validate that extraction result contains required fields."""
        if "error" in result:
            return False
        
        # Check for key fields
        required_fields = ["parts_required", "actions"]
        
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing required field: {field}")
                return False
        
        return True
    
    def batch_extract(
        self, 
        step_images: List[List[str]], 
        use_primary: bool = True,
        cache_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract information from multiple steps in batch.
        
        Args:
            step_images: List of step image lists
            use_primary: Whether to use primary VLM only
            cache_context: Optional context to differentiate cache entries between manuals
        
        Returns:
            List of extracted step information
        """
        logger.info(f"Batch extracting {len(step_images)} steps...")
        
        results = []
        for i, image_paths in enumerate(step_images):
            logger.info(f"Processing step {i + 1}/{len(step_images)}")
            result = self.extract_step(image_paths, step_number=i + 1, use_primary=use_primary, cache_context=cache_context)
            results.append(result)
        
        logger.info(f"Batch extraction complete. Extracted {len(results)} steps")
        return results
    
    def refine_extraction(
        self, 
        initial_result: Dict[str, Any], 
        image_paths: List[str],
        refinement_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refine an initial extraction with additional context or corrections.
        
        Args:
            initial_result: Initial extraction result
            image_paths: Original step images
            refinement_prompt: Optional custom refinement instructions
        
        Returns:
            Refined extraction result
        """
        logger.info("Refining extraction...")
        
        # Use primary VLM for refinement
        client = self.clients.get(self.primary_vlm)
        
        if not client:
            logger.error(f"Primary VLM {self.primary_vlm} not available")
            return initial_result
        
        # TODO: Implement refinement logic
        # This would involve sending the initial result back to VLM with refinement instructions
        
        logger.info("Refinement not yet implemented, returning initial result")
        return initial_result
    
    def extract_part_identifiers(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract and normalize part identifiers from extraction result.
        
        Args:
            result: Extraction result
        
        Returns:
            List of normalized part identifiers
        """
        parts = result.get("parts_required", [])
        
        identifiers = []
        for part in parts:
            identifier = {
                "description": part.get("description", ""),
                "color": part.get("color", "").lower(),
                "shape": part.get("shape", ""),
                "part_id": part.get("part_id", None),
                "quantity": part.get("quantity", 1)
            }
            identifiers.append(identifier)
        
        return identifiers

