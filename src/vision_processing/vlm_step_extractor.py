"""
VLM Step Extractor: Uses Vision-Language Models to extract structured information
from LEGO instruction steps. Manages multiple VLM providers with fallback logic.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..api.qwen_vlm import QwenVLMClient
from ..api.deepseek_api import DeepSeekClient
from ..api.kimi_api import KimiClient
from ..api.openai_api import OpenAIVisionClient
from ..api.anthropic_api import AnthropicVisionClient
from ..api.gemini_api import GeminiVisionClient
from ..utils.config import get_config
from ..utils.cache import get_cache
from .build_memory import BuildMemory
from .token_budget import TokenBudgetManager

class VLMStepExtractor:
    """Extracts structured step information using VLMs with fallback support."""

    def __init__(self):
        config = get_config()
        self.primary_vlm = config.models.primary_vlm
        self.secondary_vlm = config.models.secondary_vlm
        self.fallback_vlm = config.models.fallback_vlm
        self.cache = get_cache()  # Initialize cache for batch processing
        self.build_memory: Optional[BuildMemory] = None  # Context-aware memory
        self.token_budget: Optional[TokenBudgetManager] = None  # Token management
        
        # Initialize VLM clients
        self.clients = {
            # Chinese VLMs
            "qwen-vl-max": QwenVLMClient(),
            "qwen-vl-plus": QwenVLMClient(),
            "deepseek-v2": DeepSeekClient(),
            "kimi-vision": KimiClient(),

            # International VLMs
            "gpt-4o": OpenAIVisionClient(),
            "gpt-4o-mini": OpenAIVisionClient(),
            "gpt-4-vision": OpenAIVisionClient(),
            "gpt-4-turbo": OpenAIVisionClient(),
            "claude-3-opus": AnthropicVisionClient(),
            "claude-3-sonnet": AnthropicVisionClient(),
            "claude-3-5-sonnet": AnthropicVisionClient(),
            "claude-3-haiku": AnthropicVisionClient(),
            "gemini-2.5-flash": GeminiVisionClient(),
            "gemini-2.0-flash": GeminiVisionClient(),
            "gemini-2.0-flash-exp": GeminiVisionClient(),
            "gemini-2.0-flash-thinking-exp": GeminiVisionClient(),
            "gemini-flash-latest": GeminiVisionClient(),
            "gemini-1.5-pro": GeminiVisionClient(),
            "gemini-1.5-pro-latest": GeminiVisionClient(),
            "gemini-1.5-flash": GeminiVisionClient(),
            "gemini-1.5-flash-latest": GeminiVisionClient(),
            "gemini-pro-vision": GeminiVisionClient(),
        }
        
        logger.info(f"VLM Step Extractor initialized with primary: {self.primary_vlm}")

    def initialize_memory(self, main_build: str, window_size: int = 5, max_tokens: int = 1_000_000):
        """
        Initialize memory systems for context-aware extraction.

        Args:
            main_build: Name of the main build being assembled
            window_size: Number of previous steps in sliding window (default: 5)
            max_tokens: Maximum context window size (default: 1M for Gemini)
        """
        self.build_memory = BuildMemory(main_build, window_size)
        self.token_budget = TokenBudgetManager(max_tokens)
        logger.info(f"Initialized context-aware memory (window={window_size}, main_build={main_build})")

    def extract_step(
        self,
        image_paths: List[str],
        step_number: Optional[int] = None,
        use_primary: bool = True,
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured information from a single step.

        NEW: If build_memory is initialized, uses context-aware extraction.

        Args:
            image_paths: List of paths to step images
            step_number: Optional step number
            use_primary: Whether to use primary VLM (True) or try all (False)
            cache_context: Optional context to differentiate cache entries between manuals

        Returns:
            Extracted step information
        """
        if use_primary:
            # Get context from memory systems if available
            context = None
            if self.build_memory:
                context = self.build_memory.get_full_context()

                # Check token budget if available
                if self.token_budget:
                    budget_check = self.token_budget.check_budget(
                        image_count=len(image_paths),
                        sliding_window_size=len(self.build_memory.sliding_window.window),
                        long_term_memory_size=self.token_budget.estimate_tokens(context["long_term"]),
                        prompt_size=500
                    )

                    if not budget_check["fits"]:
                        logger.warning(f"Token budget exceeded: {budget_check['estimated_usage']} > {budget_check['available']}")
                        # Auto-adjust window size
                        if "reduce_window_to" in budget_check["recommendations"]:
                            new_size = budget_check["recommendations"]["reduce_window_to"]
                            from collections import deque
                            self.build_memory.sliding_window.window = deque(
                                list(self.build_memory.sliding_window.window)[-new_size:],
                                maxlen=new_size
                            )
                            # Re-get context with adjusted window
                            context = self.build_memory.get_full_context()

            result = self._extract_with_vlm_and_context(
                self.primary_vlm,
                image_paths,
                step_number,
                context,
                cache_context
            )

            # Update memory with extraction result if no error
            if self.build_memory and "error" not in result:
                self.build_memory.add_step(result)

            return result
        else:
            return self._extract_with_fallback(image_paths, step_number, cache_context)
    
    def _extract_with_vlm(
        self,
        vlm_name: str,
        image_paths: List[str],
        step_number: Optional[int],
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract using a specific VLM (legacy, without context)."""
        return self._extract_with_vlm_and_context(
            vlm_name, image_paths, step_number, None, cache_context
        )

    def _extract_with_vlm_and_context(
        self,
        vlm_name: str,
        image_paths: List[str],
        step_number: Optional[int],
        context: Optional[Dict[str, str]],
        cache_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract using a specific VLM with context.

        NEW: Passes context to VLM for context-aware extraction.
        """
        client = self.clients.get(vlm_name)

        if not client:
            raise ValueError(f"Unknown VLM: {vlm_name}")

        logger.info(f"Extracting step info using {vlm_name}{' with context' if context else ''}")

        try:
            # Build enhanced prompt with context if available
            if context:
                prompt = self._build_context_aware_prompt(step_number, context)

                # Use context-aware extraction if client supports it
                if hasattr(client, 'extract_step_info_with_context'):
                    result = client.extract_step_info_with_context(
                        image_paths,
                        step_number,
                        custom_prompt=prompt,
                        cache_context=cache_context
                    )
                else:
                    # Fallback: client doesn't support custom prompts
                    logger.warning(f"{vlm_name} doesn't support context-aware extraction, using standard extraction")
                    import inspect
                    sig = inspect.signature(client.extract_step_info)
                    if 'cache_context' in sig.parameters:
                        result = client.extract_step_info(image_paths, step_number, cache_context=cache_context)
                    else:
                        result = client.extract_step_info(image_paths, step_number)
            else:
                # No context, use standard extraction
                if hasattr(client, 'extract_step_info'):
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

    def _build_context_aware_prompt(
        self,
        step_number: Optional[int],
        context: Dict[str, str]
    ) -> str:
        """
        Build enhanced prompt with context.

        Includes:
        - Sliding window (recent steps)
        - Long-term memory (build state)
        - Instructions to relate to previous work
        """
        step_context = f"Step {step_number}: " if step_number else ""

        prompt_parts = []

        # Add long-term context (high-level overview)
        if context.get("long_term"):
            prompt_parts.append(f"""
BUILD CONTEXT:
{context['long_term']}
""")

        # Add sliding window context (recent steps)
        if context.get("sliding_window"):
            prompt_parts.append(f"""
{context['sliding_window']}
""")

        # Main extraction instructions
        prompt_parts.append(f"""
CURRENT TASK:
Analyze {step_context}this LEGO instruction image.

Extract detailed information and return ONLY valid JSON:

{{
  "step_number": <number or null>,
  "parts_required": [
    {{
      "description": "part description",
      "color": "color name",
      "shape": "brick type and dimensions",
      "part_id": "LEGO part ID if visible",
      "quantity": <number>
    }}
  ],
  "existing_assembly": "description of already assembled parts shown",
  "new_parts_to_add": [
    "description of each new part being added in this step"
  ],
  "actions": [
    {{
      "action_verb": "attach|connect|place|align|rotate",
      "target": "what is being attached",
      "destination": "where it's being attached",
      "orientation": "directional cues"
    }}
  ],
  "spatial_relationships": {{
    "position": "top|bottom|left|right|front|back|center",
    "rotation": "rotation description if any",
    "alignment": "alignment instructions"
  }},
  "dependencies": "which previous steps are prerequisites",
  "notes": "any special instructions or warnings",

  "subassembly_hint": {{
    "is_new_subassembly": true/false,
    "name": "descriptive name if new (e.g., 'wheel_assembly')",
    "description": "what is being built (e.g., '4-wheel chassis')",
    "continues_previous": true/false
  }},

  "context_references": {{
    "references_previous_steps": true/false,
    "which_steps": [list of step numbers referenced],
    "reference_description": "what is being referenced (e.g., 'the base from step 4')"
  }}
}}

IMPORTANT INSTRUCTIONS:
1. Consider the build context and recent steps when analyzing this image
2. If this step references previous work (e.g., "attach to base"), identify which step in "context_references"
3. Determine if this is starting a new subassembly or continuing the current one
4. If the image shows parts already assembled in previous steps, mention them in "existing_assembly"
5. Focus on what's NEW in this step, not what was already built

Be detailed and precise. If information is unclear, mark as null or "unclear".
""")

        return "\n".join(prompt_parts)
    
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
        cache_context: Optional[str] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract information from multiple steps in batch.
        Groups steps to reduce API calls and avoid rate limits.
        
        Args:
            step_images: List of step image lists
            use_primary: Whether to use primary VLM only
            cache_context: Optional context to differentiate cache entries between manuals
            batch_size: Number of steps to process in a single API call (default: 10)
        
        Returns:
            List of extracted step information
        """
        logger.info(f"Batch extracting {len(step_images)} steps (batch size: {batch_size})...")
        
        results = []
        total_steps = len(step_images)
        
        # Process in batches to reduce API calls
        for batch_start in range(0, total_steps, batch_size):
            batch_end = min(batch_start + batch_size, total_steps)
            batch_steps = step_images[batch_start:batch_end]
            
            logger.info(f"Processing batch: steps {batch_start + 1}-{batch_end} of {total_steps}")
            
            # Try batch processing first (more efficient)
            batch_result = self._extract_batch_multi_step(
                batch_steps, 
                start_step=batch_start + 1,
                use_primary=use_primary,
                cache_context=cache_context
            )
            
            if batch_result and len(batch_result) == len(batch_steps):
                # Batch processing succeeded
                results.extend(batch_result)
                logger.info(f"✓ Batch processed {len(batch_result)} steps successfully")
            else:
                # Fallback to individual processing if batch fails
                logger.warning("Batch processing failed, falling back to individual extraction")
                for i, image_paths in enumerate(batch_steps):
                    step_num = batch_start + i + 1
                    logger.info(f"  Processing step {step_num}/{total_steps} individually")
                    result = self.extract_step(
                        image_paths, 
                        step_number=step_num, 
                        use_primary=use_primary, 
                        cache_context=cache_context
                    )
                    results.append(result)
        
        logger.info(f"Batch extraction complete. Extracted {len(results)} steps")
        return results
    
    def _extract_batch_multi_step(
        self,
        batch_steps: List[List[str]],
        start_step: int,
        use_primary: bool,
        cache_context: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract multiple steps in a single API call.
        Combines all images from multiple steps and asks VLM to analyze them together.
        """
        vlm_name = self.primary_vlm if use_primary else self.fallback_vlm
        client = self.clients.get(vlm_name)
        
        if not client:
            logger.warning(f"VLM {vlm_name} not available for batch processing")
            return None
        
        try:
            # Flatten all images from all steps in this batch
            all_images = []
            for step_images in batch_steps:
                all_images.extend(step_images)
            
            # Check cache for this batch
            cache_suffix = f":{cache_context}" if cache_context else ""
            cache_key = f"{vlm_name}:batch_{start_step}-{start_step + len(batch_steps) - 1}:{','.join(all_images[:3])}{cache_suffix}"
            cached = self.cache.get(vlm_name, cache_key, all_images) if hasattr(self, 'cache') else None
            
            if cached:
                logger.info(f"✓ Cache hit for batch {start_step}-{start_step + len(batch_steps) - 1}")
                return cached
            
            # Build batch prompt
            batch_prompt = self._build_batch_prompt(batch_steps, start_step)
            
            # Call VLM with all images (if client supports it)
            logger.info(f"Calling {vlm_name} for batch of {len(batch_steps)} steps ({len(all_images)} images)")
            
            # Check what parameters the client's extract_step_info supports
            import inspect
            sig = inspect.signature(client.extract_step_info)
            params = sig.parameters
            
            # Build kwargs based on what the client accepts
            kwargs = {
                "image_paths": all_images,
                "step_number": None,  # Indicate this is a batch
            }
            
            # Only add parameters if the client supports them
            if 'use_json_mode' in params:
                kwargs['use_json_mode'] = True
            if 'cache_context' in params:
                kwargs['cache_context'] = f"{cache_context}_batch_{start_step}" if cache_context else f"batch_{start_step}"
            
            # Call the client's extract_step_info with appropriate params
            result = client.extract_step_info(**kwargs)
            
            # Parse batch result - expecting array of step extractions
            if isinstance(result, dict) and "steps" in result:
                steps_data = result["steps"]
            elif isinstance(result, list):
                steps_data = result
            else:
                # Single result returned - can't use for batch
                logger.warning("VLM returned single result instead of batch, falling back")
                return None
            
            # Cache the batch result
            if hasattr(self, 'cache'):
                self.cache.set(vlm_name, cache_key, steps_data, all_images)
            
            return steps_data
            
        except Exception as e:
            logger.warning(f"Batch extraction failed: {e}")
            return None
    
    def _build_batch_prompt(self, batch_steps: List[List[str]], start_step: int) -> str:
        """Build a prompt for processing multiple steps at once."""
        num_steps = len(batch_steps)
        return f"""Analyze these {num_steps} consecutive LEGO instruction steps (steps {start_step} to {start_step + num_steps - 1}).
Each step is shown with one or more images. Extract information for ALL steps and return as a JSON array.

Return format:
{{
  "steps": [
    {{
      "step_number": {start_step},
      "parts_required": [...],
      "actions": [...],
      ... (same structure as before)
    }},
    {{
      "step_number": {start_step + 1},
      ...
    }},
    ... (for all {num_steps} steps)
  ]
}}"""
    
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

