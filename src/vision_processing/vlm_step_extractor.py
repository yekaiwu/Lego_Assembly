"""
VLM Step Extractor: Uses Vision-Language Models to extract structured information
from LEGO instruction steps. Manages multiple VLM providers with fallback logic.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..api.litellm_vlm import UnifiedVLMClient
from ..utils.config import get_config
from ..utils.cache import get_cache
from .build_memory import BuildMemory
from .token_budget import TokenBudgetManager

class VLMStepExtractor:
    """Extracts structured step information using VLMs with fallback support."""

    def __init__(self):
        config = get_config()
        self.ingestion_vlm = config.models.ingestion_vlm
        self.ingestion_secondary_vlm = config.models.ingestion_secondary_vlm
        self.ingestion_fallback_vlm = config.models.ingestion_fallback_vlm
        self.cache = get_cache()  # Initialize cache for step processing
        self.build_memory: Optional[BuildMemory] = None  # Context-aware memory
        self.token_budget: Optional[TokenBudgetManager] = None  # Token management

        # Cache for unified VLM clients (created on-demand)
        self._client_cache = {}

        logger.info(f"VLM Step Extractor initialized with ingestion VLM: {self.ingestion_vlm}")

    def _get_client(self, vlm_name: str) -> UnifiedVLMClient:
        """
        Get or create a VLM client for the given model name.

        Args:
            vlm_name: LiteLLM model identifier (e.g., "gemini/gemini-2.5-flash", "gpt-4o")

        Returns:
            UnifiedVLMClient instance
        """
        if vlm_name not in self._client_cache:
            self._client_cache[vlm_name] = UnifiedVLMClient(vlm_name)
        return self._client_cache[vlm_name]

    def initialize_memory(self, main_build: str, window_size: int = 2, max_tokens: int = 1_000_000):
        """
        Initialize memory systems for context-aware extraction.

        Args:
            main_build: Name of the main build being assembled
            window_size: Number of previous steps in sliding window (default: 2)
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
    ) -> List[Dict[str, Any]]:
        """
        Extract structured information from instruction page(s).

        Pages may contain multiple steps. Returns array of all steps found.
        If build_memory is initialized, uses context-aware extraction.

        Args:
            image_paths: List of paths to step images
            step_number: Optional step number (usually None to let VLM detect)
            use_primary: Whether to use primary VLM (True) or try all (False)
            cache_context: Optional context to differentiate cache entries between manuals

        Returns:
            List of extracted step dictionaries (1 or more steps per page)
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

            results = self._extract_with_vlm_and_context(
                self.ingestion_vlm,
                image_paths,
                step_number,
                context,
                cache_context
            )

            # Update memory with each extracted step
            if self.build_memory:
                for result in results:
                    if "error" not in result:
                        self.build_memory.add_step(result)

            return results
        else:
            return self._extract_with_fallback(image_paths, step_number, cache_context)
    
    def _extract_with_vlm(
        self,
        vlm_name: str,
        image_paths: List[str],
        step_number: Optional[int],
        cache_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract using a specific VLM (without context)."""
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
    ) -> List[Dict[str, Any]]:
        """
        Extract using a specific VLM with context.
        Returns array of steps (VLM client now returns lists).
        """
        try:
            client = self._get_client(vlm_name)
        except Exception as e:
            logger.error(f"Failed to create client for {vlm_name}: {e}")
            raise ValueError(f"Unknown VLM: {vlm_name}") from e

        logger.info(f"Extracting step info using {vlm_name}{' with context' if context else ''}")

        try:
            # Build enhanced prompt with context if available
            if context:
                prompt = self._build_context_aware_prompt(step_number, context)

                # Use context-aware extraction if client supports it
                if hasattr(client, 'extract_step_info_with_context'):
                    results = client.extract_step_info_with_context(
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
                        results = client.extract_step_info(image_paths, step_number, cache_context=cache_context)
                    else:
                        results = client.extract_step_info(image_paths, step_number)
            else:
                # No context, use standard extraction
                if hasattr(client, 'extract_step_info'):
                    import inspect
                    sig = inspect.signature(client.extract_step_info)
                    if 'cache_context' in sig.parameters:
                        results = client.extract_step_info(image_paths, step_number, cache_context=cache_context)
                    else:
                        results = client.extract_step_info(image_paths, step_number)
                else:
                    results = client.extract_step_info(image_paths, step_number)

            # Validate each step in the results array
            validated_results = []
            for result in results:
                if self._validate_extraction(result):
                    validated_results.append(result)
                else:
                    logger.warning(f"Step {result.get('step_number', 'unknown')} failed validation")
                    validated_results.append({"error": "Validation failed", "raw_result": result})

            if validated_results:
                logger.info(f"Successfully extracted {len(validated_results)} step(s) using {vlm_name}")
                return validated_results
            else:
                return [{"error": "No valid steps extracted"}]

        except Exception as e:
            logger.error(f"Error extracting with {vlm_name}: {e}")
            return [{"error": str(e)}]

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
IMPORTANT: This page may contain ONE or MORE assembly steps. Analyze carefully and extract ALL steps shown.

{step_context}Return a JSON ARRAY containing ALL steps found on this page:

[
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
]

IMPORTANT INSTRUCTIONS:
1. Look carefully - the page may show 1, 2, or more steps (typically identified by step numbers in the image)
2. Return ALL steps as an array, even if there's only one step
3. Consider the build context and recent steps when analyzing
4. If a step references previous work, identify which step in "context_references"
5. Determine if each step starts a new subassembly or continues the current one
6. Focus on what's NEW in each step, not what was already built

RESPONSE CONSTRAINTS (CRITICAL):
- Keep descriptions CONCISE (max 10-15 words per field)
- Use short phrases, not full sentences
- Prioritize accuracy over detail
- If information is unclear, mark as null or "unclear"
- Limit "actions" array to 3 most important actions per step
- Keep "existing_assembly" under 20 words
- Return ONLY the JSON array, no additional text

Example formats:
- One step on page: [{{step 1 data}}]
- Two steps on page: [{{step 1 data}}, {{step 2 data}}]
""")

        return "\n".join(prompt_parts)
    
    def _extract_with_fallback(
        self,
        image_paths: List[str],
        step_number: Optional[int],
        cache_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract with fallback logic through multiple VLMs."""
        vlm_sequence = [self.ingestion_vlm, self.ingestion_secondary_vlm, self.ingestion_fallback_vlm]

        for vlm_name in vlm_sequence:
            logger.info(f"Trying VLM: {vlm_name}")

            try:
                results = self._extract_with_vlm(vlm_name, image_paths, step_number, cache_context)

                # If extraction succeeded, return (results is now an array)
                # Check if any result has an error
                if not any("error" in r for r in results):
                    return results

                logger.warning(f"{vlm_name} failed, trying next VLM...")

            except Exception as e:
                logger.error(f"{vlm_name} raised exception: {e}")
                continue

        # All VLMs failed - return error as array for consistency
        logger.error("All VLMs failed to extract step information")
        return [{
            "error": "All VLMs failed",
            "step_number": step_number,
            "image_paths": image_paths
        }]
    
    def _validate_extraction(self, result: Dict[str, Any]) -> bool:
        """Validate that extraction result contains required fields."""
        if "error" in result:
            return False

        # Must have a step number (can be null, but field must exist)
        if "step_number" not in result:
            logger.warning("Missing step_number field")
            return False

        # Check for key fields - allow them to be empty lists/null
        required_fields = ["parts_required", "actions"]

        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing required field: {field}")
                return False

            # Ensure field is not None (should be at least an empty list/string)
            if result[field] is None:
                logger.warning(f"Field {field} is None, setting to empty list")
                result[field] = []

        return True

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
        try:
            client = self._get_client(self.ingestion_vlm)
        except Exception as e:
            logger.error(f"Primary VLM {self.ingestion_vlm} not available: {e}")
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

