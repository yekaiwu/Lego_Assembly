"""
Prompt Management System for VLM interactions.
Handles prompt loading, versioning, template variables, and usage tracking.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import re


class PromptManager:
    """Manages VLM prompts with versioning and template support."""
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt files (defaults to ./prompts)
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent / "prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self.prompts_cache = {}
        self.active_versions = {}  # prompt_name -> active version
        
        # Load default active versions
        self._load_active_versions()
        
        logger.info(f"PromptManager initialized with prompts from {self.prompts_dir}")
    
    def get_prompt(
        self,
        prompt_name: str,
        version: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get a prompt by name with optional versioning and context injection.
        
        Args:
            prompt_name: Name of prompt (e.g., 'state_analysis', 'error_detection')
            version: Optional version (e.g., 'v2'). If None, uses active version.
            context: Optional context dictionary for template variable substitution
        
        Returns:
            Formatted prompt string
        
        Example:
            prompt = manager.get_prompt(
                'state_analysis',
                context={'manual_id': '6454922', 'expected_step': 5}
            )
        """
        # Determine version
        if version is None:
            version = self.active_versions.get(prompt_name, 'v1')
        
        # Build cache key
        cache_key = f"{prompt_name}:{version}"
        
        # Check cache
        if cache_key in self.prompts_cache:
            template = self.prompts_cache[cache_key]
        else:
            # Load from file
            template = self._load_prompt_file(prompt_name, version)
            self.prompts_cache[cache_key] = template
        
        # Apply context substitution if provided
        if context:
            return self._substitute_variables(template, context)
        
        return template
    
    def _load_prompt_file(self, prompt_name: str, version: str) -> str:
        """
        Load prompt file from disk.
        
        Looks for:
        1. {prompt_name}_{version}.txt (e.g., state_analysis_v2.txt)
        2. {prompt_name}_prompt.txt (default, unversioned)
        """
        # Try versioned file first
        versioned_path = self.prompts_dir / f"{prompt_name}_{version}.txt"
        if versioned_path.exists():
            logger.debug(f"Loading prompt: {versioned_path.name}")
            return versioned_path.read_text(encoding='utf-8')
        
        # Try default file
        default_path = self.prompts_dir / f"{prompt_name}_prompt.txt"
        if default_path.exists():
            logger.debug(f"Loading prompt: {default_path.name}")
            return default_path.read_text(encoding='utf-8')
        
        # Fallback: just prompt_name.txt
        fallback_path = self.prompts_dir / f"{prompt_name}.txt"
        if fallback_path.exists():
            logger.debug(f"Loading prompt: {fallback_path.name}")
            return fallback_path.read_text(encoding='utf-8')
        
        logger.error(f"Prompt file not found: {prompt_name} (version: {version})")
        raise FileNotFoundError(f"Prompt '{prompt_name}' not found in {self.prompts_dir}")
    
    def _substitute_variables(self, template: str, context: Dict[str, Any]) -> str:
        """
        Substitute template variables with context values.
        
        Supports {{variable}} syntax.
        
        Example:
            template: "Manual ID: {{manual_id}}, Step: {{current_step}}"
            context: {"manual_id": "6454922", "current_step": 5}
            result: "Manual ID: 6454922, Step: 5"
        """
        result = template
        
        # Find all {{variable}} patterns
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, template)
        
        for var_name in matches:
            if var_name in context:
                value = context[var_name]
                
                # Handle different types
                if isinstance(value, (dict, list)):
                    # For complex types, use JSON representation
                    value_str = json.dumps(value, indent=2, ensure_ascii=False)
                else:
                    value_str = str(value)
                
                # Replace
                result = result.replace(f"{{{{{var_name}}}}}", value_str)
            else:
                logger.warning(f"Template variable '{var_name}' not found in context")
                # Leave as-is if not in context
        
        return result
    
    def set_active_version(self, prompt_name: str, version: str):
        """
        Set the active version for a prompt.
        
        Args:
            prompt_name: Prompt name
            version: Version to set as active (e.g., 'v2')
        """
        self.active_versions[prompt_name] = version
        logger.info(f"Set {prompt_name} active version to {version}")
        
        # Save to config
        self._save_active_versions()
    
    def list_available_prompts(self) -> Dict[str, list]:
        """
        List all available prompts and their versions.
        
        Returns:
            Dictionary mapping prompt names to list of available versions
        """
        prompts = {}
        
        if not self.prompts_dir.exists():
            return prompts
        
        for prompt_file in self.prompts_dir.glob("*.txt"):
            name = prompt_file.stem
            
            # Parse name (handle versioned and unversioned)
            if '_v' in name:
                # Versioned: prompt_name_v2
                base_name, version = name.rsplit('_v', 1)
                version = f"v{version}"
            elif name.endswith('_prompt'):
                # Unversioned: prompt_name_prompt
                base_name = name[:-7]  # Remove '_prompt'
                version = 'v1'
            else:
                # Simple name
                base_name = name
                version = 'v1'
            
            if base_name not in prompts:
                prompts[base_name] = []
            
            if version not in prompts[base_name]:
                prompts[base_name].append(version)
        
        return prompts
    
    def _load_active_versions(self):
        """Load active versions from config file."""
        config_path = self.prompts_dir / "active_versions.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.active_versions = json.load(f)
                logger.debug(f"Loaded active versions: {self.active_versions}")
            except Exception as e:
                logger.warning(f"Failed to load active versions: {e}")
                self.active_versions = {}
        else:
            # Default versions
            self.active_versions = {
                'part_association': 'v1',
                'subassembly_identification': 'v1',
                'state_analysis': 'v1',
                'error_detection': 'v1'
            }
    
    def _save_active_versions(self):
        """Save active versions to config file."""
        config_path = self.prompts_dir / "active_versions.json"
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.active_versions, f, indent=2)
            logger.debug("Saved active versions")
        except Exception as e:
            logger.warning(f"Failed to save active versions: {e}")
    
    def get_prompt_info(self, prompt_name: str) -> Dict[str, Any]:
        """
        Get information about a prompt.
        
        Returns:
            Dictionary with prompt metadata
        """
        available_versions = self.list_available_prompts().get(prompt_name, [])
        active_version = self.active_versions.get(prompt_name, 'v1')
        
        return {
            "name": prompt_name,
            "available_versions": available_versions,
            "active_version": active_version,
            "has_versions": len(available_versions) > 1
        }


# Singleton instance
_prompt_manager_instance = None


def get_prompt_manager() -> PromptManager:
    """Get PromptManager singleton instance."""
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance

