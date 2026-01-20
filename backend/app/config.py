"""
Configuration management for LEGO RAG Backend.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys (LiteLLM will auto-detect these from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    dashscope_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    moonshot_api_key: Optional[str] = None

    # Vector Database
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "lego_manuals"

    # VLM Settings (Backend) - Using LiteLLM model identifiers
    # Format: "provider/model" or just "model" for OpenAI
    diagram_vlm: str = "gemini/gemini-robotics-er-1.5-preview"  # VLM for diagram descriptions
    embedding_vlm: str = "gemini/gemini-2.5-flash"  # VLM for multimodal embeddings
    state_analysis_vlm: str = "gemini/gemini-2.5-flash"  # VLM for analyzing user assembly images

    # LLM Settings - Using LiteLLM model identifiers
    # Examples:
    #   OpenAI: "gpt-4", "gpt-3.5-turbo"
    #   Anthropic: "claude-3-5-sonnet-20241022"
    #   Gemini: "gemini/gemini-2.5-flash"
    #   Qwen: "dashscope/qwen-max"
    #   DeepSeek: "deepseek/deepseek-chat"
    rag_llm_model: str = "gemini/gemini-2.5-flash"  # For text generation
    rag_embedding_model: str = "gemini/text-embedding-004"  # For embeddings

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Data Paths (configurable via environment variables)
    output_dir: Path = Path("./output")  # Changed from ../output to ./output for project root execution
    temp_pages_dir: Path = Path("./output/temp_pages")

    @model_validator(mode='after')
    def resolve_paths(self):
        """
        Resolve output paths to handle both backend/ and project root execution contexts.

        If output_dir from .env is '../output' (for backend/ context) but we're running
        from project root, adjust to './output' instead.
        """
        # Convert to Path if string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.temp_pages_dir, str):
            self.temp_pages_dir = Path(self.temp_pages_dir)

        # Resolve the path relative to current working directory
        resolved_output = self.output_dir.resolve()

        # If the resolved path doesn't exist, try alternative locations
        if not resolved_output.exists():
            # Try ./output relative to cwd (project root execution)
            alternative = Path.cwd() / "output"
            if alternative.exists():
                self.output_dir = Path("./output")
                self.temp_pages_dir = Path("./output/temp_pages")
                print(f"INFO: Adjusted output_dir from {resolved_output} to {alternative}")

        return self
    
    # RAG Settings
    top_k_results: int = 5
    similarity_threshold: float = 0.2  # Used as minimum threshold for hybrid search
    max_context_length: int = 4000

    # Processing Features
    enable_spatial_relationships: bool = True

    class Config:
        # Look for .env in current directory, then parent directory
        env_file = [".env", "../.env"]
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from Phase 1 .env
    
    def get_api_keys_dict(self) -> dict:
        """
        Get all API keys as a dictionary for LiteLLM.

        LiteLLM will automatically use the correct key based on the model.
        """
        api_keys = {}

        if self.openai_api_key:
            api_keys["OPENAI_API_KEY"] = self.openai_api_key
        if self.anthropic_api_key:
            api_keys["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.gemini_api_key:
            api_keys["GEMINI_API_KEY"] = self.gemini_api_key
        if self.dashscope_api_key:
            api_keys["DASHSCOPE_API_KEY"] = self.dashscope_api_key
        if self.deepseek_api_key:
            api_keys["DEEPSEEK_API_KEY"] = self.deepseek_api_key
        if self.moonshot_api_key:
            api_keys["MOONSHOT_API_KEY"] = self.moonshot_api_key

        return api_keys


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()

