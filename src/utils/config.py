"""
Configuration management for LEGO Assembly System.
Loads environment variables and provides centralized config access.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables with override to ensure .env takes precedence
load_dotenv(override=True)

class APIConfig(BaseModel):
    """API configuration for VLM services."""
    # Chinese VLM API Keys
    dashscope_api_key: str = Field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    deepseek_api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    moonshot_api_key: str = Field(default_factory=lambda: os.getenv("MOONSHOT_API_KEY", ""))
    
    # International VLM API Keys
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    
    # Other API Keys
    rebrickable_api_key: str = Field(default_factory=lambda: os.getenv("REBRICKABLE_API_KEY", ""))
    
    # Chinese VLM Endpoints
    qwen_vl_endpoint: str = Field(default="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
    deepseek_endpoint: str = Field(default="https://api.deepseek.com/v1")
    moonshot_endpoint: str = Field(default="https://api.moonshot.cn/v1")
    
    # International VLM Endpoints
    openai_endpoint: str = Field(default_factory=lambda: os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"))
    anthropic_endpoint: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_ENDPOINT", "https://api.anthropic.com/v1"))
    gemini_endpoint: str = Field(default_factory=lambda: os.getenv("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models"))
    
    # Model names (configurable via env vars)
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    anthropic_model: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

class ModelConfig(BaseModel):
    """VLM model selection configuration."""
    # Phase 1: Ingestion VLMs (for extracting steps from instruction pages)
    ingestion_vlm: str = Field(default_factory=lambda: os.getenv("INGESTION_VLM", "gemini-robotics-er-1.5-preview"))
    ingestion_secondary_vlm: str = Field(default_factory=lambda: os.getenv("INGESTION_SECONDARY_VLM", "gpt-4o-mini"))
    ingestion_fallback_vlm: str = Field(default_factory=lambda: os.getenv("INGESTION_FALLBACK_VLM", "gemini-2.5-flash"))

    # Backend: Diagram VLM (for generating diagram descriptions)
    diagram_vlm: str = Field(default_factory=lambda: os.getenv("DIAGRAM_VLM", "gemini-robotics-er-1.5-preview"))

    # Embedding VLM (for generating embeddings)
    embedding_vlm: str = Field(default_factory=lambda: os.getenv("EMBEDDING_VLM", "gemini-2.5-flash"))

    # General settings
    max_retries: int = Field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    request_timeout: int = Field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60")))

class PathConfig(BaseModel):
    """File system path configuration."""
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    cache_dir: Path = Field(default_factory=lambda: Path(os.getenv("CACHE_DIR", "./cache")))
    parts_db_path: Path = Field(default_factory=lambda: Path(os.getenv("PARTS_DB_PATH", "./data/parts_database.db")))
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parts_db_path.parent.mkdir(parents=True, exist_ok=True)

class SystemConfig(BaseModel):
    """Overall system configuration."""
    api: APIConfig = Field(default_factory=APIConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    cache_enabled: bool = Field(default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true")  # Enabled by default to prevent data loss

# Global config instance
config = SystemConfig()

def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config

