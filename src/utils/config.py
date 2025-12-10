"""
Configuration management for LEGO Assembly System.
Loads environment variables and provides centralized config access.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class APIConfig(BaseModel):
    """API configuration for VLM services."""
    dashscope_api_key: str = Field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    deepseek_api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    moonshot_api_key: str = Field(default_factory=lambda: os.getenv("MOONSHOT_API_KEY", ""))
    rebrickable_api_key: str = Field(default_factory=lambda: os.getenv("REBRICKABLE_API_KEY", ""))
    
    qwen_vl_endpoint: str = Field(default="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
    deepseek_endpoint: str = Field(default="https://api.deepseek.com/v1")
    moonshot_endpoint: str = Field(default="https://api.moonshot.cn/v1")

class ModelConfig(BaseModel):
    """VLM model selection configuration."""
    primary_vlm: str = Field(default="qwen-vl-max")
    secondary_vlm: str = Field(default="deepseek-v2")
    fallback_vlm: str = Field(default="kimi-vision")
    max_retries: int = Field(default=3)
    request_timeout: int = Field(default=60)

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
    cache_enabled: bool = Field(default=True)

# Global config instance
config = SystemConfig()

def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config

