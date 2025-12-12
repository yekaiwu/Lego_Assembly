"""
Configuration management for LEGO RAG Backend.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    dashscope_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    moonshot_api_key: Optional[str] = None
    
    # Vector Database
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "lego_manuals"
    
    # LLM Settings
    rag_llm_provider: str = "qwen"  # qwen, deepseek, or moonshot
    rag_llm_model: str = "qwen-max"
    rag_embedding_provider: str = "qwen"
    rag_embedding_model: str = "text-embedding-v2"
    
    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    # Data Paths
    output_dir: Path = Path("../output")
    temp_pages_dir: Path = Path("../output/temp_pages")
    
    # RAG Settings
    top_k_results: int = 5
    similarity_threshold: float = 0.2  # Used as minimum threshold for hybrid search
    max_context_length: int = 4000
    
    class Config:
        # Look for .env in current directory, then parent directory
        env_file = [".env", "../.env"]
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from Phase 1 .env
    
    def get_llm_api_key(self) -> str:
        """Get API key for configured LLM provider."""
        if self.rag_llm_provider == "qwen":
            if not self.dashscope_api_key:
                raise ValueError("DASHSCOPE_API_KEY not set")
            return self.dashscope_api_key
        elif self.rag_llm_provider == "deepseek":
            if not self.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY not set")
            return self.deepseek_api_key
        elif self.rag_llm_provider == "moonshot":
            if not self.moonshot_api_key:
                raise ValueError("MOONSHOT_API_KEY not set")
            return self.moonshot_api_key
        else:
            raise ValueError(f"Unknown LLM provider: {self.rag_llm_provider}")


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()

