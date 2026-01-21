"""
Shared pytest fixtures for the LEGO Assembly System tests.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock

import pytest


# ============================================================================
# Path and Directory Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test isolation."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_output_dir(temp_dir):
    """Create a temporary output directory structure."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "temp_pages").mkdir(exist_ok=True)
    (output_dir / "graphs").mkdir(exist_ok=True)

    return output_dir


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_manual_id() -> str:
    """Return a sample manual ID for testing."""
    return "6454922"


@pytest.fixture
def sample_extracted_step() -> Dict:
    """Return a sample extracted step data."""
    return {
        "step_number": 1,
        "parts_required": [
            {
                "description": "2x4 red brick",
                "color": "red",
                "quantity": 2,
                "part_id": "3001"
            }
        ],
        "actions": [
            {
                "action": "Place brick on baseplate",
                "details": "Position at center"
            }
        ],
        "notes": "Start of construction",
        "page_number": 1,
        "images": ["page_001.png"]
    }


@pytest.fixture
def sample_extracted_steps(sample_extracted_step) -> List[Dict]:
    """Return a list of sample extracted steps."""
    return [
        sample_extracted_step,
        {
            "step_number": 2,
            "parts_required": [
                {
                    "description": "2x2 blue brick",
                    "color": "blue",
                    "quantity": 1,
                    "part_id": "3003"
                }
            ],
            "actions": [
                {
                    "action": "Stack on previous brick",
                    "details": "Align edges"
                }
            ],
            "notes": "Continue building",
            "page_number": 2,
            "images": ["page_002.png"]
        }
    ]


@pytest.fixture
def sample_graph_data(sample_manual_id) -> Dict:
    """Return sample hierarchical graph data."""
    return {
        "assembly_id": sample_manual_id,
        "metadata": {
            "total_parts": 3,
            "total_subassemblies": 1,
            "total_steps": 2,
            "max_depth": 2
        },
        "nodes": [
            {
                "id": "main",
                "type": "assembly",
                "name": "Main Assembly",
                "children": ["sub_1"],
                "parts": [],
                "steps": []
            },
            {
                "id": "sub_1",
                "type": "subassembly",
                "name": "Subassembly 1",
                "children": [],
                "parts": ["part_1", "part_2"],
                "steps": [1, 2]
            }
        ]
    }


# ============================================================================
# Mock Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_settings():
    """Mock application settings."""
    settings = Mock()
    settings.cache_enabled = True
    settings.paths = Mock()
    settings.paths.cache_dir = Path("/tmp/test_cache")
    settings.paths.output_dir = Path("/tmp/test_output")
    settings.paths.data_dir = Path("/tmp/test_data")

    settings.api = Mock()
    settings.api.gemini_api_key = "test_key_123"
    settings.api.openai_api_key = None
    settings.api.anthropic_api_key = None

    settings.models = Mock()
    settings.models.ingestion_vlm = "gemini/gemini-2.0-flash"
    settings.models.llm = "gpt-4"

    return settings


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture
def mock_chroma_manager():
    """Mock ChromaDB manager."""
    manager = MagicMock()
    manager.collection = MagicMock()
    manager.add_documents = MagicMock(return_value=True)
    manager.query = MagicMock(return_value={
        "documents": [["Sample document"]],
        "metadatas": [[{"source": "test"}]],
        "distances": [[0.5]]
    })
    return manager


@pytest.fixture
def mock_vlm_client():
    """Mock VLM (Vision Language Model) client."""
    client = MagicMock()
    client.model = "test-vlm-model"
    client.extract_step_info = MagicMock(return_value=[{
        "step_number": 1,
        "parts_required": [],
        "actions": [],
        "notes": "Test step"
    }])
    return client


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for text generation."""
    client = MagicMock()
    client.generate = MagicMock(return_value="Test response from LLM")
    client.chat = MagicMock(return_value={
        "choices": [{
            "message": {
                "content": "Test chat response"
            }
        }]
    })
    return client


@pytest.fixture
def mock_graph_manager(sample_graph_data):
    """Mock graph manager with sample data."""
    manager = MagicMock()
    manager.graph = sample_graph_data
    manager.load_graph = MagicMock(return_value=sample_graph_data)
    manager.get_node = MagicMock(return_value=sample_graph_data["nodes"][0])
    manager.get_nodes_by_type = MagicMock(return_value=[sample_graph_data["nodes"][0]])
    return manager


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def api_client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from backend.app.main import app

    return TestClient(app)


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary test files after each test."""
    yield
    # Cleanup logic runs after each test
    temp_test_dirs = [
        Path("/tmp/test_cache"),
        Path("/tmp/test_output"),
        Path("/tmp/test_data")
    ]
    for dir_path in temp_test_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)


# ============================================================================
# Environment Variable Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
