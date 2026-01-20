"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ==================== Manual Metadata ====================

class ManualMetadata(BaseModel):
    """Metadata for a LEGO manual."""
    manual_id: str
    total_steps: int
    generated_at: datetime
    source: Optional[str] = None
    manual_pages: Optional[int] = None
    status: str = "ingested"


class ManualListResponse(BaseModel):
    """Response containing list of available manuals."""
    manuals: List[ManualMetadata]
    total: int


# ==================== Step Information ====================

class PartInfo(BaseModel):
    """Information about a LEGO part."""
    part_id: Optional[str] = None
    description: str
    color: str
    shape: Optional[str] = None
    quantity: int = 1
    position: Optional[Dict[str, float]] = None
    rotation: Optional[Dict[str, float]] = None


class StepInfo(BaseModel):
    """Detailed information about a single step."""
    manual_id: str
    step_number: int
    description: str
    parts: List[PartInfo]
    actions: List[Dict[str, Any]]
    spatial_relationships: Dict[str, str]
    dependencies: List[int]
    notes: Optional[str] = None
    image_path: Optional[str] = None


class StepListResponse(BaseModel):
    """Response containing steps for a manual."""
    manual_id: str
    steps: List[StepInfo]
    total_steps: int


# ==================== Ingestion ====================

class IngestionRequest(BaseModel):
    """Request to ingest a manual."""
    manual_id: str
    extracted_json_path: str
    plan_json_path: str
    dependencies_json_path: str
    plan_txt_path: str
    images_dir: str


class IngestionResponse(BaseModel):
    """Response after ingesting a manual."""
    manual_id: str
    status: str
    message: str
    steps_ingested: int
    parts_ingested: int
    chunks_created: int


# ==================== Query/RAG ====================

class TextQueryRequest(BaseModel):
    """Request for text-based query."""
    manual_id: str
    question: str
    include_images: bool = True
    max_results: int = 5
    session_id: Optional[str] = None  # For multimodal queries with uploaded images


class ImageQueryRequest(BaseModel):
    """Request for image-based query."""
    manual_id: str
    image: bytes
    question: Optional[str] = None


class MultimodalQueryRequest(BaseModel):
    """Request for multimodal query (text + user images)."""
    manual_id: str
    question: str
    session_id: str  # Session ID from image upload
    include_images: bool = True
    max_results: int = 5


class RetrievalResult(BaseModel):
    """Single retrieval result from vector store."""
    step_number: int
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    image_path: Optional[str] = None


# ==================== Vision/State Analysis (Forward declarations) ====================

class DetectedPart(BaseModel):
    """Part detected in user's assembly photos."""
    description: str
    color: str
    shape: Optional[str] = None
    part_id: Optional[str] = None
    quantity: int = 1
    location: Optional[str] = None
    confidence: float = 0.5


class ImageAnalysisResult(BaseModel):
    """VLM analysis results from user's uploaded images."""
    detected_parts: List[DetectedPart] = Field(default_factory=list)
    confidence: float = 0.0
    matched_node_ids: List[str] = Field(default_factory=list)  # Node IDs that were matched
    unmatched_parts: List[DetectedPart] = Field(default_factory=list)  # Parts detected but not in catalog


class QueryResponse(BaseModel):
    """Response to a query."""
    answer: str
    sources: List[RetrievalResult]
    current_step: Optional[int] = None
    next_step: Optional[int] = None
    guidance: Optional[str] = None
    parts_needed: Optional[List[PartInfo]] = None
    image_analysis: Optional[ImageAnalysisResult] = None  # VLM analysis if images uploaded


# ==================== Health & Status ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    vector_store_connected: bool
    total_manuals: int
    total_chunks: int


# ==================== Vision/State Analysis (Additional Classes) ====================

class AssembledStructure(BaseModel):
    """Description of assembled structure or subassembly."""
    description: str
    size: Optional[str] = None
    completeness: Optional[str] = None


class PartConnection(BaseModel):
    """Connection between two parts."""
    part_a: str
    part_b: str
    connection_type: str
    orientation: Optional[str] = None


class SpatialLayout(BaseModel):
    """Spatial layout information."""
    overall_shape: Optional[str] = None
    front_view: Optional[str] = None
    top_view: Optional[str] = None
    complexity: Optional[str] = None


class StateAnalysisRequest(BaseModel):
    """Request for assembly state analysis."""
    manual_id: str
    # Image paths will be provided after upload


class StateAnalysisResponse(BaseModel):
    """Response with state analysis and guidance."""
    # Detected State
    detected_parts: List[DetectedPart]
    assembled_structures: List[AssembledStructure]
    connections: List[PartConnection]
    spatial_layout: SpatialLayout
    detection_confidence: float
    
    # Progress Information
    completed_steps: List[int]
    current_step: int
    progress_percentage: float
    total_steps: int
    
    # Guidance
    instruction: str
    next_step_number: Optional[int] = None
    parts_needed: List[PartInfo]
    reference_image: Optional[str] = None
    
    # Errors and Corrections
    errors: List[Dict[str, str]]
    error_corrections: List[str]
    missing_parts: List[Dict[str, Any]]
    
    # Metadata
    encouragement: str
    confidence: float
    status: str


class ImageUploadResponse(BaseModel):
    """Response after image upload."""
    uploaded_files: List[str]
    session_id: str
    message: str
    status: str


class AssemblyError(BaseModel):
    """Description of an assembly error."""
    error_type: str
    severity: str  # "warning", "error", "critical"
    message: str
    suggested_fix: Optional[str] = None

