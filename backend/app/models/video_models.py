"""Pydantic models for video analysis API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class VideoUploadResponse(BaseModel):
    """Response for video upload."""
    video_id: str
    filename: str
    size_mb: float
    duration_sec: float
    fps: float
    resolution: List[int]
    status: str = "uploaded"


class VideoAnalysisRequest(BaseModel):
    """Request to start video analysis."""
    manual_id: str
    video_id: str


class VideoAnalysisResponse(BaseModel):
    """Response for analysis start."""
    analysis_id: str
    status: str = "processing"
    estimated_time_sec: int = 180
    message: str


class AssemblyEvent(BaseModel):
    """Single assembly event detected in video."""
    step_id: int
    step_number: Optional[int] = None
    start_seconds: float
    end_seconds: float
    anchor_timestamp: float
    instruction: str
    action: str = "attach"
    target_box_2d: Optional[List[int]] = None
    assembly_box_2d: Optional[List[int]] = None
    confidence: float = 0.8
    reasoning: Optional[str] = None
    parts_required: Optional[List[Dict[str, Any]]] = None
    reference_image: Optional[str] = None


class AnalysisStatus(BaseModel):
    """Status of ongoing analysis."""
    analysis_id: str
    status: str  # processing, completed, error
    progress_percentage: Optional[int] = None
    current_step: Optional[str] = None


class AnalysisResults(BaseModel):
    """Complete analysis results."""
    analysis_id: str
    status: str = "completed"
    results: Optional[Dict[str, Any]] = None
    processing_time_sec: Optional[float] = None


class OverlayOptions(BaseModel):
    """Options for overlay generation."""
    show_target_marker: bool = True
    show_hud_panel: bool = True
    show_instruction_card: bool = True
    show_debug_grid: bool = False


class OverlayGenerationResponse(BaseModel):
    """Response for overlay generation request."""
    overlay_id: str
    status: str = "processing"
    estimated_time_sec: int = 60


class StepAtTimestampRequest(BaseModel):
    """Request for step at specific timestamp."""
    analysis_id: str
    timestamp_sec: float


class ActiveStepResponse(BaseModel):
    """Response with active step information."""
    timestamp_sec: float
    active_step: Optional[AssemblyEvent] = None
    message: Optional[str] = None
