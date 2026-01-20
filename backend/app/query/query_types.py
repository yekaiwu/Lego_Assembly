"""
Query Types - Data models for query requests and responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class QueryType(Enum):
    """Supported query types."""
    NEXT_STEP = "next_step"           # "What's next?", "What do I do now?"
    PARTS_NEEDED = "parts_needed"     # "What parts do I need?", "Which pieces?"
    CURRENT_STEP = "current_step"     # "What step am I on?"
    HELP = "help"                     # "How do I attach this?", "Help with..."
    VERIFICATION = "verification"     # "Did I do this right?", "Is this correct?"
    UNKNOWN = "unknown"


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str = Field(..., description="User's text query")
    manual_id: str = Field(..., description="Manual identifier")
    current_step: Optional[int] = Field(None, description="Current step number if known")
    image_path: Optional[str] = Field(None, description="Path to user's photo")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Previous conversation context"
    )


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str = Field(..., description="Text response to user")
    query_type: str = Field(..., description="Classified query type")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    current_step: Optional[int] = Field(None, description="Detected or provided current step")
    next_step: Optional[int] = Field(None, description="Next step number")
    target_step: Optional[int] = Field(None, description="Target step for query")
    parts_needed: Optional[List[Dict[str, Any]]] = Field(None, description="Parts needed")
    actions: Optional[List[str]] = Field(None, description="Assembly actions")
    visual_aids: Optional[List[str]] = Field(None, description="Paths to helpful images")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if any")
    requires_image: Optional[bool] = Field(None, description="Whether image is required")
    requires_clarification: Optional[bool] = Field(None, description="Whether clarification needed")
    is_complete: Optional[bool] = Field(None, description="Whether assembly is complete")
