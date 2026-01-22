"""
FastAPI Main Application for LEGO RAG System.
Provides REST API endpoints for manual ingestion and querying.
"""

from fastapi import FastAPI, HTTPException, Query, Path as PathParam, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from typing import Optional, List, Dict
import uuid
import shutil
import json
import os
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .config import get_settings
from .auth import require_admin, get_current_user, verify_api_key_or_token
from .security import (
    sanitize_filename,
    validate_image_upload,
    validate_video_upload,
    validate_json_upload,
    sanitize_session_id,
    sanitize_manual_id,
    validate_path_traversal
)
from .models.schemas import (
    IngestionResponse,
    QueryResponse,
    ManualListResponse,
    ManualMetadata,
    StepListResponse,
    StepInfo,
    TextQueryRequest,
    MultimodalQueryRequest,
    HealthResponse,
    StateAnalysisRequest,
    StateAnalysisResponse,
    ImageUploadResponse,
    DetectedPart,
    AssembledStructure,
    PartConnection,
    SpatialLayout,
    PartInfo,
    RetrievalResult
)
from .ingestion.ingest_service import get_ingestion_service
from .rag.pipeline import get_rag_pipeline
from .vector_store.chroma_manager import get_chroma_manager
from .vision import get_state_analyzer, get_state_comparator, get_guidance_generator
from .graph.graph_manager import get_graph_manager

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])

# Security audit logger
security_logger = logger.bind(audit=True)

# Initialize FastAPI app
app = FastAPI(
    title="LEGO Assembly RAG API",
    description="Retrieval-Augmented Generation system for LEGO assembly guidance",
    version="1.0.0"
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS with specific origins
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
PRODUCTION_FRONTEND = os.getenv("PRODUCTION_FRONTEND", "")
ALLOWED_ORIGINS = [FRONTEND_URL]
if PRODUCTION_FRONTEND:
    ALLOWED_ORIGINS.append(PRODUCTION_FRONTEND)

logger.info(f"CORS configured for origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Add HSTS in production
    if os.getenv("ENVIRONMENT") == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Initialize services
settings = get_settings()
ingestion_service = get_ingestion_service()
rag_pipeline = get_rag_pipeline()
chroma_manager = get_chroma_manager()
graph_manager = get_graph_manager()


# ==================== Health & Status ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    try:
        total_manuals = len(chroma_manager.get_all_manuals())
        total_chunks = chroma_manager.count_documents()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            vector_store_connected=True,
            total_manuals=total_manuals,
            total_chunks=total_chunks
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            vector_store_connected=False,
            total_manuals=0,
            total_chunks=0
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return await root()


# ==================== Ingestion API ====================

@app.post("/api/ingest/manual/{manual_id}", response_model=IngestionResponse)
@limiter.limit("10/minute")
async def ingest_manual(
    request: Request,
    manual_id: str = PathParam(..., description="Manual identifier (e.g., 6454922)"),
    admin: Dict = Depends(require_admin)
):
    """
    Ingest a specific manual into the vector store.
    The manual files must exist in the output directory.

    **Requires admin authentication.**
    """
    try:
        # Sanitize manual_id
        manual_id = sanitize_manual_id(manual_id)

        security_logger.info(f"Manual ingestion started: {manual_id} by user {admin.get('sub')}")

        result = ingestion_service.ingest_manual(manual_id)

        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail="Ingestion failed")

        security_logger.info(f"Manual ingestion completed: {manual_id}")
        return IngestionResponse(**result)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Manual files not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/ingest/all")
@limiter.limit("1/hour")
async def ingest_all_manuals(
    request: Request,
    admin: Dict = Depends(require_admin)
):
    """
    Ingest all available manuals from the output directory.

    **Requires admin authentication.**
    **Rate limited to 1 request per hour due to high cost.**
    """
    try:
        security_logger.warning(f"Bulk ingestion started by user {admin.get('sub')}")
        result = ingestion_service.ingest_all_manuals()
        security_logger.info(f"Bulk ingestion completed")
        return result
    except Exception as e:
        logger.error(f"Bulk ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/manual/{manual_id}")
@limiter.limit("5/hour")
async def delete_manual(
    request: Request,
    manual_id: str = PathParam(..., description="Manual identifier"),
    admin: Dict = Depends(require_admin)
):
    """
    Remove a manual from the vector store.

    **Requires admin authentication.**
    **Destructive operation - use with caution.**
    """
    try:
        # Sanitize manual_id
        manual_id = sanitize_manual_id(manual_id)

        security_logger.warning(f"Manual deletion requested: {manual_id} by user {admin.get('sub')}")

        result = ingestion_service.remove_manual(manual_id)

        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail="Deletion failed")

        security_logger.info(f"Manual deleted: {manual_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deletion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/upload/manual/{manual_id}")
@limiter.limit("20/hour")
async def upload_manual_files(
    request: Request,
    manual_id: str = PathParam(..., description="Manual identifier"),
    extracted_json: UploadFile = File(..., description="extracted.json file"),
    plan_json: UploadFile = File(..., description="plan.json file"),
    dependencies_json: UploadFile = File(..., description="dependencies.json file"),
    plan_txt: UploadFile = File(None, description="plan.txt file (optional)"),
    graph_json: UploadFile = File(None, description="graph.json file (optional)"),
    images: List[UploadFile] = File(None, description="Step images (temp_pages/*.png)"),
    segmented_parts: List[UploadFile] = File(None, description="Segmented part images (segmented_parts/step_*/*.png)"),
    admin: Dict = Depends(require_admin)
):
    """
    Upload manual files to backend.

    **Requires admin authentication.**

    Uploads all necessary files for a manual:
    - extracted.json, plan.json, dependencies.json (required)
    - plan.txt, graph.json (optional)
    - Images from temp_pages/ (optional)

    After upload, call /api/ingest/manual/{manual_id} to ingest into vector store.
    """
    try:
        # Sanitize manual_id
        manual_id = sanitize_manual_id(manual_id)

        security_logger.info(f"File upload started for manual {manual_id} by user {admin.get('sub')}")

        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create temp_pages/{manual_id} directory if images are provided
        temp_pages_dir = output_dir / "temp_pages" / manual_id
        if images:
            temp_pages_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files = []

        # Validate and upload required JSON files (directly to output directory)
        for file, filename in [
            (extracted_json, f"{manual_id}_extracted.json"),
            (plan_json, f"{manual_id}_plan.json"),
            (dependencies_json, f"{manual_id}_dependencies.json"),
        ]:
            await validate_json_upload(file)
            file_path = output_dir / filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            uploaded_files.append(filename)
            logger.info(f"Uploaded {filename}")

        # Upload optional files (directly to output directory)
        if plan_txt:
            file_path = output_dir / f"{manual_id}_plan.txt"
            with open(file_path, "wb") as f:
                shutil.copyfileobj(plan_txt.file, f)
            uploaded_files.append(f"{manual_id}_plan.txt")

        if graph_json:
            await validate_json_upload(graph_json)
            file_path = output_dir / f"{manual_id}_graph.json"
            with open(file_path, "wb") as f:
                shutil.copyfileobj(graph_json.file, f)
            uploaded_files.append(f"{manual_id}_graph.json")

        # Validate and upload images (to temp_pages/{manual_id}/)
        if images:
            for image in images:
                await validate_image_upload(image)
                # Sanitize filename
                safe_filename = sanitize_filename(image.filename or f"page_{len(uploaded_files)}.png")
                file_path = temp_pages_dir / safe_filename
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(image.file, f)
                uploaded_files.append(f"temp_pages/{manual_id}/{safe_filename}")

        # Validate and upload segmented part images (to segmented_parts/{manual_id}/)
        if segmented_parts:
            segmented_base_dir = output_dir / "segmented_parts" / manual_id
            segmented_base_dir.mkdir(parents=True, exist_ok=True)

            for part_image in segmented_parts:
                await validate_image_upload(part_image)
                # Filename should be in format: step_XXX/filename.png
                original_filename = part_image.filename or "unknown.png"

                # Parse step directory from filename (e.g., "step_001/part.png")
                if "/" in original_filename:
                    step_dir_name, filename = original_filename.split("/", 1)
                    step_dir = segmented_base_dir / sanitize_filename(step_dir_name)
                    step_dir.mkdir(parents=True, exist_ok=True)
                    safe_filename = sanitize_filename(filename)
                    file_path = step_dir / safe_filename
                else:
                    # Fallback: save to root if no directory structure
                    safe_filename = sanitize_filename(original_filename)
                    file_path = segmented_base_dir / safe_filename

                with open(file_path, "wb") as f:
                    shutil.copyfileobj(part_image.file, f)
                uploaded_files.append(f"segmented_parts/{manual_id}/{original_filename}")

        security_logger.info(f"File upload completed for manual {manual_id}: {len(uploaded_files)} files")

        return {
            "status": "success",
            "manual_id": manual_id,
            "uploaded_files": uploaded_files,
            "message": f"Successfully uploaded {len(uploaded_files)} files. Call /api/ingest/manual/{manual_id} to ingest."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading manual files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ==================== Query API ====================

@app.post("/api/query/text", response_model=QueryResponse)
async def query_text(request: TextQueryRequest):
    """
    Process a text-based query about a manual.
    Supports optional multimodal queries with user assembly images.
    
    Example request (text-only):
    ```json
    {
        "manual_id": "6454922",
        "question": "What parts do I need for step 5?",
        "include_images": true,
        "max_results": 5
    }
    ```
    
    Example request (multimodal with images):
    ```json
    {
        "manual_id": "6454922",
        "question": "What's next?",
        "session_id": "uuid-from-image-upload",
        "include_images": true,
        "max_results": 5
    }
    ```
    """
    try:
        # Check if user provided images via session_id
        user_images = None
        if request.session_id:
            upload_dir = Path(f"/tmp/lego_assembly_uploads/{request.session_id}")
            if upload_dir.exists():
                user_images = sorted([str(p) for p in upload_dir.glob("image_*.*")])
                logger.info(f"Found {len(user_images)} user images for multimodal query")
            else:
                logger.warning(f"Session {request.session_id} not found, proceeding with text-only query")
        
        response = rag_pipeline.process_text_query(
            manual_id=request.manual_id,
            question=request.question,
            max_results=request.max_results,
            include_images=request.include_images,
            user_images=user_images
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/multimodal", response_model=QueryResponse)
async def query_multimodal(request: MultimodalQueryRequest):
    """
    Process a multimodal query (text + user assembly images).
    
    User must first upload images via /api/vision/upload-images to get a session_id,
    then use that session_id in this request.
    
    Example request:
    ```json
    {
        "manual_id": "6454922",
        "question": "What's next?",
        "session_id": "uuid-from-upload",
        "include_images": true,
        "max_results": 5
    }
    ```
    """
    try:
        # Get uploaded image paths
        upload_dir = Path(f"/tmp/lego_assembly_uploads/{request.session_id}")
        if not upload_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="Session not found. Please upload images first via /api/vision/upload-images"
            )
        
        user_images = sorted([str(p) for p in upload_dir.glob("image_*.*")])
        if not user_images:
            raise HTTPException(
                status_code=404,
                detail="No images found for this session"
            )
        
        logger.info(f"Processing multimodal query with {len(user_images)} images")
        
        response = rag_pipeline.process_text_query(
            manual_id=request.manual_id,
            question=request.question,
            max_results=request.max_results,
            include_images=request.include_images,
            user_images=user_images
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multimodal query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def clean_markdown(text: str) -> str:
    """
    Remove markdown formatting symbols from text for plain display.

    Args:
        text: Text with markdown formatting

    Returns:
        Cleaned text without markdown symbols
    """
    import re

    # Remove bold/italic markers
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)       # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_

    # Remove code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)         # `code`

    # Clean up headers (keep the text, remove #)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove extra blank lines (more than 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


@app.get("/api/manual/{manual_id}/step/{step_number}", response_model=QueryResponse)
async def get_step(
    manual_id: str = PathParam(..., description="Manual identifier"),
    step_number: int = PathParam(..., description="Step number", ge=1),
    use_llm: bool = Query(False, description="Use LLM to generate formatted response (slower)")
):
    """
    Get detailed information for a specific step.

    By default, returns the raw step data from ChromaDB (fast, no LLM).
    Set use_llm=true to get an LLM-generated formatted response (slower, uses RAG).
    """
    try:
        # Fast path: Just retrieve from ChromaDB without LLM
        if not use_llm:
            # Get step data from ChromaDB
            results = chroma_manager.collection.get(
                where={
                    "$and": [
                        {"manual_id": manual_id},
                        {"chunk_type": "step"},
                        {"step_number": step_number}
                    ]
                }
            )

            if not results or not results['documents'] or len(results['documents']) == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Step {step_number} not found in manual {manual_id}"
                )

            # Extract the first result
            content = results['documents'][0]
            metadata = results['metadatas'][0]

            # Clean markdown formatting from content
            cleaned_content = clean_markdown(content)

            # Create a simple response without LLM generation
            source = RetrievalResult(
                step_number=metadata.get('step_number', step_number),
                content=cleaned_content,
                similarity_score=1.0,
                metadata=metadata,
                image_path=metadata.get('image_path', '')
            )

            return QueryResponse(
                answer=cleaned_content,  # Return cleaned content as answer
                sources=[source],
                current_step=step_number,
                next_step=step_number + 1 if step_number < metadata.get('total_steps', 100) else None,
                guidance=None,
                parts_needed=None
            )

        # Slow path: Use RAG pipeline with LLM
        else:
            response = rag_pipeline.get_step_details(manual_id, step_number)
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Step retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Manual Management ====================

@app.get("/api/debug/files")
async def inspect_output_directory():
    """
    Debug endpoint to inspect output directory structure.
    Lists all files and directories in the output folder.
    """
    try:
        output_dir = Path(settings.output_dir)

        if not output_dir.exists():
            return {
                "error": "Output directory does not exist",
                "path": str(output_dir.resolve())
            }

        files_list = []

        # Walk through directory
        for item in output_dir.rglob("*"):
            relative_path = item.relative_to(output_dir)
            files_list.append({
                "path": str(relative_path),
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            })

        return {
            "output_directory": str(output_dir.resolve()),
            "total_items": len(files_list),
            "files": sorted(files_list, key=lambda x: x["path"])
        }
    except Exception as e:
        logger.error(f"Error inspecting output directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vector-store/inspect")
async def inspect_vector_store(
    manual_id: Optional[str] = None,
    step_number: Optional[int] = None,
    limit: int = 50,
    show_content: bool = True
):
    """
    Inspect vector store contents (for debugging).
    
    Query parameters:
    - manual_id: Filter by specific manual (optional)
    - step_number: Filter by specific step (optional, requires manual_id)
    - limit: Max documents to return (default: 50, max: 500)
    - show_content: Include document content in response (default: true)
    """
    try:
        collection = chroma_manager.collection
        
        # Get collection stats
        total_docs = collection.count()
        
        # Get manual IDs
        manual_ids = chroma_manager.get_all_manuals()
        
        # Get stats per manual
        manual_stats = {}
        for mid in manual_ids:
            count = chroma_manager.count_documents(mid)
            manual_stats[mid] = count
        
        # Build where filter
        where_filter = None
        if manual_id and step_number:
            where_filter = {
                "$and": [
                    {"manual_id": manual_id},
                    {"step_number": step_number}
                ]
            }
        elif manual_id:
            where_filter = {"manual_id": manual_id}
        
        # Limit max to prevent performance issues
        limit = min(limit, 500)
        
        # Get documents
        if where_filter:
            all_data = collection.get(where=where_filter, limit=limit)
        else:
            all_data = collection.get(limit=limit)
        
        # Format documents
        documents = []
        for i in range(len(all_data['ids'])):
            doc = {
                "id": all_data['ids'][i],
                "metadata": all_data['metadatas'][i]
            }
            if show_content:
                content = all_data['documents'][i]
                doc["content"] = content[:500] + "..." if len(content) > 500 else content
                doc["content_length"] = len(content)
            documents.append(doc)
        
        # Get step numbers for manual if filtering
        step_numbers = []
        if manual_id:
            manual_data = collection.get(
                where={
                    "$and": [
                        {"manual_id": manual_id},
                        {"chunk_type": "step"}
                    ]
                },
                limit=1000
            )
            step_numbers = sorted([
                m.get('step_number', 0) 
                for m in manual_data['metadatas'] 
                if m.get('step_number', 0) > 0
            ])
        
        return {
            "total_documents": total_docs,
            "total_manuals": len(manual_ids),
            "manual_stats": manual_stats,
            "filters": {
                "manual_id": manual_id,
                "step_number": step_number,
                "limit": limit
            },
            "step_numbers": step_numbers if manual_id else None,
            "documents_returned": len(documents),
            "documents": documents,
            "collection_name": chroma_manager.settings.collection_name,
            "persist_directory": str(chroma_manager.settings.chroma_persist_dir)
        }
    except Exception as e:
        logger.error(f"Error inspecting vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manuals", response_model=ManualListResponse)
async def list_manuals():
    """
    Get list of all available manuals.
    """
    try:
        manual_ids = chroma_manager.get_all_manuals()
        
        manuals = []
        for manual_id in manual_ids:
            # Get metadata from vector store
            try:
                results = chroma_manager.collection.get(
                    where={
                        "$and": [
                            {"manual_id": manual_id},
                            {"chunk_type": "metadata"}
                        ]
                    },
                    limit=1
                )
                
                if results and results['metadatas']:
                    metadata = results['metadatas'][0]
                    manuals.append(ManualMetadata(
                        manual_id=manual_id,
                        total_steps=metadata.get('total_steps', 0),
                        generated_at=metadata.get('generated_at', ''),
                        status="available"
                    ))
                else:
                    # Fallback if no metadata found
                    doc_count = chroma_manager.count_documents(manual_id)
                    manuals.append(ManualMetadata(
                        manual_id=manual_id,
                        total_steps=doc_count - 1,  # Exclude metadata chunk
                        generated_at="",
                        status="available"
                    ))
            except Exception as e:
                logger.warning(f"Could not fetch metadata for {manual_id}: {e}")
                continue
        
        return ManualListResponse(
            manuals=manuals,
            total=len(manuals)
        )
        
    except Exception as e:
        logger.error(f"Error listing manuals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manual/{manual_id}/steps")
async def list_manual_steps(
    manual_id: str = PathParam(..., description="Manual identifier")
):
    """
    Get all steps for a specific manual.
    """
    try:
        # Get all step documents for this manual
        results = chroma_manager.collection.get(
            where={
                "$and": [
                    {"manual_id": manual_id},
                    {"chunk_type": "step"}
                ]
            }
        )
        
        if not results or not results['metadatas']:
            raise HTTPException(
                status_code=404,
                detail=f"Manual {manual_id} not found"
            )
        
        steps = []
        for metadata, content in zip(results['metadatas'], results['documents']):
            steps.append({
                "step_number": metadata.get('step_number', 0),
                "has_parts": metadata.get('has_parts', False),
                "parts_count": metadata.get('parts_count', 0),
                "has_dependencies": metadata.get('has_dependencies', False),
                "image_path": metadata.get('image_path', ''),
                "preview": content[:200] + "..." if len(content) > 200 else content
            })
        
        # Sort by step number
        steps.sort(key=lambda x: x['step_number'])
        
        return {
            "manual_id": manual_id,
            "total_steps": len(steps),
            "steps": steps
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing steps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manual/{manual_id}/extracted-steps")
async def get_extracted_steps(
    manual_id: str = PathParam(..., description="Manual identifier"),
    limit: Optional[int] = Query(None, description="Limit number of steps to return", ge=1, le=100),
    step_number: Optional[int] = Query(None, description="Get specific step only", ge=1)
):
    """
    Get detailed extracted step data from Phase 1 processing.
    
    Returns the raw extracted step information including parts, actions, and notes.
    """
    try:
        extracted_path = Path(settings.output_dir) / f"{manual_id}_extracted.json"
        
        if not extracted_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Extracted data not found for manual {manual_id}. Please process the manual first using main.py"
            )
        
        with open(extracted_path, 'r', encoding='utf-8') as f:
            all_steps = json.load(f)
        
        # Filter to valid steps only
        valid_steps = [s for s in all_steps if s.get("step_number") and s.get("step_number") > 0]
        
        # Filter by step number if specified
        if step_number:
            valid_steps = [s for s in valid_steps if s.get("step_number") == step_number]
            if not valid_steps:
                raise HTTPException(
                    status_code=404,
                    detail=f"Step {step_number} not found in manual {manual_id}"
                )
        
        # Apply limit if specified
        if limit:
            valid_steps = valid_steps[:limit]
        
        return {
            "manual_id": manual_id,
            "total_steps": len([s for s in all_steps if s.get("step_number") and s.get("step_number") > 0]),
            "returned_steps": len(valid_steps),
            "steps": valid_steps
        }
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Extracted data file not found: {e}"
        )
    except Exception as e:
        logger.error(f"Error getting extracted steps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Image Serving ====================

@app.get("/api/image")
async def serve_image(
    path: str = Query(..., description="Image path")
):
    """
    Serve step images.
    """
    try:
        # Resolve path relative to project root (parent of backend/)
        # If path is relative (e.g., "output/temp_pages/page_001.png"), resolve it
        image_path = Path(path)

        # If not absolute, resolve relative to backend's parent directory (project root)
        if not image_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent  # Go up from app/main.py to project root
            image_path = project_root / image_path

        if not image_path.exists():
            logger.error(f"Image not found at: {image_path}")
            raise HTTPException(status_code=404, detail=f"Image not found: {path}")

        return FileResponse(image_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Vision/State Analysis API ====================

@app.post("/api/vision/upload-images", response_model=ImageUploadResponse)
@limiter.limit("30/hour")
async def upload_assembly_images(
    request: Request,
    images: List[UploadFile] = File(..., description="1-4 images of assembly")
):
    """
    Upload images of user's physical assembly.
    Accepts 1-4 images from different angles (2+ recommended for better accuracy).

    Returns session_id for subsequent analysis.
    """
    try:
        # Validate number of images
        if len(images) < 1 or len(images) > 4:
            raise HTTPException(
                status_code=400,
                detail="Please upload between 1 and 4 images"
            )

        # Validate all images before uploading
        for image in images:
            await validate_image_upload(image)

        # Generate secure session ID using secrets module
        import secrets
        session_id = secrets.token_urlsafe(32)

        # Create upload directory with validation
        base_upload_dir = Path("/tmp/lego_assembly_uploads")
        base_upload_dir.mkdir(parents=True, exist_ok=True)
        upload_dir = base_upload_dir / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Verify no path traversal
        if not str(upload_dir.resolve()).startswith(str(base_upload_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid path")

        # Save uploaded files
        uploaded_files = []
        for i, image in enumerate(images):
            # Sanitize file extension
            file_extension = Path(image.filename).suffix
            safe_extension = sanitize_filename(file_extension) if file_extension else ".jpg"
            if not safe_extension.startswith('.'):
                safe_extension = f".{safe_extension}"

            file_path = upload_dir / f"image_{i + 1}{safe_extension}"

            with open(file_path, "wb") as f:
                shutil.copyfileobj(image.file, f)

            uploaded_files.append(str(file_path))
            logger.info(f"Saved image: {file_path}")

        return ImageUploadResponse(
            uploaded_files=uploaded_files,
            session_id=session_id,
            message=f"Successfully uploaded {len(uploaded_files)} images",
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/vision/analyze", response_model=StateAnalysisResponse)
async def analyze_assembly_state(
    manual_id: str,
    session_id: str,
    output_dir: Optional[str] = "./output"
):
    """
    Analyze assembly state from uploaded images.
    
    This endpoint:
    1. Analyzes images using VLM to detect parts and state
    2. Compares detected state with expected plan
    3. Determines progress and identifies errors
    4. Generates next-step guidance
    
    Args:
        manual_id: Manual identifier (must be processed by Phase 1)
        session_id: Session ID from image upload
        output_dir: Directory containing Phase 1 outputs
    
    Returns:
        Complete state analysis with guidance
    """
    try:
        logger.info(f"Analyzing assembly state for manual {manual_id}, session {session_id}")
        
        # Get uploaded image paths
        upload_dir = Path(f"/tmp/lego_assembly_uploads/{session_id}")
        if not upload_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="Session not found. Please upload images first."
            )
        
        image_paths = sorted([str(p) for p in upload_dir.glob("image_*.*")])
        if not image_paths:
            raise HTTPException(
                status_code=404,
                detail="No images found for this session"
            )
        
        logger.info(f"Found {len(image_paths)} images for analysis")
        
        # Initialize vision services
        state_analyzer = get_state_analyzer()
        state_comparator = get_state_comparator()
        guidance_generator = get_guidance_generator()
        
        # Step 1: Analyze assembly state with VLM
        logger.info("Step 1: Analyzing assembly state with VLM...")
        detected_state = state_analyzer.analyze_assembly_state(
            image_paths=image_paths,
            manual_id=manual_id
        )
        
        # Step 2: Compare with plan
        logger.info("Step 2: Comparing with plan...")
        comparison_result = state_comparator.compare_with_plan(
            detected_state=detected_state,
            manual_id=manual_id,
            output_dir=output_dir
        )
        
        # Step 3: Generate guidance
        logger.info("Step 3: Generating guidance...")
        guidance = guidance_generator.generate_guidance(
            detected_state=detected_state,
            comparison_result=comparison_result,
            manual_id=manual_id,
            output_dir=output_dir
        )
        
        # Build comprehensive response
        response = StateAnalysisResponse(
            # Detected State
            detected_parts=[
                DetectedPart(**part) for part in detected_state.get("detected_parts", [])
            ],
            assembled_structures=[
                AssembledStructure(**struct) for struct in detected_state.get("assembled_structures", [])
            ],
            connections=[
                PartConnection(**conn) for conn in detected_state.get("connections", [])
            ],
            spatial_layout=SpatialLayout(**detected_state.get("spatial_layout", {})),
            detection_confidence=detected_state.get("confidence", 0.0),
            
            # Progress
            completed_steps=comparison_result.get("completed_steps", []),
            current_step=comparison_result.get("current_step", 1),
            progress_percentage=comparison_result.get("progress_percentage", 0.0),
            total_steps=comparison_result.get("total_steps", 0),
            
            # Guidance
            instruction=guidance.get("instruction", ""),
            next_step_number=guidance.get("next_step_number"),
            parts_needed=[
                PartInfo(**part) for part in guidance.get("parts_needed", [])
            ],
            reference_image=guidance.get("reference_image"),
            
            # Errors
            errors=comparison_result.get("errors", []),
            error_corrections=guidance.get("error_corrections", []),
            missing_parts=comparison_result.get("missing_parts", []),
            
            # Metadata
            encouragement=guidance.get("encouragement", ""),
            confidence=guidance.get("confidence", 0.0),
            status="success"
        )
        
        logger.info(f"Analysis complete: {response.progress_percentage:.1f}% progress")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing assembly state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/vision/session/{session_id}")
async def cleanup_session(
    session_id: str = PathParam(..., description="Session ID to cleanup")
):
    """
    Clean up uploaded images for a session.
    """
    try:
        upload_dir = Path(f"/tmp/lego_assembly_uploads/{session_id}")
        
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            logger.info(f"Cleaned up session: {session_id}")
            return {
                "status": "success",
                "message": f"Session {session_id} cleaned up"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Hierarchical Graph API ====================

@app.get("/api/manual/{manual_id}/graph/summary")
async def get_graph_summary(
    manual_id: str = PathParam(..., description="Manual identifier")
):
    """
    Get summary of hierarchical assembly graph for a manual.
    
    Returns:
        - Total nodes (parts, subassemblies)
        - Total edges (relationships)
        - Statistics by type
    """
    try:
        summary = graph_manager.get_graph_summary(manual_id)
        
        if not summary:
            raise HTTPException(
                status_code=404,
                detail=f"Graph not found for manual {manual_id}. Please process the manual first."
            )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manual/{manual_id}/graph/step/{step_number}/state")
async def get_step_state(
    manual_id: str = PathParam(..., description="Manual identifier"),
    step_number: int = PathParam(..., description="Step number", ge=1)
):
    """
    Get assembly state at a specific step from the hierarchical graph.
    
    Returns:
        - Parts added in this step
        - Subassemblies created/modified
        - Cumulative state
        - Completion percentage
    """
    try:
        step_state = graph_manager.get_step_state(manual_id, step_number)
        
        if not step_state:
            raise HTTPException(
                status_code=404,
                detail=f"Step state not found for manual {manual_id} step {step_number}"
            )
        
        return step_state
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting step state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manual/{manual_id}/graph/subassemblies")
async def get_subassemblies(
    manual_id: str = PathParam(..., description="Manual identifier")
):
    """
    Get all subassemblies in a manual.
    
    Returns list of subassemblies with:
        - Name and description
        - Step created
        - Parts contained
        - Parent relationships
    """
    try:
        subassemblies = graph_manager.get_nodes_by_type(manual_id, "subassembly")
        
        if not subassemblies and subassemblies is not None:
            # Empty list means no subassemblies found
            return {
                "manual_id": manual_id,
                "total_subassemblies": 0,
                "subassemblies": []
            }
        
        if subassemblies is None:
            raise HTTPException(
                status_code=404,
                detail=f"Graph not found for manual {manual_id}"
            )
        
        # Enrich each subassembly with its parts
        enriched = []
        for subasm in subassemblies:
            parts = graph_manager.get_subassembly_parts(manual_id, subasm['node_id'])
            enriched.append({
                "node_id": subasm['node_id'],
                "name": subasm['name'],
                "description": subasm.get('description', ''),
                "step_created": subasm.get('step_created', 0),
                "steps_used": subasm.get('steps_used', []),
                "parts": [
                    {
                        "name": p['name'],
                        "color": p.get('color', ''),
                        "shape": p.get('shape', ''),
                        "role": p.get('role', '')
                    }
                    for p in parts
                ]
            })
        
        return {
            "manual_id": manual_id,
            "total_subassemblies": len(enriched),
            "subassemblies": enriched
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting subassemblies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manual/{manual_id}/graph")
async def get_full_graph(
    manual_id: str = PathParam(..., description="Manual identifier")
):
    """
    Get complete hierarchical graph structure for visualization.

    Returns the full graph with:
        - All nodes (model, subassemblies, parts)
        - All edges (relationships)
        - Metadata (total parts, steps, depth)
        - Node attributes (type, layer, step_created, etc.)

    Use this endpoint for graph visualization with tools like:
    - ReactFlow
    - Cytoscape
    - D3.js
    - Vis.js
    """
    try:
        graph = graph_manager.load_graph(manual_id)

        if not graph:
            raise HTTPException(
                status_code=404,
                detail=f"Graph not found for manual {manual_id}. Please process the manual first."
            )

        return graph

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting full graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manual/{manual_id}/progress")
async def get_assembly_progress(
    manual_id: str = PathParam(..., description="Manual identifier"),
    session_id: Optional[str] = None
):
    """
    Calculate assembly progress from current state.

    If session_id provided, analyzes uploaded images to determine progress.
    Otherwise returns overall manual statistics.

    Returns:
        - Progress percentage (based on step completion)
        - Current step number
        - Total steps
        - Completed steps list
    """
    try:
        graph = graph_manager.load_graph(manual_id)

        if not graph:
            raise HTTPException(
                status_code=404,
                detail=f"Graph not found for manual {manual_id}"
            )

        metadata = graph.get('metadata', {})
        total_steps = metadata.get('total_steps', 0)

        # If session_id provided, analyze images to get current step
        if session_id:
            upload_dir = Path(f"/tmp/lego_assembly_uploads/{session_id}")
            if not upload_dir.exists():
                raise HTTPException(
                    status_code=404,
                    detail="Session not found"
                )

            image_paths = sorted([str(p) for p in upload_dir.glob("image_*.*")])

            if image_paths:
                # Analyze images to get detected parts
                state_analyzer = get_state_analyzer()
                detected_state = state_analyzer.analyze_assembly_state(
                    image_paths=image_paths,
                    manual_id=manual_id
                )

                # Compare with plan to get current step
                state_comparator = get_state_comparator()
                comparison = state_comparator.compare_with_plan(
                    detected_state=detected_state,
                    manual_id=manual_id
                )

                # Calculate progress from step number
                current_step = comparison.get('current_step', 1)
                completed_steps = comparison.get('completed_steps', [])
                progress = (current_step / total_steps) * 100 if total_steps > 0 else 0.0

                return {
                    "manual_id": manual_id,
                    "progress_percentage": round(progress, 1),
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "completed_steps": completed_steps
                }

        # Default: return overall statistics (no progress calculated)
        return {
            "manual_id": manual_id,
            "total_steps": total_steps,
            "total_parts": metadata.get('total_parts', 0),
            "progress_percentage": 0.0,
            "current_step": 0,
            "completed_steps": []
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assembly progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manual/{manual_id}/graph/node/{node_id}/hierarchy")
async def get_node_hierarchy(
    manual_id: str = PathParam(..., description="Manual identifier"),
    node_id: str = PathParam(..., description="Node identifier")
):
    """
    Get hierarchical path from root to a specific node.
    
    Shows how a part or subassembly fits into the overall assembly structure.
    
    Returns:
        - Path from root model to target node
        - Each node's type and description
        - Step information
    """
    try:
        path = graph_manager.get_assembly_path(manual_id, node_id)
        
        if not path:
            raise HTTPException(
                status_code=404,
                detail=f"Node {node_id} not found in manual {manual_id}"
            )
        
        return {
            "manual_id": manual_id,
            "target_node_id": node_id,
            "hierarchy_depth": len(path),
            "path": [
                {
                    "node_id": node['node_id'],
                    "type": node.get('type', 'unknown'),
                    "name": node.get('name', ''),
                    "step_created": node.get('step_created', 0),
                    "layer": node.get('layer', 0)
                }
                for node in path
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node hierarchy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Startup & Shutdown ====================

# ==================== Video Analysis API ====================

from fastapi import BackgroundTasks, Form
from .models.video_models import (
    VideoUploadResponse,
    VideoAnalysisResponse,
    AnalysisResults,
    OverlayOptions,
    OverlayGenerationResponse,
    ActiveStepResponse,
)
from .video.storage_manager import VideoStorageManager, AnalysisStorageManager
from .video.metadata_extractor import extract_video_metadata, validate_video_format
from .video.video_analyzer import VideoAnalyzer
from .video.overlay_renderer import create_overlay_video
from .utils.background_tasks import TaskManager, TaskStatus

# Initialize video services
video_storage = VideoStorageManager()
analysis_storage = AnalysisStorageManager()
task_manager = TaskManager()


@app.post("/api/video/upload", response_model=VideoUploadResponse)
@limiter.limit("10/hour")
async def upload_video(
    request: Request,
    manual_id: str = Form(...),
    video: UploadFile = File(...)
):
    """
    Upload assembly video for analysis.

    Args:
        manual_id: Manual ID this video belongs to
        video: Video file (MP4, MOV, AVI, WebM - max 500MB)

    Returns:
        Video metadata and upload confirmation
    """
    try:
        # Sanitize manual_id
        manual_id = sanitize_manual_id(manual_id)

        # Validate video
        await validate_video_upload(video)

        # Create secure temp directory
        import secrets
        import tempfile
        temp_dir = Path("/tmp/lego_videos")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename to prevent path traversal
        safe_filename = sanitize_filename(video.filename or "video.mp4")
        temp_id = secrets.token_hex(8)
        temp_path = temp_dir / f"{temp_id}_{safe_filename}"

        # Write video to temp file
        with open(temp_path, "wb") as f:
            f.write(await video.read())

        # Validate video format
        if not validate_video_format(str(temp_path)):
            temp_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="Invalid video format"
            )

        # Extract metadata
        metadata = extract_video_metadata(str(temp_path))

        # Save video
        with open(temp_path, "rb") as f:
            video_data = f.read()

        video_id, video_path = video_storage.save_uploaded_video(
            video_data,
            manual_id,
            safe_filename
        )

        # Cleanup temp file
        temp_path.unlink()

        logger.info(f"Uploaded video {video_id} for manual {manual_id}")

        return VideoUploadResponse(
            video_id=video_id,
            filename=safe_filename,
            **metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def run_video_analysis(
    video_path: str,
    manual_id: str,
    analysis_id: str,
    extracted_path: str,
    dependencies_path: Optional[str],
    reference_images: List[str]
):
    """Background task for video analysis."""
    try:
        # Update task status
        task_manager.update_task(
            analysis_id,
            status=TaskStatus.PROCESSING,
            progress=10,
            current_step="Initializing video analyzer"
        )

        # Initialize analyzer
        api_key = settings.gemini_api_key
        analyzer = VideoAnalyzer(api_key=api_key)

        task_manager.update_task(
            analysis_id,
            progress=20,
            current_step="Analyzing video with VLM"
        )

        # Run analysis
        results = analyzer.analyze_video(
            video_path=video_path,
            manual_id=manual_id,
            dependencies_path=dependencies_path,
            extracted_data_path=extracted_path,
            reference_images=reference_images
        )

        task_manager.update_task(
            analysis_id,
            progress=80,
            current_step="Saving results"
        )

        # Save results
        results['analysis_id'] = analysis_id
        analysis_storage.save_analysis_results(
            results,
            manual_id,
            analysis_id
        )

        task_manager.update_task(
            analysis_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            current_step="Analysis complete"
        )

        logger.info(f"Video analysis completed: {analysis_id}")

    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        task_manager.update_task(
            analysis_id,
            status=TaskStatus.ERROR,
            error=str(e)
        )


@app.post("/api/video/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    manual_id: str = Query(...),
    video_id: str = Query(...),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze video to detect assembly steps.

    This is a long-running operation (2-5 minutes).

    Args:
        manual_id: Manual ID
        video_id: Video ID from upload

    Returns:
        Analysis job information
    """
    try:
        # Get video path
        video_path = video_storage.get_video_path(manual_id, video_id)
        if not video_path:
            raise HTTPException(
                status_code=404,
                detail=f"Video {video_id} not found for manual {manual_id}"
            )

        # Check if manual data exists
        output_dir = Path(settings.output_dir)
        extracted_path = output_dir / f"{manual_id}_extracted.json"
        dependencies_path = output_dir / f"{manual_id}_dependencies.json"

        if not extracted_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Manual {manual_id} not processed. Run manual processing first."
            )

        # Get reference images (first 10 step images)
        temp_pages_dir = output_dir / "temp_pages"
        reference_images = []
        if temp_pages_dir.exists():
            reference_images = sorted(temp_pages_dir.glob("page_*.png"))[:10]
            reference_images = [str(p) for p in reference_images]

        # Create analysis task
        analysis_id = str(uuid.uuid4())[:8]
        task_manager.create_task(
            analysis_id,
            "video_analysis",
            {"manual_id": manual_id, "video_id": video_id}
        )

        # Start background analysis
        background_tasks.add_task(
            run_video_analysis,
            video_path=video_path,
            manual_id=manual_id,
            analysis_id=analysis_id,
            extracted_path=str(extracted_path),
            dependencies_path=str(dependencies_path) if dependencies_path.exists() else None,
            reference_images=reference_images
        )

        logger.info(f"Started video analysis: {analysis_id}")

        return VideoAnalysisResponse(
            analysis_id=analysis_id,
            message=f"Analysis started. Check status at /api/video/analysis/{analysis_id}"
        )

    except Exception as e:
        logger.error(f"Error starting video analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video/analysis/{analysis_id}", response_model=AnalysisResults)
async def get_analysis_status(analysis_id: str):
    """
    Get status of video analysis or overlay generation.

    Args:
        analysis_id: Analysis job ID or overlay job ID

    Returns:
        Analysis status or results if complete
    """
    try:
        task_info = task_manager.get_task(analysis_id)

        if not task_info:
            raise HTTPException(
                status_code=404,
                detail=f"Task {analysis_id} not found"
            )

        # Handle overlay tasks differently
        is_overlay_task = analysis_id.startswith("overlay_")

        if task_info["status"] == TaskStatus.COMPLETED:
            if is_overlay_task:
                # For overlay tasks, return the task metadata as results
                return AnalysisResults(
                    analysis_id=analysis_id,
                    status="completed",
                    results={
                        "type": "overlay",
                        "analysis_id": task_info["metadata"].get("analysis_id"),
                        "manual_id": task_info["metadata"].get("manual_id")
                    },
                    processing_time_sec=task_info.get("duration_sec")
                )
            else:
                # For analysis tasks, load results file
                manual_id = task_info["metadata"]["manual_id"]
                results = analysis_storage.load_analysis_results(manual_id, analysis_id)

                if results is None:
                    logger.warning(f"Results file not found for completed analysis {analysis_id}")
                    # Return with empty results if file missing
                    results = {}

                return AnalysisResults(
                    analysis_id=analysis_id,
                    status="completed",
                    results=results,
                    processing_time_sec=task_info.get("duration_sec")
                )
        else:
            # Return status for in-progress tasks
            return AnalysisResults(
                analysis_id=analysis_id,
                status=task_info["status"],
                results={
                    "progress_percentage": task_info["progress_percentage"],
                    "current_step": task_info["current_step"]
                }
            )

    except Exception as e:
        logger.error(f"Error getting analysis status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_overlay_generation(
    video_path: str,
    analysis_id: str,
    manual_id: str,
    overlay_id: str,
    options: OverlayOptions
):
    """Background task for overlay generation."""
    try:
        task_manager.update_task(
            overlay_id,
            status=TaskStatus.PROCESSING,
            progress=10,
            current_step="Loading analysis results"
        )

        # Load analysis results
        results = analysis_storage.load_analysis_results(manual_id, analysis_id)

        task_manager.update_task(
            overlay_id,
            progress=30,
            current_step="Rendering overlay video"
        )

        # Generate overlay
        overlay_output = f"/tmp/overlay_{overlay_id}.mp4"
        create_overlay_video(
            video_path=video_path,
            analysis_results=results,
            output_path=overlay_output,
            show_target=options.show_target_marker,
            show_hud=options.show_hud_panel,
            show_instruction=options.show_instruction_card
        )

        task_manager.update_task(
            overlay_id,
            progress=80,
            current_step="Saving overlay video"
        )

        # Save overlay
        video_id = results.get('video_path', '').split('/')[-1].split('_')[0]
        video_storage.save_overlay_video(overlay_output, manual_id, video_id)

        # Cleanup temp file
        Path(overlay_output).unlink()

        task_manager.update_task(
            overlay_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            current_step="Overlay complete"
        )

        logger.info(f"Overlay generation completed: {overlay_id}")

    except Exception as e:
        logger.error(f"Overlay generation failed: {e}")
        task_manager.update_task(
            overlay_id,
            status=TaskStatus.ERROR,
            error=str(e)
        )


@app.post("/api/video/overlay", response_model=OverlayGenerationResponse)
async def generate_overlay(
    analysis_id: str = Query(...),
    options: OverlayOptions = None,
    background_tasks: BackgroundTasks = None
):
    """
    Generate annotated video with visual overlays.

    Args:
        analysis_id: Analysis ID
        options: Overlay rendering options

    Returns:
        Overlay generation job info
    """
    try:
        if options is None:
            options = OverlayOptions()

        # Load analysis results to get video path and manual_id
        task_info = task_manager.get_task(analysis_id)
        if not task_info or task_info["status"] != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail="Analysis must be completed before generating overlay"
            )

        manual_id = task_info["metadata"]["manual_id"]
        video_id = task_info["metadata"]["video_id"]

        video_path = video_storage.get_video_path(manual_id, video_id)
        if not video_path:
            raise HTTPException(status_code=404, detail="Video not found")

        # Create overlay task
        overlay_id = f"overlay_{str(uuid.uuid4())[:8]}"
        task_manager.create_task(
            overlay_id,
            "overlay_generation",
            {"analysis_id": analysis_id, "manual_id": manual_id}
        )

        # Start background generation
        background_tasks.add_task(
            run_overlay_generation,
            video_path=video_path,
            analysis_id=analysis_id,
            manual_id=manual_id,
            overlay_id=overlay_id,
            options=options
        )

        return OverlayGenerationResponse(
            overlay_id=overlay_id,
            estimated_time_sec=90
        )

    except Exception as e:
        logger.error(f"Error starting overlay generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video/step-at-time", response_model=ActiveStepResponse)
async def get_step_at_timestamp(
    analysis_id: str = Query(...),
    timestamp_sec: float = Query(...)
):
    """
    Get which step is active at a specific timestamp.

    Args:
        analysis_id: Analysis ID
        timestamp_sec: Time in seconds

    Returns:
        Active step information
    """
    try:
        task_info = task_manager.get_task(analysis_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="Analysis not found")

        manual_id = task_info["metadata"]["manual_id"]
        results = analysis_storage.load_analysis_results(manual_id, analysis_id)

        # Find active event
        active_event = None
        for event in results.get('detected_events', []):
            if event['start_seconds'] <= timestamp_sec <= event['end_seconds']:
                active_event = event
                break

        if active_event:
            return ActiveStepResponse(
                timestamp_sec=timestamp_sec,
                active_step=active_event
            )
        else:
            return ActiveStepResponse(
                timestamp_sec=timestamp_sec,
                active_step=None,
                message="No step detected at this timestamp"
            )

    except Exception as e:
        logger.error(f"Error getting step at timestamp: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video/download/{overlay_id}")
async def download_overlay_video(
    overlay_id: str
):
    """
    Download generated overlay video.

    Args:
        overlay_id: Overlay ID

    Returns:
        Video file
    """
    try:
        # Extract analysis_id from overlay_id task
        task_info = task_manager.get_task(overlay_id)
        if not task_info or task_info["status"] != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=404,
                detail="Overlay not found or not yet complete"
            )

        manual_id = task_info["metadata"]["manual_id"]
        analysis_id = task_info["metadata"]["analysis_id"]

        # Get analysis to find video_id
        analysis_results = analysis_storage.load_analysis_results(manual_id, analysis_id)
        video_path = analysis_results.get('video_path', '')
        video_id = video_path.split('/')[-1].split('_')[0]

        # Get overlay path
        overlay_path = video_storage.get_overlay_path(manual_id, video_id)

        if not overlay_path or not Path(overlay_path).exists():
            raise HTTPException(status_code=404, detail="Overlay video not found")

        return FileResponse(
            overlay_path,
            media_type="video/mp4",
            filename=f"assembly_overlay_{overlay_id}.mp4"
        )

    except Exception as e:
        logger.error(f"Error downloading overlay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Startup & Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting LEGO RAG API server")
    logger.info(f"Vector store: {settings.chroma_persist_dir}")
    logger.info(f"Output directory: {settings.output_dir}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LEGO RAG API server")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
