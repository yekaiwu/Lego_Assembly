"""
FastAPI Main Application for LEGO RAG System.
Provides REST API endpoints for manual ingestion and querying.
"""

from fastapi import FastAPI, HTTPException, Query, Path as PathParam, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from typing import Optional, List
import uuid
import shutil
from loguru import logger

from .config import get_settings
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
    PartInfo
)
from .ingestion.ingest_service import get_ingestion_service
from .rag.pipeline import get_rag_pipeline
from .vector_store.chroma_manager import get_chroma_manager
from .vision import get_state_analyzer, get_state_comparator, get_guidance_generator

# Initialize FastAPI app
app = FastAPI(
    title="LEGO Assembly RAG API",
    description="Retrieval-Augmented Generation system for LEGO assembly guidance",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = get_settings()
ingestion_service = get_ingestion_service()
rag_pipeline = get_rag_pipeline()
chroma_manager = get_chroma_manager()


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
async def ingest_manual(
    manual_id: str = PathParam(..., description="Manual identifier (e.g., 6454922)")
):
    """
    Ingest a specific manual into the vector store.
    The manual files must exist in the output directory.
    """
    try:
        result = ingestion_service.ingest_manual(manual_id)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return IngestionResponse(**result)
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Manual files not found for {manual_id}. Please process the manual first using main.py"
        )
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/all")
async def ingest_all_manuals():
    """
    Ingest all available manuals from the output directory.
    """
    try:
        result = ingestion_service.ingest_all_manuals()
        return result
    except Exception as e:
        logger.error(f"Bulk ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/manual/{manual_id}")
async def delete_manual(
    manual_id: str = PathParam(..., description="Manual identifier")
):
    """
    Remove a manual from the vector store.
    """
    try:
        result = ingestion_service.remove_manual(manual_id)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return result
        
    except Exception as e:
        logger.error(f"Deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api/manual/{manual_id}/step/{step_number}", response_model=QueryResponse)
async def get_step(
    manual_id: str = PathParam(..., description="Manual identifier"),
    step_number: int = PathParam(..., description="Step number", ge=1)
):
    """
    Get detailed information for a specific step.
    """
    try:
        response = rag_pipeline.get_step_details(manual_id, step_number)
        return response
        
    except Exception as e:
        logger.error(f"Step retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Manual Management ====================

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


# ==================== Image Serving ====================

@app.get("/api/image")
async def serve_image(
    path: str = Query(..., description="Image path")
):
    """
    Serve step images.
    """
    try:
        image_path = Path(path)
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(image_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Vision/State Analysis API ====================

@app.post("/api/vision/upload-images", response_model=ImageUploadResponse)
async def upload_assembly_images(
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
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create upload directory
        upload_dir = Path(f"/tmp/lego_assembly_uploads/{session_id}")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        uploaded_files = []
        for i, image in enumerate(images):
            # Validate file type
            if not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {image.filename} is not an image"
                )
            
            # Save file
            file_extension = Path(image.filename).suffix or ".jpg"
            file_path = upload_dir / f"image_{i + 1}{file_extension}"
            
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
        logger.error(f"Error uploading images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

