# Video Analysis Integration

## Overview

The Video Analysis feature integrates aDIoT's video-based step tracking capabilities into the Lego_Assembly system. This allows users to upload assembly videos and automatically identify which steps are completed at each timestamp, with visual overlay generation support.

## Features

### 1. Video Upload & Analysis
- Upload assembly videos (MP4, MOV, AVI)
- Automatic step detection using Gemini Robotics ER VLM
- Temporal event extraction with start/end times
- Coordinate-based target tracking (normalized 0-1000 scale)

### 2. Interactive Video Player
- Timeline visualization with step markers
- Click-to-seek functionality
- Real-time active step tracking
- Confidence scores for each detected step

### 3. Visual Overlay Generation
- Green pulsing target markers at attachment points
- HUD panel showing step progression
- Instruction cards with current assembly actions
- Downloadable annotated videos

## Architecture

### Backend Components

#### 1. Video Analysis Module (`backend/app/video/`)
- **video_analyzer.py**: Gemini VLM integration for step detection
- **overlay_renderer.py**: Visual guidance overlay generation
- **coordinate_utils.py**: Coordinate validation and transformation
- **storage_manager.py**: Video and analysis result storage
- **metadata_extractor.py**: Video metadata extraction
- **video_prompts.py**: VLM prompt templates

#### 2. API Endpoints (`backend/app/main.py`)
- `POST /api/video/upload`: Upload video file
- `POST /api/video/analyze`: Start video analysis
- `GET /api/video/analysis/{analysis_id}`: Poll analysis status
- `POST /api/video/overlay`: Generate overlay video
- `GET /api/video/step-at-time`: Get step at specific timestamp
- `GET /api/video/download/{overlay_id}`: Download overlay video

### Frontend Components

#### 1. React Components (`frontend/components/`)
- **VideoUpload.tsx**: Video upload with progress tracking
- **VideoStepPlayer.tsx**: Interactive video player with timeline

#### 2. API Integration (`frontend/lib/api/client.ts`)
- Video upload with FormData
- Analysis status polling
- Overlay generation requests
- TypeScript type definitions

## Usage Flow

### 1. Process Manual (Prerequisite)
```bash
uv run python main.py https://example.com/manual.pdf
```

This creates:
- `{manual_id}_extracted.json` - Step data
- `{manual_id}_dependencies.json` - Step dependencies
- `output/temp_pages/` - Reference images

### 2. Upload Video via Frontend
1. Navigate to http://localhost:3000
2. Select a manual
3. Click "Video Analysis" tab
4. Upload assembly video (drag & drop or click)
5. Video is uploaded and analysis starts automatically

### 3. Analysis Process (2-5 minutes)
- Video uploaded to backend
- Gemini Robotics ER VLM analyzes video with manual context
- Detects assembly steps with timing and coordinates
- Validates and saves results

### 4. View Results
- Interactive timeline with color-coded step segments
- Click segments to seek to specific steps
- Active step card shows:
  - Step instruction
  - Required parts
  - Confidence score
  - Reference image from manual

### 5. Generate Overlay (Optional)
- Click "Generate Visual Overlay"
- Overlay video renders with:
  - Target markers
  - HUD panel
  - Instruction cards
- Download annotated video

## API Response Formats

### Upload Response
```json
{
  "video_id": "abc123",
  "filename": "assembly.mp4",
  "size_mb": 25.3,
  "duration_sec": 45.2,
  "fps": 30,
  "resolution": [1920, 1080],
  "status": "uploaded"
}
```

### Analysis Results
```json
{
  "analysis_id": "xyz789",
  "status": "completed",
  "results": {
    "manual_id": "6262059",
    "total_duration_sec": 45.2,
    "fps": 30,
    "detected_events": [
      {
        "step_id": 1,
        "start_seconds": 0.81,
        "end_seconds": 3.45,
        "anchor_timestamp": 2.1,
        "instruction": "Place tan 2x4 brick",
        "confidence": 0.92,
        "target_box_2d": [389, 497, 422, 532],
        "assembly_box_2d": [338, 381, 442, 617],
        "parts_required": [...]
      }
    ],
    "total_steps_detected": 7,
    "coverage_percentage": 100.0,
    "average_confidence": 0.89
  },
  "processing_time_sec": 142.5
}
```

## Configuration

### Backend Environment Variables (`.env`)
```bash
# Google API key for Gemini Robotics ER
GOOGLE_API_KEY=your_api_key_here

# Video settings (optional)
VIDEO_MAX_UPLOAD_SIZE_MB=500
VIDEO_ALLOWED_FORMATS=mp4,mov,avi
```

### Dependencies

#### Backend (pyproject.toml)
```toml
dependencies = [
    "google-generativeai>=0.3.0",  # Video analysis
    "opencv-python>=4.8.0",        # Video processing
    "python-multipart>=0.0.6",     # File uploads
    "aiofiles>=23.2.1",            # Async file ops
]
```

#### Frontend (package.json)
```json
{
  "dependencies": {
    // Existing dependencies work, no new packages needed
  }
}
```

## Storage Structure

```
Lego_Assembly/
├── uploads/videos/{manual_id}/
│   ├── {video_id}_assembly.mp4      # Original video
│   └── {video_id}_overlay.mp4       # Generated overlay
│
├── output/video_analysis/{manual_id}/
│   └── {analysis_id}_results.json   # Analysis results
│
└── backend/cache/analysis_status/
    └── {analysis_id}.json           # Task progress tracking
```

## Technical Details

### Coordinate System
- All coordinates use **normalized 0-1000 scale**
- Format: `[ymin, xmin, ymax, xmax]` (Y first!)
- Converted to pixels only during overlay rendering
- Resolution-independent for any video size

### VLM Prompt Strategy
- Includes manual steps as context
- Provides step dependencies for ordering
- Sends reference images from manual
- Specifies coordinate format explicitly
- Validates spatial containment (target inside assembly box)

### Background Processing
- Video analysis runs as FastAPI background task
- Status polling every 5 seconds via frontend
- Task manager tracks progress (0-100%)
- Results cached for instant re-access

### Overlay Rendering
- Uses PIL for high-quality graphics
- OpenCV for video encoding
- Pulsing animations via numpy sin function
- H.264 codec for compatibility

## Error Handling

### Frontend
- File format validation before upload
- Progress indicators during analysis
- Graceful error messages with retry
- Timeout handling for long operations

### Backend
- Coordinate validation and clamping
- Video format verification
- Manual existence checks
- Task status error tracking

## Performance Considerations

### Video Analysis Time
- 1-minute video: ~2-3 minutes
- 5-minute video: ~5-8 minutes
- Factors: VLM processing, frame count, complexity

### Overlay Generation Time
- 1-minute video: ~30-60 seconds
- Depends on: resolution, FPS, number of steps

### Optimization Tips
1. Use H.264 encoded videos for faster upload
2. Limit video length to 5 minutes for faster analysis
3. Ensure stable internet for VLM API calls
4. Videos are processed sequentially (queue if needed)

## Troubleshooting

### Analysis Fails
- Check GOOGLE_API_KEY is set correctly
- Verify manual was processed first
- Ensure video format is supported
- Check video file isn't corrupted

### Overlay Not Generating
- Confirm analysis completed successfully
- Check disk space for output videos
- Verify OpenCV installation

### Timeline Not Showing Steps
- Check analysis_results.detected_events is populated
- Verify step_id matches manual steps
- Inspect browser console for errors

## Future Enhancements

### Planned Features
1. **Real-time Analysis**: Live video stream processing
2. **Error Detection**: Identify incorrect part placements
3. **Multi-angle Support**: Analyze from multiple camera angles
4. **Voice Guidance**: Audio narration overlay
5. **AR Mode**: Live camera overlay (mobile)
6. **Collaborative Sharing**: Share annotated videos

### Optimization Opportunities
1. Video compression before analysis
2. Frame sampling (analyze every Nth frame)
3. Parallel VLM requests for independent segments
4. GPU acceleration for overlay rendering
5. CDN for video storage and streaming

## Credits

This integration combines:
- **aDIoT System**: Video analysis and coordinate tracking
- **Lego_Assembly**: Manual processing and RAG system
- **Gemini Robotics ER**: VLM for step detection
- **OpenCV**: Video processing
- **PIL/Pillow**: Visual overlay rendering

## License

MIT License - Same as parent project
