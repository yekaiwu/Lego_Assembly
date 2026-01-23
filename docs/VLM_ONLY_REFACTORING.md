# VLM-Only Refactoring Documentation

## Overview

This document describes the refactoring from a complex multi-stage pipeline to a simplified VLM-only approach for LEGO assembly state detection.

## Motivation

The original implementation had three stages:

1. **VLM Part Detection** (`StateAnalyzer.analyze_assembly_state`)
   - Detected parts, structures, and connections from photos
   - Returned structured JSON with part details

2. **Text-Based Matching** (`StateMatcher.match_state`)
   - Converted detected parts to node IDs using fuzzy string matching
   - Matched against step_states using F1 scoring
   - Handled color:shape signature matching with normalization

3. **Visual Matching** (`VisualMatcher.match_user_assembly_to_graph`)
   - SAM3 segmentation to isolate the assembly
   - ORB feature extraction and matching
   - Compared against reference step images
   - Combined text + visual with weighted scores

### Problems with Complex Pipeline

- **Multiple failure points**: SAM3 API, ORB features, fuzzy matching
- **Complex configuration**: Multiple weight parameters to tune
- **External dependencies**: Roboflow SAM3 API required
- **Processing time**: Multiple API calls and feature extraction
- **Maintenance burden**: Three separate systems to debug and maintain
- **Accuracy issues**: Part descriptions often don't match exactly

## New Simplified Approach

### Single VLM Call

Replace the entire pipeline with one VLM call that directly detects the current step:

```python
# Old (complex)
detected_state = state_analyzer.analyze_assembly_state(images, manual_id)
visual_matches = visual_matcher.match_user_assembly_to_graph(...)
text_matches = state_matcher.match_state(...)
combined = combine_matches(visual_matches, text_matches)

# New (simple)
step_detection = direct_analyzer.detect_current_step(images, manual_id)
```

### Architecture

```
User Photos → DirectStepAnalyzer → Single VLM Call → Step Number + Confidence
                                                            ↓
                                                    GuidanceGenerator
                                                            ↓
                                                    Next Step Guidance
```

### Benefits

1. **Simplicity**: One VLM call instead of multiple stages
2. **Speed**: Faster execution, fewer API calls
3. **Reliability**: Fewer failure points
4. **Maintainability**: Easier to debug and improve
5. **Accuracy**: Modern VLMs can reason about assembly state directly
6. **No external dependencies**: No SAM3, no ORB processing

## Implementation Details

### New Components

#### 1. DirectStepAnalyzer
**File**: `backend/app/vision/direct_step_analyzer.py`

Main class for simplified step detection:

```python
class DirectStepAnalyzer:
    def detect_current_step(
        self,
        image_paths: List[str],
        manual_id: str
    ) -> Dict[str, Any]:
        """
        Directly detect which step the user is on.

        Returns:
            {
                "step_number": int,
                "confidence": float,
                "reasoning": str,
                "next_step": int or None,
                "assembly_status": {...}
            }
        """
```

#### 2. Direct Step Detection Prompt
**File**: `backend/app/vision/prompts/direct_step_detection_prompt.txt`

VLM prompt that:
- Provides manual steps as context
- Asks VLM to identify current step from photos
- Requests structured JSON response with reasoning

Key features:
- Uses cumulative assembly logic (step N includes all parts from 1-N)
- Provides clear confidence scoring guidelines
- Handles edge cases (no parts, completion, errors)

#### 3. Updated API Endpoint
**File**: `backend/app/main.py:980-1100`

Simplified workflow:
```python
@app.post("/api/vision/analyze")
async def analyze_assembly_state(...):
    # Step 1: Direct VLM detection
    step_detection = direct_analyzer.detect_current_step(
        image_paths=image_paths,
        manual_id=manual_id
    )

    # Step 2: Generate guidance
    guidance = guidance_generator.generate_guidance_for_step(
        manual_id=manual_id,
        current_step=step_detection["step_number"],
        next_step=step_detection["next_step"],
        ...
    )

    # Step 3: Return response
    return StateAnalysisResponse(...)
```

#### 4. Enhanced GuidanceGenerator
**File**: `backend/app/vision/guidance_generator.py`

New method for simplified workflow:
```python
def generate_guidance_for_step(
    self,
    manual_id: str,
    current_step: int,
    next_step: Optional[int],
    ...
) -> Dict[str, Any]:
```

Generates faster, simpler instructions without LLM overhead.

### Removed Components

#### Deleted Files
- `backend/app/vision/visual_matcher.py` - SAM3 + ORB visual matching
- `backend/app/graph/state_matcher.py` - Text-based F1 matching
- `backend/app/vision/state_comparator.py` - TF-IDF comparison

#### Deprecated Components
- `StateAnalyzer.match_state_to_graph()` - Now returns simplified fallback
  - Kept for RAG pipeline backward compatibility
  - Logs deprecation warnings

### Testing

#### Unit Tests
**File**: `tests/unit/test_direct_step_analyzer.py`

Comprehensive tests covering:
- Successful step detection
- Edge cases (no parts, completion, errors)
- Validation and clamping
- Prompt building
- Error handling

#### Integration Tests
**File**: `tests/integration/test_vlm_only_workflow.py`

End-to-end workflow tests:
- Complete workflow for different steps
- Low confidence handling
- Multiple image angles
- Progress calculation
- Error scenarios
- Performance verification (single VLM call)

## Configuration Changes

### Removed Environment Variables

The following are **no longer needed**:

- `VISUAL_MATCH_WEIGHT` - Visual matching weight (deprecated)
- `TEXT_MATCH_WEIGHT` - Text matching weight (deprecated)
- `ENABLE_ROBOFLOW_SAM3` - SAM3 toggle (no longer used)
- `ROBOFLOW_API_KEY` - SAM3 API key (no longer used)
- `ROBOFLOW_SAM3_CONFIDENCE` - SAM3 confidence threshold (no longer used)

### Retained Variables

- `STATE_ANALYSIS_VLM` - VLM model for state analysis (still used)
- `RAG_LLM_PROVIDER` - LLM for guidance generation (still used)
- `RAG_LLM_MODEL` - LLM model name (still used)

## Migration Guide

### For New Code

Use the new simplified approach:

```python
from backend.app.vision import DirectStepAnalyzer, GuidanceGenerator

# Detect step
analyzer = DirectStepAnalyzer()
detection = analyzer.detect_current_step(image_paths, manual_id)

# Generate guidance
guidance_gen = GuidanceGenerator()
guidance = guidance_gen.generate_guidance_for_step(
    manual_id=manual_id,
    current_step=detection["step_number"],
    next_step=detection["next_step"],
    output_dir=output_dir
)
```

### For Existing Code (RAG Pipeline)

Legacy StateAnalyzer still works but logs warnings:

```python
from backend.app.vision import StateAnalyzer  # Legacy

analyzer = StateAnalyzer()
# analyze_assembly_state() still works
# match_state_to_graph() returns simplified results
```

## Performance Comparison

### Old Pipeline
```
┌─────────────────────────────────────┐
│ 1. VLM Part Detection (~2-3s)      │
│    ├─ Analyze images                │
│    └─ Extract parts JSON            │
├─────────────────────────────────────┤
│ 2. Visual Matching (~5-7s)          │
│    ├─ SAM3 Segmentation (API call)  │
│    ├─ ORB Feature Extraction        │
│    └─ Compare with reference images │
├─────────────────────────────────────┤
│ 3. Text Matching (~1-2s)            │
│    ├─ Build part signatures         │
│    ├─ Fuzzy matching                │
│    └─ F1 scoring                    │
├─────────────────────────────────────┤
│ 4. Combine Results (~0.5s)          │
│    └─ Weighted fusion               │
└─────────────────────────────────────┘
Total: ~9-13 seconds
```

### New Pipeline
```
┌─────────────────────────────────────┐
│ 1. Direct Step Detection (~2-3s)   │
│    └─ Single VLM call               │
├─────────────────────────────────────┤
│ 2. Generate Guidance (~0.5s)        │
│    └─ Format instructions           │
└─────────────────────────────────────┘
Total: ~2.5-3.5 seconds
```

**Speed improvement**: ~3-4x faster

## Accuracy Considerations

### Advantages of VLM-Only

1. **Better context understanding**: VLM sees the full assembly, not just extracted features
2. **Handles variations**: Can reason about different angles, lighting, partial builds
3. **Natural reasoning**: Can explain why a step matches
4. **No matching errors**: No fuzzy string matching or part signature issues

### Potential Limitations

1. **VLM model quality**: Depends on VLM's vision capabilities
2. **Prompt engineering**: Requires well-designed prompts
3. **No feature-level comparison**: Doesn't use ORB features (but this may be unnecessary)

### Recommended VLM Models

- **Primary**: Qwen2-VL (good balance of speed/accuracy)
- **Alternative**: GPT-4V (higher accuracy, slower)
- **Experimental**: Gemini Pro Vision (fast, good spatial reasoning)

## Future Improvements

### Potential Enhancements

1. **Multi-step reasoning**: Ask VLM to verify detected step
2. **Error detection**: Enhanced prompts for common mistakes
3. **Confidence calibration**: Learn optimal confidence thresholds
4. **Few-shot examples**: Include example detections in prompt
5. **Step sequence validation**: Ensure logical step progression

### Monitoring

Track these metrics:
- **Detection accuracy**: % of correct step identifications
- **Confidence correlation**: How well confidence predicts accuracy
- **Speed**: Average detection time
- **Error rate**: % of failures or low-confidence detections

## Rollback Plan

If needed, old components can be restored:

1. Restore deleted files from git history:
   ```bash
   git checkout main~1 -- backend/app/vision/visual_matcher.py
   git checkout main~1 -- backend/app/graph/state_matcher.py
   git checkout main~1 -- backend/app/vision/state_comparator.py
   ```

2. Revert API endpoint changes in `main.py`

3. Re-enable environment variables in `.env`

## Questions or Issues

For questions about the refactoring, see:
- **Design Rationale**: This document, "Motivation" section
- **Implementation**: Code in `backend/app/vision/direct_step_analyzer.py`
- **Tests**: `tests/unit/test_direct_step_analyzer.py`
- **API Changes**: `backend/app/main.py:980-1100`

---

**Last Updated**: 2026-01-23
**Author**: Refactoring Team
**Version**: 2.0 (VLM-Only Approach)
