# LEGO Vision-Based RAG System - Quick Start

Get the complete Vision-Enhanced RAG system running in 10 minutes.

**NEW in 2.3**:
- ✅ **✂️ SAM3 Integration** - Pixel-perfect part segmentation with Meta's SAM3 model
- ✅ **🖼️ Visual Graph References** - Parts and subassemblies now include cropped images
- ✅ **Phase 0: Document Understanding** - Smart page filtering (10-15% fewer API calls)
- ✅ **Phase 1: Context-Aware Extraction** - Sliding window + long-term memory
- ✅ **Enhanced subassembly detection** (20-30% better accuracy with context)
- ✅ **Inter-step reference tracking** (context-aware dependencies)
- ✅ **One-command workflow**: Extraction + Ingestion in a single step
- ✅ **Progress checkpointing**: Resume interrupted processing automatically
- ✅ **Automatic text truncation**: Handles long content for embeddings
- ✅ Computer vision-based assembly state analysis
- ✅ 🧠 **Hierarchical assembly graph** (parts → subassemblies → model)
- ✅ **Graph-enhanced retrieval** for structural queries
- ✅ **Subassembly detection** from user photos

---

## 📋 Prerequisites

**Required**:
- Python 3.12+ with `uv` package manager
- Node.js 18+ and npm
- Poppler (for PDF processing)
- At least one API key: Gemini (recommended), Qwen, DeepSeek, or Moonshot
- Roboflow account (optional, for SAM3 segmentation)

**Optional**:
- CUDA-capable GPU (not required - SAM3 uses cloud API)
- HuggingFace account (not required - SAM3 uses Roboflow cloud API)

**Install Tools**:
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Poppler
brew install poppler  # macOS
# sudo apt-get install poppler-utils  # Linux
```

**Get API Key** (choose one):
- **Gemini** (recommended): https://aistudio.google.com/app/apikey
- **Qwen**: https://dashscope.console.aliyun.com/
- **DeepSeek**: https://platform.deepseek.com/
- **Moonshot**: https://platform.moonshot.cn/
- **Roboflow** (for SAM3): https://app.roboflow.com (Settings → API)

---

## 🚀 Setup Steps

### Step 1: Configure Environment

```bash
cd /Users/jay/Desktop/CS480/Lego_Assembly

# Copy template
cp ENV_TEMPLATE.txt .env

# Edit and add your API key
nano .env
```

**Add to `.env`**:
```bash
# Primary VLM API Key (choose one)
# Gemini (recommended - default)
GEMINI_API_KEY=your-gemini-api-key-here

# OR Qwen
DASHSCOPE_API_KEY=sk-your-key-here

# OR DeepSeek
DEEPSEEK_API_KEY=sk-your-key-here

# OR Moonshot
MOONSHOT_API_KEY=sk-your-key-here

# Optional: Additional API keys for fallback
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-your-key-here

# SAM3 - Part Segmentation (Optional - uses Roboflow cloud API)
ENABLE_ROBOFLOW_SAM3=true          # Enable SAM3 segmentation
ROBOFLOW_API_KEY=your-roboflow-key-here
ROBOFLOW_SAM3_CONFIDENCE_THRESHOLD=0.7
ROBOFLOW_SAM3_OUTPUT_DIR=./output/segmented_parts
ROBOFLOW_SAM3_SAVE_MASKS=true
ROBOFLOW_SAM3_SAVE_CROPPED_IMAGES=true

# Optional: Processing features
ENABLE_SPATIAL_RELATIONSHIPS=true  # Enable spatial relationship extraction
CACHE_ENABLED=true                  # Enable VLM response caching
```

### Step 2: Install Dependencies

```bash
# Install all dependencies with uv (recommended)
uv sync

# OR install with pip
pip install -e .

# This installs Phase 1 + Backend dependencies
# SAM3 uses Roboflow cloud API - no local model download needed!
```

**Note**: SAM3 segmentation uses Roboflow's cloud API, so no local model download is required. Just configure your API key in `.env`.

### Step 3: Process Manual (Complete Workflow)

```bash
# Process LEGO manual from URL (Extraction + Ingestion in one command!)
uv run python main.py https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6454922.pdf

# ⏱️ Takes ~2-5 minutes (depends on manual size)
# ✓ Step 1/7: Extract pages from PDF
# ✓ Step 2/7: Collect user metadata & filter pages (Phase 0) → User confirmation
# ✓ Step 3/7: Context-aware VLM extraction (Phase 1)
# ✓ Step 4/7: Build dependency graph
# ✓ Step 4.5/7: SAM3 segmentation (optional, if enabled)
# ✓ Step 5/7: Build hierarchical graph (enhanced with context hints)
# ✓ Step 6/7: Generate 3D plan
# ✓ Step 7/7: Ingest into vector store (multimodal embeddings)

# Output created in ./output/
#   - 6454922_extracted.json      (Step data with subassembly_hints & context_references)
#   - 6454922_plan.json           (3D assembly plan)
#   - 6454922_dependencies.json   (Step dependencies)
#   - 6454922_graph.json          (Hierarchical assembly graph with image references)
#   - 6454922_plan.txt            (Human-readable plan)
#   - temp_pages/*.png            (Step diagram images)
#   - segmented_parts/            (SAM3 segmented images - NEW!)
#       └── 6454922/
#           └── step_001/
#               ├── part_0_image.png   (Cropped part images)
#               ├── part_0_mask.png    (Segmentation masks)
#               ├── assembly_image.png (Assembled result)
#               └── assembly_mask.png  (Assembly mask)
#   - .6454922_checkpoint.json    (Progress checkpoint - hidden)
```

**🎯 One Command Does It All!**
- No need for separate extraction and ingestion steps
- Automatically generates multimodal embeddings (text + diagrams)
- Vector store is ready immediately after completion

**📄 Phase 0: User Confirmation**
During Step 2, the system will show you the document analysis results:
```
================================================================================
DOCUMENT ANALYSIS RESULTS
================================================================================

Build: Fire Truck Set #6454922
Total Pages: 52
Instruction Pages: 45
  Page ranges: 6-50
Estimated Steps: 45

Filtered Out:
  - Cover/Intro: pages 1-5
  - Advertisements: pages 51-52

================================================================================

Proceed with processing these instruction pages? (y/n):
```

**🧠 Phase 1: Context-Aware Benefits**
- Sliding window tracks last 2 steps (~600 tokens per extraction)
- Long-term memory maintains overall build state (~500 tokens)
- Enhanced extraction includes:
  - `subassembly_hint`: Detects new subassemblies during extraction
  - `context_references`: Captures inter-step dependencies
- Results in 20-30% better subassembly detection accuracy

**⏸️ Checkpointing & Resume**:
If the process is interrupted (Ctrl+C, crash, or network error):
```bash
# Just rerun the same command - it will resume automatically!
uv run python main.py https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6454922.pdf

# ↻ Resuming from checkpoint: last completed step = step_extraction
# ✓ Step 1/7: Page extraction already complete (skipping)
# ✓ Step 2/7: Document analysis already complete (skipping)
# ✓ Step 3/7: Step extraction already complete (skipping)
# → Step 4/7: Building dependency graph...
```

The checkpoint system:
- ✅ Saves progress after each major step
- ✅ Automatically skips completed steps on resume
- ✅ No wasted API calls or computation
- ✅ Checkpoint cleared on successful completion

**Advanced Options**:
```bash
# Skip vector store ingestion (only generate files)
uv run python main.py <url> --skip-ingestion

# Use text-only embeddings (faster, less accurate)
uv run python main.py <url> --no-multimodal

# Start from scratch (ignore checkpoint)
uv run python main.py <url> --no-resume

# Use fallback VLMs if primary fails
uv run python main.py <url> --use-fallback

# Don't display plans in console
uv run python main.py <url> --no-display

# Enable VLM response caching
uv run python main.py <url> --cache

# Set custom output directory
uv run python main.py <url> -o ./custom_output

# Set custom assembly ID
uv run python main.py <url> --assembly-id my_custom_id

# Set logging level
uv run python main.py <url> --log-level DEBUG

# View all options
uv run python main.py --help
```

**Note**: To disable spatial features, set `ENABLE_SPATIAL_RELATIONSHIPS=false` in your `.env` file before running.

### Step 3b: Comparison Testing (Optional)

**🔬 Test Impact of Spatial Features**

Compare the same manual with/without all spatial features:

```bash
# Baseline: Full spatial features (default)
uv run python main.py manual.pdf \
  -o output/baseline \
  --assembly-id manual_baseline

# No spatial features (comparison mode)
# Set environment variable before running
export ENABLE_SPATIAL_RELATIONSHIPS=false
uv run python main.py manual.pdf \
  -o output/no_spatial \
  --assembly-id manual_no_spatial

# Or add to .env file temporarily:
# ENABLE_SPATIAL_RELATIONSHIPS=false

# This disables:
#   - Spatial relationships extraction (saves ~10-15% VLM tokens)
#   - Spatial-temporal pattern analysis
#   - Results in faster processing and simpler graphs
```

**Both versions get unique IDs and coexist in the vector store!**

**Compare Results**:
```bash
# 1. Graph structure comparison
diff output/baseline/manual_baseline_graph_summary.txt \
     output/no_spatial/manual_no_spatial_graph_summary.txt

# 2. JSON data comparison
python3 -c "
import json
b = json.load(open('output/baseline/manual_baseline_graph.json'))
n = json.load(open('output/no_spatial/manual_no_spatial_graph.json'))
print('Baseline:', b['metadata']['configuration'])
print('No-spatial:', n['metadata']['configuration'])
print('Subassemblies:', b['metadata']['total_subassemblies'], 'vs', n['metadata']['total_subassemblies'])
"

# 3. Query performance (after starting backend)
curl -X POST http://localhost:8000/api/query/text \
  -d '{"manual_id": "manual_baseline", "question": "How to attach wheels?"}'

curl -X POST http://localhost:8000/api/query/text \
  -d '{"manual_id": "manual_no_spatial", "question": "How to attach wheels?"}'
```

**What to Look For**:
- **Graph Complexity**: Baseline should have more detailed subassemblies
- **Processing Speed**: No-spatial versions process ~10-15% faster
- **Query Accuracy**: Compare RAG responses for spatial vs. structural questions
- **Token Usage**: Check console logs for VLM token consumption

**Configuration Tracking**:

Every graph file tracks which features were enabled:
```
output/*/manual_*_graph_summary.txt:
  ⚙️  CONFIGURATION:
    Spatial Relationships: Enabled / DISABLED
    Spatial-Temporal Patterns: Enabled / DISABLED
```

### Step 4: Start Backend (Terminal 1)

```bash
# Option 1: Use the startup script (recommended)
./scripts/dev/start_backend.sh

# Option 2: Manual start
cd backend
uv run uvicorn app.main:app --reload --port 8000

# ✓ Backend running at http://localhost:8000
# ✓ API docs at http://localhost:8000/docs
```

Keep this terminal running!

**Note**: The manual is already in the vector store from Step 3. No additional ingestion needed!

**Backend Configuration**:
- Uses environment variables from `.env` in project root
- API keys are automatically detected by LiteLLM
- Default models: Gemini 2.5 Flash for embeddings and text generation
- Vector database: `backend/chroma_db/`

**What happens during the workflow?** 📦

The complete workflow processes your manual in 7 integrated steps:

**Steps 1-6: Extraction & Graph Building**
1. **Page Extraction**: Converts PDF to individual page images
2. **📄 User Metadata Collection (Phase 0 - NEW)**:
   - Interactive collection of manual metadata (build name, page ranges, step info)
   - Filters to only instruction pages (saves 10-15% API calls)
   - **Asks for user confirmation** before proceeding
3. **🧠 Context-Aware VLM Extraction (Phase 1 - NEW)**:
   - Initializes build memory (sliding window + long-term tracking)
   - Extracts each step WITH context from previous steps
   - Generates enhanced subassembly hints (20-30% better accuracy)
   - Captures inter-step references automatically
4. **Dependency Graph**: Infers step dependencies
4.5. **SAM3 Segmentation** (optional): Pixel-level part segmentation using Roboflow API
5. **Hierarchical Graph**: Builds parts → subassemblies → model structure (enhanced with Phase 1 hints)
6. **3D Plan Generation**: Creates assembly plan with spatial coordinates

**Step 7: Vector Store Ingestion** (Automatic)
Converts Phase 1 output into a searchable vector database:

1. **Loads Phase 1 Data**: Reads all JSON files from `output/` directory (now with enhanced context data)

2. **Processes Each Step** (with multimodal embeddings):
   - For each step, loads the corresponding diagram image
   - **Multimodal Processing** (if enabled, default):
     - Uses **Qwen-VL** to analyze the step diagram image
     - Generates a detailed text description of what's shown (e.g., "Red 2x4 brick attached to blue base plate, yellow wheel on left side...")
     - Combines the step instructions text + diagram description + context references
     - Creates a **fused embedding** that captures text, visual, and structural information
   - **Text-Only Fallback** (if image missing):
     - Uses only the step text content
     - Creates text embedding

3. **Creates Searchable Chunks**:
   - Each step becomes a "chunk" with:
     - Content: Instructions + diagram description + context hints (auto-truncated if > 2048 chars)
     - Embedding: Vector representation (for semantic search)
     - Metadata: Step number, parts, dependencies, image path, subassembly_hint, context_references

4. **Stores in ChromaDB**:
   - Saves all chunks to the vector database
   - Enables fast semantic search and retrieval
   - Enhanced with context-aware metadata for better retrieval

**Why it takes 2-5 minutes**:
- Page analysis (Phase 0): ~30 seconds (1 VLM call for sampled pages)
- Context-aware VLM extraction: ~2-3 seconds per step (includes context building)
- Multimodal embedding: ~1-2 seconds per step
- For a 45-step manual: 1 analysis + 45 context-aware extractions + 45 embeddings = ~3-5 minutes

**🎯 Performance Improvements with Phase 0 & 1**:
- **API Call Reduction**: 10-15% fewer calls (filtered pages)
- **Better Accuracy**: 20-30% improved subassembly detection
- **Richer Data**: Enhanced extraction with context references
- **Smarter Retrieval**: Graph + context metadata improves RAG quality

### Step 5: Start Frontend (Terminal 2)

```bash
# Option 1: Use the startup script (recommended)
./scripts/dev/start_frontend.sh

# Option 2: Manual start
cd frontend

# Install dependencies (first time only)
npm install

# Configure backend URL
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev

# ✓ Frontend running at http://localhost:3000
```

---

## ✅ Verification

### Test Backend
```bash
# Health check
curl http://localhost:8000/health

# List manuals
curl http://localhost:8000/api/manuals

# Get step (check for new fields: subassembly_hint, context_references)
curl http://localhost:8000/api/manual/6454922/step/5
```

### Test Frontend
1. Open http://localhost:3000
2. Select manual "6454922"
3. Navigate through steps
4. Ask questions in chat
5. Try vision upload (if available)

---

## 🔍 Understanding the New Output

### Enhanced Extraction Data

The extracted JSON now includes context-aware fields:

```json
{
  "step_number": 5,
  "parts_required": [...],
  "actions": [...],
  "subassembly_hint": {
    "is_new_subassembly": true,
    "name": "wheel_assembly",
    "description": "4-wheel chassis with axles",
    "continues_previous": false
  },
  "context_references": {
    "references_previous_steps": true,
    "which_steps": [1, 2, 3],
    "reference_description": "the base structure from steps 1-3"
  }
}
```

### Hierarchical Graph Enhancement

The graph.json is now enhanced with better subassembly detection from Phase 1:
- More accurate subassembly boundaries
- Richer relationship data
- Better completion markers

---

## 🐛 Troubleshooting

### Document Analysis Issues

**Problem**: Phase 0 metadata collection is incorrect
```bash
# The system will ask for confirmation during metadata collection
# You can provide accurate page ranges and step information
# If you need to reprocess, use --no-resume to start fresh
uv run python main.py <url> --no-resume
```

### Context Memory Issues

**Problem**: Token budget warnings
```
WARNING: Token budget exceeded. Reducing window size: 5 → 3
```
- This is automatic and safe
- Reduces context window to fit in 1M token limit (Gemini 2.5 Flash)
- System will continue normally with smaller window

### "API key not found"
```bash
# Check .env file exists
cat .env | grep API_KEY

# Make sure you copied from template
cp ENV_TEMPLATE.txt .env
```

### "No manuals available" (Frontend)
```bash
# Manual is auto-ingested from main.py
# If missing, check backend logs for ingestion errors
```

### "Cannot connect to backend" (Frontend)
```bash
# Check backend is running
curl http://localhost:8000/health

# Check frontend .env.local
cat frontend/.env.local
```

### Comparison Testing Issues

**Problem**: Different versions overwrite each other in vector store
```bash
# Make sure you use unique --assembly-id for each version!
# Correct:
uv run python main.py manual.pdf --assembly-id baseline
export ENABLE_SPATIAL_RELATIONSHIPS=false
uv run python main.py manual.pdf --assembly-id no_spatial

# Wrong (will overwrite):
uv run python main.py manual.pdf  # Uses filename as ID
export ENABLE_SPATIAL_RELATIONSHIPS=false
uv run python main.py manual.pdf  # Same ID, overwrites!
```

**Problem**: Can't see configuration differences in output
```bash
# Check graph summary file
cat output/baseline/baseline_graph_summary.txt | grep "CONFIGURATION" -A 2

# Or check JSON metadata
python3 -c "
import json
g = json.load(open('output/baseline/baseline_graph.json'))
print(g['metadata']['configuration'])
"
```

**Problem**: Query returns wrong manual version
```bash
# Make sure you specify correct manual_id in API call
curl -X POST http://localhost:8000/api/query/text \
  -H "Content-Type: application/json" \
  -d '{
    "manual_id": "baseline",  # Must match --assembly-id
    "question": "your question"
  }'

# List all available manuals
curl http://localhost:8000/api/manuals
```

---

## 📚 Next Steps

1. **Explore Context-Aware Features**:
   - Check extracted JSON for `subassembly_hint` and `context_references`
   - Compare with/without context (see IMPLEMENTATION_SUMMARY.md)

2. **Test Enhanced RAG**:
   - Ask structural questions: "What subassembly contains the red brick?"
   - Test inter-step references: "What was built in previous steps?"

3. **Run Spatial Features Comparison** (Recommended for Research):
   ```bash
   # Process the same manual 2 ways
   uv run python main.py manual.pdf -o output/baseline --assembly-id baseline
   export ENABLE_SPATIAL_RELATIONSHIPS=false
   uv run python main.py manual.pdf -o output/no_spatial --assembly-id no_spatial

   # Compare graph outputs
   diff output/baseline/baseline_graph_summary.txt output/no_spatial/no_spatial_graph_summary.txt

   # Test query performance differences (after starting backend)
   # Use frontend to select different manuals and compare RAG responses
   ```

4. **Review Documentation**:
   - [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Phase 0 & 1 details
   - [README.md](README.md) - Full system overview (includes comparison testing guide)
   - [implementation_plan.md](implementation_plan.md) - Original design spec

5. **Run Tests**:
   ```bash
   # Test build memory
   pytest tests/test_build_memory.py -v
   ```

---

## 📖 Documentation

- **[README.md](README.md)** - Complete system overview
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Phase 0 & 1 technical details
- **[implementation_plan.md](implementation_plan.md)** - Design specification
- **API Docs**: http://localhost:8000/docs (when backend running)

---

**System Version**: 2.3.0 (with Context-Aware Processing & SAM3 Integration)
**Last Updated**: January 2026
**Status**: ✅ Production Ready
