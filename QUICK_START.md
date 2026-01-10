# LEGO Vision-Based RAG System - Quick Start

Get the complete Vision-Enhanced RAG system running in 10 minutes.

**NEW in 2.2**: 
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
- Python 3.9+ with `uv`
- Node.js 18+ and npm
- Poppler (for PDF processing)
- At least one API key: Qwen (recommended), DeepSeek, or Moonshot

**Install Tools**:
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Poppler
brew install poppler  # macOS
# sudo apt-get install poppler-utils  # Linux
```

**Get API Key** (choose one):
- **Qwen** (recommended): https://dashscope.console.aliyun.com/
- **DeepSeek**: https://platform.deepseek.com/
- **Moonshot**: https://platform.moonshot.cn/

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
# If using Qwen (recommended)
DASHSCOPE_API_KEY=sk-your-key-here
RAG_LLM_PROVIDER=qwen
RAG_LLM_MODEL=qwen-max

# OR if using DeepSeek
DEEPSEEK_API_KEY=sk-your-key-here
RAG_LLM_PROVIDER=deepseek
RAG_LLM_MODEL=deepseek-chat

# OR if using Moonshot
MOONSHOT_API_KEY=sk-your-key-here
RAG_LLM_PROVIDER=moonshot
RAG_LLM_MODEL=moonshot-v1-32k
```

### Step 2: Install Dependencies

```bash
# Install all dependencies with uv
uv sync

# This installs Phase 1 + Backend dependencies
```

### Step 3: Process Manual (Complete Workflow)

```bash
# Process LEGO manual from URL (Extraction + Ingestion in one command!)
uv run python main.py https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6454922.pdf

# ⏱️ Takes ~2-5 minutes (depends on manual size)
# ✓ Step 1/6: Extract pages from PDF
# ✓ Step 2/6: VLM extraction of steps
# ✓ Step 3/6: Build dependency graph
# ✓ Step 4/6: Build hierarchical graph
# ✓ Step 5/6: Generate 3D plan
# ✓ Step 6/6: Ingest into vector store (multimodal embeddings)

# Output created in ./output/
#   - 6454922_extracted.json      (Step data)
#   - 6454922_plan.json           (3D assembly plan)
#   - 6454922_dependencies.json   (Step dependencies)
#   - 6454922_graph.json          (Hierarchical assembly graph)
#   - 6454922_plan.txt            (Human-readable plan)
#   - temp_pages/*.png            (Step diagram images)
#   - .6454922_checkpoint.json    (Progress checkpoint - hidden)
```

**🎯 One Command Does It All!**
- No need for separate extraction and ingestion steps
- Automatically generates multimodal embeddings (text + diagrams)
- Vector store is ready immediately after completion

**⏸️ Checkpointing & Resume**:
If the process is interrupted (Ctrl+C, crash, or network error):
```bash
# Just rerun the same command - it will resume automatically!
uv run python main.py https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6454922.pdf

# ↻ Resuming from checkpoint: last completed step = step_extraction
# ✓ Step 1/6: Page extraction already complete (skipping)
# ✓ Step 2/6: Step extraction already complete (skipping)
# → Step 3/6: Building dependency graph...
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

# View all options
uv run python main.py --help
```

### Step 4: Start Backend (Terminal 1)

```bash
cd backend

# Start backend server (manual already ingested!)
uv run uvicorn app.main:app --reload --port 8000

# ✓ Backend running at http://localhost:8000
# ✓ API docs at http://localhost:8000/docs
```

Keep this terminal running!

**Note**: The manual is already in the vector store from Step 3. No additional ingestion needed!

**What happens during the workflow?** 📦

The complete workflow processes your manual in 6 integrated steps:

**Steps 1-5: Extraction & Graph Building**
1. **Page Extraction**: Converts PDF to individual page images
2. **VLM Extraction**: Uses VLM to extract step information (parts, actions, spatial relationships)
3. **Dependency Graph**: Infers step dependencies
4. **Hierarchical Graph**: Builds parts → subassemblies → model structure
5. **3D Plan Generation**: Creates assembly plan with spatial coordinates

**Step 6: Vector Store Ingestion** (Automatic)
Converts Phase 1 output into a searchable vector database:

1. **Loads Phase 1 Data**: Reads all JSON files from `output/` directory

2. **Processes Each Step** (with multimodal embeddings):
   - For each step, loads the corresponding diagram image
   - **Multimodal Processing** (if enabled, default):
     - Uses **Qwen-VL** to analyze the step diagram image
     - Generates a detailed text description of what's shown (e.g., "Red 2x4 brick attached to blue base plate, yellow wheel on left side...")
     - Combines the step instructions text + diagram description
     - Creates a **fused embedding** that captures both text and visual information
   - **Text-Only Fallback** (if image missing):
     - Uses only the step text content
     - Creates text embedding

3. **Creates Searchable Chunks**:
   - Each step becomes a "chunk" with:
     - Content: Instructions + diagram description (auto-truncated if > 2048 chars)
     - Embedding: Vector representation (for semantic search)
     - Metadata: Step number, parts, dependencies, image path

4. **Stores in ChromaDB**:
   - Saves all chunks to the vector database
   - Enables fast semantic search and retrieval

**Why it takes 2-5 minutes**: 
- VLM extraction: ~1-2 seconds per step
- Multimodal embedding: ~1-2 seconds per step
- For a 50-step manual: 50 VLM calls + 50 embedding calls = ~2-5 minutes
- **Result**: High-quality extraction + searchable vector store!

**Benefits of Multimodal Embeddings**:
- ✅ Better visual query understanding ("What does step 5 look like?")
- ✅ Improved part matching from images
- ✅ Enhanced context for vague queries
- ✅ More accurate step estimation from user photos

### Step 5: Start Frontend (Terminal 2)

```bash
# Open new terminal
cd frontend

# Install Node.js dependencies
npm install

# Configure backend URL
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start frontend
npm run dev

# ✓ Frontend running at http://localhost:3000
```

Keep this terminal running!

### Step 6: Use the System

Open browser: **http://localhost:3000**

#### Mode 1: Text Chat (Traditional)
1. **Select manual** "6454922" from dropdown
2. **Navigate steps** with Previous/Next buttons
3. **Ask questions** in chat tab:
   - "What do I do in step 5?"
   - "What parts do I need?"
   - "How do I attach this piece?"

#### Mode 2: Photo Analysis (NEW! ⭐)
1. **Select manual** "6454922" from dropdown
2. **Switch to "Photo Analysis" tab**
3. **Upload 2-4 photos** of your physical assembly:
   - Take photos from different angles (front, back, sides, top)
   - Ensure good lighting and clear focus
4. **Click "Analyze My Assembly"**
5. **Get intelligent guidance**:
   - See your progress percentage
   - View completed steps
   - Get next-step instructions
   - See reference images
   - Identify any errors or missing parts

---

## 🎯 Quick Test

Once everything is running, test each component:

### Test Backend
```bash
# Health check
curl http://localhost:8000/health

# List manuals
curl http://localhost:8000/api/manuals

# Ask question (text mode)
curl -X POST http://localhost:8000/api/query/text \
  -H "Content-Type: application/json" \
  -d '{"manual_id": "6454922", "question": "What is step 1?"}'

# 🧠 Test hierarchical graph endpoints (NEW!)
curl http://localhost:8000/api/manual/6454922/graph/summary
curl http://localhost:8000/api/manual/6454922/graph/subassemblies
curl http://localhost:8000/api/manual/6454922/progress
```

### Test Frontend - Text Chat
- Visit http://localhost:3000
- Select manual from dropdown
- Stay on "Text Chat" tab
- Click "Next" to navigate steps
- Type question in chat and press Enter

### Test Frontend - Photo Analysis (NEW! ⭐)
- Visit http://localhost:3000
- Select manual "6454922"
- Switch to "Photo Analysis" tab
- Upload 2-4 test photos of a LEGO assembly
- Click "Analyze My Assembly"
- View progress, guidance, and next steps

**Pro Tip**: Build the first few steps of set 6454922 physically, take photos, and see the system track your progress automatically!

---

## 🚀 Advanced: Multimodal Queries (Text + Images)

**NEW!** Ask vague questions with your assembly photos and get accurate answers!

### Try Multimodal Query Flow

#### 1. Upload Your Assembly Photos
```bash
# Upload 2-4 photos via API
curl -X POST http://localhost:8000/api/vision/upload-images \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "images=@photo3.jpg"

# Response: {"session_id": "abc-123-def", ...}
# Save this session_id!
```

#### 2. Ask Vague Questions
```bash
# Now ask without specifying step numbers
curl -X POST http://localhost:8000/api/query/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "manual_id": "6454922",
    "question": "What'\''s next?",
    "session_id": "abc-123-def"
  }'
```

**What happens:**
1. System analyzes your photos (detects parts, colors, structure)
2. Estimates you're at steps 17-19
3. Augments "What's next?" → "What is the next step after steps 17-19 using red 2x4 brick?"
4. Returns accurate next step instructions

#### 3. Compare: Without vs. With Images

**Without Images (Traditional)**:
```bash
curl -X POST http://localhost:8000/api/query/text \
  -d '{"manual_id": "6454922", "question": "What'\''s next?"}'

# Response: ❌ "I need more context. Which step are you on?"
```

**With Images (Multimodal)**:
```bash
curl -X POST http://localhost:8000/api/query/multimodal \
  -d '{"manual_id": "6454922", "question": "What'\''s next?", "session_id": "abc-123"}'

# Response: ✅ "Based on your assembly showing red 2x4 brick and blue plate at step 17,
#              next attach the yellow 1x2 plate to the top-right corner..."
```

### More Vague Queries That Work

With images uploaded:
- ❓ "What's next?"
- ❓ "Help, I'm stuck!"
- ❓ "Am I doing this right?"
- ❓ "What should I do?"
- ❓ "Is this correct?"

Without images, these would fail. With multimodal RAG, they work perfectly!

---

## 🔧 Helper Scripts

### Start Backend (Easy Way)
```bash
./start_backend.sh
# Checks dependencies, activates env, starts server
```

### Start Frontend (Easy Way)
```bash
./start_frontend.sh
# Installs dependencies if needed, starts dev server
```

### Ingest Manual (Easy Way)
```bash
./ingest_manual.sh 6454922
# Checks files exist, runs ingestion
```

---

## 🐛 Troubleshooting

### "API key not found"
```bash
# Check .env file
cat .env | grep API_KEY

# Should show your key
# If not, edit .env and add it
```

### "No manuals available" in frontend
```bash
# Run ingestion first
cd backend
uv run python -m app.scripts.ingest_manual 6454922
```

### "Cannot connect to backend"
```bash
# Check backend is running
curl http://localhost:8000/health

# Should return: {"status": "healthy", ...}
# If not, restart backend
```

### "Module not found"
```bash
# Reinstall dependencies
uv sync

# Or clear and reinstall
rm -rf .venv
uv sync
```

### Port already in use
```bash
# Find process
lsof -i :8000  # Backend
lsof -i :3000  # Frontend

# Kill process
kill -9 <PID>
```

### "Multimodal queries not working well"
```bash
# Re-ingest manual with multimodal embeddings
cd backend

# Delete old embeddings
curl -X DELETE http://localhost:8000/api/manual/6454922

# Re-ingest with multimodal support
uv run python -m app.scripts.ingest_manual 6454922
```

**Why?** Old ingestions only have text embeddings. Re-ingesting generates multimodal embeddings (text + diagram descriptions) for better visual query results.

---

## 📦 What You Get

After setup, you have:

✅ **Phase 1**: Manual processing with VLMs (ground truth extraction)  
✅ **Phase 2**: Multimodal RAG backend with state analysis  
✅ **Phase 3**: Beautiful web UI with dual-mode interface

**Text Chat Mode - Try these queries**:
- "What parts do I need for step 5?"
- "How do I attach the red brick?"
- "What's the next step?"
- "Show me dependencies"
- 🧠 **NEW: Structural queries**:
  - "What subassembly contains the red brick?"
  - "What is the wheel assembly made from?"
  - "Show me the hierarchy of step 5"

**Photo Analysis Mode - Try this workflow** ⭐:
1. Build steps 1-3 of your LEGO set
2. Take 3 photos from different angles
3. Upload and analyze
4. Get automatic progress tracking: "You've completed steps 1-3 (15%)"
5. Receive next-step guidance: "Next: attach blue 2x4 brick..."
6. Build next step and repeat!

**Multimodal Query Mode - Try vague questions** 🆕:
1. Upload 2-4 assembly photos (get session_id)
2. Ask vague questions: "What's next?", "Help!", "Am I doing this right?"
3. System automatically:
   - Detects parts from photos
   - Estimates current step
   - Augments your query with context
   - Returns accurate, specific guidance
4. No need to specify step numbers!

---

## 📊 Graph Visualization & Debugging

The system generates hierarchical assembly graphs that map parts → subassemblies → model structure. Use these tools to verify and debug graph quality.

### View Graph Structure

```bash
# Basic visualization
python3 backend/app/scripts/visualize_graph.py output/6454922_graph.json

# With subassembly details
python3 backend/app/scripts/visualize_graph.py output/6454922_graph.json --show-subassemblies

# With step progression
python3 backend/app/scripts/visualize_graph.py output/6454922_graph.json --show-steps 10

# Full details
python3 backend/app/scripts/visualize_graph.py output/6454922_graph.json \
  --show-subassemblies --show-steps 10 --max-parts 5
```

### Output Files

When a graph is generated, you get two files:

1. **`{manual_id}_graph.json`** - Full hierarchical graph data (for system use)
2. **`{manual_id}_graph_summary.txt`** - Human-readable summary (for verification)

```bash
# Review the summary
cat output/6454922_graph_summary.txt
```

### Verify Graph Quality

**Good Indicators:**
- ✅ Subassembly names are descriptive (e.g., "Front Wheel Assembly")
- ✅ Hierarchy depth > 1 (parts nested under subassemblies)
- ✅ Completeness markers have sensible values
- ✅ Matches manual structure when compared with PDF

**Bad Indicators:**
- ❌ Generic names like "Assembly from step 7"
- ❌ Max depth = 0 (flat structure, all parts directly under root)
- ❌ Every step creates a subassembly (over-detection)
- ❌ No subassemblies detected (under-detection)

**Example Good Hierarchy:**
```
📦 LEGO Model [root]
├── 🔧 Vehicle Base [step 3]
│   ├── 🔧 Front Wheel Assembly [step 5]
│   │   ├── 🧱 black wheel
│   │   └── 🧱 grey axle
│   └── 🧱 black base plate
└── 🔧 Vehicle Body [step 10]
```

### Test Graph Regeneration

To test the enhanced graph builder with an existing manual:

```bash
# Regenerate graph with latest implementation
python3 backend/app/scripts/test_graph_regeneration.py

# This will:
# - Load existing extracted steps for manual 6454922
# - Rebuild graph with enhanced 2-stage system
# - Generate summary files
# - Compare with old graph (if exists)
```

### Debug Issues

If graph quality is poor:

1. **Check the logs** - Look for "Building Hierarchical Assembly Graph" section
2. **Review summary file** - `cat output/{manual_id}_graph_summary.txt`
3. **Visualize structure** - Use visualize_graph.py
4. **Compare with manual PDF** - Verify detected subassemblies match actual structure

**Common Issues:**

- **No subassemblies**: Manual may be very simple, or detection heuristics too strict
- **Too many subassemblies**: Detection heuristics too loose, every step creates one
- **Flat hierarchy**: Parent-child relationships not being established correctly
- **Wrong names**: Name inference not finding functional keywords

See inline code comments in `src/plan_generation/graph_builder.py` for tuning options.

---

## 🎓 Next Steps

1. **Try multimodal queries**: Upload photos and ask vague questions
2. **Try graph queries**: Ask about subassemblies and relationships 🧠 NEW
3. **Process more manuals**: Run `uv run python main.py <url>` for other sets
4. **Ingest them**: Run `uv run python -m app.scripts.ingest_manual <id>` (with multimodal embeddings)
5. **Explore graph API**: Check `/api/manual/{id}/graph/*` endpoints
6. **Customize**: Edit prompts in `backend/app/rag/generator.py`
7. **Explore**: Check API docs at http://localhost:8000/docs
8. **Read more**: See implementation docs:
   - `MULTIMODAL_IMPLEMENTATION_SUMMARY.md` - Multimodal features
   - `IMPLEMENTATION_COMPLETE.md` - Hierarchical graph 🧠 NEW

---

## 📚 More Information

- **Full Documentation**: See [README.md](README.md)
- **API Reference**: http://localhost:8000/docs
- **Configuration**: Edit `.env` file
- **Troubleshooting**: See README.md troubleshooting section

---

**Quick Start Version**: 2.1.0 (with Hierarchical Graph 🧠)  
**Setup Time**: ~10 minutes  
**Status**: Ready to use! 🎉
