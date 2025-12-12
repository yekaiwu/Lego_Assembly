# LEGO Assembly Vision-Based RAG System

**Complete AI-Powered LEGO Assembly Assistant with Computer Vision**

Production-ready system combining Vision-Language Models (VLMs) for manual processing with Retrieval-Augmented Generation (RAG) and **Computer Vision-Based Assembly State Analysis** for intelligent, state-aware guidance.

---

## 🎯 System Overview

This system provides end-to-end LEGO assembly assistance through three integrated phases:

### **Phase 1: Manual Processing**
- VLM-based step extraction from PDF instruction manuals
- 3D plan generation with spatial reasoning
- Dependency graph construction
- Part database integration with Rebrickable

### **Phase 2: Multimodal RAG Backend** ⭐ NEW
- **🎨 Multimodal Embeddings**: Text + diagram descriptions for better visual retrieval
- **🤖 LLM Query Augmentation**: Understands vague queries like "What's next?"
- **📸 Computer Vision State Analysis**: Upload photos to track progress automatically
- **🔍 VLM-based Part Detection**: Identifies visible parts, colors, and connections
- **📊 Progress Mapping**: Compares detected state with expected plan
- **⚠️ Error Detection**: Identifies missing parts and incorrect placements
- **💡 Intelligent Guidance**: Generates next-step instructions based on current state
- **🧠 Image-Aware Retrieval**: Boosts results matching detected parts
- ChromaDB vector database for semantic search
- Qwen/DeepSeek/Moonshot LLM integration
- FastAPI REST API with 15+ endpoints

### **Phase 3: Frontend UI** ⭐ ENHANCED
- Next.js 14 web application with TypeScript
- **Photo Upload Interface**: Multi-image capture (2-4 angles)
- **Visual Progress Tracking**: Real-time assembly state display
- **Dual Mode**: Text chat + Photo analysis tabs
- Manual selection and browsing interface
- Step-by-step navigator with images
- Real-time AI chat assistant

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│          Frontend (Next.js)                      │
│  Manual Selector │ Step Navigator               │
│  Text Chat │ Photo Upload & Analysis ⭐ NEW     │
└─────────────┬────────────────────────────────────┘
              │ REST API
┌─────────────┴────────────────────────────────────┐
│            Backend (FastAPI)                     │
│  ┌──────────────────────────────────────────┐   │
│  │  Vision Analysis Pipeline ⭐ NEW         │   │
│  │  Photos → VLM → State Comparison →       │   │
│  │  Progress Mapping → Guidance Generation  │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │  RAG Pipeline (Text Queries)             │   │
│  │  Retrieval → Context → LLM               │   │
│  └──────────────────────────────────────────┘   │
│  ChromaDB Vector Store (Qwen Embeddings)        │
└──────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────┐
│    Phase 1: Manual Processing                   │
│  VLM Extraction → 3D Planning → JSON             │
│  (Ground Truth for Vision Comparison)            │
└──────────────────────────────────────────────────┘
```

### Project Structure

```
Lego_Assembly/
├── main.py                      # Phase 1 orchestrator
├── ENV_TEMPLATE.txt             # Environment configuration template
├── pyproject.toml               # Python dependencies (uv)
├── QUICK_START.md               # Quick setup guide
│
├── src/                         # Phase 1: Manual Processing
│   ├── api/                    # VLM clients (Qwen, DeepSeek, Kimi)
│   ├── vision_processing/      # PDF extraction & VLM analysis
│   ├── plan_generation/        # 3D planning & part database
│   └── utils/                  # Configuration & caching
│
├── backend/                     # Phase 2: Vision-Enhanced RAG
│   ├── app/
│   │   ├── main.py            # FastAPI application (15+ endpoints)
│   │   ├── config.py          # Configuration
│   │   ├── models/            # Pydantic schemas
│   │   ├── llm/               # LLM clients (Qwen/DeepSeek/Moonshot)
│   │   ├── vision/            # ⭐ NEW: Vision analysis module
│   │   │   ├── state_analyzer.py      # VLM-based part detection
│   │   │   ├── state_comparator.py    # Progress mapping
│   │   │   └── guidance_generator.py  # Next-step guidance
│   │   ├── vector_store/      # ChromaDB integration
│   │   ├── ingestion/         # Data processing
│   │   ├── rag/               # RAG pipeline (text queries)
│   │   └── scripts/           # CLI tools
│   └── chroma_db/             # Vector database (auto-created)
│
├── frontend/                    # Phase 3: Enhanced Web UI
│   ├── app/                   # Next.js pages (with vision tabs)
│   ├── components/            # React components
│   │   ├── ImageUpload.tsx          # ⭐ NEW: Photo upload
│   │   ├── VisualGuidance.tsx       # ⭐ NEW: Analysis display
│   │   ├── ChatInterface.tsx        # Text chat
│   │   ├── ManualSelector.tsx       # Manual selection
│   │   └── StepNavigator.tsx        # Step navigation
│   └── lib/                   # API client & state (vision APIs)
│
├── output/                      # Generated outputs
│   ├── {manual_id}_*.json     # Structured data
│   └── temp_pages/*.png       # Step images
│
└── data/
    └── parts_database.db       # LEGO parts cache
```

---

## 🚀 Quick Start

See [QUICK_START.md](QUICK_START.md) for detailed setup instructions.

### Prerequisites

- Python 3.9+ with `uv` package manager
- Node.js 18+ and npm
- Poppler (for PDF processing)
- At least one API key: Qwen (DashScope), DeepSeek, or Moonshot

### 1. Configure Environment

```bash
# Copy template and add your API keys
cp ENV_TEMPLATE.txt .env
nano .env  # Add your API keys
```

**Required**: At least one of:
- `DASHSCOPE_API_KEY` - For Qwen-VL and Qwen-Max
- `DEEPSEEK_API_KEY` - For DeepSeek-Chat
- `MOONSHOT_API_KEY` - For Moonshot/Kimi

### 2. Install Dependencies

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 3. Process a Manual (Phase 1)

```bash
# Process LEGO manual from URL
python main.py https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6454922.pdf

# Output created in ./output/:
#   6454922_extracted.json
#   6454922_plan.json
#   6454922_dependencies.json
#   6454922_plan.txt
#   temp_pages/*.png
```

### 4. Start RAG Backend

```bash
# Navigate to backend
cd backend

# Ingest manual into vector store (with multimodal embeddings)
python -m app.scripts.ingest_manual 6454922

# Start FastAPI server
uvicorn app.main:app --reload --port 8000

# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

**Note**: Ingestion now generates multimodal embeddings (text + diagram descriptions) using Qwen-VL. This takes ~30-60 seconds per manual but significantly improves retrieval quality for visual queries.

### 5. Start Frontend

```bash
# Navigate to frontend (new terminal)
cd frontend

# Install dependencies
npm install

# Configure backend URL
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev

# Access at: http://localhost:3000
```

---

## 📚 Features & Capabilities

### 🆕 Vision-Based Assembly State Analysis

**The Star Feature**: Upload photos of your physical assembly and get intelligent, state-aware guidance!

**How It Works**:
1. **Upload 2-4 Photos**: Take pictures from different angles (front, back, sides, top)
2. **VLM Analysis**: Qwen-VL analyzes images to detect parts, colors, and connections
3. **State Comparison**: System compares detected state with Phase 1 plan data
4. **Progress Mapping**: Determines which steps are completed (e.g., "Steps 1-5 done, currently on step 6")
5. **Error Detection**: Identifies missing parts or incorrect placements
6. **Guidance Generation**: Provides clear next-step instructions with reference images

**Example Workflow**:
```
1. User builds steps 1-3 of LEGO set
2. Takes 3 photos from different angles
3. Uploads to system via frontend
4. System responds:
   ✅ Progress: 15% (3/20 steps completed)
   ✅ Current State: Yellow base plate with 4 red bricks attached
   ⚠️  Missing: 1 blue 2x2 brick for step 3
   📋 Next Step: Attach green 2x4 brick to right side
   🖼️  [Shows reference image for step 4]
```

**Key Capabilities**:
- ✅ Multi-view image analysis (2-4 photos)
- ✅ Automatic part detection (colors, shapes, part IDs)
- ✅ Progress percentage calculation
- ✅ Error detection and correction suggestions
- ✅ Next-step prediction with visual references
- ✅ Iterative workflow: build → photo → guidance → repeat

**API Endpoints**:
- `POST /api/vision/upload-images` - Upload assembly photos
- `POST /api/vision/analyze` - Analyze state and generate guidance
- `DELETE /api/vision/session/{session_id}` - Cleanup session

---

### 🆕 Multimodal Query System

**The Enhanced Feature**: Ask vague questions with your assembly photos and get accurate, context-aware answers!

**How It Works**:
1. **Upload Photos**: Same as vision analysis - 2-4 photos from different angles
2. **Ask Vague Questions**: "What's next?", "Help!", "Am I doing this right?"
3. **Automatic Context**: System analyzes images, detects parts, estimates current step
4. **Query Augmentation**: LLM expands vague query with visual context
5. **Smart Retrieval**: Searches for steps matching detected parts
6. **Accurate Answer**: Gets precise guidance without requiring step numbers

**Example Workflow**:
```
Traditional (without multimodal):
User: "What's next?"
System: ❌ "I need more context. Which step are you on?"

With Multimodal RAG:
User: "What's next?" + [uploads 3 photos]
System: 
  1. Detects: red 2x4 brick, blue plate, yellow base
  2. Estimates: Steps 17-19
  3. Augments query: "What is the next step after steps 17-19 using red 2x4 brick?"
  4. Retrieves: Step 18 with 92% confidence
  5. Answers: ✅ "Based on your assembly, attach the yellow 1x2 plate to the top-right corner..."
```

**Key Features**:
- ✅ **Vague Query Understanding**: No need to specify step numbers
- ✅ **Image-Aware Retrieval**: Boosts results matching your detected parts
- ✅ **Automatic Step Estimation**: Infers where you are from photos
- ✅ **Query Expansion**: Turns "Help!" into specific, searchable queries
- ✅ **Session-Based**: Upload once, ask multiple questions

**API Endpoints**:
- `POST /api/query/multimodal` - Query with uploaded images
- `POST /api/query/text` - Text query (optionally with session_id)

**Usage Example**:
```bash
# 1. Upload images
curl -X POST http://localhost:8000/api/vision/upload-images \
  -F "images=@photo1.jpg" -F "images=@photo2.jpg"
# Returns: {"session_id": "abc-123", ...}

# 2. Ask vague question with context
curl -X POST http://localhost:8000/api/query/multimodal \
  -d '{
    "manual_id": "6454922",
    "question": "What should I do next?",
    "session_id": "abc-123"
  }'
```

**Re-Ingestion for Best Results**:
```bash
# Delete old embeddings
curl -X DELETE http://localhost:8000/api/manual/6454922

# Re-ingest with multimodal embeddings
curl -X POST http://localhost:8000/api/ingest/manual/6454922
```

This generates fused embeddings (text + diagram descriptions) for improved visual retrieval.

---

### Phase 1: Manual Processing

**Input**: PDF instruction manual or image directory  
**Output**: Structured JSON + step images (used as ground truth for vision analysis)

- **VLM Extraction**: Uses Qwen-VL, DeepSeek, or Kimi to analyze manual pages
- **Step Detection**: Automatically identifies step boundaries
- **Part Recognition**: Extracts part descriptions, colors, and quantities
- **Spatial Analysis**: Determines 3D positions and relationships
- **Dependency Graph**: Builds assembly order with parallel paths
- **Part Matching**: Integrates with Rebrickable for part IDs

**Example Usage**:
```bash
python main.py <manual_url> -o ./output --assembly-id 6454922
```

### Phase 2: RAG Backend

**Input**: User questions about assembly  
**Output**: Context-aware AI responses

- **Semantic Search**: ChromaDB with Qwen text-embedding-v2
- **Data Ingestion**: Processes Phase 1 outputs into vector store
- **Hybrid Retrieval**: Combines vector similarity with metadata filtering
- **Multi-LLM Support**: 
  - Qwen-Max (primary, best for Chinese)
  - DeepSeek-Chat (cost-effective)
  - Moonshot (strong context understanding)
- **Context Assembly**: Retrieves relevant steps, parts, and dependencies
- **REST API**: FastAPI with automatic OpenAPI docs

**Supported Queries**:
- "What do I do in step 5?"
- "What parts do I need?"
- "How do I attach the red brick?"
- "What's the next step?"
- "Show me dependencies for step 10"

**API Endpoints**:
```
GET  /health                          # Health check
POST /api/ingest/manual/{id}          # Ingest manual with multimodal embeddings
POST /api/query/text                  # Ask question (optionally with session_id)
POST /api/query/multimodal            # Query with uploaded images (NEW!)
GET  /api/manual/{id}/step/{num}      # Get step details
GET  /api/manuals                     # List all manuals
GET  /api/manual/{id}/steps           # List manual steps
GET  /api/image?path={path}           # Serve images
POST /api/vision/upload-images        # Upload assembly photos (NEW!)
POST /api/vision/analyze              # Analyze assembly state (NEW!)
DELETE /api/vision/session/{id}       # Cleanup session
```

### Phase 3: Frontend UI

**Access**: http://localhost:3000

- **Manual Selector**: Browse and select from ingested manuals
- **Step Navigator**: 
  - View current step with image
  - Navigate with Previous/Next buttons
  - Progress bar showing completion
  - Parts list for current step
- **AI Chat Interface**:
  - Natural language questions
  - Real-time responses from RAG backend
  - Message history
  - Quick question shortcuts
- **LEGO Theme**: Custom Tailwind CSS with LEGO colors
- **Responsive Design**: Works on desktop and mobile

---

## ⚙️ Configuration

### Environment Variables

All configuration in `.env` file (copy from `ENV_TEMPLATE.txt`):

```bash
# VLM API Keys (Phase 1 & RAG LLM)
DASHSCOPE_API_KEY=sk-...          # Alibaba Cloud for Qwen
DEEPSEEK_API_KEY=sk-...            # DeepSeek
MOONSHOT_API_KEY=sk-...            # Moonshot/Kimi

# RAG System Settings
RAG_LLM_PROVIDER=qwen              # qwen, deepseek, or moonshot
RAG_LLM_MODEL=qwen-max             # Model name
RAG_EMBEDDING_PROVIDER=qwen        # Only qwen supported for embeddings
RAG_EMBEDDING_MODEL=text-embedding-v2

# Vector Database
CHROMA_PERSIST_DIR=./backend/chroma_db
COLLECTION_NAME=lego_manuals

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Data Paths
OUTPUT_DIR=./output
TEMP_PAGES_DIR=./output/temp_pages

# RAG Pipeline
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
MAX_CONTEXT_LENGTH=4000

# Part Database
PARTS_DB_PATH=./data/parts_database.db
REBRICKABLE_API_KEY=...           # Optional, for part enrichment
```

### LLM Provider Selection

**Qwen (Recommended)**:
- Best bilingual (Chinese/English) support
- Native embedding model included
- Cost: ~¥0.12/1K tokens (~$0.017)

**DeepSeek**:
- Cost-effective alternative
- Good performance
- Cost: ~¥0.01/1K tokens (~$0.0014)
- Note: No native embedding API (uses Qwen for embeddings)

**Moonshot (Kimi)**:
- Strong context understanding
- 32K context window
- Cost: ~¥0.12/1K tokens (~$0.017)
- Note: Uses Qwen for embeddings

---

## 🔧 Development

### Phase 1: Adding New VLM

1. Create client in `src/api/new_vlm.py`
2. Add to `VLMStepExtractor` in `src/vision_processing/vlm_step_extractor.py`
3. Update `ENV_TEMPLATE.txt` with new API key

### Phase 2: Customizing RAG

**Modify Prompts**: Edit `backend/app/rag/generator.py`
```python
system_prompt = """Your custom system prompt here..."""
```

**Adjust Retrieval**: Edit `backend/app/rag/retrieval.py`
```python
top_k = 10  # Retrieve more results
similarity_threshold = 0.8  # Stricter threshold
```

**Add New Endpoint**: Edit `backend/app/main.py`
```python
@app.get("/api/custom")
async def custom_endpoint():
    return {"data": "your response"}
```

### Phase 3: Frontend Customization

**Change Theme**: Edit `frontend/tailwind.config.ts`
```typescript
colors: {
  lego: {
    red: '#D01012',    // Your custom colors
    // ...
  }
}
```

**Add Component**: Create in `frontend/components/`
```typescript
export default function NewComponent() {
  // Your component logic
}
```

---

## 📊 Performance & Costs

### Processing Speed
- **Phase 1**: ~10-20 seconds for 20-step manual
- **Phase 2 Ingestion**: ~30 seconds for 50-step manual
- **Phase 2 Query**: 1-3 seconds (first query), <1s (cached)
- **Phase 3 Load**: <1 second

### API Costs (Qwen)
- **Manual Processing**: ~¥5-10 per manual (~$0.70-1.40)
- **Embedding**: ~¥0.0001 per text (~$0.000014)
- **Query**: ~¥0.1-0.5 per query (~$0.014-0.070)

### Storage
- **Vector DB**: ~10MB per manual
- **Image Cache**: ~50MB per manual
- **Part Database**: ~5MB (shared)

---

## 🐛 Troubleshooting

### "API key not found"
```bash
# Check .env file exists
cat .env | grep API_KEY

# Make sure you copied from template
cp ENV_TEMPLATE.txt .env
```

### "No manuals available" (Frontend)
```bash
# Ingest manual first
cd backend
python -m app.scripts.ingest_manual 6454922
```

### "Cannot connect to backend" (Frontend)
```bash
# Check backend is running
curl http://localhost:8000/health

# Check frontend .env.local
cat frontend/.env.local
```

### "Module not found" errors
```bash
# Reinstall dependencies
uv sync

# Or with pip
pip install -e .
```

### Port already in use
```bash
# Find and kill process
lsof -i :8000  # Backend
lsof -i :3000  # Frontend
kill -9 <PID>
```

---

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Step-by-step setup guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[MULTIMODAL_IMPLEMENTATION_SUMMARY.md](MULTIMODAL_IMPLEMENTATION_SUMMARY.md)** - Multimodal RAG technical documentation
- **API Docs**: http://localhost:8000/docs (when backend running)

---

## 🎯 Roadmap

### Current Features
- ✅ Multi-VLM manual processing
- ✅ 3D plan generation
- ✅ Multimodal embeddings (text + diagrams)
- ✅ LLM query augmentation for vague queries
- ✅ RAG pipeline with semantic search
- ✅ Image upload for visual queries
- ✅ Part recognition with computer vision
- ✅ Progress tracking with VLM
- ✅ Web UI with chat interface
- ✅ Chinese LLM support

### Planned Enhancements
- [ ] Progress saving and resume
- [ ] Multi-manual comparison
- [ ] AR overlay guidance
- [ ] Voice interface
- [ ] 3D visualization
- [ ] Community features
- [ ] Mobile app
- [ ] Offline mode

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **LEGO®** is a trademark of the LEGO Group
- VLM Providers: Alibaba Cloud (Qwen), DeepSeek, Moonshot AI
- Vector DB: ChromaDB
- Frameworks: FastAPI, Next.js, React

---

## 📮 Support

For issues or questions:
1. Check [QUICK_START.md](QUICK_START.md) troubleshooting section
2. Review API documentation at `/docs` endpoint
3. Check configuration in `.env` file

**System Version**: 2.0.0  
**Last Updated**: December 2025  
**Status**: ✅ Production Ready
