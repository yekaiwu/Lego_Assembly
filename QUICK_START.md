# LEGO Vision-Based RAG System - Quick Start

Get the complete Vision-Enhanced RAG system running in 10 minutes.

**NEW**: Now with computer vision-based assembly state analysis!

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

### Step 3: Process a Manual (Phase 1)

```bash
# Process LEGO manual from URL
uv run python main.py https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6454922.pdf

# Wait ~30-60 seconds...
# ✓ Output created in ./output/
#   - 6454922_extracted.json
#   - 6454922_plan.json
#   - 6454922_dependencies.json
#   - temp_pages/*.png
```

### Step 4: Start Backend (Terminal 1)

```bash
cd backend

# Ingest the manual into vector store (with multimodal embeddings)
uv run python -m app.scripts.ingest_manual 6454922
# ⏱️ Takes ~30-60 seconds (generates diagram descriptions)

# Start backend server
uv run uvicorn app.main:app --reload --port 8000

# ✓ Backend running at http://localhost:8000
# ✓ API docs at http://localhost:8000/docs
```

Keep this terminal running!

**What's happening?** Ingestion now uses Qwen-VL to generate diagram descriptions, creating multimodal embeddings (text + visual) for better retrieval.

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

## 🎓 Next Steps

1. **Try multimodal queries**: Upload photos and ask vague questions
2. **Process more manuals**: Run `uv run python main.py <url>` for other sets
3. **Ingest them**: Run `uv run python -m app.scripts.ingest_manual <id>` (with multimodal embeddings)
4. **Customize**: Edit prompts in `backend/app/rag/generator.py`
5. **Explore**: Check API docs at http://localhost:8000/docs
6. **Read more**: See `MULTIMODAL_IMPLEMENTATION_SUMMARY.md` for technical details

---

## 📚 More Information

- **Full Documentation**: See [README.md](README.md)
- **API Reference**: http://localhost:8000/docs
- **Configuration**: Edit `.env` file
- **Troubleshooting**: See README.md troubleshooting section

---

**Quick Start Version**: 2.0.0  
**Setup Time**: ~10 minutes  
**Status**: Ready to use! 🎉
