#!/bin/bash

# LEGO RAG System - Backend Startup Script

echo "🧱 Starting LEGO RAG Backend..."
echo ""

# Navigate to backend directory
cd "$(dirname "$0")/../../backend"

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating from template..."
    cp ENV_BACKEND_TEMPLATE.txt .env
    echo "❌ Please edit backend/.env and add your OPENAI_API_KEY"
    exit 1
fi

# Check if OpenAI key is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "❌ OpenAI API key not configured in .env"
    echo "Please edit backend/.env and add your OPENAI_API_KEY"
    exit 1
fi

# Check if chroma_db exists
if [ ! -d "chroma_db" ]; then
    echo "⚠️  Vector database not found!"
    echo "You may need to ingest manuals first:"
    echo "  python -m app.scripts.ingest_manual <manual_id>"
    echo ""
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Starting backend server on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
uvicorn app.main:app --reload --port 8000




