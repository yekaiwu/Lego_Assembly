#!/bin/bash

# LEGO RAG System - Manual Ingestion Script

if [ -z "$1" ]; then
    echo "Usage: ./ingest_manual.sh <manual_id>"
    echo "Example: ./ingest_manual.sh 6454922"
    echo ""
    echo "Available manuals in output/:"
    ls -1 ../../output/ | grep "_extracted.json" | sed 's/_extracted.json//' | sed 's/^/  - /'
    exit 1
fi

MANUAL_ID=$1

echo "🧱 Ingesting LEGO Manual: $MANUAL_ID"
echo ""

# Check if manual files exist
if [ ! -f "../../output/${MANUAL_ID}_extracted.json" ]; then
    echo "❌ Manual files not found for $MANUAL_ID"
    echo "Please process the manual first:"
    echo "  python main.py <manual_url>"
    exit 1
fi

# Navigate to backend
cd "$(dirname "$0")/../../backend"

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Backend not configured. Please run setup first."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "📦 Processing and embedding manual data..."
echo ""

# Run ingestion
python -m app.scripts.ingest_manual $MANUAL_ID

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully ingested manual $MANUAL_ID"
    echo "🚀 Start the backend and frontend to use it!"
else
    echo ""
    echo "❌ Ingestion failed. Check the error messages above."
    exit 1
fi




