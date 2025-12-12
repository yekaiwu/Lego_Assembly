#!/bin/bash

# LEGO RAG System - Frontend Startup Script

echo "🧱 Starting LEGO RAG Frontend..."
echo ""

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Check if .env.local exists
if [ ! -f .env.local ]; then
    echo "⚠️  .env.local file not found!"
    echo "Creating with default settings..."
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

echo "✅ Starting frontend on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the development server
npm run dev


