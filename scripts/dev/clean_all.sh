#!/bin/bash
# clean_all.sh - Complete cache and output cleanup

echo "🧹 Cleaning all caches and outputs..."

# Clear VLM and Python caches
rm -rf ../../cache ../../backend/cache

# Clear EVERYTHING from output directory
echo "   Clearing output directory..."
rm -rf ../../output/*
rm -rf ../../output/.* 2>/dev/null  # Remove hidden files/dirs (ignore errors if none exist)

# Clear ChromaDB vector store
echo "   Clearing ChromaDB vector store..."
rm -rf ../../backend/chroma_db/*

# Clear Python bytecode
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "✅ All cleared! Ready for fresh extraction."
echo "⚠️  NOTE: Restart the backend server to clear in-memory caches."
