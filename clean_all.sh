#!/bin/bash
  # clean_all.sh - Complete cache and output cleanup

  echo "🧹 Cleaning all caches and outputs..."

  # Clear VLM and Python caches
  rm -rf ./cache ./backend/cache

  # Clear ALL JSON files in output directory
  rm -f output/*.json output/.*json

  # Clear Python bytecode
  find . -type f -name "*.pyc" -delete
  find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

  echo "✅ All cleared! Ready for fresh extraction."
