#!/bin/bash
# Install pre-commit hooks for LEGO Assembly System

set -e

echo "=================================="
echo "Installing Pre-commit Hooks"
echo "=================================="
echo ""

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Error: pre-commit not found. Installing..."
    pip install pre-commit
fi

# Install the hooks
echo "Installing git hooks..."
pre-commit install
pre-commit install --hook-type pre-push

echo ""
echo "=================================="
echo "✓ Pre-commit hooks installed!"
echo "=================================="
echo ""
echo "The following hooks will run before each commit:"
echo "  - Code formatting (Black)"
echo "  - Linting (Ruff)"
echo "  - Basic file checks"
echo "  - Pytest (fast tests only)"
echo ""
echo "To run hooks manually:"
echo "  pre-commit run --all-files"
echo ""
echo "To skip hooks for a specific commit:"
echo "  SKIP=pytest-check git commit -m 'message'"
echo ""
echo "To update hooks to latest versions:"
echo "  pre-commit autoupdate"

exit 0
