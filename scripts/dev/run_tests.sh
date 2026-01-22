#!/bin/bash
# Test runner script for LEGO Assembly System
# Runs tests in order: unit tests first (fast), then integration tests

set -e  # Exit on error

echo "=================================="
echo "LEGO Assembly System - Test Runner"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found. Install uv first:${NC}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Parse arguments
RUN_SLOW=false
RUN_API=false
COVERAGE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --slow)
            RUN_SLOW=true
            shift
            ;;
        --api)
            RUN_API=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--slow] [--api] [--coverage] [-v|--verbose]"
            exit 1
            ;;
    esac
done

# Build pytest command using uv
PYTEST_CMD="uv run pytest tests/"
PYTEST_ARGS="-v"

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -vv"
fi

# Add marker filters
MARKER_FILTERS="not slow and not requires_api"
if [ "$RUN_SLOW" = true ]; then
    MARKER_FILTERS="not requires_api"
fi
if [ "$RUN_API" = true ]; then
    MARKER_FILTERS=""
fi

if [ -n "$MARKER_FILTERS" ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m \"$MARKER_FILTERS\""
fi

# Add coverage if requested
if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=backend/app --cov=src --cov-report=term-missing --cov-report=html"
fi

# Step 1: Run unit tests
echo -e "${YELLOW}Step 1: Running unit tests...${NC}"
echo ""
eval "$PYTEST_CMD tests/unit/ $PYTEST_ARGS" || {
    echo -e "${RED}Unit tests failed!${NC}"
    exit 1
}
echo -e "${GREEN}✓ Unit tests passed${NC}"
echo ""

# Step 2: Run integration tests
echo -e "${YELLOW}Step 2: Running integration tests...${NC}"
echo ""
eval "$PYTEST_CMD tests/integration/ $PYTEST_ARGS" || {
    echo -e "${RED}Integration tests failed!${NC}"
    exit 1
}
echo -e "${GREEN}✓ Integration tests passed${NC}"
echo ""

# Step 3: Run remaining tests
echo -e "${YELLOW}Step 3: Running other tests...${NC}"
echo ""
eval "$PYTEST_CMD $PYTEST_ARGS --ignore=tests/unit --ignore=tests/integration" || {
    echo -e "${RED}Other tests failed!${NC}"
    exit 1
}
echo -e "${GREEN}✓ All other tests passed${NC}"
echo ""

# Summary
echo "=================================="
echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
echo "=================================="

if [ "$COVERAGE" = true ]; then
    echo ""
    echo "Coverage report generated:"
    echo "  HTML: htmlcov/index.html"
    echo "  Open with: open htmlcov/index.html"
fi

exit 0
