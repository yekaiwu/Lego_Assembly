#!/bin/bash
# Coverage check script for LEGO Assembly System
# Runs tests with coverage and fails if below threshold

set -e

echo "=================================="
echo "Coverage Check"
echo "=================================="
echo ""

# Minimum coverage threshold
MIN_COVERAGE=70

# Run tests with coverage using uv
uv run pytest tests/ \
    -v \
    -m "not slow and not requires_api" \
    --cov=backend/app \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    --cov-fail-under=$MIN_COVERAGE

echo ""
echo "=================================="
echo "✓ Coverage check passed (>= ${MIN_COVERAGE}%)"
echo "=================================="
echo ""
echo "Coverage reports generated:"
echo "  - HTML: htmlcov/index.html"
echo "  - XML: coverage.xml"
echo "  - Terminal output above"
echo ""
echo "To view HTML report: open htmlcov/index.html"

exit 0
