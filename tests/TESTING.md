# Testing Guide - LEGO Assembly System

## Overview

This project uses **pytest** for testing with comprehensive coverage tracking and pre-commit hooks to ensure code quality.

## Quick Start

### 1. Install Dependencies

```bash
# Install all dependencies including dev dependencies
pip install -e '.[dev]'
```

### 2. Run Tests

```bash
# Run all tests (fast tests only, excludes slow and API tests)
pytest tests/ -v

# Or use the test runner script
./scripts/run_tests.sh
```

### 3. Install Pre-commit Hooks

```bash
# Install hooks (required before first commit)
./scripts/install_hooks.sh

# Or manually
pre-commit install
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_graph_manager.py
│   ├── test_state_matcher.py
│   └── ...
├── integration/             # Integration tests (slower, multiple components)
│   ├── test_visual_matching.py
│   └── test_api_endpoints.py
└── test_*.py                # Other tests
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_graph_manager.py

# Run specific test function
pytest tests/unit/test_graph_manager.py::test_load_graph_from_file
```

### Test Markers

Tests are organized with pytest markers:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only tests that don't require API keys
pytest -m "not requires_api"

# Combine markers
pytest -m "unit and not slow"
```

Available markers:
- `unit` - Unit tests for individual components
- `integration` - Integration tests
- `api` - API endpoint tests
- `vision` - Vision/VLM model tests
- `slow` - Tests that take >5 seconds
- `requires_api` - Tests requiring external API keys

### Using Test Runner Scripts

```bash
# Run all fast tests
./scripts/run_tests.sh

# Include slow tests
./scripts/run_tests.sh --slow

# Include tests requiring API keys
./scripts/run_tests.sh --api

# Run with coverage
./scripts/run_tests.sh --coverage

# Verbose output
./scripts/run_tests.sh -v
```

## Coverage

### Generate Coverage Report

```bash
# Run tests with coverage
pytest tests/ --cov=backend/app --cov=src --cov-report=html

# Or use the coverage script
./scripts/check_coverage.sh

# View HTML report
open htmlcov/index.html
```

### Coverage Threshold

The project requires minimum **70% coverage**. The coverage check will fail if coverage drops below this threshold.

```bash
# Check coverage (fails if < 70%)
./scripts/check_coverage.sh
```

## Pre-commit Hooks

Pre-commit hooks run automatically before each commit to catch issues early.

### Installation

```bash
./scripts/install_hooks.sh
```

### What Runs on Commit

1. **Code Formatting** - Black formats code automatically
2. **Linting** - Ruff checks for code quality issues
3. **Basic Checks** - Trailing whitespace, JSON/YAML validation
4. **Fast Tests** - Unit tests run before commit

### Manual Hook Execution

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run pytest-check --all-files
pre-commit run black --all-files
```

### Skip Hooks

```bash
# Skip all hooks for a commit
git commit --no-verify -m "message"

# Skip specific hook
SKIP=pytest-check git commit -m "WIP: work in progress"
```

## Code Quality Tools

### Black (Code Formatter)

```bash
# Format all code
black .

# Check without modifying
black --check .

# Format specific file
black src/some_file.py
```

Configuration in `pyproject.toml`:
- Line length: 100 characters
- Target: Python 3.8+

### Ruff (Linter)

```bash
# Lint all code
ruff check .

# Auto-fix issues
ruff check --fix .

# Check specific file
ruff check src/some_file.py
```

Configuration in `pyproject.toml`:
- Line length: 100 characters
- Rules: pycodestyle, pyflakes, isort, bugbear, comprehensions

## Writing Tests

### Example Unit Test

```python
import pytest

@pytest.mark.unit
def test_something(mock_settings):
    """Test description."""
    # Arrange
    expected = "value"

    # Act
    result = some_function()

    # Assert
    assert result == expected
```

### Example Integration Test

```python
import pytest

@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline(sample_data, temp_output_dir):
    """Test description."""
    # Test implementation
    pass
```

### Using Fixtures

Common fixtures available in `tests/conftest.py`:

```python
def test_example(
    temp_dir,              # Temporary directory
    sample_manual_id,      # Sample manual ID
    sample_graph_data,     # Sample graph structure
    mock_settings,         # Mock settings
    mock_vlm_client,       # Mock VLM client
    mock_chroma_manager,   # Mock ChromaDB
):
    # Use fixtures in test
    pass
```

## Continuous Integration

### Pre-commit Checks

Before each commit:
1. Code is formatted with Black
2. Code is linted with Ruff
3. Fast tests are run (unit tests, no API calls)

### Recommended CI Pipeline

```yaml
# .github/workflows/test.yml (example)
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -e '.[dev]'
      - name: Run tests
        run: ./scripts/run_tests.sh --coverage
      - name: Check coverage
        run: ./scripts/check_coverage.sh
```

## Troubleshooting

### Tests Failing Locally

```bash
# Clear cache and retry
pytest --cache-clear tests/

# Run in verbose mode to see details
pytest tests/ -vv

# Run specific failing test
pytest tests/unit/test_graph_manager.py::test_load_graph -vv
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Pre-commit Hook Issues

```bash
# Update hooks
pre-commit autoupdate

# Reinstall hooks
pre-commit uninstall
pre-commit install

# Run manually to debug
pre-commit run --all-files --verbose
```

### Coverage Not Generated

```bash
# Ensure pytest-cov is installed
pip install pytest-cov

# Run with explicit coverage
pytest tests/ --cov=backend/app --cov-report=html
```

## Best Practices

1. **Write tests first** - Follow TDD when possible
2. **Keep tests isolated** - Use fixtures, avoid shared state
3. **Mock external dependencies** - Don't call real APIs in tests
4. **Use descriptive names** - Test names should describe what they test
5. **One assertion per test** - Keep tests focused
6. **Mark tests appropriately** - Use markers for slow/API tests
7. **Test edge cases** - Not just happy paths
8. **Keep tests fast** - Unit tests should run in milliseconds

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/mark.html)
- [Black documentation](https://black.readthedocs.io/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [pre-commit documentation](https://pre-commit.com/)
