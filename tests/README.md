# BTUT Test Suite

Comprehensive test suite for the BTUT platform.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
│   └── test_simulator.py
├── integration/       # Integration tests for API and services
│   └── test_api.py
├── e2e/              # End-to-end workflow tests
│   └── test_workflow.py
├── performance/      # Performance and benchmark tests
│   └── test_scaling.py
├── pytest.ini        # Pytest configuration
└── README.md         # This file
```

## Running Tests

### Install Dependencies

```bash
pip install pytest pytest-cov pytest-xdist pytest-timeout requests
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=btut --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# E2E tests only
pytest tests/e2e/

# Exclude slow tests
pytest -m "not slow"

# Run only slow tests
pytest -m slow
```

### Run Specific Test Files

```bash
# Run simulator tests
pytest tests/unit/test_simulator.py

# Run API tests
pytest tests/integration/test_api.py

# Run workflow tests
pytest tests/e2e/test_workflow.py
```

### Run Specific Test Functions

```bash
# Run single test
pytest tests/unit/test_simulator.py::TestSimulatorInitialization::test_default_initialization

# Run test class
pytest tests/unit/test_simulator.py::TestSimulatorInitialization
```

## Test Categories

### Unit Tests

Test individual components in isolation.

**What's tested:**
- Simulator initialization
- Parameter validation
- Convergence logic
- Result formatting
- Presets and utilities

**Run:**
```bash
pytest tests/unit/ -v
```

### Integration Tests

Test API endpoints and service integration.

**What's tested:**
- Health check endpoint
- Simulation endpoint (sync/async)
- CORS headers
- Error handling
- Rate limiting

**Prerequisites:**
- API server running at `http://localhost:8000`

**Run:**
```bash
# Start API server first
cd api
uvicorn main:app --reload

# Run tests in another terminal
pytest tests/integration/ -v
```

### End-to-End Tests

Test complete user workflows.

**What's tested:**
- Python SDK workflows
- API workflows
- Research workflows
- Production workflows
- Documentation examples

**Run:**
```bash
pytest tests/e2e/ -v
```

### Performance Tests

Benchmark and performance validation.

**What's tested:**
- Scaling with agent count
- O(N) complexity verification
- Convergence time
- Memory usage

**Run:**
```bash
pytest tests/performance/ -v
```

## Test Markers

Tests are marked with categories for selective execution:

```python
@pytest.mark.slow          # Slow-running tests
@pytest.mark.integration   # Integration tests
@pytest.mark.e2e          # End-to-end tests
@pytest.mark.asynctest    # Async API tests
@pytest.mark.performance  # Performance benchmarks
@pytest.mark.unit         # Unit tests
```

**Usage:**
```bash
# Run only fast tests
pytest -m "not slow"

# Run integration and e2e
pytest -m "integration or e2e"

# Run everything except performance
pytest -m "not performance"
```

## Coverage Reports

### Generate HTML Coverage Report

```bash
pytest --cov=btut --cov-report=html
open htmlcov/index.html  # macOS
# or
start htmlcov/index.html  # Windows
```

### Generate Terminal Coverage Report

```bash
pytest --cov=btut --cov-report=term-missing
```

### Coverage Targets

- **Unit tests**: 90%+ coverage
- **Integration tests**: 80%+ coverage
- **Overall**: 85%+ coverage

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Pushes to main
- Nightly builds

### Local Pre-commit Hook

```bash
# Install pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
pytest tests/unit/ tests/integration/ -x --tb=short
EOF

chmod +x .git/hooks/pre-commit
```

## Writing Tests

### Test Structure

```python
import pytest
from btut import Simulator

class TestFeature:
    """Test description"""

    def test_basic_behavior(self):
        """Test basic functionality"""
        # Arrange
        sim = Simulator(agents=1000, gamma=1.5)

        # Act
        result = sim.run()

        # Assert
        assert result.converged is True

    def test_edge_case(self):
        """Test edge case"""
        with pytest.raises(ValueError):
            Simulator(agents=-1)
```

### Best Practices

1. **One assertion per test**: Tests should be focused
2. **Descriptive names**: `test_simulation_converges_with_high_gamma`
3. **AAA pattern**: Arrange, Act, Assert
4. **Use fixtures**: Share setup across tests
5. **Mark slow tests**: Use `@pytest.mark.slow`
6. **Test edge cases**: Boundaries, errors, invalid input

### Fixtures

```python
@pytest.fixture
def simulator():
    """Standard simulator for testing"""
    return Simulator(agents=1000, gamma=1.5)

def test_with_fixture(simulator):
    result = simulator.run()
    assert result.converged is True
```

## Debugging Tests

### Run with debugger

```bash
pytest --pdb  # Drop into debugger on failure
```

### Show print statements

```bash
pytest -s  # Show stdout
```

### Show local variables on failure

```bash
pytest -l  # Show locals in traceback
```

### Verbose failure info

```bash
pytest -vv --tb=long
```

## Performance Benchmarks

### Run Benchmarks

```bash
pytest tests/performance/ --benchmark-only
```

### Expected Performance

| Test | Expected Time |
|------|--------------|
| 1K agents | < 50ms |
| 10K agents | < 200ms |
| 100K agents | < 2s |
| 1M agents | < 20s |

## Troubleshooting

### Tests Fail: API Not Running

**Problem:** Integration tests fail with connection errors

**Solution:**
```bash
cd api
uvicorn main:app --reload
```

### Tests Fail: Import Errors

**Problem:** Cannot import btut module

**Solution:**
```bash
# Install in development mode
cd python-sdk
pip install -e .
```

### Tests Timeout

**Problem:** Tests hang or timeout

**Solution:**
```bash
# Increase timeout
pytest --timeout=600

# Or skip slow tests
pytest -m "not slow"
```

### Memory Errors

**Problem:** Out of memory with large tests

**Solution:**
```bash
# Reduce agent counts in tests
# Or run tests serially
pytest -n 0
```

## Test Data

### Generate Test Data

```bash
python tests/generate_test_data.py
```

### Mock Data

Tests use generated mock data for:
- Network structures
- Agent strategies
- Convergence histories

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - run: pip install -r requirements-test.txt
      - run: pytest --cov=btut --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Test Maintenance

### Regular Tasks

- **Weekly**: Review and update slow tests
- **Monthly**: Update test data
- **Per release**: Full test suite + benchmarks
- **After bugs**: Add regression tests

### Adding Tests

1. Create test file in appropriate directory
2. Add test class and methods
3. Mark with appropriate markers
4. Update this README if needed
5. Ensure tests pass locally
6. Submit PR

## Support

- Issues: https://github.com/direncode/btut/issues
- Discussions: https://github.com/direncode/btut/discussions
