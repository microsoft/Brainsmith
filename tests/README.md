# Brainsmith Test Suite

Integration tests for Brainsmith's core systems: blueprint parsing, design space exploration, and plugin framework.

## Test Coverage

### Blueprint Parser (`test_blueprint_parser.py`)
- YAML blueprint parsing and validation
- Configuration inheritance (single and multi-level)
- Dynamic step operations (insert, replace, remove)
- Design space and kernel validation

### DSE Execution (`test_dse_execution.py`)
- Exploration tree construction and traversal
- Segment-based execution with artifact sharing
- Tree structure validation (nodes, branches, efficiency)
- Directory structure and result management

### Plugin System (`test_plugin_system.py`)
- Transform, kernel, and step plugin registration
- Framework integration (FINN/QONNX transforms)
- Plugin discovery and metadata queries
- Backend selection for hardware kernels

## Running Tests

```bash
# All tests (run inside container)
./smithy pytest tests/

# Specific component
./smithy pytest tests/integration/test_plugin_system.py

# With coverage
./smithy pytest tests/ --cov=brainsmith.core
```

## Test Structure

```
tests/
├── integration/      # Core system tests
├── fixtures/         # Test plugins and utilities
└── conftest.py      # Shared pytest fixtures
```