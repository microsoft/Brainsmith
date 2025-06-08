# Brainsmith Week 1 Test Suite

Comprehensive test suite for the Week 1 implementation of the extensible Brainsmith architecture using existing components.

## Overview

This test suite validates all core components implemented in Week 1:

- **DesignSpaceOrchestrator**: Main orchestration engine with hierarchical exit points
- **FINNInterface**: Legacy FINN support with 4-hook placeholder
- **WorkflowManager**: High-level workflow coordination
- **Python API**: Main API functions with backward compatibility
- **Legacy Support**: Compatibility layer for existing functionality

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── test_core_orchestrator.py      # Core orchestration engine tests
├── test_finn_interface.py         # FINN interface tests
├── test_workflow_manager.py       # Workflow management tests
├── test_api.py                    # Python API tests
├── test_legacy_support.py         # Legacy compatibility tests
├── test_integration.py            # End-to-end integration tests
├── run_all_tests.py              # Comprehensive test runner
└── README.md                      # This file
```

## Running Tests

### Run All Tests
```bash
# Comprehensive test runner with detailed reporting
python tests/run_all_tests.py

# Quiet mode
python tests/run_all_tests.py --quiet

# Verbose mode
python tests/run_all_tests.py --verbose
```

### Run Individual Test Modules
```bash
# Core orchestrator tests
python -m unittest tests.test_core_orchestrator -v

# FINN interface tests
python -m unittest tests.test_finn_interface -v

# Workflow manager tests
python -m unittest tests.test_workflow_manager -v

# API tests
python -m unittest tests.test_api -v

# Legacy support tests
python -m unittest tests.test_legacy_support -v

# Integration tests
python -m unittest tests.test_integration -v
```

### Run Specific Test Cases
```bash
# Run specific test class
python -m unittest tests.test_core_orchestrator.TestDesignSpaceOrchestrator -v

# Run specific test method
python -m unittest tests.test_api.TestAPIFunctions.test_brainsmith_explore_roofline -v
```

## Test Categories

### Unit Tests
- **test_core_orchestrator.py**: Tests individual orchestrator functionality
- **test_finn_interface.py**: Tests FINN interface components
- **test_workflow_manager.py**: Tests workflow coordination
- **test_api.py**: Tests API function behavior
- **test_legacy_support.py**: Tests backward compatibility

### Integration Tests
- **test_integration.py**: Tests complete end-to-end workflows
- Tests component interaction and data flow
- Tests error handling and recovery mechanisms

## Key Test Scenarios

### Hierarchical Exit Points
Tests all three exit points with existing components:

1. **Roofline Analysis** (~30s)
   - Quick analytical bounds using existing analysis tools
   - Performance bounds estimation
   - Memory bandwidth analysis

2. **Dataflow Analysis** (~2min)
   - Transform application using existing transforms
   - Kernel mapping to existing custom operations
   - Performance estimation without RTL generation

3. **Dataflow Generation** (~10min)
   - Complete optimization using existing strategies
   - RTL/HLS generation using existing FINN flow
   - Full synthesis results and performance metrics

### Backward Compatibility
- Legacy `explore_design_space()` function support
- Existing blueprint format compatibility
- Graceful migration guidance with deprecation warnings

### Error Handling
- File not found scenarios
- Invalid configuration handling
- Component failure recovery
- Graceful degradation mechanisms

## Test Fixtures and Mocking

### Mock Components
All tests use mock components to avoid dependencies on external libraries:

- **MockBlueprint**: Simulates blueprint configuration
- **MockOrchestrator**: Simulates orchestration behavior
- **Mock FINN Components**: Simulates FINN build processes

### Temporary Files
Tests that require files create temporary directories:

```python
self.temp_dir = tempfile.mkdtemp()
self.model_path = os.path.join(self.temp_dir, "test_model.onnx")
self.blueprint_path = os.path.join(self.temp_dir, "test_blueprint.yaml")
```

### Cleanup
All tests properly clean up temporary resources in `tearDown()` methods.

## Expected Test Results

### Success Criteria
- **≥95% pass rate**: Excellent - Week 1 ready for production
- **≥85% pass rate**: Good - Week 1 ready for Week 2 implementation
- **≥70% pass rate**: Warning - Consider fixes before Week 2
- **<70% pass rate**: Critical - Fix issues before proceeding

### Typical Test Output
```
🧪======================================================================
🚀 Brainsmith Week 1 Implementation - Comprehensive Test Suite
🧪======================================================================

📦 Running tests.test_core_orchestrator...
✅ test_orchestrator_initialization (TestDesignSpaceOrchestrator) ... ok
✅ test_roofline_exit_point (TestDesignSpaceOrchestrator) ... ok
✅ test_dataflow_analysis_exit_point (TestDesignSpaceOrchestrator) ... ok
✅ test_dataflow_generation_exit_point (TestDesignSpaceOrchestrator) ... ok

📊 COMPREHENSIVE TEST SUMMARY
===============================================================================
⏱️  Total execution time: 2.45 seconds
📝 Tests run: 45
✅ Passed: 43
❌ Failed: 0
💥 Errors: 0
⏭️  Skipped: 2
📈 Success rate: 95.6%

🎉 EXCELLENT: Week 1 implementation is production-ready!
🚀 Week 2 Readiness: READY FOR WEEK 2
```

## Mock Strategy

### Why Mocking?
- **Independence**: Tests don't depend on external FINN libraries
- **Speed**: Tests run quickly without actual compilation
- **Reliability**: Tests are deterministic and don't fail due to environment issues
- **Coverage**: Can test error scenarios that would be hard to reproduce

### What's Mocked?
- FINN DataflowBuildConfig and build processes
- File system operations (where appropriate)
- External library imports
- Time-consuming operations

### What's Real?
- Core logic and control flow
- Data structure manipulation
- Configuration parsing and validation
- Error handling mechanisms

## Troubleshooting

### Common Issues

#### Import Errors
```
Warning: Could not import core components: No module named 'brainsmith.core'
```
**Solution**: Ensure Python path includes project root:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python tests/run_all_tests.py
```

#### Missing Dependencies
```
ImportError: No module named 'click'
```
**Solution**: Install development dependencies (though tests should work without them due to mocking)

#### All Tests Skipped
```
⏭️  All tests skipped - components not available
```
**Solution**: Check that core modules are properly implemented and importable

### Debug Mode
Enable debug mode by setting verbosity to maximum:
```bash
python tests/run_all_tests.py --verbose
```

## Contributing

### Adding New Tests
1. Create test methods following naming convention: `test_<functionality>`
2. Use descriptive docstrings explaining what's being tested
3. Include both positive and negative test cases
4. Add proper setup and teardown for resources

### Test Guidelines
- Test one thing per test method
- Use clear, descriptive assertions
- Mock external dependencies
- Clean up resources in tearDown()
- Document complex test scenarios

### Integration with Week 2
These tests provide the foundation for Week 2 library implementation:
- Placeholder library tests will be replaced with real implementations
- New tests will be added for library-specific functionality
- Integration tests will be extended for new library features

## Week 2 Transition

### What Changes
- Placeholder library classes will be replaced with real implementations
- New tests will be added for kernels, transforms, and optimization libraries
- Integration tests will cover library-specific workflows

### What Stays
- Core orchestration tests remain the same
- API interface tests remain the same
- Legacy compatibility tests remain the same
- Test infrastructure and runner remain the same

The test suite is designed to grow incrementally as new libraries are implemented in Week 2 and beyond.