# BrainSmith Phase 1 Test Suite - Core Functionality

Clean, focused test suite for the restructured Brainsmith three-layer architecture, designed with minimal mocking to validate core functionality for stakeholder delivery.

## 🎯 Overview

This test suite replaces the bloated legacy test system with a streamlined approach that:
- **Tests real components** wherever possible (minimal mocking)
- **Validates the three-layer architecture** (Core → Infrastructure → Libraries)
- **Ensures backward compatibility** is maintained 100%
- **Provides stakeholder confidence** in the restructured platform

## 🏗️ Architecture Alignment

The tests are organized to match the new three-layer architecture:

```
new_tests/
├── core/                    # 🎯 Core Layer Tests
│   ├── test_forge_api.py    # Main forge() function validation
│   ├── test_cli.py          # CLI interface and commands
│   ├── test_metrics.py      # DSEMetrics system validation
│   └── test_validation.py   # Input validation and error handling
│
├── infrastructure/          # 🏗️ Infrastructure Layer Tests  
│   ├── test_design_space.py # Design space engine components
│   └── test_package_imports.py # Import system and fallbacks
│
└── fixtures/                # 🛠️ Test Support
    ├── mock_helpers.py      # Minimal mocking utilities
    └── test_data/           # Sample models and blueprints
```

## 🚀 Quick Start

### Run All Phase 1 Tests
```bash
# Custom test runner with detailed reporting
python new_tests/run_phase1_tests.py

# Standard pytest (all tests)
pytest new_tests/ -v

# Fast smoke test
pytest new_tests/ -m "not slow" -x
```

### Run Specific Test Categories
```bash
# Core layer only
pytest new_tests/core/ -v

# Infrastructure layer only  
pytest new_tests/infrastructure/ -v

# Specific test file
pytest new_tests/core/test_forge_api.py -v
```

### Run with Coverage
```bash
pytest new_tests/ --cov=brainsmith --cov-report=html
```

## 📊 Test Categories

### 🎯 Core Layer Tests (`core/`)

#### **test_forge_api.py** - Main API Validation
- ✅ Basic forge() function with valid inputs
- ✅ Hardware graph mode (`is_hw_graph=True/False`)
- ✅ Build modes (`build_core=True/False`)
- ✅ Input validation and error handling
- ✅ Output directory handling
- ✅ Fallback behavior when components unavailable

#### **test_cli.py** - CLI Interface  
- ✅ Command parsing and execution
- ✅ Help and version commands
- ✅ Error handling and user feedback
- ✅ Output formatting and UX
- ✅ Argument validation

#### **test_metrics.py** - Metrics System
- ✅ DSEMetrics, PerformanceMetrics, ResourceMetrics classes
- ✅ Metric calculations and scoring
- ✅ Serialization/deserialization (JSON, dict)
- ✅ Validation and edge cases
- ✅ Optimization score computation

#### **test_validation.py** - Input Validation
- ✅ File existence and format validation
- ✅ Blueprint validation system
- ✅ Objectives and constraints validation
- ✅ Error message quality and clarity

### 🏗️ Infrastructure Layer Tests (`infrastructure/`)

#### **test_design_space.py** - DSE Engine
- ✅ ParameterDefinition, DesignPoint, DesignSpace classes
- ✅ Parameter validation and type checking
- ✅ Design point sampling and generation
- ✅ Blueprint integration and parsing
- ✅ Serialization and data persistence

#### **test_package_imports.py** - Import System
- ✅ Core, infrastructure, and library imports
- ✅ Graceful fallback for missing components
- ✅ __all__ exports validation
- ✅ Package structure and metadata
- ✅ Circular import prevention

## 🧪 Testing Philosophy

### Minimal Mocking Approach
- **Real Components**: Test actual implementations wherever possible
- **External Dependencies Only**: Mock only FINN, file I/O when necessary  
- **Fallback Testing**: Validate graceful degradation when components missing
- **Integration Focus**: Ensure components work together correctly

### Key Testing Principles
1. **Fast Execution**: Complete suite runs in < 30 seconds
2. **Clear Results**: Obvious pass/fail with actionable error messages
3. **Stakeholder Ready**: Validates core workflows work end-to-end
4. **Maintainable**: Simple structure, minimal setup, easy to extend

## 📈 Success Metrics

### Expected Results
- **≥95% pass rate**: Excellent - Ready for stakeholder delivery
- **≥85% pass rate**: Good - Ready for Phase 2 implementation  
- **≥70% pass rate**: Warning - Consider fixes before Phase 2
- **<70% pass rate**: Critical - Fix issues before proceeding

### Sample Output
```
🧪======================================================================
🚀 BrainSmith Phase 1 Implementation - Core Functionality Test Suite
🧪======================================================================

🧪 Running Core Layer Tests...
  📝 test_forge_api.py        ✅ PASSED
  📝 test_cli.py             ✅ PASSED  
  📝 test_metrics.py         ✅ PASSED
  📝 test_validation.py      ✅ PASSED

🧪 Running Infrastructure Layer Tests...
  📝 test_design_space.py    ✅ PASSED
  📝 test_package_imports.py ⏭️  SKIPPED (components not available)

🎯 PHASE 1 CORE FUNCTIONALITY TEST SUMMARY
📊 OVERALL RESULTS:
   📁 Total test files: 6
   ✅ Passed: 5
   ❌ Failed: 0  
   ⏭️  Skipped: 1
   📈 Success rate: 83.3%
   🚀 GOOD: Phase 1 ready for Phase 2 implementation
```

## 🔧 Fixtures and Test Data

### Shared Fixtures (`conftest.py`)
- **sample_model_path**: Temporary ONNX model file
- **sample_blueprint_path**: Valid blueprint YAML
- **invalid_blueprint_path**: Invalid blueprint for error testing
- **expected_forge_result**: Standard result structure validation
- **temp_test_dir**: Isolated temporary directory

### Mock Helpers (`fixtures/mock_helpers.py`)
- **MockFINNInterface**: FINN build simulation
- **MockDSEResults**: DSE result generation
- **MockONNXModel**: ONNX model representation
- **Context Managers**: For mocking unavailable dependencies

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```
Warning: Could not import core components
```
**Solution**: Ensure Python path includes project root:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python new_tests/run_phase1_tests.py
```

#### All Tests Skipped
```
⏭️  All tests skipped - components not available
```
**Solution**: Check that core modules are implemented and importable

#### Dependency Missing
```
ImportError: No module named 'click'
```
**Solution**: Install test dependencies:
```bash
pip install pytest click
```

### Debug Mode
```bash
# Maximum verbosity
pytest new_tests/ -vvv --tb=long

# Stop on first failure
pytest new_tests/ -x

# Run specific test method
pytest new_tests/core/test_forge_api.py::TestForgeAPI::test_forge_basic_functionality -v
```

## 🔮 Future Phases

### Phase 2: Library Robustness Tests
- Kernels library validation
- Transforms library testing
- Analysis tools verification
- Automation system testing

### Phase 3: Hooks and Advanced Features
- Hooks system validation
- Extension point testing
- Advanced DSE scenarios
- Performance benchmarking

## 🤝 Contributing

### Adding New Tests
1. Follow the naming convention: `test_<functionality>.py`
2. Use clear, descriptive test method names
3. Include both positive and negative test cases
4. Add proper docstrings explaining what's being tested
5. Use existing fixtures and helpers where possible

### Test Guidelines
- **One thing per test**: Each test method should test one specific functionality
- **Clear assertions**: Use descriptive assertion messages
- **Minimal mocking**: Only mock external dependencies
- **Resource cleanup**: Always clean up temporary resources
- **Documentation**: Document complex test scenarios

### Extending for Stakeholder Needs
- Add tests in `contrib/` directories for stakeholder-specific validation
- Follow the minimal mocking approach
- Ensure tests run quickly (< 5 seconds per test file)
- Include comprehensive error reporting

## 📚 Related Documentation

- **[Implementation Checklist](../NEW_TESTS_IMPLEMENTATION_CHECKLIST.md)** - Task tracking
- **[Repository Restructuring Plan](../BRAINSMITH_REPOSITORY_RESTRUCTURING_PLAN_FINAL.md)** - Architecture overview
- **[Legacy Tests](../tests/)** - Original test suite (for reference)

---

**This test suite provides stakeholders with confidence that the restructured Brainsmith platform maintains full functionality while offering a clean, extensible foundation for future development.**