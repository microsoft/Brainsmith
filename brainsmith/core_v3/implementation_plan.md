# DSE V3 Implementation Plan

## Overview
This document provides a step-by-step implementation plan for the new Brainsmith Core V3 architecture, starting with Phase 1 (Design Space Constructor).

## Directory Structure
```
brainsmith/core_v3/
├── __init__.py
├── implementation_plan.md         # This file
├── phase1/                       # Design Space Constructor
│   ├── __init__.py
│   ├── data_structures.py        # Core data structures
│   ├── parser.py                 # Blueprint parser
│   ├── validator.py              # Design space validator
│   ├── forge.py                  # Main Forge API
│   └── exceptions.py             # Custom exceptions
├── phase2/                       # Design Space Explorer
│   ├── __init__.py
│   ├── explorer.py
│   ├── combination_generator.py
│   └── results_aggregator.py
├── phase3/                       # Build Runner
│   ├── __init__.py
│   ├── runner.py
│   ├── preprocessor.py
│   ├── postprocessor.py
│   └── backends/
└── tests/
    ├── __init__.py
    ├── unit/
    ├── integration/
    └── fixtures/
```

## Implementation Phases

### Phase 1: Design Space Constructor (Week 1-2)

#### Step 1: Core Setup
- [ ] Create directory structure for core_v3
- [ ] Create __init__.py files with proper imports
- [ ] Set up basic project structure

#### Step 2: Data Structures
- [ ] Create `data_structures.py` with all dataclasses
  - [ ] KernelOption (name, backends)
  - [ ] TransformOption (name, optional, mutually_exclusive)
  - [ ] HWCompilerSpace
  - [ ] ProcessingSpace
  - [ ] SearchConfig (constraints, strategy)
  - [ ] GlobalConfig
  - [ ] DesignSpace
- [ ] Add helper methods for combination counting
- [ ] Add string representation methods

#### Step 3: Exceptions
- [ ] Create `exceptions.py` with custom exceptions
  - [ ] BrainsmithError (base)
  - [ ] BlueprintParseError
  - [ ] ValidationError
  - [ ] ConfigurationError

#### Step 4: Blueprint Parser
- [ ] Create `parser.py` with BlueprintParser class
- [ ] Implement version validation
- [ ] Implement kernel parsing
  - [ ] Handle simple string format
  - [ ] Handle tuple format (name, backends)
  - [ ] Handle mutually exclusive groups
  - [ ] Handle optional kernels (~)
- [ ] Implement transform parsing
  - [ ] Handle flat list format
  - [ ] Handle phase-based format
  - [ ] Handle optional transforms (~)
  - [ ] Handle mutually exclusive transforms
- [ ] Implement search config parsing
- [ ] Implement global config parsing
- [ ] Add comprehensive error messages

#### Step 5: Validator
- [ ] Create `validator.py` with DesignSpaceValidator
- [ ] Implement basic structure validation
- [ ] Implement kernel validation
  - [ ] Check for empty kernel lists
  - [ ] Validate tuple formats
  - [ ] Check backend names (if registry available)
- [ ] Implement transform validation
  - [ ] Check transform names
  - [ ] Validate phase names if phase-based
- [ ] Implement constraint validation
- [ ] Implement combination count warnings
- [ ] Add validation result reporting

#### Step 6: Forge API
- [ ] Create `forge.py` with ForgeAPI class
- [ ] Implement main forge() method
- [ ] Add file loading and error handling
- [ ] Implement logging and progress reporting
- [ ] Add summary generation
- [ ] Create convenience functions

#### Step 7: Unit Tests
- [ ] Create test fixtures (sample blueprints)
- [ ] Test data structure creation
- [ ] Test parser with various formats
  - [ ] Simple kernels
  - [ ] Kernels with backends
  - [ ] Mutually exclusive kernels
  - [ ] Optional elements
  - [ ] Both transform formats
- [ ] Test validator edge cases
- [ ] Test error handling
- [ ] Test combination counting

#### Step 8: Integration Tests
- [ ] Create end-to-end forge test
- [ ] Test with real ONNX models
- [ ] Test with complex blueprints
- [ ] Test error scenarios
- [ ] Performance test for large design spaces

### Phase 2: Design Space Explorer (Week 3-4)

#### Step 1: Core Explorer
- [ ] Create `explorer.py` with ExplorerEngine
- [ ] Implement exploration lifecycle
- [ ] Add progress tracking
- [ ] Implement result collection

#### Step 2: Combination Generator
- [ ] Create `combination_generator.py`
- [ ] Implement exhaustive generation
- [ ] Handle kernel combinations
- [ ] Handle transform combinations
- [ ] Handle optional elements
- [ ] Apply constraints

#### Step 3: Results Aggregator
- [ ] Create `results_aggregator.py`
- [ ] Implement result storage
- [ ] Add analysis methods
- [ ] Implement recommendation generation

#### Step 4: Hook System
- [ ] Define hook interfaces
- [ ] Implement hook registry
- [ ] Add default hooks (logging, caching)
- [ ] Document extension points

#### Step 5: Testing
- [ ] Unit tests for each component
- [ ] Integration tests with mock backends
- [ ] Performance tests

### Phase 3: Build Runner (Week 5-6)

#### Step 1: Build Runner Core
- [ ] Create `runner.py` with BuildRunner
- [ ] Implement build lifecycle
- [ ] Add error handling and recovery
- [ ] Implement metrics collection

#### Step 2: Preprocessing
- [ ] Create `preprocessor.py`
- [ ] Define preprocessing interface
- [ ] Implement basic preprocessing steps
- [ ] Add extensibility

#### Step 3: Backend Integration
- [ ] Create backend interface
- [ ] Implement FINN backend adapter
- [ ] Add backend factory
- [ ] Implement result parsing

#### Step 4: Postprocessing
- [ ] Create `postprocessor.py`
- [ ] Implement metrics extraction
- [ ] Add analysis steps
- [ ] Generate reports

#### Step 5: Testing
- [ ] Unit tests for each component
- [ ] Integration tests with FINN
- [ ] End-to-end DSE tests

## Testing Strategy

### Unit Testing
```python
# Example test structure
tests/unit/
├── test_data_structures.py
├── test_parser.py
├── test_validator.py
├── test_forge.py
├── test_explorer.py
├── test_combination_generator.py
└── test_runner.py
```

### Integration Testing
```python
# Example integration test
def test_complete_dse_flow():
    """Test complete DSE flow from blueprint to results"""
    # Create design space
    design_space = forge("model.onnx", "blueprint.yaml")
    
    # Explore design space
    results = explore(design_space)
    
    # Verify results
    assert len(results.evaluations) > 0
    assert results.best_config is not None
```

### Test Fixtures
```yaml
# fixtures/simple_blueprint.yaml
version: "3.0"
hw_compiler:
  kernels:
    - "matmul"
    - ("gemm", ["rtl", "hls"])
  transforms:
    - "quantization"
global:
  output_stage: "rtl"
```

## Implementation Guidelines

### Code Style
- Use type hints throughout
- Follow PEP 8 conventions
- Add comprehensive docstrings
- Keep functions focused and testable

### Error Handling
- Use custom exceptions
- Provide helpful error messages
- Log warnings appropriately
- Fail fast with clear diagnostics

### Documentation
- Document all public APIs
- Add usage examples
- Keep README updated
- Document design decisions

## Milestones

### Milestone 1: Phase 1 Complete (End of Week 2)
- [ ] All Phase 1 components implemented
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation complete
- [ ] Code review completed

### Milestone 2: Phase 2 Complete (End of Week 4)
- [ ] Explorer engine functional
- [ ] Combination generation working
- [ ] Hook system implemented
- [ ] Tests passing

### Milestone 3: Phase 3 Complete (End of Week 6)
- [ ] Build runner integrated
- [ ] FINN backend connected
- [ ] End-to-end flow working
- [ ] Performance validated

## Next Steps After Implementation

1. **Migration Planning**
   - Map existing blueprints to V3 format
   - Create migration scripts
   - Update documentation

2. **Performance Optimization**
   - Profile combination generation
   - Optimize memory usage
   - Add caching layers

3. **Extended Features**
   - Blueprint inheritance
   - Advanced search strategies
   - ML-guided exploration
   - Visual design space browser

## Success Criteria

1. **Functionality**
   - Can parse all blueprint formats
   - Generates correct combinations
   - Integrates with FINN successfully
   - Produces accurate results

2. **Performance**
   - Handles 1000+ combinations efficiently
   - Parallel execution works correctly
   - Memory usage is reasonable

3. **Usability**
   - Clear error messages
   - Intuitive API
   - Good documentation
   - Easy to extend

4. **Code Quality**
   - 90%+ test coverage
   - No critical linting issues
   - Well-documented code
   - Modular design