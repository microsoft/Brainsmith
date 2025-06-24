# FINN API V2 Implementation Plan

**Date**: December 2024  
**Purpose**: Step-by-step plan to implement the new FINN-Brainsmith API

## Overview

This plan outlines the implementation of the new FINN-Brainsmith API that replaces the flawed "6-entrypoint" system with a clean, strategy-based approach leveraging the plugin system.

## Phase 1: Core Infrastructure (Week 1)

### 1.1 Base Data Classes
- [ ] Create `brainsmith/core/finn_v2/__init__.py`
- [ ] Implement `CompilationConfig` dataclass
- [ ] Implement `CompilationStage` enum
- [ ] Implement `TransformSequence` class
- [ ] Implement `KernelSelection` class
- [ ] Add comprehensive type hints and docstrings

### 1.2 Strategy Base Class
- [ ] Create `brainsmith/core/finn_v2/strategies.py`
- [ ] Implement abstract `CompilationStrategy` base class
- [ ] Define abstract methods for kernel selection
- [ ] Define abstract methods for transform sequences
- [ ] Define abstract methods for FINN parameters

### 1.3 Concrete Strategies
- [ ] Implement `HighPerformanceStrategy`
- [ ] Implement `AreaOptimizedStrategy` 
- [ ] Implement `BalancedStrategy`
- [ ] Add unit tests for each strategy

## Phase 2: Compiler Implementation (Week 1-2)

### 2.1 FINNCompiler Core
- [ ] Create `brainsmith/core/finn_v2/compiler.py`
- [ ] Implement `FINNCompiler` class
- [ ] Add strategy registration system
- [ ] Implement workflow orchestration
- [ ] Add error handling and validation

### 2.2 Transform Application
- [ ] Implement transform sequence execution
- [ ] Add transform parameter handling
- [ ] Integrate with plugin registry
- [ ] Add progress tracking and logging

### 2.3 Kernel Backend Selection
- [ ] Implement kernel backend resolution
- [ ] Add fallback mechanisms
- [ ] Validate kernel availability
- [ ] Handle missing backends gracefully

## Phase 3: Legacy FINN Integration (Week 2)

### 3.1 Legacy Adapter
- [ ] Create `brainsmith/core/finn_v2/legacy_adapter.py`
- [ ] Implement `LegacyFINNAdapter` class
- [ ] Add DataflowBuildConfig generation
- [ ] Implement model serialization handling

### 3.2 Step Function Generation
- [ ] Convert transform sequences to FINN steps
- [ ] Map strategy parameters to FINN config
- [ ] Handle custom step injection
- [ ] Maintain step execution order

### 3.3 Metrics Extraction
- [ ] Implement FINN output parsing
- [ ] Extract performance metrics
- [ ] Extract resource utilization
- [ ] Handle missing metrics gracefully

## Phase 4: Integration Testing (Week 2-3)

### 4.1 Test Infrastructure
- [ ] Create `tests/unit/core/finn_v2/` directory
- [ ] Set up test fixtures and mocks
- [ ] Create sample models for testing
- [ ] Add integration test framework

### 4.2 Strategy Testing
- [ ] Test each strategy with sample models
- [ ] Verify kernel selection logic
- [ ] Validate transform sequences
- [ ] Check FINN parameter generation

### 4.3 End-to-End Testing
- [ ] Test full compilation workflow
- [ ] Verify FINN integration
- [ ] Test error handling paths
- [ ] Benchmark performance

## Phase 5: Migration Support (Week 3)

### 5.1 Migration Utilities
- [ ] Create `brainsmith/core/finn_v2/migration.py`
- [ ] Add 6-entrypoint to strategy converter
- [ ] Implement configuration migration
- [ ] Add deprecation warnings

### 5.2 Documentation
- [ ] Write API reference documentation
- [ ] Create migration guide
- [ ] Add usage examples
- [ ] Document best practices

### 5.3 Backward Compatibility
- [ ] Create compatibility shim for old API
- [ ] Add transition period support
- [ ] Log usage of deprecated features
- [ ] Plan deprecation timeline

## Phase 6: Advanced Features (Week 4)

### 6.1 Custom Strategy Support
- [ ] Add custom strategy registration
- [ ] Implement strategy composition
- [ ] Add strategy validation
- [ ] Create strategy builder helpers

### 6.2 Dynamic Workflows
- [ ] Add conditional transform execution
- [ ] Implement transform dependencies
- [ ] Add checkpoint/restart capability
- [ ] Enable parallel transform paths

### 6.3 Enhanced Debugging
- [ ] Add detailed execution tracing
- [ ] Implement intermediate model inspection
- [ ] Add performance profiling
- [ ] Create debugging utilities

## Implementation Checklist

### Week 1: Foundation
- [ ] Complete Phase 1 (Core Infrastructure)
- [ ] Begin Phase 2 (Compiler Implementation)
- [ ] Set up CI/CD for new module
- [ ] Create initial documentation

### Week 2: Integration
- [ ] Complete Phase 2 (Compiler Implementation)
- [ ] Complete Phase 3 (Legacy FINN Integration)
- [ ] Begin Phase 4 (Integration Testing)
- [ ] Update existing tests

### Week 3: Testing & Migration
- [ ] Complete Phase 4 (Integration Testing)
- [ ] Complete Phase 5 (Migration Support)
- [ ] Run performance benchmarks
- [ ] Get user feedback

### Week 4: Polish & Advanced Features
- [ ] Complete Phase 6 (Advanced Features)
- [ ] Finalize documentation
- [ ] Prepare release notes
- [ ] Plan deprecation announcement

## Code Structure

```
brainsmith/core/finn_v2/
├── __init__.py              # Public API exports
├── config.py                # Configuration classes
├── strategies.py            # Strategy implementations
├── compiler.py              # Main compiler class
├── legacy_adapter.py        # FINN compatibility
├── migration.py             # Migration utilities
├── utils.py                 # Helper functions
└── exceptions.py            # Custom exceptions

tests/unit/core/finn_v2/
├── test_config.py
├── test_strategies.py
├── test_compiler.py
├── test_legacy_adapter.py
├── test_migration.py
└── test_integration.py
```

## Risk Mitigation

### Technical Risks
1. **FINN API Changes**: Maintain adapter pattern for isolation
2. **Plugin System Issues**: Add fallback mechanisms
3. **Performance Regression**: Benchmark against current system
4. **Breaking Changes**: Provide compatibility layer

### Timeline Risks
1. **Scope Creep**: Focus on MVP first, advanced features later
2. **Testing Delays**: Parallelize testing efforts
3. **Integration Issues**: Early integration testing
4. **Documentation Lag**: Write docs alongside code

## Success Criteria

### Functional Requirements
- [ ] All existing models compile successfully
- [ ] Performance matches or exceeds current system
- [ ] Clean API with no "6-entrypoint" concept
- [ ] Full plugin system integration

### Non-Functional Requirements
- [ ] Clear and comprehensive documentation
- [ ] 90%+ test coverage
- [ ] Migration path for existing users
- [ ] Maintainable and extensible codebase

## Next Steps

1. **Review and Approval**: Get design approval from team
2. **Environment Setup**: Prepare development environment
3. **Implementation Start**: Begin with Phase 1
4. **Regular Check-ins**: Weekly progress reviews
5. **User Testing**: Early access program for feedback

## Conclusion

This implementation plan provides a clear path to replace the current FINN integration with a cleaner, more maintainable system. The phased approach allows for incremental progress while maintaining system stability.