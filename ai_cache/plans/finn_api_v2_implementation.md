# FINN-Brainsmith API V2 Implementation Plan

**Version**: 1.0  
**Date**: December 2024  
**Objective**: Implement the new FINN-Brainsmith API to replace the flawed 6-entrypoint system

## Phase 1: Core Abstractions (Week 1)

### Tasks:

- [ ] Create core data models
  - [ ] Implement `Kernel` dataclass
  - [ ] Implement `KernelBackend` dataclass  
  - [ ] Implement `Transform` dataclass
  - [ ] Implement `CompilationStage` and `TransformCategory` enums
  - [ ] Implement `CompilationStrategy` dataclass
  - [ ] Implement `CompilationContext` dataclass
  - [ ] Implement `CompilationResult` dataclass

- [ ] Create base interfaces
  - [ ] Define `IKernelRegistry` interface
  - [ ] Define `ITransformRegistry` interface
  - [ ] Define `IStageExecutor` interface
  - [ ] Define `ICompiler` interface

### Deliverables:
- `brainsmith/core/finn_v2/models.py` - Core data models
- `brainsmith/core/finn_v2/interfaces.py` - Base interfaces
- Unit tests for all models

## Phase 2: Registry Implementation (Week 2)

### Tasks:

- [ ] Implement KernelRegistry
  - [ ] Core registry functionality
  - [ ] Built-in FINN kernel registration
  - [ ] Built-in BrainSmith kernel registration
  - [ ] Kernel validation logic
  - [ ] Backend selection logic

- [ ] Implement TransformRegistry  
  - [ ] Core registry functionality
  - [ ] Built-in cleanup transforms
  - [ ] Built-in optimization transforms
  - [ ] Transform categorization
  - [ ] Stage-based retrieval

- [ ] Create registry configuration
  - [ ] YAML/JSON configuration support
  - [ ] Dynamic loading mechanism
  - [ ] Validation framework

### Deliverables:
- `brainsmith/core/finn_v2/kernel_registry.py`
- `brainsmith/core/finn_v2/transform_registry.py`
- `brainsmith/core/finn_v2/registry_config.py`
- Integration tests for registries

## Phase 3: Transform Implementation (Week 3)

### Tasks:

- [ ] Port existing transforms
  - [ ] Port cleanup transforms
  - [ ] Port streamlining transforms  
  - [ ] Port hardware inference transforms
  - [ ] Port optimization transforms

- [ ] Create transform wrappers
  - [ ] FINN transform adapter
  - [ ] BrainSmith transform adapter
  - [ ] Transform composition utilities

- [ ] Implement transform pipeline
  - [ ] Sequential execution
  - [ ] Conditional execution
  - [ ] Transform dependencies

### Deliverables:
- `brainsmith/core/finn_v2/transforms/` directory structure
- `brainsmith/core/finn_v2/transform_pipeline.py`
- Transform unit tests

## Phase 4: Compiler Implementation (Week 4)

### Tasks:

- [ ] Implement StageExecutor
  - [ ] Stage orchestration logic
  - [ ] Transform application
  - [ ] Context management
  - [ ] Error handling

- [ ] Implement FINNCompiler
  - [ ] Main compilation flow
  - [ ] Strategy application
  - [ ] Progress tracking
  - [ ] Result extraction

- [ ] Create compilation utilities
  - [ ] Model loading/validation
  - [ ] Output management
  - [ ] Metrics collection

### Deliverables:
- `brainsmith/core/finn_v2/stage_executor.py`
- `brainsmith/core/finn_v2/compiler.py`
- `brainsmith/core/finn_v2/compilation_utils.py`
- End-to-end compilation tests

## Phase 5: Legacy Adapter (Week 5)

### Tasks:

- [ ] Implement LegacyFINNAdapter
  - [ ] Context to DataflowBuildConfig conversion
  - [ ] Step function generation
  - [ ] Parameter mapping
  - [ ] Enum conversions

- [ ] Create compatibility layer
  - [ ] Map new transforms to FINN steps
  - [ ] Handle FINN-specific parameters
  - [ ] Maintain backward compatibility

- [ ] Test with existing FINN builds
  - [ ] BERT model testing
  - [ ] CNN model testing
  - [ ] Performance comparison

### Deliverables:
- `brainsmith/core/finn_v2/legacy_adapter.py`
- `brainsmith/core/finn_v2/compatibility.py`
- Compatibility test suite

## Phase 6: Integration and Migration (Week 6)

### Tasks:

- [ ] Update DSE integration
  - [ ] Replace FINNEvaluationBridge usage
  - [ ] Update metrics extraction
  - [ ] Maintain API compatibility

- [ ] Create migration utilities
  - [ ] Convert 6-entrypoint configs to strategies
  - [ ] Blueprint compatibility layer
  - [ ] Migration validation

- [ ] Documentation and examples
  - [ ] API documentation
  - [ ] Migration guide
  - [ ] Example strategies

### Deliverables:
- Updated `brainsmith/core/api_v2.py`
- `brainsmith/core/finn_v2/migration.py`
- Documentation in `docs/finn_v2/`
- Example scripts

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock external dependencies
- Achieve >90% code coverage

### Integration Tests
- Test registry interactions
- Test transform pipelines
- Test compilation flows

### System Tests
- End-to-end BERT compilation
- Performance benchmarks
- Resource utilization tests

### Compatibility Tests
- Ensure existing blueprints work
- Verify FINN builds succeed
- Compare metrics with old system

## Risk Mitigation

### Technical Risks
1. **FINN API changes**: Abstract behind interfaces
2. **Performance regression**: Benchmark throughout
3. **Breaking changes**: Maintain compatibility layer

### Schedule Risks
1. **Complex transforms**: Start with simple ones
2. **FINN integration issues**: Early prototype testing
3. **Testing delays**: Parallel test development

## Success Criteria

1. **Functional**: All existing models compile successfully
2. **Performance**: No regression in compilation time or quality
3. **Maintainable**: Clear separation of concerns, easy to extend
4. **Compatible**: Existing blueprints work without modification
5. **Documented**: Complete API docs and migration guide

## Code Structure

```
brainsmith/core/finn_v2/
├── __init__.py
├── models.py              # Core data models
├── interfaces.py          # Base interfaces
├── kernel_registry.py     # Kernel management
├── transform_registry.py  # Transform management
├── compiler.py           # Main compiler
├── stage_executor.py     # Stage execution
├── legacy_adapter.py     # FINN compatibility
├── transforms/           # Transform implementations
│   ├── __init__.py
│   ├── cleanup.py
│   ├── optimization.py
│   ├── hardware.py
│   └── system.py
├── kernels/             # Kernel specifications
│   ├── __init__.py
│   ├── finn_kernels.py
│   └── brainsmith_kernels.py
└── strategies/          # Pre-built strategies
    ├── __init__.py
    ├── bert.py
    ├── cnn.py
    └── default.py
```

## Next Steps

1. Review and approve design document
2. Set up development branch
3. Begin Phase 1 implementation
4. Weekly progress reviews
5. Continuous integration setup