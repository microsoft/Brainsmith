# FINN-Brainsmith API V2 Design Summary

**Date**: December 2024  
**Status**: Design Complete, Ready for Implementation

## Executive Summary

We have designed a completely new FINN-Brainsmith API that replaces the flawed "6-entrypoint" system with a cleaner, more intuitive architecture based on proper abstractions:

1. **Kernels & Backends**: Hardware operations and their implementations
2. **Transforms**: Graph transformations organized by compilation stage
3. **Compilation Strategies**: High-level specifications for model compilation
4. **Stage-Based Execution**: Clear progression through compilation phases

## Key Design Decisions

### 1. Abandoning "6-Entrypoints"

The "6-entrypoint" concept was fundamentally flawed because:
- It mixed different levels of abstraction (operations, transforms, parameters)
- It created artificial boundaries that don't match compilation flow
- It added complexity without providing real benefits

Instead, we now have:
- **Kernels**: What operations to accelerate
- **Transforms**: How to optimize the graph
- **Strategies**: Complete compilation specifications

### 2. Registry Pattern

All components (kernels, transforms) are managed through registries:
- Centralized management
- Easy extension without core changes
- Clear discovery and validation
- Type-safe access

### 3. Stage-Based Compilation

Clear compilation stages replace mixed build steps:
1. **Graph Cleanup**: Basic optimizations
2. **Topology Optimization**: Model-level transforms
3. **Kernel Mapping**: Hardware lowering
4. **Kernel Optimization**: Operation-specific opts
5. **Graph Optimization**: System-level opts

### 4. Clean Separation of Concerns

- **Kernels**: WHAT operations to accelerate
- **Backends**: HOW to implement kernels
- **Transforms**: Graph modifications
- **Strategies**: Complete compilation recipes

## Architecture Components

### Core Classes

```python
# Data Models
- Kernel: Hardware operation specification
- KernelBackend: Specific implementation
- Transform: Graph transformation
- CompilationStrategy: Complete compilation spec

# Registries
- KernelRegistry: Manages kernels/backends
- TransformRegistry: Manages transforms

# Execution
- StageExecutor: Executes compilation stages
- FINNCompiler: Main orchestrator
- LegacyFINNAdapter: FINN compatibility
```

### Data Flow

```
CompilationStrategy
    ↓
FINNCompiler
    ↓
StageExecutor (for each stage)
    ↓
Transform Application
    ↓
LegacyFINNAdapter
    ↓
FINN DataflowBuildConfig
```

## Benefits Over Current System

1. **Clarity**: Clear abstractions that match mental models
2. **Extensibility**: Easy to add kernels, transforms, strategies
3. **Type Safety**: Strong typing throughout
4. **Maintainability**: Clean separation of concerns
5. **Flexibility**: Strategies compose kernels and transforms
6. **Compatibility**: Clean adapter for legacy FINN

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)
- Data models
- Base interfaces
- Basic registries

### Phase 2: Component Implementation (Weeks 2-3)
- Kernel registry with built-ins
- Transform registry with built-ins
- Transform implementations

### Phase 3: Compilation Engine (Week 4)
- Stage executor
- Main compiler
- Result extraction

### Phase 4: Integration (Weeks 5-6)
- Legacy FINN adapter
- DSE integration
- Migration utilities

## Example Usage

```python
# Create strategy
strategy = CompilationStrategy(
    name="bert_optimized",
    kernels=[
        KernelSpec(kernel="MatMul", backend="hls"),
        KernelSpec(kernel="LayerNorm", backend="brainsmith_rtl")
    ],
    transforms=[
        TransformSpec(transform="Streamline", config={"level": 2}),
        TransformSpec(transform="SetFolding", config={"target_fps": 3000})
    ],
    parameters={
        "target_frequency_mhz": 200,
        "folding_config": "bert_folding.json"
    }
)

# Compile
compiler = FINNCompiler()
result = compiler.compile("bert.onnx", strategy, "./output")
```

## Migration Path

1. Implement new API alongside existing system
2. Create adapters for backward compatibility
3. Migrate existing code incrementally
4. Deprecate old system once stable

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| FINN API changes | Abstract behind interfaces |
| Performance regression | Continuous benchmarking |
| Breaking changes | Compatibility layer |
| Complex transforms | Start with simple ones |

## Success Metrics

- All existing models compile successfully
- No performance regression
- Improved code maintainability
- Easier to add new features
- Clear documentation

## Conclusion

The new FINN-Brainsmith API V2 provides a clean, extensible architecture that:
- Properly separates kernels, transforms, and strategies
- Uses clear abstractions instead of misleading concepts
- Maintains FINN compatibility while enabling future growth
- Significantly improves code maintainability

This design is ready for implementation and will provide a solid foundation for the future of the Brainsmith platform.