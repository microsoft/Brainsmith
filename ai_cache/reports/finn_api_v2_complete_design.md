# FINN-Brainsmith API V2: Complete Design Package

**Date**: December 2024  
**Status**: Complete Design Ready for Implementation  
**Purpose**: Comprehensive design to replace the 6-entrypoint system

## Executive Summary

This document summarizes the complete design package for FINN-Brainsmith API V2, which replaces the flawed "6-entrypoint" system with a clean, strategy-based approach that leverages the new plugin system.

## Problem Statement

The current FINN integration system has fundamental issues:

1. **Artificial "6-Entrypoint" Concept**: The system claims to have 6 compilation entrypoints, but this is a fabricated abstraction that doesn't exist in FINN
2. **Mixed Concerns**: Conflates DSE components with build steps, kernels with transforms
3. **Over-Complexity**: Multiple unnecessary translation layers create confusion
4. **Poor Separation**: No clear distinction between hardware implementations and graph operations

## Solution: Strategy-Based Architecture

### Core Concept
Replace artificial entrypoints with **Compilation Strategies** that:
- Select appropriate kernels and backends
- Compose transform sequences for each compilation stage  
- Configure FINN parameters for the desired optimization goal

### Key Abstractions

```python
@dataclass
class CompilationConfig:
    """High-level compilation configuration."""
    output_dir: str
    target_device: str
    target_frequency_mhz: float
    constraints: Dict[str, Any]

class CompilationStrategy(ABC):
    """Base class for compilation approaches."""
    
    @abstractmethod
    def select_kernels(self) -> KernelSelection:
        """Choose kernels and their backends."""
        
    @abstractmethod 
    def build_transform_sequence(self) -> Dict[CompilationStage, TransformSequence]:
        """Build transform pipeline."""
        
    @abstractmethod
    def get_finn_parameters(self) -> Dict[str, Any]:
        """Configure FINN backend."""

class FINNCompiler:
    """Main orchestrator."""
    
    def compile(self, model_path: str, strategy: str, config: CompilationConfig):
        """Execute compilation workflow."""
```

## Built-in Strategies

### 1. HighPerformanceStrategy
- **Goal**: Maximize throughput
- **Kernels**: Prefer RTL backends for compute-intensive operations
- **Transforms**: Aggressive parallelization, minimal cleanup
- **FINN Config**: Wide MVAU, large FIFOs, full precision

### 2. AreaOptimizedStrategy  
- **Goal**: Minimize resource usage
- **Kernels**: Prefer HLS backends for resource sharing
- **Transforms**: Aggressive cleanup, bit-width reduction
- **FINN Config**: Narrow MVAU, small FIFOs, standalone thresholds

### 3. BalancedStrategy
- **Goal**: Trade-offs between performance and area
- **Kernels**: Mix of RTL (compute) and HLS (control) backends
- **Transforms**: Moderate optimizations
- **FINN Config**: Default settings with constraint-based tuning

## Plugin System Integration

### Leverages Existing Infrastructure
- All kernels registered via `@kernel` decorator
- All backends registered via `@backend` decorator  
- All transforms registered via `@transform` decorator
- Automatic discovery from `brainsmith/kernels/` and `brainsmith/transforms/`

### Clean Separation
- **Kernels**: Hardware implementations (MatMul, LayerNorm, etc.)
- **Transforms**: Graph operations (RemoveIdentityOps, StreamlineActivations, etc.)
- **Strategies**: High-level compilation approaches

## Usage Examples

### Simple Compilation
```python
compiler = FINNCompiler()
result = compiler.compile("model.onnx", strategy="balanced")
```

### Custom Configuration
```python
config = CompilationConfig(
    output_dir="./my_build",
    target_device="U250", 
    target_frequency_mhz=300.0,
    constraints={"target_fps": 5000}
)

result = compiler.compile("model.onnx", strategy="performance", config=config)
```

### Custom Strategy
```python
class BERTOptimizedStrategy(CompilationStrategy):
    def select_kernels(self):
        return KernelSelection().use_kernel("MatMul", "MatMulRTL")
    
    def build_transform_sequence(self):
        return {
            CompilationStage.TOPOLOGY_OPTIMIZATION: TransformSequence().add(
                "FuseAttentionOps", {"num_heads": 12}
            )
        }
    
    def get_finn_parameters(self):
        return {"mvau_wwidth_max": 64, "folding_config_file": "bert_folding.json"}

bert_strategy = BERTOptimizedStrategy(config)
result = compiler.compile("bert.onnx", strategy=bert_strategy)
```

## Legacy FINN Integration

### LegacyFINNAdapter
- Converts modern API to DataflowBuildConfig
- Maintains full FINN compatibility
- Isolates FINN dependencies for easy updates
- Handles step function generation and parameter mapping

### Migration Path
- Compatibility shim for transition period
- Migration utilities to convert old 6-entrypoint configs
- Clear deprecation timeline with upgrade guidance

## Benefits

### 1. Eliminates Conceptual Confusion
- No more artificial "6-entrypoint" abstraction
- Clear mapping to actual compilation stages
- Direct relationship to FINN's architecture

### 2. Clean Separation of Concerns
- Kernels vs transforms clearly distinguished
- Strategies encapsulate high-level decisions
- Plugin system handles registration and discovery

### 3. Leverages Plugin Infrastructure  
- Uses existing `@transform`, `@kernel`, `@backend` decorators
- Automatic discovery and registration
- Easy contribution model for community

### 4. Flexible and Extensible
- Easy to add new strategies
- Custom strategies are first-class citizens
- Dynamic transform composition

### 5. Maintains FINN Compatibility
- Full DataflowBuildConfig support
- All existing FINN features available
- Clean adapter pattern for future FINN updates

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] `CompilationConfig`, `CompilationStrategy` base classes
- [ ] `HighPerformanceStrategy`, `AreaOptimizedStrategy`, `BalancedStrategy`
- [ ] `TransformSequence`, `KernelSelection` utilities

### Phase 2: Compiler Implementation (Week 1-2)  
- [ ] `FINNCompiler` orchestration
- [ ] Transform sequence execution
- [ ] Kernel backend selection and validation

### Phase 3: Legacy Integration (Week 2)
- [ ] `LegacyFINNAdapter` implementation
- [ ] DataflowBuildConfig generation
- [ ] Metrics extraction from FINN outputs

### Phase 4: Testing & Migration (Week 2-3)
- [ ] Comprehensive test suite
- [ ] Migration utilities for old configs
- [ ] Documentation and examples

### Phase 5: Advanced Features (Week 3-4)
- [ ] Custom strategy helpers
- [ ] Enhanced debugging capabilities
- [ ] Performance optimizations

## File Structure

```
brainsmith/core/finn_v2/
├── __init__.py              # Public API exports
├── config.py                # CompilationConfig and enums
├── strategies.py            # Strategy implementations  
├── compiler.py              # FINNCompiler main class
├── legacy_adapter.py        # FINN compatibility
├── migration.py             # Migration utilities
└── exceptions.py            # Custom exceptions

tests/unit/core/finn_v2/
├── test_config.py
├── test_strategies.py
├── test_compiler.py
├── test_legacy_adapter.py
└── test_integration.py
```

## Success Criteria

### Functional
- [ ] All existing models compile successfully
- [ ] Performance matches or exceeds current system
- [ ] Clean API with intuitive abstractions
- [ ] Full plugin system integration

### Non-Functional  
- [ ] Clear and comprehensive documentation
- [ ] 90%+ test coverage
- [ ] Easy migration path for existing users
- [ ] Maintainable and extensible codebase

## Migration Guide

### From 6-Entrypoint System
```python
# Old approach (confusing)
old_config = {
    'entrypoint_1': ['LayerNorm', 'Softmax'],
    'entrypoint_2': ['cleanup', 'streamlining'], 
    'entrypoint_3': ['MatMul', 'LayerNorm'],
    'entrypoint_4': ['matmul_hls', 'layernorm_rtl'],
    'entrypoint_5': ['target_fps_parallelization'],
    'entrypoint_6': ['set_fifo_depths']
}

# New approach (clear)
config = CompilationConfig(
    target_device="Pynq-Z1",
    target_frequency_mhz=200.0,
    constraints={"target_fps": 1000}
)

result = compiler.compile("model.onnx", strategy="balanced", config=config)
```

### Key Changes
1. **No More Entrypoints**: Use strategies instead of arbitrary entrypoint groupings
2. **Clear Abstractions**: Kernels, transforms, and strategies are distinct concepts
3. **Plugin Integration**: All components use the existing plugin system
4. **Simplified Configuration**: Direct parameter specification instead of complex mappings

## Documents Created

1. **`ai_cache/designs/finn_api_v2_clean_design.md`**: Complete architecture design
2. **`ai_cache/plans/finn_api_v2_implementation_plan.md`**: 4-week implementation roadmap  
3. **`ai_cache/examples/finn_api_v2_usage.py`**: Comprehensive usage examples
4. **`ai_cache/designs/finn_api_v2_example_implementation.py`**: Working code examples

## Next Steps

1. **Design Review**: Get team approval for the new architecture
2. **Implementation Kickoff**: Begin Phase 1 development
3. **Testing Setup**: Establish CI/CD for the new module
4. **User Engagement**: Get feedback from early adopters
5. **Migration Planning**: Plan transition timeline from old system

## Conclusion

The FINN-Brainsmith API V2 provides a modern, clean interface that:
- Eliminates the conceptual confusion of artificial "entrypoints"
- Properly separates hardware implementations from graph operations
- Leverages the plugin system for community contributions
- Maintains full backward compatibility with FINN
- Provides clear extension points for future development

This design removes significant technical debt while positioning BrainSmith as a truly extensible platform for FPGA AI acceleration.