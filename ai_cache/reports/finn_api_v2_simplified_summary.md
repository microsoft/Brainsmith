# FINN-Brainsmith API V2: Simplified Implementation

**Date**: December 2024  
**Status**: Implemented and Tested  
**Purpose**: Simple, clean replacement for the 6-entrypoint system

## Executive Summary

Successfully implemented a simplified FINN-Brainsmith API that eliminates the confusing "6-entrypoint" concept and provides a clean, direct interface for FINN builds. The implementation consists of just two core classes and directly leverages the existing plugin system.

## Implementation

### Core Components

#### 1. FINNBuildSpec
A simple dataclass that specifies what you want in a FINN build:

```python
@dataclass
class FINNBuildSpec:
    kernels: List[str]                    # e.g., ["MatMul", "LayerNorm"]
    kernel_backends: Dict[str, str]       # e.g., {"MatMul": "hls", "LayerNorm": "rtl"}
    transforms: Dict[str, List[str]]      # stage -> [transform_names]
    output_dir: str = "./finn_output"
    target_device: str = "Pynq-Z1"
    target_frequency_mhz: float = 200.0
    target_fps: Optional[int] = None
    folding_config_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 2. FINNLegacyConverter
Converts the clean API to legacy FINN DataflowBuildConfig:

```python
class FINNLegacyConverter:
    def convert(self, spec: FINNBuildSpec) -> Dict[str, Any]:
        """Convert to DataflowBuildConfig parameters."""
        
    def create_dataflow_config(self, spec: FINNBuildSpec):
        """Create actual DataflowBuildConfig object."""
        
    def execute_build(self, model_path: str, spec: FINNBuildSpec) -> Dict[str, Any]:
        """Execute complete FINN build."""
```

## Usage Examples

### Simple Build
```python
spec = FINNBuildSpec(
    kernels=["MatMul", "LayerNorm"],
    kernel_backends={"MatMul": "hls", "LayerNorm": "rtl"},
    transforms={
        "graph_cleanup": ["RemoveIdentityOps"],
        "topology_optimization": ["ExpandNorms"]
    },
    output_dir="./build"
)

converter = FINNLegacyConverter()
result = converter.execute_build("model.onnx", spec)
```

### BERT Optimization
```python
spec = FINNBuildSpec(
    kernels=["MatMul", "LayerNorm", "Softmax", "GELU"],
    kernel_backends={
        "MatMul": "rtl",
        "LayerNorm": "rtl", 
        "Softmax": "hls",
        "GELU": "hls"
    },
    transforms={
        "graph_cleanup": ["RemoveIdentityOps", "FoldConstants"],
        "topology_optimization": ["ExpandNorms", "FuseAttentionOps"],
        "kernel_optimization": ["BERTFoldingOptimization"]
    },
    target_frequency_mhz=250.0,
    target_fps=3000,
    folding_config_file="configs/bert_folding.json"
)
```

### Integration with Existing Hardware Step
```python
spec = FINNBuildSpec(
    kernels=["MatMul", "LayerNorm", "Softmax"],
    transforms={
        "kernel_mapping": ["infer_hardware_step"]  # Use existing step
    }
)
```

## Files Created

```
brainsmith/core/finn_v2/
├── __init__.py              # Package exports
├── api.py                   # Core API implementation
└── examples.py              # Usage examples

test_finn_api_v2.py                    # Test suite
example_finn_api_integration.py       # Integration examples
```

## Benefits Achieved

### 1. Eliminates Confusion
- **No more "6-entrypoint" concept** - was artificial and didn't exist in FINN
- **Clear separation** - kernels vs transforms are distinct
- **Direct specification** - say exactly what you want

### 2. Leverages Plugin System
- Uses existing `@kernel`, `@backend`, `@transform` decorators
- Automatic discovery through PluginRegistry
- Easy community contributions

### 3. Simple Integration
- Works with existing hardware steps (e.g., `infer_hardware_step`)
- Clean conversion to legacy FINN DataflowBuildConfig
- Easy Blueprint V2 integration

### 4. No Over-Engineering
- Just 2 core classes (~300 lines total)
- No complex strategy hierarchies
- No artificial workflow engines
- Direct mapping to FINN's actual architecture

## Comparison: Old vs New

### Old Approach (6-Entrypoint - Confusing)
```python
entrypoint_config = {
    'entrypoint_1': ['LayerNorm', 'Softmax'],
    'entrypoint_2': ['cleanup', 'streamlining'], 
    'entrypoint_3': ['MatMul', 'LayerNorm'],
    'entrypoint_4': ['matmul_hls', 'layernorm_rtl'],
    'entrypoint_5': ['target_fps_parallelization'],
    'entrypoint_6': ['set_fifo_depths']
}
# → Complex mapping through multiple layers
# → Artificial 'entrypoint' concept that doesn't exist
# → Mixed kernels and transforms
```

### New Approach (Direct - Clear)
```python
spec = FINNBuildSpec(
    kernels=["MatMul", "LayerNorm"],
    kernel_backends={"MatMul": "hls", "LayerNorm": "rtl"},
    transforms={
        "graph_cleanup": ["RemoveIdentityOps"],
        "topology_optimization": ["ExpandNorms", "StreamlineActivations"],
        "kernel_optimization": ["TargetFPSParallelization"]
    },
    target_fps=1000
)
# → Direct specification of what you want
# → Clear separation of concerns
# → Uses plugin system automatically
```

## Test Results

All tests pass successfully:
- ✅ FINNBuildSpec creation and validation
- ✅ FINNLegacyConverter configuration generation  
- ✅ Integration with existing hardware steps
- ✅ BERT optimization example
- ✅ Blueprint V2 constraint mapping

## Integration Points

### With Plugin System
- Automatically uses registered transforms and kernels
- Validates availability through PluginRegistry
- Supports both built-in and community plugins

### With Existing Code
- Works with existing `brainsmith/libraries/transforms/steps/hardware.py`
- Maintains full FINN compatibility
- Easy migration from current DSE system

### With Blueprint V2
- CompilationConfig maps to Blueprint constraints
- Automatic kernel/backend selection based on optimization goals
- Direct integration with DSE evaluation

## Next Steps

1. **Register Existing Components**: Port existing transforms and kernels to plugin system
2. **Blueprint Integration**: Add automatic FINNBuildSpec generation from Blueprint configs
3. **DSE Integration**: Use in evaluation bridge for real FINN builds
4. **Deprecate Old System**: Mark 6-entrypoint system as deprecated

## Conclusion

The simplified FINN-Brainsmith API successfully:
- **Eliminates conceptual confusion** of artificial "entrypoints"
- **Provides clear abstractions** for kernels, transforms, and builds
- **Leverages existing infrastructure** (plugin system, FINN interface)
- **Requires minimal code** (no over-engineering)
- **Maintains compatibility** while enabling future improvements

This implementation provides exactly what was requested: a clean API object for FINN builds and simple conversion to legacy FINN format, without any unnecessary complexity.