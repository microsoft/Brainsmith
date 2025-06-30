# Unified Plugin System - Final Design

## Overview

The unified plugin system provides explicit, type-specific decorators for registering components while maintaining a single, powerful registry underneath. This design eliminates all compatibility layers and legacy patterns, following Prime Directive 1 (Break Fearlessly).

## Key Design Decisions

1. **Explicit Decorators**: `@transform`, `@kernel`, `@backend` instead of generic `@plugin`
2. **Unified Registry**: Single registry for all component types
3. **Smart Transform Behavior**: Transforms with `kernel` parameter become kernel inference
4. **No Stage for Kernel Inference**: Kernel inference transforms have `stage=None` since they attach to kernels, not stages
5. **Backend Declaration**: Backends declare their kernel association
6. **Optional Metadata**: Version, author, and description are optional

## Component Types

### 1. Regular Transforms
Transforms that modify the computational graph at specific compilation stages.

```python
@transform(
    name="ExpandNorms",
    stage="topology_opt",        # Required for regular transforms
    description="Expand normalization operations",
    author="brainsmith-team",    # Optional
    version="1.0.0"              # Optional
)
class ExpandNorms(Transformation):
    def apply(self, model):
        # Transform implementation
        return model, graph_modified
```

### 2. Kernel Inference Transforms
Transforms that detect patterns and convert them to hardware kernels. These attach to kernels, not stages.

```python
@transform(
    name="InferLayerNorm",
    kernel="LayerNorm",          # Makes this kernel inference
    stage=None,                  # Always None for kernel inference
    description="Detect and convert LayerNorm patterns"
)
class InferLayerNorm(Transformation):
    def apply(self, model):
        # Pattern matching and conversion
        return model, graph_modified
```

**Important**: When `kernel` is specified:
- The transform is registered as type "kernel_inference"
- `stage` must be None (these transforms attach to kernels, not pipeline stages)
- The transform will be associated with the specified kernel

### 3. Kernels
Hardware operations that can be implemented in HLS or RTL.

```python
@kernel(
    name="LayerNorm",
    op_type="LayerNorm",         # Optional: ONNX op type
    domain="brainsmith.kernels.layernorm",  # Optional: ONNX domain
    description="Hardware LayerNorm implementation"
)
class LayerNorm(HWCustomOp):
    def get_nodeattr_types(self):
        # Define node attributes
        pass
```

### 4. Backends
Concrete implementations of kernels in HLS or RTL.

```python
@backend(
    name="LayerNormHLS",
    kernel="LayerNorm",          # Required: which kernel this implements
    backend_type="hls",          # Required: "hls" or "rtl"
    description="HLS C++ implementation of LayerNorm"
)
class LayerNormHLS(LayerNorm, HLSBackend):
    def emit(self):
        # Generate HLS code
        pass

@backend(
    name="LayerNormRTL",
    kernel="LayerNorm",
    backend_type="rtl",
    description="Optimized RTL implementation"
)
class LayerNormRTL(LayerNorm, RTLBackend):
    def emit(self):
        # Generate Verilog
        pass
```

## Decorator Specifications

### @transform

```python
def transform(
    name: str,                    # Required: unique identifier
    stage: Optional[str] = None,  # Required for regular, None for kernel inference
    kernel: Optional[str] = None, # Makes this kernel inference
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None,
    **kwargs                      # Additional metadata
) -> Callable[[Type], Type]
```

**Validation Rules**:
- If `kernel` is specified: `stage` must be None, registers as "kernel_inference"
- If `kernel` is not specified: `stage` is required, registers as "transform"
- Valid stages: "cleanup", "topology_opt", "kernel_opt", "dataflow_opt"

### @kernel

```python
def kernel(
    name: str,                    # Required: unique identifier
    op_type: Optional[str] = None,
    domain: Optional[str] = None,
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None,
    **kwargs
) -> Callable[[Type], Type]
```

### @backend

```python
def backend(
    name: str,                    # Required: unique identifier
    kernel: str,                  # Required: kernel this implements
    backend_type: str,            # Required: "hls" or "rtl"
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None,
    **kwargs
) -> Callable[[Type], Type]
```

**Validation Rules**:
- `backend_type` must be either "hls" or "rtl"
- `kernel` must reference an existing kernel name

## Registry Architecture

### Storage Format
```python
{
    "transform:ExpandNorms": {
        "type": "transform",
        "name": "ExpandNorms",
        "class": ExpandNorms,
        "stage": "topology_opt",
        "description": "...",
        # ... other metadata
    },
    "kernel_inference:InferLayerNorm": {
        "type": "kernel_inference",
        "name": "InferLayerNorm",
        "class": InferLayerNorm,
        "kernel": "LayerNorm",
        "stage": None,
        # ... other metadata
    },
    "kernel:LayerNorm": {
        "type": "kernel",
        "name": "LayerNorm",
        "class": LayerNorm,
        # ... other metadata
    },
    "backend:LayerNormHLS": {
        "type": "backend",
        "name": "LayerNormHLS",
        "class": LayerNormHLS,
        "kernel": "LayerNorm",
        "backend_type": "hls",
        # ... other metadata
    }
}
```

### Query Examples

```python
from brainsmith.plugin.core import get_registry

registry = get_registry()

# Get specific component
transform = registry.get("transform", "ExpandNorms")
kernel = registry.get("kernel", "LayerNorm")

# Find all transforms for a stage
topology_transforms = registry.query(type="transform", stage="topology_opt")

# Find kernel inference transforms for a kernel
inferences = registry.query(type="kernel_inference", kernel="LayerNorm")

# Find all backends for a kernel
backends = registry.query(type="backend", kernel="LayerNorm")
# Returns: [LayerNormHLS info, LayerNormRTL info]

# Find specific backend type
hls_backends = registry.query(type="backend", backend_type="hls")
```

## Complete Feature Example

```python
from brainsmith.plugin.core import transform, kernel, backend

# 1. Define the kernel
@kernel(
    name="Softmax",
    op_type="Softmax",
    description="Hardware softmax operation"
)
class Softmax(HWCustomOp):
    """Computes softmax activation."""
    pass

# 2. Add HLS backend
@backend(
    name="SoftmaxHLS",
    kernel="Softmax",
    backend_type="hls",
    description="HLS implementation using logarithmic approximation"
)
class SoftmaxHLS(Softmax, HLSBackend):
    pass

# 3. Add RTL backend
@backend(
    name="SoftmaxRTL",
    kernel="Softmax",
    backend_type="rtl",
    description="Optimized RTL with pipelined architecture"
)
class SoftmaxRTL(Softmax, RTLBackend):
    pass

# 4. Add kernel inference
@transform(
    name="InferSoftmax",
    kernel="Softmax",
    stage=None,  # Always None for kernel inference
    description="Convert ONNX Softmax to hardware implementation"
)
class InferSoftmax(Transformation):
    def apply(self, model):
        # Find softmax patterns and replace with Softmax kernel
        return model, graph_modified
```

## Migration from Old System

### Old Pattern
```python
from brainsmith.plugin.decorators import transform, kernel_inference_transform

@transform(
    name="ExpandNorms",
    stage="topology_opt",
    category="streamline"  # Old system had categories
)
class ExpandNorms(Transformation):
    pass

@kernel_inference_transform(
    name="InferLayerNorm",
    kernel="LayerNorm",
    stage="kernel_opt"  # Old system put these in stages
)
class InferLayerNorm(Transformation):
    pass
```

### New Pattern
```python
from brainsmith.plugin.core import transform

@transform(
    name="ExpandNorms",
    stage="topology_opt"  # No category
)
class ExpandNorms(Transformation):
    pass

@transform(
    name="InferLayerNorm",
    kernel="LayerNorm",
    stage=None  # Kernel inference has no stage
)
class InferLayerNorm(Transformation):
    pass
```

## Compilation Pipeline Integration

### Stage-Based Transforms
```python
def run_stage(model, stage_name):
    """Run all transforms for a specific stage."""
    registry = get_registry()
    transforms = registry.query(type="transform", stage=stage_name)
    
    for transform_info in transforms:
        transform_cls = transform_info["class"]
        model, _ = transform_cls().apply(model)
    
    return model

# Run pipeline stages
for stage in ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]:
    model = run_stage(model, stage)
```

### Kernel Inference
```python
def apply_kernel_inferences(model, kernel_name):
    """Apply all inference transforms for a specific kernel."""
    registry = get_registry()
    inferences = registry.query(type="kernel_inference", kernel=kernel_name)
    
    for inference_info in inferences:
        inference_cls = inference_info["class"]
        model, _ = inference_cls().apply(model)
    
    return model

# Apply kernel-specific inferences (not tied to stages)
for kernel_name in ["LayerNorm", "Softmax", "MatMul"]:
    model = apply_kernel_inferences(model, kernel_name)
```

## Key Design Rules

1. **Kernel Inference Has No Stage**: Transforms with `kernel` parameter must have `stage=None`
2. **Explicit Over Implicit**: All relationships declared explicitly
3. **No Compatibility Layers**: Clean break from old systems
4. **Fail Fast**: Validation at decoration time
5. **One Registry**: All queries through unified registry
6. **Optional Metadata**: Version, author, description are optional for flexibility

## Benefits

1. **Clarity**: Explicit decorators show component type immediately
2. **Type Safety**: Each decorator has appropriate typed parameters
3. **Flexibility**: Kernel inference naturally handled by transform decorator
4. **Discoverability**: Powerful query system for finding components
5. **Simplicity**: One registry, one query API, no fallbacks
6. **Extensibility**: Easy to add new component types

## Future Extensions

New component types follow the same pattern:

```python
def optimizer(name: str, target: str, **kwargs):
    """Decorator for optimization passes."""
    def decorator(cls):
        # Same pattern as other decorators
        metadata = {
            "type": "optimizer",
            "name": name,
            "target": target,
            **kwargs
        }
        cls._plugin_metadata = metadata
        registry = get_registry()
        registry.register("optimizer", name, cls, **metadata)
        return cls
    return decorator
```

## Conclusion

This unified plugin system provides a clean, extensible foundation for BrainSmith's compilation pipeline. The design achieves simplicity through explicit decorators while maintaining the power of a unified registry. The clear separation between stage-based transforms and kernel inference transforms reflects their different roles in the compilation process.