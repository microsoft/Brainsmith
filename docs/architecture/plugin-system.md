# Plugin System

The plugin system is Brainsmith's core extensibility mechanism, enabling custom kernels, transforms, and build steps.

## Overview

The plugin registry is a singleton that manages all extensible components with decorator-based registration:

```python
from brainsmith.core.plugins import transform, kernel, backend, step

@transform(name="MyTransform")
class MyTransform:
    def apply(self, model):
        # Your transformation logic
        return model

@kernel(name="MyKernel")
class MyKernel:
    pass

@backend(kernel="MyKernel", name="rtl", default=True)
class MyKernelRTL:
    pass

@step(name="my_step")
def my_custom_step(model, cfg):
    # Custom build step
    return model
```

## Plugin Types

### Transforms

Graph transformations that modify the ONNX/QONNX model:

```python
@transform(name="RemoveIdentity", framework="brainsmith")
class RemoveIdentity(Transformation):
    def apply(self, model):
        # Remove identity nodes
        for node in model.graph.node:
            if node.op_type == "Identity":
                # Remove node logic
                pass
        return model
```

**Access:**
```python
from brainsmith.core.plugins import get_transform

transform_cls = get_transform("RemoveIdentity")
transform = transform_cls()
model = transform.apply(model)
```

### Kernels

Hardware operator implementations:

```python
@kernel(name="MatMul")
class MatMulKernel:
    """Matrix multiplication kernel."""

    def __init__(self, onnx_node):
        self.onnx_node = onnx_node

    def make_weight_file(self):
        # Generate weight files
        pass
```

**Access:**
```python
from brainsmith.core.plugins import get_kernel

kernel_cls = get_kernel("MatMul")
kernel = kernel_cls(onnx_node)
```

### Backends

RTL or HLS implementations per kernel:

```python
@backend(kernel="MatMul", name="rtl", default=True)
class MatMulRTL:
    """RTL backend for MatMul."""

    def generate_hdl(self):
        # Generate RTL code
        pass

@backend(kernel="MatMul", name="hls")
class MatMulHLS:
    """HLS backend for MatMul."""

    def generate_hls(self):
        # Generate HLS code
        pass
```

**Access:**
```python
from brainsmith.core.plugins import get_backend

backend_cls = get_backend("MatMul", "rtl")
backend = backend_cls()
```

### Steps

Build pipeline operations:

```python
@step(name="my_optimization")
def my_optimization_step(model, cfg):
    """Custom optimization step."""
    # Apply optimizations
    return model
```

**Access:**
```python
from brainsmith.core.plugins import get_step

step_fn = get_step("my_optimization")
model = step_fn(model, config)
```

## Framework Integration

Brainsmith integrates FINN and QONNX transforms automatically:

```python
# FINN transforms
get_transform("finn:Streamline")
get_transform("finn:InferShapes")

# QONNX transforms
get_transform("qonnx:ConvertSubToAdd")
get_transform("qonnx:InferDataTypes")
```

These are automatically discovered and wrapped with the `framework` prefix.

## Registration API

### Manual Registration

```python
from brainsmith.core.plugins.registry import _registry

_registry.register(
    plugin_type='transform',
    name='CustomTransform',
    cls=CustomTransform,
    framework='brainsmith',
    metadata={'version': '1.0'}
)
```

### Query Functions

```python
from brainsmith.core.plugins import (
    list_transforms,
    list_kernels,
    has_transform,
    has_kernel
)

# List all transforms
transforms = list_transforms()

# List all kernels
kernels = list_kernels()

# Check if transform exists
if has_transform("MyTransform"):
    ...

# Check if kernel exists
if has_kernel("MatMul"):
    ...
```

## Discovery and Loading

Plugins are lazy-loaded on first access:

1. **Auto-discovery**: Registry imports `brainsmith.{transforms,kernels,steps,operators}`
2. **Framework adapters**: FINN/QONNX plugins loaded on first framework prefix access
3. **Decorator registration**: `@transform`, `@kernel`, etc. register during module import

```python
# First access triggers discovery
transform_cls = get_transform("MyTransform")  # Auto-discovers all transforms

# Subsequent access uses cache
transform_cls2 = get_transform("MyTransform")  # From cache
```

## Blueprint Integration

Reference plugins declaratively in blueprints:

```yaml
design_space:
  kernels:
    - name: MatMul
      backends:
        - rtl  # References MatMulRTL backend
        - hls  # References MatMulHLS backend

  steps:
    - cleanup  # References cleanup step
    - qonnx_to_finn
    - my_optimization  # References custom step
    - finn:Streamline  # References FINN transform
```

## Best Practices

### Naming Conventions

- **Transforms**: PascalCase, descriptive (e.g., `RemoveUnusedNodes`)
- **Kernels**: PascalCase, operator name (e.g., `Conv2d`, `MatMul`)
- **Backends**: `{Kernel}{Backend}` (e.g., `MatMulRTL`, `Conv2dHLS`)
- **Steps**: snake_case, action-oriented (e.g., `step_create_partition`)

### Metadata

Add metadata for discoverability:

```python
@transform(
    name="MyTransform",
    framework="brainsmith",
    version="1.0",
    description="Optimizes graph structure"
)
class MyTransform:
    pass
```

### Testing

Test plugins in isolation:

```python
def test_my_transform():
    from brainsmith.core.plugins import get_transform

    transform_cls = get_transform("MyTransform")
    transform = transform_cls()

    # Test transformation
    result = transform.apply(model)
    assert result is not None
```

## API Reference

For detailed API documentation, see:

- [Plugin Registry API](../api-reference/plugins.md)
- [Core API](../api-reference/core.md)
