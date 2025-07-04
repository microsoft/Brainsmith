# Brainsmith Plugin System

A high-performance plugin registry system for the Brainsmith FPGA compiler, providing transform, kernel, backend, and step management with zero discovery overhead.

## Overview

The Brainsmith plugin system provides:
- **Decoration-time registration** - Plugins register automatically when decorated
- **Direct class access** - No wrapper overhead, return actual classes
- **Universal framework support** - All plugin types support framework qualification
- **Multiple access patterns** - Attribute access, dictionary lookup, framework scoping
- **Blueprint optimization** - Load only required plugins for production
- **Framework integration** - Seamless QONNX/FINN plugin integration

## Quick Start

### Creating Plugins

```python
from brainsmith.core.plugins import transform, kernel, backend, step

# Define a transform
@transform(name="MyTransform", stage="topology_opt")
class MyTransform:
    def apply(self, model):
        # Transform logic here
        return model, False

# Define a kernel
@kernel(name="MyKernel", op_type="MyOp")
class MyKernel:
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

# Define a backend
@backend(name="MyKernelHLS", kernel="MyKernel", language="hls", default=True)
class MyKernelHLS(MyKernel):
    def generate_hls(self):
        return "// HLS implementation"

# Define a step
@step(name="MyStep", category="build")
class MyStep:
    def __call__(self, build_context):
        # Step logic here
        return build_context
```

### Using Plugins - Multiple Access Patterns

```python
from brainsmith.plugins import transforms, kernels, backends, steps

# Direct attribute access (returns actual classes)
transform_cls = transforms.MyTransform
kernel_cls = kernels.MyKernel
backend_cls = backends.MyKernelHLS
step_cls = steps.MyStep

# Dictionary-style access
transform_cls = transforms["MyTransform"]
kernel_cls = kernels["MyKernel"]
backend_cls = backends["MyKernelHLS"]
step_cls = steps["MyStep"]

# Framework-qualified attribute access
qonnx_transform = transforms.qonnx.BatchNormToAffine
finn_kernel = kernels.finn.MatrixVectorUnit
finn_backend = backends.finn.LayerNormHLS
finn_step = steps.finn.CreateDataflowPartition

# Framework-qualified dictionary access
qonnx_transform = transforms["qonnx:BatchNormToAffine"]
finn_kernel = kernels["finn:MatrixVectorUnit"]
finn_backend = backends["finn:LayerNormHLS"]
finn_step = steps["finn:CreateDataflowPartition"]

# Step category access
build_step = steps.build.MyStep
test_step = steps.testing.MyTestStep

# Use plugins (direct instantiation)
model = model.transform(transforms.MyTransform())
kernel_instance = kernels.MyKernel()
backend_instance = backends.MyKernelHLS()
result = steps.MyStep()(build_context)
```

## Plugin Types

### Transforms
Modify ONNX models during compilation. Must specify either `stage` or `kernel`.

**Available stages:**
- `pre_proc` - Pre-processing operations
- `cleanup` - Graph cleanup operations  
- `topology_opt` - Topology optimizations
- `kernel_opt` - Kernel-specific optimizations
- `dataflow_opt` - Dataflow optimizations
- `post_proc` - Post-processing operations

### Kernels
Define hardware acceleration interfaces for specific operations.

### Backends
Provide implementation-specific code generation for kernels. Multiple backends can exist per kernel.

**Supported languages:**
- `hls` - High-Level Synthesis C++
- `rtl` or `verilog` - RTL/Verilog implementation
- `systemc` - SystemC implementation

### Steps
Build system operations that execute during hardware compilation.

**Common categories:**
- `build` - Build process steps
- `testing` - Test and validation steps
- `analysis` - Analysis and reporting steps

## Architecture

The plugin system consists of five core components:

1. **Registry** (`registry.py`) - Central storage with pre-computed indexes
2. **Decorators** (`decorators.py`) - Auto-registration at decoration time
3. **Collections** (`plugin_collections.py`) - Natural access patterns
4. **Blueprint Loader** (`blueprint_loader.py`) - Production optimization
5. **Framework Adapters** (`framework_adapters.py`) - External integration

### How It Works

1. **Registration**: When a class is decorated, it's immediately registered in the global registry
2. **Storage**: The registry maintains dictionaries for fast O(1) lookups and pre-computed indexes
3. **Access**: Collections provide natural access that delegates directly to the registry
4. **Direct Classes**: Collections return actual plugin classes, not wrapper objects

## Advanced Usage

### Backend Search and Selection

```python
from brainsmith.plugins import backends

# Find backends by criteria
hls_backends = backends.find(language="hls")
print(f"HLS backends: {[b.__name__ for b in hls_backends]}")

# Get backends for specific kernel
kernel_backends = backends.list_for_kernel("LayerNorm")
print(f"LayerNorm backends: {kernel_backends}")

# Get first matching backend for kernel
backend_cls = backends.get_for_kernel("LayerNorm", language="hls")
if backend_cls:
    backend_instance = backend_cls()
```

### Registry Queries

```python
from brainsmith.core.plugins import get_registry

registry = get_registry()

# Get statistics
stats = registry.get_stats()
print(f"Total plugins: {stats['total_plugins']}")
print(f"Frameworks: {stats['frameworks']}")

# List available plugins
transforms_list = registry.list_available_transforms()
steps_list = registry.list_available_steps()
kernels_list = registry.list_available_kernels()

# Get framework-specific plugins
qonnx_transforms = registry.get_framework_transforms("qonnx")
finn_kernels = registry.get_framework_kernels("finn")

# Find plugins by criteria
hls_backend_names = registry.find_backends(language="hls")
```

### Blueprint Optimization

For production deployments, load only required plugins:

```yaml
# blueprint.yaml
hw_compiler:
  transforms:
    cleanup:
      - RemoveIdentityOps
      - GiveReadableTensorNames
    topology_opt:
      - MyTransform
  kernels:
    - LayerNorm
  steps:
    - GenerateReferenceIO
    - CreateDataflowPartition
```

```python
from brainsmith.core.plugins.blueprint_loader import load_blueprint_plugins

# Load optimized subset
collections = load_blueprint_plugins('blueprint.yaml')
transforms = collections['transforms']
steps = collections['steps']

# Use normally - only blueprint plugins available
model = model.transform(transforms.RemoveIdentityOps())
result = steps.GenerateReferenceIO()(build_context)
```

### Multiple Backend Implementations

```python
# Register multiple backends for different optimization goals
@backend(name="LayerNormHLS_Fast", kernel="LayerNorm", language="hls", 
         optimization="throughput", resource_usage="high")
class LayerNormHLS_Fast(LayerNorm):
    pass

@backend(name="LayerNormHLS_Small", kernel="LayerNorm", language="hls",
         optimization="area", resource_usage="low")
class LayerNormHLS_Small(LayerNorm):
    pass

# Select backend by criteria
fast_backend = backends.get_for_kernel("LayerNorm", optimization="throughput")
small_backend = backends.get_for_kernel("LayerNorm", optimization="area")
```

## API Reference

### Decorators

- `@transform(name, stage, kernel, framework, **metadata)` - Register a transform
- `@kernel(name, op_type, framework, **metadata)` - Register a kernel  
- `@backend(name, kernel, language, default, framework, **metadata)` - Register a backend
- `@step(name, category, framework, **metadata)` - Register a step
- `@plugin(type, name, **metadata)` - Generic plugin registration

### Collections

All collections support:
- Attribute access: `collection.PluginName`
- Dictionary access: `collection["PluginName"]`
- Framework scoping: `collection.framework.PluginName` or `collection["framework:PluginName"]`

- `transforms` - Access transform plugins
- `kernels` - Access kernel plugins
- `backends` - Access backend plugins (with search methods)
- `steps` - Access step plugins (with category access)

### Collection Methods

**Backend Collection:**
- `backends.find(**criteria)` - Find backends matching criteria
- `backends.list_for_kernel(kernel_name)` - List backend names for kernel
- `backends.get_for_kernel(kernel_name, **criteria)` - Get first matching backend

**Transform Collection:**
- `transforms.list_by_stage(stage)` - List transforms for stage
- `transforms.get_by_stage(stage)` - Get transforms for stage as dict
- `transforms.list_stages()` - List available stages

**Step Collection:**
- `steps.list_by_category(category)` - List steps for category
- `steps.list_categories()` - List available categories
- `steps.category.StepName` - Access steps by category

### Registry Methods

- `get_registry()` - Get the global registry instance
- `registry.get_transform(name, framework=None)` - Get transform by name
- `registry.get_kernel(name, framework=None)` - Get kernel by name
- `registry.get_backend(name, framework=None)` - Get backend by name
- `registry.get_step(name, framework=None)` - Get step by name
- `registry.find_backends(**criteria)` - Find backends matching criteria

## Key Improvements

### Direct Class Access
Collections now return actual plugin classes, not wrapper objects:

```python
# Returns the actual class, not a wrapper
transform_cls = transforms.MyTransform
instance = transform_cls()  # Direct instantiation
```

### Universal Framework Support
All plugin types support framework qualification:

```python
# Before: Only transforms had framework support
transforms.qonnx.BatchNormToAffine

# Now: All plugin types support frameworks
kernels.finn.MatrixVectorUnit
backends.qonnx.LayerNormHLS
steps.finn.CreateDataflowPartition
```

### Steps as First-Class Plugins
Steps are no longer treated as transforms:

```python
# Before: Steps were transforms with special metadata
@transform(name="MyStep", stage="build", plugin_type="step")

# Now: Steps have their own registry and decorators
@step(name="MyStep", category="build")
```

## Migration Guide

### From Wrapper-Based Access
```python
# Before: Wrapper objects with methods
kernel = kernels.LayerNorm
backend = kernel.hls()  # Convenience method

# After: Direct class access with explicit backend selection
kernel_cls = kernels.LayerNorm
backend_cls = backends.get_for_kernel("LayerNorm", language="hls")
backend = backend_cls()
```

### From Step-as-Transform
```python
# Before: Steps registered as transforms
@transform(name="MyStep", stage="build", plugin_type="step")

# After: Steps have their own decorator
@step(name="MyStep", category="build")
```

## Development

### Testing
```bash
# Run plugin system tests
pytest brainsmith/core/plugins/tests/

# Debug plugin registration
python -m brainsmith.core.plugins.debug
```

### Common Issues

**Plugin not found**: Ensure the module containing the plugin is imported
**Backend not available**: Check that backend is registered with correct kernel name
**Step not found**: Ensure steps use `@step` decorator, not `@transform`

## License

Part of the Brainsmith project. See LICENSE for details.