# Brainsmith Plugin System

A high-performance plugin registry system for the Brainsmith FPGA compiler, providing transform, kernel, and backend management with zero discovery overhead.

## Overview

The Brainsmith plugin system provides:
- **Decoration-time registration** - Plugins register automatically when decorated
- **Direct registry lookups** - O(1) access to any plugin without discovery
- **Natural access patterns** - Use plugins via intuitive dot notation
- **Blueprint optimization** - Load only required plugins for production
- **Framework integration** - Seamless QONNX/FINN transform integration

## Quick Start

### Creating a Plugin

```python
from brainsmith.core.plugins import transform, kernel, backend

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
```

### Using Plugins

```python
from brainsmith.plugins import transforms as tfm, kernels as kn

# Use transforms
model = model.transform(tfm.MyTransform())
model = model.transform(tfm.qonnx.RemoveIdentityOps())
model = model.transform(tfm.finn.Streamline())

# Use kernels and backends
kernel = kn.MyKernel
hls_impl = kernel()  # Gets default backend (HLS)
hls_impl = kernel.hls()  # Explicitly get HLS backend
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

## Architecture

The plugin system consists of five core components:

1. **Registry** (`registry.py`) - Central storage with pre-computed indexes
2. **Decorators** (`decorators.py`) - Auto-registration at decoration time
3. **Collections** (`collections.py`) - Natural access patterns
4. **Blueprint Loader** (`blueprint_loader.py`) - Production optimization
5. **Framework Adapters** (`framework_adapters.py`) - External integration

### How It Works

1. **Registration**: When a class is decorated with `@transform`, `@kernel`, or `@backend`, it's immediately registered in the global registry
2. **Storage**: The registry maintains dictionaries for fast O(1) lookups and pre-computed indexes for efficient queries
3. **Access**: Collections provide natural dot-notation access that delegates directly to the registry
4. **Optimization**: Blueprint loader creates subset registries containing only required plugins

## Advanced Usage

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
```

```python
from brainsmith.core.plugins.blueprint_loader import load_blueprint_plugins

# Load optimized subset
collections = load_blueprint_plugins('blueprint.yaml')
tfm = collections['transforms']

# Use normally - only blueprint plugins available
model = model.transform(tfm.RemoveIdentityOps())
```

### Registry Queries

```python
from brainsmith.core.plugins import get_registry

registry = get_registry()

# List backends for a kernel
backends = registry.list_backends_by_kernel("LayerNorm")
# Returns: ["LayerNormHLS", "LayerNormRTL"]

# Find backends by criteria
hls_backends = registry.find_backends(language="hls")
area_optimized = registry.find_backends(optimization="area")

# Get plugin metadata
metadata = registry.get_plugin_metadata("MyTransform")
print(f"Stage: {metadata.get('stage')}")
print(f"Framework: {metadata.get('framework')}")
```

### Multiple Backends

```python
# Register multiple backends for optimization choices
@backend(name="KernelHLS_Fast", kernel="MyKernel", language="hls", 
         optimization="throughput", resource_usage="high")
class KernelHLS_Fast(MyKernel):
    pass

@backend(name="KernelHLS_Small", kernel="MyKernel", language="hls",
         optimization="area", resource_usage="low")
class KernelHLS_Small(MyKernel):
    pass

# Find backend by criteria
kernel = kn.MyKernel
fast_impl = kernel.find_backend(optimization="throughput")
small_impl = kernel.find_backend(optimization="area")
```

## API Reference

### Decorators

- `@transform(name, stage, kernel, framework, **metadata)` - Register a transform
- `@kernel(name, op_type, framework, **metadata)` - Register a kernel  
- `@backend(name, kernel, language, default, **metadata)` - Register a backend
- `@plugin(type, name, **metadata)` - Generic plugin registration

### Collections

- `transforms` - Access transform plugins
- `kernels` - Access kernel plugins
- `backends` - Access backend plugins
- `steps` - Access step plugins (treated as transforms)

### Registry Methods

- `get_registry()` - Get the global registry instance
- `registry.get_transform(name)` - Get transform by name
- `registry.get_kernel(name)` - Get kernel by name
- `registry.get_backend(name)` - Get backend by name
- `registry.find_backends(**criteria)` - Find backends matching criteria
- `registry.list_backends_by_kernel(kernel)` - List backend names for kernel

## Backward Compatibility

The system maintains compatibility with existing code through:
- Bridge module at `brainsmith.plugins` 
- Support for both `backend_type` and `language` parameters
- Framework-specific accessors (`transforms.qonnx.*`, `transforms.finn.*`)

## Examples

See the `examples/` directory for complete examples:
- `basic_plugin.py` - Simple transform and kernel definition
- `multi_backend.py` - Multiple backend implementations
- `blueprint_usage.py` - Production blueprint optimization
- `framework_integration.py` - QONNX/FINN integration

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
**Transform validation warnings**: Ensure either `stage` or `kernel` is specified, not both

## License

Part of the Brainsmith project. See LICENSE for details.