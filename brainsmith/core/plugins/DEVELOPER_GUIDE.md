# Brainsmith Plugin System - Developer Guide

This guide provides comprehensive documentation for developing with the Brainsmith plugin system, including creating plugins, using the registry, and optimizing for production.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Creating Plugins](#creating-plugins)
3. [Using Plugins](#using-plugins)
4. [Registry API](#registry-api)
5. [Backend System](#backend-system)
6. [Blueprint Optimization](#blueprint-optimization)
7. [Debugging](#debugging)
8. [Best Practices](#best-practices)

## Quick Start

### Basic Plugin Creation

```python
from brainsmith.core.plugins import transform, kernel, backend

# Create a transform
@transform(name="OptimizeModel", stage="topology_opt")
class OptimizeModel:
    def apply(self, model):
        # Your optimization logic
        return model, True  # (modified_model, was_changed)

# Create a kernel
@kernel(name="CustomOp", op_type="CustomOperation")
class CustomOp:
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

# Create a backend
@backend(name="CustomOpHLS", kernel="CustomOp", language="hls", default=True)
class CustomOpHLS(CustomOp):
    def generate_hls(self):
        return "// HLS implementation"
```

### Using Plugins

```python
from brainsmith.plugins import transforms as tfm, kernels as kn

# Use transforms
model = model.transform(tfm.OptimizeModel())
model = model.transform(tfm.qonnx.InferDataTypes())

# Use kernels with backends
op = kn.CustomOp()  # Gets default backend (HLS)
```

## Creating Plugins

### Transform Plugins

Transforms modify ONNX models during compilation. They must specify either a `stage` or `kernel`.

#### Stage-Based Transforms

```python
from brainsmith.core.plugins import transform

@transform(
    name="RemoveRedundantOps",  # Optional, defaults to class name
    stage="cleanup",            # Required for stage-based
    framework="brainsmith",     # Optional, defaults to "brainsmith"
    description="Remove redundant operations",
    author="dev-team",
    version="1.0.0"
)
class RemoveRedundantOps:
    def apply(self, model):
        """
        Apply the transform to the model.
        
        Returns:
            tuple: (modified_model, was_changed)
        """
        # Implementation
        changed = False
        # ... modify model ...
        return model, changed
```

#### Kernel-Specific Transforms (Inference)

```python
@transform(
    name="InferCustomOp",
    kernel="CustomOp",  # Required for kernel-specific
    description="Convert patterns to CustomOp kernel"
)
class InferCustomOp:
    def apply(self, model):
        # Detect patterns and convert to CustomOp
        nodes_converted = 0
        # ... pattern matching logic ...
        return model, nodes_converted > 0
```

#### Available Stages

- `pre_proc` - Pre-processing operations
- `cleanup` - Graph cleanup and normalization
- `topology_opt` - Topology-level optimizations
- `kernel_opt` - Kernel-specific optimizations  
- `dataflow_opt` - Dataflow and scheduling optimizations
- `post_proc` - Post-processing and finalization

### Kernel Plugins

Kernels define hardware acceleration interfaces.

```python
from brainsmith.core.plugins import kernel

@kernel(
    name="MatrixMultiply",
    op_type="MatMul",  # ONNX operator type
    description="Hardware accelerated matrix multiplication"
)
class MatrixMultiply:
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        # Initialize from ONNX node
    
    def get_nodeattr_types(self):
        """Define node attributes and their types."""
        return {
            "transA": ("i", False, 0),    # (type, required, default)
            "transB": ("i", False, 0),
            "alpha": ("f", False, 1.0)
        }
    
    def make_shape_compatible_op(self, model):
        """Make operation shape-compatible if needed."""
        return model
    
    def get_input_shapes(self):
        """Return expected input shapes."""
        return self.input_shapes
    
    def get_output_shapes(self):
        """Return output shapes."""
        return self.output_shapes
```

### Backend Plugins

Backends provide concrete implementations of kernels. Multiple backends can exist per kernel.

```python
from brainsmith.core.plugins import backend

# HLS Backend (C++ High-Level Synthesis)
@backend(
    name="MatrixMultiplyHLS",
    kernel="MatrixMultiply",
    language="hls",  # Implementation language
    default=True,    # Mark as default for this kernel
    optimization="balanced",
    resource_usage="medium",
    description="Balanced HLS implementation"
)
class MatrixMultiplyHLS(MatrixMultiply):
    def generate_hls(self):
        """Generate HLS C++ code."""
        return """
        void matmul(stream<T> &a, stream<T> &b, stream<T> &out) {
            // HLS implementation
        }
        """
    
    def get_compile_flags(self):
        return ["-O3", "-std=c++11"]

# RTL Backend (Verilog)
@backend(
    name="MatrixMultiplyRTL",
    kernel="MatrixMultiply", 
    language="verilog",
    optimization="throughput",
    resource_usage="high",
    description="High-throughput RTL implementation"
)
class MatrixMultiplyRTL(MatrixMultiply):
    def generate_verilog(self):
        """Generate Verilog RTL."""
        return """
        module matmul(clk, rst, a, b, out);
            // RTL implementation
        endmodule
        """
```

## Using Plugins

### Natural Access Patterns

```python
from brainsmith.plugins import transforms as tfm, kernels as kn

# Direct access
transform = tfm.RemoveRedundantOps()

# Framework-specific access
qonnx_transform = tfm.qonnx.InferDataTypes()
finn_transform = tfm.finn.Streamline()

# Apply transforms
model = model.transform(transform)
```

### Working with Kernels and Backends

```python
# Get kernel with default backend
kernel = kn.MatrixMultiply()  # Uses MatrixMultiplyHLS (marked as default)

# Get specific backend by language
hls_impl = kn.MatrixMultiply.hls()
rtl_impl = kn.MatrixMultiply.rtl()

# Get backend by name
specific = kn.MatrixMultiply.get_backend("MatrixMultiplyRTL")

# Find backend by criteria
fast_impl = kn.MatrixMultiply.find_backend(optimization="throughput")
small_impl = kn.MatrixMultiply.find_backend(optimization="area")

# List available backends
backends = kn.MatrixMultiply.list_backends()
# Returns: ["MatrixMultiplyHLS", "MatrixMultiplyRTL"]
```

## Registry API

The registry provides low-level access to all plugins and their metadata.

### Accessing the Registry

```python
from brainsmith.core.plugins import get_registry

registry = get_registry()
```

### Registry Statistics

```python
stats = registry.get_stats()
print(f"Total plugins: {stats['total_plugins']}")
print(f"Transforms: {stats['transforms']}")
print(f"Kernels: {stats['kernels']}")
print(f"Backends: {stats['backends']}")
print(f"Stages: {stats['stages']}")
print(f"Frameworks: {stats['frameworks']}")
```

### Querying Plugins

```python
# Check if plugins exist
if "RemoveRedundantOps" in registry.transforms:
    cls = registry.get_transform("RemoveRedundantOps")

# List plugins by type
all_transforms = list(registry.transforms.keys())
all_kernels = list(registry.kernels.keys())
all_backends = list(registry.backends.keys())

# List by framework
qonnx_transforms = list(registry.get_framework_transforms("qonnx").keys())
finn_transforms = list(registry.get_framework_transforms("finn").keys())

# List by stage
cleanup_transforms = registry.list_transforms_by_stage("cleanup")
```

### Plugin Metadata

```python
# Get metadata for any plugin
metadata = registry.get_plugin_metadata("MatrixMultiplyHLS")
print(f"Type: {metadata['type']}")
print(f"Kernel: {metadata['kernel']}")
print(f"Language: {metadata['language']}")
print(f"Optimization: {metadata.get('optimization')}")

# List all plugins with metadata
for plugin in registry.list_all_plugins():
    print(f"{plugin['name']}: {plugin['metadata']}")
```

### Backend Queries

```python
# List backends for a kernel
backends = registry.list_backends_by_kernel("MatrixMultiply")
# Returns: ["MatrixMultiplyHLS", "MatrixMultiplyRTL"]

# Find backends by criteria
hls_backends = registry.find_backends(language="hls")
throughput_backends = registry.find_backends(optimization="throughput")
mm_hls = registry.find_backends(kernel="MatrixMultiply", language="hls")

# Get default backend
default = registry.get_default_backend("MatrixMultiply")
```

## Backend System

The backend system supports multiple implementations per kernel with rich metadata for selection.

### Backend Registration

```python
# Multiple optimization variants
@backend(name="OptimizerHLS_Fast", kernel="Optimizer", 
         language="hls", optimization="throughput", 
         resource_usage="high", pipeline_depth=16)
class OptimizerHLS_Fast(Optimizer):
    pass

@backend(name="OptimizerHLS_Small", kernel="Optimizer",
         language="hls", optimization="area",
         resource_usage="low", pipeline_depth=4)
class OptimizerHLS_Small(Optimizer):
    pass

@backend(name="OptimizerHLS_Balanced", kernel="Optimizer",
         language="hls", optimization="balanced", 
         default=True, pipeline_depth=8)
class OptimizerHLS_Balanced(Optimizer):
    pass
```

### Backend Selection

```python
kernel = kn.Optimizer

# Automatic selection
default_impl = kernel()  # Gets OptimizerHLS_Balanced

# By optimization strategy
fast = kernel.find_backend(optimization="throughput")
small = kernel.find_backend(optimization="area")

# By multiple criteria
specific = kernel.find_backend(
    language="hls",
    optimization="throughput", 
    pipeline_depth=16
)

# List all options
backends_info = kernel.list_backends_with_metadata()
for info in backends_info:
    print(f"{info['name']}: {info['metadata']}")
```

### Backend Discovery Methods

```python
# Registry-level queries
all_hls = registry.find_backends(language="hls")
all_fast = registry.find_backends(optimization="throughput")
low_resource = registry.find_backends(resource_usage="low")

# Complex queries
candidates = registry.find_backends(
    kernel="Optimizer",
    language="hls",
    optimization="balanced"
)
```

## Blueprint Optimization

For production deployments, use blueprint loading to include only required plugins.

### Blueprint Format

```yaml
# production_blueprint.yaml
hw_compiler:
  transforms:
    cleanup:
      - RemoveIdentityOps
      - RemoveUnusedNodes
    topology_opt:
      - OptimizeModel
      - FuseOperations
    kernel_opt:
      - ~ConvertToHWLayers  # Optional (~ prefix)
  
  kernels:
    - MatrixMultiply
    - LayerNorm
    
  backends:
    - MatrixMultiply:hls  # Specific backend
    - LayerNorm:*         # All backends
```

### Using Blueprint Loading

```python
from brainsmith.core.plugins.blueprint_loader import load_blueprint_plugins

# Load subset registry with only required plugins
collections = load_blueprint_plugins('production_blueprint.yaml')

# Use optimized collections
tfm = collections['transforms']
kn = collections['kernels']

# Only blueprint-specified plugins are available
model = model.transform(tfm.RemoveIdentityOps())
model = model.transform(tfm.OptimizeModel())

# Attempting to use non-blueprint plugins will fail
# tfm.SomeOtherTransform()  # AttributeError
```

### Blueprint Performance Analysis

```python
from brainsmith.core.plugins.blueprint_loader import analyze_blueprint_requirements

# Analyze blueprint impact
stats = analyze_blueprint_requirements('production_blueprint.yaml')
print(f"Plugins loaded: {stats['total_loaded_plugins']}")
print(f"Memory reduction: {stats['performance_improvement']}")
```

## Debugging

### Plugin Status

```python
from brainsmith.core.plugins import plugin_status

status = plugin_status()
print(f"Registry stats: {status}")
```

### Command-Line Debugging

```bash
# Show plugin summary
python -m brainsmith.core.plugins.debug

# List all transforms
python -m brainsmith.core.plugins.debug list

# Search for plugins
python -m brainsmith.core.plugins.debug search "Optimize"

# Show plugin details
python -m brainsmith.core.plugins.debug show OptimizeModel

# List by framework
python -m brainsmith.core.plugins.debug framework qonnx

# List by stage
python -m brainsmith.core.plugins.debug stage cleanup
```

### Common Issues

#### Plugin Not Found

```python
# Check if module is imported
import my_plugin_module  # Must import to trigger registration

# Verify registration
from brainsmith.core.plugins import get_registry
registry = get_registry()
assert "MyPlugin" in registry.transforms
```

#### Backend Not Available

```python
# Check backend is registered for kernel
backends = registry.list_backends_by_kernel("MyKernel")
print(f"Available backends: {backends}")

# Verify backend metadata
metadata = registry.get_plugin_metadata("MyBackendHLS")
assert metadata['kernel'] == "MyKernel"
```

#### Validation Warnings

```python
# Transform must specify stage XOR kernel
@transform(name="Bad")  # Warning: no stage or kernel
class Bad:
    pass

@transform(name="Bad2", stage="cleanup", kernel="Foo")  # Warning: both specified
class Bad2:
    pass
```

## Best Practices

### Plugin Development

1. **Use convenience decorators** - Cleaner than generic `@plugin`
2. **Unique names** - Especially for backends
3. **Rich metadata** - Helps with discovery and debugging
4. **Validate early** - Check requirements in `__init__`
5. **Document behavior** - Clear docstrings

### Performance

1. **Import plugins early** - Registration happens at import
2. **Use blueprint loading** - Reduce memory in production
3. **Cache registry reference** - `registry = get_registry()`
4. **Use indexed queries** - `find_backends()` over manual filtering

### Organization

1. **Group by type** - Separate files for transforms/kernels/backends
2. **Framework prefixes** - Clear naming for framework-specific plugins  
3. **Consistent metadata** - Use standard attributes
4. **Version plugins** - Track changes over time

### Testing

```python
import pytest
from brainsmith.core.plugins import transform, get_registry

def test_plugin_registration():
    # Define test plugin
    @transform(name="TestTransform", stage="cleanup")
    class TestTransform:
        def apply(self, model):
            return model, False
    
    # Verify registration
    registry = get_registry()
    assert "TestTransform" in registry.transforms
    
    # Test usage
    from brainsmith.plugins import transforms as tfm
    instance = tfm.TestTransform()
    assert instance is not None
```

## Migration Notes

### From Old Plugin System

```python
# Old system (no longer supported)
from brainsmith.plugin.decorators import op_transform
from brainsmith.plugin import get_plugin_manager

# New system
from brainsmith.core.plugins import transform
from brainsmith.plugins import transforms as tfm
```

### Import Paths

- Old: `brainsmith.plugin.*`
- New: `brainsmith.core.plugins.*`
- Bridge: `brainsmith.plugins` (backward compatible)

### Key Changes

1. No discovery needed - plugins register at decoration
2. No manager layer - direct registry access
3. No caching - all lookups are O(1)
4. Cleaner decorators - type-specific options

## Summary

The Brainsmith plugin system provides a high-performance, easy-to-use framework for extending the compiler with custom transforms, kernels, and backends. Key features:

- **Zero-overhead registration** through decorators
- **Natural access patterns** with collections
- **Rich backend system** with metadata-driven selection
- **Blueprint optimization** for production deployments
- **Comprehensive debugging** tools

Focus on your plugin logic - the system handles registration, discovery, and optimization automatically.