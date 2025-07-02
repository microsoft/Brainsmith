# BrainSmith Perfect Code Plugin System - Developer Guide

## Table of Contents
1. [Quick Start](#quick-start-5-minutes)
2. [Plugin Types Reference](#plugin-types-reference)
3. [Registry Interaction](#registry-interaction)
4. [Plugin Discovery and Querying](#plugin-discovery-and-querying)
5. [Backend Selection and Management](#backend-selection-and-management)
6. [Advanced Patterns](#advanced-patterns)
7. [Performance Best Practices](#performance-best-practices)
8. [Migration Guide](#migration-from-old-system)

## Quick Start (5 Minutes)

### 1. Creating a Plugin

The Brainsmith plugin system provides convenience decorators for each plugin type, making plugin development clean and intuitive.

#### Using Convenience Decorators (Recommended)

```python
from brainsmith.core.plugins import transform

@transform(name="MyTransform", stage="topology_opt")
class MyTransform:
    """A simple transform that does something useful."""
    
    def apply(self, model):
        # Transform logic here
        model_changed = False
        return model, model_changed
```

#### Using Generic Decorator (Alternative)

```python
from brainsmith.core.plugins import plugin

@plugin(type="transform", name="MyTransform", stage="topology_opt")
class MyTransform:
    """A simple transform that does something useful."""
    
    def apply(self, model):
        # Transform logic here
        model_changed = False
        return model, model_changed
```

That's it! Your plugin is automatically registered and available at decoration time.

### 2. Using Plugins

```python
from brainsmith.plugins import transforms as tfm

# Direct access
model = model.transform(tfm.MyTransform())

# Framework-specific access
model = model.transform(tfm.qonnx.RemoveIdentityOps())
model = model.transform(tfm.finn.Streamline())

# With parameters
model = model.transform(tfm.MyTransform(param1="value1"))
```

### 3. Blueprint Optimization (Production)

```yaml
# blueprint.yaml
hw_compiler:
  transforms:
    cleanup:
      - RemoveIdentityOps
      - MyTransform
```

```python
from brainsmith.core.plugins.blueprint_loader import load_blueprint_plugins

# Load only required plugins - 86.7% memory savings
collections = load_blueprint_plugins('blueprint.yaml')
tfm = collections['transforms']

# Use normally
model = model.transform(tfm.MyTransform())
```

## Plugin Types Reference

### Transform Plugins

Transforms modify models during the compilation pipeline. Use the `@transform` decorator for clean, type-safe registration.

```python
# Convenience decorator (recommended)
from brainsmith.core.plugins import transform

# Stage-based transform
@transform(
    name="MyTransform",        # Optional, defaults to class name
    stage="topology_opt",      # Required for stage-based transforms
    framework="brainsmith",    # Optional, defaults to "brainsmith"
    description="Optimizes model topology",
    author="your-name",
    version="1.0.0"
)
class MyTransform:
    def apply(self, model):
        """Transform the model, return (model, changed_bool)."""
        return model, False

# Kernel-specific transform (for inference)
@transform(
    name="InferMyKernel", 
    kernel="MyKernel",         # Required for kernel-specific transforms
    description="Convert ONNX patterns to MyKernel"
)
class InferMyKernel:
    def apply(self, model):
        """Detect and mark nodes that can use MyKernel."""
        return model, False

# Framework-specific transform
@transform(
    name="QONNXOptimization", 
    stage="cleanup", 
    framework="qonnx"          # Available as tfm.qonnx.QONNXOptimization
)
class QONNXOptimization:
    def apply(self, model):
        return model, True
```

**Transform Parameters:**
- `name` (Optional[str]): Plugin name, defaults to class name
- `stage` (Optional[str]): Transform stage - **required for stage-based transforms**
- `kernel` (Optional[str]): Associated kernel name - **required for kernel inference**
- `framework` (str): Framework name, defaults to "brainsmith"
- **Note**: Must specify either `stage` OR `kernel`, not both

**Available Stages:**
- `cleanup` - Graph cleanup operations
- `topology_opt` - Topology optimizations
- `kernel_opt` - Kernel-specific optimizations
- `dataflow_opt` - Dataflow optimizations
- `streamlining` - Model streamlining
- `metadata` - Metadata operations

### Kernel Plugins

Kernels define hardware acceleration interfaces. Use the `@kernel` decorator to register kernel definitions.

```python
# Convenience decorator (recommended)
from brainsmith.core.plugins import kernel

@kernel(
    name="MyAccelerator",
    op_type="MyCustomOp",     # ONNX operation type
    description="Custom accelerator kernel",
    author="your-name",
    version="1.0.0"
)
class MyAccelerator:
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
    
    def get_nodeattr_types(self):
        """Return dict of node attributes and their types."""
        return {
            "param1": ("i", True, ""),    # Integer, required, no default
            "param2": ("f", False, 1.0)   # Float, optional, default 1.0
        }
    
    def make_shape_compatible_op(self, model):
        """Transform model to be shape-compatible."""
        return model

# Or generic decorator
@plugin(type="kernel", name="MyAccelerator", op_type="MyCustomOp")
class MyAccelerator:
    pass
```

**Kernel Parameters:**
- `name` (Optional[str]): Kernel name, defaults to class name
- `op_type` (Optional[str]): ONNX operator type this kernel handles
- `framework` (str): Framework name, defaults to "brainsmith"

### Backend Plugins

Backends provide implementation-specific code generation for kernels. The updated system supports multiple backends per kernel with rich metadata.

```python
# Convenience decorator (recommended)
from brainsmith.core.plugins import backend

# HLS backend (marked as default)
@backend(
    name="MyAcceleratorHLS",
    kernel="MyAccelerator",      # Required: which kernel this implements
    language="hls",              # New: use 'language' instead of 'backend_type'
    default=True,                # New: mark as default backend
    optimization="balanced",     # Rich metadata supported
    description="HLS backend for MyAccelerator",
    author="your-name"
)
class MyAcceleratorHLS(MyAccelerator):
    def generate_hls(self):
        """Generate HLS implementation."""
        return "// HLS code here"

# RTL backend with different optimization
@backend(
    name="MyAcceleratorRTL_Fast",
    kernel="MyAccelerator",
    language="verilog",          # More specific than just "rtl"
    optimization="throughput",   # Optimization strategy metadata
    resource_usage="high",       # Resource usage metadata
    description="High-throughput RTL backend"
)
class MyAcceleratorRTL_Fast(MyAccelerator):
    def generate_verilog(self):
        """Generate optimized Verilog."""
        return "// Fast RTL code here"

# Another RTL backend optimized for area
@backend(
    name="MyAcceleratorRTL_Small",
    kernel="MyAccelerator",
    language="verilog",
    optimization="area",
    resource_usage="low",
    description="Area-efficient RTL backend"
)
class MyAcceleratorRTL_Small(MyAccelerator):
    def generate_verilog(self):
        """Generate area-optimized Verilog."""
        return "// Small RTL code here"
```

**Backend Parameters:**
- `name` (Optional[str]): Backend name, defaults to class name (must be unique)
- `kernel` (str): **Required** - Associated kernel name
- `backend_type` (str): Legacy parameter, use `language` instead
- `language` (str): Implementation language ("hls", "verilog", "systemc", etc.)
- `default` (bool): Whether this is the default backend for the kernel
- `optimization` (str): Optimization strategy ("throughput", "area", "balanced", etc.)
- `framework` (str): Framework name, defaults to "brainsmith"

### Step Plugins

```python
# Convenience decorator (recommended)
from brainsmith.core.plugins import step

@step(
    name="MyStep",
    category="preprocessing"   # Required: categorizes the step
)
def my_step(model, config):
    """Execute the step on the model."""
    # Step logic here
    return model

# Or class-based with generic decorator
@plugin(type="step", name="MyStep", category="preprocessing")
class MyStep:
    def execute(self, model, config):
        return model
```

## Registry Interaction

The plugin registry provides the foundation for all plugin operations. Understanding how to interact with it directly enables advanced use cases and debugging.

### Getting the Registry

```python
from brainsmith.core.plugins import get_registry

# Get the global registry instance
registry = get_registry()

# Registry statistics
stats = registry.get_stats()
print(f"Total plugins: {stats['total_plugins']}")
print(f"Transforms: {stats['transforms']}")
print(f"Kernels: {stats['kernels']}")
print(f"Backends: {stats['backends']}")
```

### Basic Registry Operations

```python
# Check if plugins exist
if "MyTransform" in registry.transforms:
    transform_cls = registry.get_transform("MyTransform")

if "MyKernel" in registry.kernels:
    kernel_cls = registry.get_kernel("MyKernel")

if "MyBackendHLS" in registry.backends:
    backend_cls = registry.get_backend("MyBackendHLS")

# List all plugins of each type
all_transforms = list(registry.transforms.keys())
all_kernels = list(registry.kernels.keys())
all_backends = list(registry.backends.keys())
```

### Plugin Metadata Access

```python
# Get metadata for any plugin
metadata = registry.get_plugin_metadata("MyTransform")
print(f"Type: {metadata.get('type')}")
print(f"Stage: {metadata.get('stage')}")
print(f"Framework: {metadata.get('framework')}")
print(f"Author: {metadata.get('author')}")
print(f"Description: {metadata.get('description')}")

# Iterate through all plugins with metadata
all_plugins = registry.list_all_plugins()
for plugin in all_plugins:
    print(f"{plugin['name']}: {plugin['metadata'].get('description', 'No description')}")
```

### Framework and Stage Queries

```python
# Get transforms by framework
qonnx_transforms = registry.get_framework_transforms("qonnx")
finn_transforms = registry.get_framework_transforms("finn") 
brainsmith_transforms = registry.get_framework_transforms("brainsmith")

# Get transforms by stage
cleanup_transforms = registry.transforms_by_stage.get("cleanup", {})
topology_transforms = registry.transforms_by_stage.get("topology_opt", {})

# List stage names with transforms
stages_with_transforms = list(registry.transforms_by_stage.keys())
```

### Backend and Kernel Relationships

```python
# Get all backends for a kernel
backends = registry.list_backends_by_kernel("LayerNorm")
print(f"LayerNorm backends: {backends}")

# Get default backend for a kernel
default_backend = registry.get_default_backend("LayerNorm")
print(f"Default backend: {default_backend}")

# Check which kernels have backends
kernels_with_backends = list(registry.backends_by_kernel.keys())

# Find backends by criteria
hls_backends = registry.find_backends(language="hls")
area_optimized = registry.find_backends(optimization="area")
layernorm_hls = registry.find_backends(kernel="LayerNorm", language="hls")
```

### Direct Registration (Advanced)

```python
# Register plugins programmatically (less common)
registry.register_transform(
    name="ProgrammaticTransform",
    transform_class=MyTransformClass,
    stage="cleanup",
    framework="brainsmith",
    description="Registered programmatically"
)

registry.register_kernel(
    name="ProgrammaticKernel", 
    kernel_class=MyKernelClass,
    op_type="CustomOp"
)

registry.register_backend(
    name="ProgrammaticBackend",
    backend_class=MyBackendClass,
    kernel="ProgrammaticKernel",
    language="hls",
    default=True
)
```

## Plugin Discovery and Querying

### Finding Plugins by Criteria

```python
# Find transforms by various criteria
cleanup_transforms = [name for name, meta in registry.plugin_metadata.items() 
                     if meta.get('type') == 'transform' and meta.get('stage') == 'cleanup']

qonnx_transforms = [name for name, meta in registry.plugin_metadata.items()
                   if meta.get('framework') == 'qonnx']

# Find plugins by author
author_plugins = [name for name, meta in registry.plugin_metadata.items()
                 if meta.get('author') == 'brainsmith-team']

# Find kernel inference transforms
kernel_inferences = [name for name, meta in registry.plugin_metadata.items()
                    if meta.get('type') == 'transform' and meta.get('kernel')]
```

### Cross-Reference Queries

```python
# For each kernel, show its ecosystem
for kernel_name in registry.kernels.keys():
    print(f"\nKernel: {kernel_name}")
    
    # Backends
    backends = registry.list_backends_by_kernel(kernel_name)
    print(f"  Backends: {backends}")
    
    # Default backend
    default = registry.get_default_backend(kernel_name)
    print(f"  Default: {default.__name__ if default else 'None'}")
    
    # Inference transforms
    inferences = [name for name, meta in registry.plugin_metadata.items()
                 if meta.get('kernel') == kernel_name]
    print(f"  Inference transforms: {inferences}")
```

### Plugin Validation and Debugging

```python
# Check for missing components
kernels_without_backends = []
for kernel_name in registry.kernels.keys():
    if kernel_name not in registry.backends_by_kernel:
        kernels_without_backends.append(kernel_name)

if kernels_without_backends:
    print(f"Kernels without backends: {kernels_without_backends}")

# Check for orphaned backends
all_kernel_names = set(registry.kernels.keys())
backend_kernels = set()
for backend_name in registry.backends.keys():
    meta = registry.get_plugin_metadata(backend_name)
    backend_kernels.add(meta.get('kernel'))

orphaned_backends = backend_kernels - all_kernel_names
if orphaned_backends:
    print(f"Backends with missing kernels: {orphaned_backends}")
```

## Backend Selection and Management

### Multiple Backend Support

The registry supports unlimited backends per kernel with rich metadata for selection.

```python
from brainsmith.plugins import kernels

# Access kernel with multiple backends
layernorm = kernels.LayerNorm

# List all available backends
backends = layernorm.list_backends()
print(f"Available backends: {backends}")

# Get backends with metadata
backends_meta = layernorm.list_backends_with_metadata()
for b in backends_meta:
    meta = b['metadata']
    print(f"{b['name']}: {meta.get('language')} - {meta.get('description')}")
```

### Backend Selection Strategies

```python
# Get default backend (automatic selection)
backend = layernorm()  # Uses default backend

# Get specific backend by name
hls_backend = layernorm.get_backend("LayerNormHLS")
rtl_backend = layernorm.get_backend("LayerNormRTL")

# Find backend by criteria
area_backend = layernorm.find_backend(optimization="area")
throughput_backend = layernorm.find_backend(optimization="throughput")

# Language-based selection (convenience methods)
hls_impl = layernorm.hls()      # Gets HLS backend
rtl_impl = layernorm.rtl()      # Gets RTL/Verilog backend
```

### Backend Query Methods

```python
# Registry-level backend queries
all_hls = registry.find_backends(language="hls")
all_rtl = registry.find_backends(language="verilog")

# Find backends by optimization
fast_backends = registry.find_backends(optimization="throughput")
small_backends = registry.find_backends(optimization="area")

# Multi-criteria queries
layernorm_hls = registry.find_backends(kernel="LayerNorm", language="hls")
default_backends = registry.find_backends(default=True)

# Complex queries with metadata
high_performance = registry.find_backends(
    language="hls", 
    optimization="throughput",
    resource_usage="high"
)
```

### Backend Development Patterns

```python
# Pattern: Multiple optimization variants
@backend(name="KernelHLS_Fast", kernel="MyKernel", language="hls", 
         optimization="throughput", resource_usage="high")
class KernelHLS_Fast(MyKernel):
    pass

@backend(name="KernelHLS_Small", kernel="MyKernel", language="hls",
         optimization="area", resource_usage="low") 
class KernelHLS_Small(MyKernel):
    pass

@backend(name="KernelHLS_Balanced", kernel="MyKernel", language="hls",
         optimization="balanced", default=True)
class KernelHLS_Balanced(MyKernel):
    pass

# Usage: automatic selection based on requirements
kernel = kernels.MyKernel
fast_impl = kernel.find_backend(optimization="throughput")
small_impl = kernel.find_backend(optimization="area")
default_impl = kernel()  # Gets balanced version
```

### Decorator Validation and Auto-Registration

All decorators provide automatic validation and immediate registration at decoration time.

#### Validation Examples

```python
# This will warn: Transform must specify either 'stage' or 'kernel'
@transform(name="BadTransform")
class BadTransform:
    pass

# This will warn: Transform cannot specify both 'stage' and 'kernel'  
@transform(name="BadTransform2", stage="cleanup", kernel="MyKernel")
class BadTransform2:
    pass

# This will warn: Backend must specify 'kernel'
@backend(name="BadBackend", language="hls")
class BadBackend:
    pass

# This will warn: Invalid stage 'invalid_stage'
@transform(name="BadStage", stage="invalid_stage")
class BadStage:
    pass
```

#### Auto-Registration Process

```python
from brainsmith.core.plugins import transform, get_registry

# Plugin registers automatically at decoration time
@transform(name="AutoRegistered", stage="cleanup")
class AutoRegistered:
    pass

# Plugin is immediately available in registry
registry = get_registry()
assert "AutoRegistered" in registry.transforms

# And available through collections
from brainsmith.plugins import transforms as tfm
transform_instance = tfm.AutoRegistered()
```

#### Common Parameters for All Decorators

- `name` (Optional[str]): Plugin name, defaults to class name
- `framework` (str): Framework name, defaults to "brainsmith"
- `description` (str): Human-readable description
- `author` (str): Plugin author
- `version` (str): Version string
- `**kwargs`: Additional metadata

#### Migration from Generic Decorator

```python
# Before - Generic decorator
@plugin(type="transform", name="MyTransform", stage="topology_opt")
class MyTransform:
    pass

# After - Convenience decorator (recommended)
@transform(name="MyTransform", stage="topology_opt")
class MyTransform:
    pass

# Benefits of convenience decorators:
# 1. Type safety with validation
# 2. Cleaner, more readable code
# 3. Better IDE support
# 4. Self-documenting intent
```

## Advanced Patterns

### Creating Framework-Specific Transforms

```python
# For QONNX framework
@plugin(
    type="transform",
    name="QONNXSpecificTransform",
    stage="cleanup",
    framework="qonnx"         # Makes it available as tfm.qonnx.QONNXSpecificTransform
)
class QONNXSpecificTransform:
    def apply(self, model):
        # QONNX-specific logic
        return model, True
```

### Multi-Backend Kernels

```python
# Base kernel
@plugin(type="kernel", name="AdvancedKernel")
class AdvancedKernel:
    pass

# HLS backend
@plugin(type="backend", name="AdvancedKernelHLS", kernel="AdvancedKernel", backend_type="hls")
class AdvancedKernelHLS(AdvancedKernel):
    def generate_hls(self):
        return "// Optimized HLS"

# RTL backend
@plugin(type="backend", name="AdvancedKernelRTL", kernel="AdvancedKernel", backend_type="rtl")
class AdvancedKernelRTL(AdvancedKernel):
    def generate_rtl(self):
        return "// Hand-optimized RTL"

# Usage
from brainsmith.plugins import kernels as kn

kernel = kn.AdvancedKernel
hls_impl = kernel.hls()      # Gets HLS backend
rtl_impl = kernel.rtl()      # Gets RTL backend
```

### Kernel Inference Transforms

```python
@plugin(
    type="kernel_inference",
    name="InferMyKernel",
    kernel="MyKernel"         # Which kernel this infers
)
class InferMyKernel:
    def apply(self, model):
        """Detect and mark nodes that can use MyKernel."""
        # Analysis logic
        return model, nodes_marked > 0
```

## Performance Best Practices

### 1. Use Blueprint Loading in Production

```python
# Development - full flexibility
from brainsmith.plugins import transforms as tfm
transform = tfm.MyTransform()  # All plugins available

# Production - optimized loading
collections = load_blueprint_plugins('production.yaml')
tfm = collections['transforms']
transform = tfm.MyTransform()  # Only required plugins loaded (86.7% savings)
```

### 2. Organize Transforms by Stage

```python
# Good - proper stage assignment
@plugin(type="transform", stage="cleanup")
class RemoveDeadNodes:
    pass

@plugin(type="transform", stage="topology_opt")  
class FuseOperations:
    pass

# This enables efficient stage-based access
cleanup_transforms = tfm.list_by_stage("cleanup")
```

### 3. Framework Attribution

```python
# Register external transforms with proper framework
from brainsmith.core.plugins.framework_adapters import register_external_plugin

register_external_plugin(
    plugin_class=ExternalTransform,
    name="ExternalTransform",
    plugin_type="transform",
    framework="external",     # Clear framework attribution
    stage="cleanup"
)

# Now accessible as
transform = tfm.external.ExternalTransform()
```

## Debugging and Inspection

### Check Plugin Status

```python
from brainsmith.core.plugins import get_registry

registry = get_registry()
stats = registry.get_stats()

print(f"Total plugins: {stats['total_plugins']}")
print(f"Transforms: {stats['transforms']}")
print(f"Kernels: {stats['kernels']}")
print(f"Backends: {stats['backends']}")
print(f"Stages: {stats['stages']}")
print(f"Frameworks: {stats['frameworks']}")
print(f"Kernels with backends: {stats['indexed_backends']}")
```

### List Available Plugins

```python
from brainsmith.core.plugins import get_registry

registry = get_registry()

# List all plugins of each type
all_transforms = list(registry.transforms.keys())
all_kernels = list(registry.kernels.keys())
all_backends = list(registry.backends.keys())

# List by framework
qonnx_transforms = list(registry.get_framework_transforms("qonnx").keys())
finn_transforms = list(registry.get_framework_transforms("finn").keys())

# List by stage
cleanup_transforms = registry.list_transforms_by_stage("cleanup")
topology_transforms = registry.list_transforms_by_stage("topology_opt")

# List backends by kernel
for kernel_name in registry.kernels.keys():
    backends = registry.list_backends_by_kernel(kernel_name)
    print(f"{kernel_name}: {backends}")
```

### Plugin Metadata Inspection

```python
# Inspect all plugins with metadata
all_plugins = registry.list_all_plugins()
for plugin in all_plugins:
    name = plugin['name']
    metadata = plugin['metadata']
    plugin_type = metadata.get('type')
    
    print(f"{name} ({plugin_type}):")
    for key, value in metadata.items():
        if key != 'type':
            print(f"  {key}: {value}")

# Find specific metadata patterns
for name, metadata in registry.plugin_metadata.items():
    if metadata.get('author') == 'brainsmith-team':
        print(f"Brainsmith plugin: {name}")
```

### Registry Architecture Inspection

```python
# Check registry indexes
print("Transform stages:")
for stage, transforms in registry.transforms_by_stage.items():
    print(f"  {stage}: {len(transforms)} transforms")

print("\nFramework transforms:")
for framework, transforms in registry.framework_transforms.items():
    print(f"  {framework}: {len(transforms)} transforms")

print("\nBackend indexes:")
for attr, index in registry.backend_indexes.items():
    print(f"  {attr}: {list(index.keys())}")

print("\nDefault backends:")
for kernel, backend in registry.default_backends.items():
    print(f"  {kernel}: {backend}")
```

### Advanced Query Examples

```python
# Find plugins matching complex criteria
def find_plugins_by_criteria(**criteria):
    """Find plugins matching multiple criteria."""
    results = []
    for name, metadata in registry.plugin_metadata.items():
        if all(metadata.get(key) == value for key, value in criteria.items()):
            results.append(name)
    return results

# Usage examples
hls_plugins = find_plugins_by_criteria(language="hls")
brainsmith_transforms = find_plugins_by_criteria(type="transform", framework="brainsmith")
cleanup_plugins = find_plugins_by_criteria(stage="cleanup")

# Kernel ecosystem analysis
def analyze_kernel_ecosystem(kernel_name):
    """Analyze complete ecosystem for a kernel."""
    print(f"Kernel: {kernel_name}")
    
    # Basic info
    if kernel_name in registry.kernels:
        kernel_cls = registry.kernels[kernel_name]
        print(f"  Class: {kernel_cls}")
        
        # Metadata
        metadata = registry.get_plugin_metadata(kernel_name)
        for key, value in metadata.items():
            if key != 'type':
                print(f"    {key}: {value}")
    
    # Backends
    backends = registry.list_backends_by_kernel(kernel_name)
    print(f"  Backends ({len(backends)}):")
    for backend in backends:
        meta = registry.get_plugin_metadata(backend)
        lang = meta.get('language', meta.get('backend_type'))
        default = " (default)" if meta.get('default') else ""
        print(f"    {backend}: {lang}{default}")
    
    # Inference transforms
    inferences = [name for name, meta in registry.plugin_metadata.items()
                 if meta.get('kernel') == kernel_name]
    print(f"  Inference transforms: {inferences}")

# Use it
for kernel in registry.kernels.keys():
    analyze_kernel_ecosystem(kernel)
    print()
```

## Common Patterns

### 1. Plugin Development Workflow

```python
# Step 1: Create plugin file (my_transforms.py)
from brainsmith.core.plugins import plugin

@plugin(type="transform", name="OptimizeBatchNorm", stage="topology_opt")
class OptimizeBatchNorm:
    def apply(self, model):
        # Implementation
        return model, True

# Step 2: Import to register (in __init__.py or main script)
import my_transforms  # Auto-registers on import

# Step 3: Use immediately
from brainsmith.plugins import transforms as tfm
model = model.transform(tfm.OptimizeBatchNorm())
```

### 2. Conditional Plugin Registration

```python
# Register plugin only if dependency available
try:
    import special_library
    
    @plugin(type="transform", name="SpecialTransform", stage="kernel_opt")
    class SpecialTransform:
        def apply(self, model):
            return special_library.optimize(model), True
            
except ImportError:
    # Plugin won't be registered if dependency missing
    pass
```

### 3. Plugin Composition

```python
@plugin(type="transform", name="CompositeTransform", stage="cleanup")
class CompositeTransform:
    def __init__(self):
        # Compose other transforms
        self.sub_transforms = [
            tfm.RemoveIdentityOps(),
            tfm.RemoveUnusedNodes(),
            tfm.SortGraph()
        ]
    
    def apply(self, model):
        changed = False
        for transform in self.sub_transforms:
            model, sub_changed = transform.apply(model)
            changed |= sub_changed
        return model, changed
```

## Migration from Old System

### Old Way
```python
# Complex setup required
from brainsmith.plugin import get_plugin_manager
manager = get_plugin_manager()
manager.discover_plugins(modes=['full'])

# Plugin might not be available immediately
# May need cache warming, retries, etc.
```

### New Way
```python
# Just use it
from brainsmith.plugins import transforms as tfm
# Ready immediately - no setup required
```

## FAQ

**Q: How do I know my plugin was registered?**
A: Check `plugin_status()` or try to access it. Registration happens immediately at decoration time.

**Q: Can I register plugins dynamically at runtime?**
A: Yes, use `get_registry().register_transform()` directly, but prefer decorators for clarity.

**Q: What happens if two plugins have the same name?**
A: The registry uses the last registered plugin. Use unique names or framework namespacing.

**Q: How do I unregister a plugin?**
A: Use `reset_plugin_system()` to clear all plugins, or modify the registry directly for specific plugins.

**Q: Can I use plugins without the collections API?**
A: Yes, access the registry directly with `get_registry()`, but collections provide a nicer interface.

## Performance Tips

### Registry Performance

1. **Import plugins early** - Registration happens at import time, indexes are built immediately
2. **Use direct registry queries** - `registry.find_backends(language="hls")` uses O(1) indexes
3. **Avoid metadata scanning** - Use indexed attributes (language, optimization, etc.) for fast queries
4. **Cache registry instance** - `registry = get_registry()` once, reuse the reference

### Plugin Organization

5. **Use blueprint loading** - 86.7% memory savings in production environments
6. **Framework attribution** - Use framework parameter for clear namespacing and efficient queries
7. **Stage assignment** - Proper stages enable efficient filtering and pipeline optimization
8. **Default backends** - Mark default backends to avoid selection overhead

### Backend Selection Optimization

```python
# Fast - uses pre-computed indexes
hls_backends = registry.find_backends(language="hls")
layernorm_backends = registry.find_backends(kernel="LayerNorm")

# Slow - requires metadata scanning  
custom_backends = registry.find_backends(custom_attribute="value")

# Optimal - get default backend directly
default = registry.get_default_backend("LayerNorm")

# Good - use kernel collections for natural access
backend = kernels.LayerNorm.hls()
```

### Development Best Practices

9. **Avoid dynamic registration** - Decoration-time registration is fastest and most reliable
10. **Use convenience decorators** - They provide validation and cleaner code
11. **Unique backend names** - Enables multiple variants per kernel without conflicts
12. **Rich metadata** - Add optimization and resource metadata for smart selection

## Summary

The Perfect Code Plugin System makes plugin development effortless:

### Core Features
- **One decorator** to register your plugin with automatic validation
- **Zero configuration** required - works immediately
- **Immediate availability** after decoration through auto-registration
- **Natural access patterns** preserved and enhanced

### Registry Capabilities  
- **Direct registry access** for advanced queries and debugging
- **Multi-criteria searches** with O(1) performance via indexes
- **Rich metadata support** for backend selection and organization
- **Unlimited backends** per kernel with flexible naming

### Backend System
- **Multiple implementations** per kernel (HLS, RTL, variants)
- **Smart selection** by optimization, language, or custom criteria
- **Default backend** support for automatic selection
- **Rich metadata** for resource usage, optimization strategy, etc.

### Performance Optimizations
- **86.7% memory reduction** with blueprint loading in production
- **O(1) queries** through pre-computed indexes
- **Zero discovery overhead** through decoration-time registration
- **Efficient cross-references** between kernels, backends, and transforms

Focus on writing your plugin logic - the system handles registration, discovery, and optimization automatically!