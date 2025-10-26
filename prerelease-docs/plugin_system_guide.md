# Brainsmith Plugin System Guide

**Version**: 0.1.0
**Audience**: Plugin developers and advanced users

---

## Overview

Brainsmith's plugin system provides a unified way to extend the framework with custom **steps** (transformation pipelines), **kernels** (custom ONNX operators), and **backends** (code generators for HLS/RTL).

**Key Features**:
- **Lazy loading**: Components load on-demand for fast CLI startup
- **Source namespacing**: Components from different sources (brainsmith, finn, user, project) coexist without conflicts
- **Manifest caching**: Discovery results cached for performance (auto-invalidates on file changes)
- **Simple registration**: Components self-register via decorators

---

## Quick Start

### Using Components

```python
from brainsmith import get_kernel, get_step, list_kernels

# Get a kernel
LayerNorm = get_kernel('LayerNorm')
kernel_inst = LayerNorm(onnx_node)

# Get a step
streamline = get_step('streamline')
model = streamline(model, config)

# List available components
print(list_kernels())  # ['brainsmith:LayerNorm', 'brainsmith:Softmax', ...]
```

### Creating a Plugin

```python
# my_plugin/__init__.py
from brainsmith.component_helpers import create_lazy_module

COMPONENTS = {
    'steps': {'my_step': '.my_step'},
    'kernels': {'MyKernel': '.my_kernel'},
    'backends': {'MyKernel_hls': '.my_backend'},
}

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

```python
# my_plugin/my_step.py
from brainsmith import step

@step
def my_step(model, cfg):
    """Custom transformation step."""
    # Transform model
    return model
```

---

## Component Types

### 1. Steps

**Purpose**: Reusable transformation functions in the compilation pipeline.

**Decorator**: `@step`

**Function Signature**: `(model, config) -> model`

**Example**:
```python
from brainsmith import step

@step
def streamline(model, cfg):
    """Apply ONNX streamlining optimizations."""
    # Apply transformations
    return optimized_model

# With explicit name
@step(name='custom_streamline')
def my_streamline_impl(model, cfg):
    return model
```

**Usage**:
```python
from brainsmith import get_step

streamline_fn = get_step('streamline')
model = streamline_fn(model, config)
```

---

### 2. Kernels

**Purpose**: Custom ONNX operators with hardware implementations.

**Decorator**: `@kernel`

**Base Class**: `finn.custom_op.fpgadataflow.hwcustomop.HWCustomOp` (for hardware kernels)

**Example**:
```python
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from brainsmith import kernel

@kernel
class LayerNorm(HWCustomOp):
    """Hardware LayerNorm implementation."""

    def get_nodeattr_types(self):
        return {
            "Channels": ("i", True, 0),  # (type, required, default)
            "Epsilon": ("f", False, 1e-5),
        }

    def execute_node(self, context, graph):
        # Simulation logic for testing
        pass

    # Optional: InferTransform for graph conversion
    @property
    def infer_transform(self):
        from .infer_layernorm import InferLayerNorm
        return InferLayerNorm
```

**With metadata**:
```python
@kernel(
    name='CustomKernel',
    domain='custom.ops',
    infer_transform=InferCustomKernel
)
class CustomKernel(HWCustomOp):
    pass
```

**Usage**:
```python
from brainsmith import get_kernel, get_kernel_infer

# Get kernel class
LayerNorm = get_kernel('LayerNorm')
inst = LayerNorm(onnx_node)

# Get InferTransform for graph conversion
InferLayerNorm = get_kernel_infer('LayerNorm')
model = model.transform(InferLayerNorm())
```

---

### 3. Backends

**Purpose**: Generate synthesizable code (HLS C++ or RTL Verilog) for kernels.

**Decorator**: `@backend`

**Base Classes**: `HLSBackend` or `RTLBackend`

**Naming Convention**: `{KernelName}_{language}` (e.g., `LayerNorm_hls`)

**Example**:
```python
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith import backend
from .layernorm import LayerNorm

@backend
class LayerNorm_hls(LayerNorm, HLSBackend):
    """HLS backend for LayerNorm."""

    # Automatically inferred:
    # - name: 'LayerNorm_hls' (from class name)
    # - target_kernel: 'brainsmith:LayerNorm' (auto-detected)
    # - language: 'hls' (from HLSBackend)

    def global_includes(self):
        return ['#include "layernorm.hpp"']

    def defines(self, var):
        channels = self.get_nodeattr("Channels")
        return [f"#define CHANNELS {channels}"]

    def docompute(self):
        return '#include "layernorm.cpp"'
```

**With explicit metadata**:
```python
@backend(
    target_kernel='user:CustomKernel',
    language='hls',
    variant='optimized'
)
class CustomKernel_hls_fast(CustomKernel, HLSBackend):
    pass
```

**Usage**:
```python
from brainsmith import get_backend, list_backends_for_kernel

# Get specific backend
backend_cls = get_backend('LayerNorm_hls')

# Find all backends for a kernel
backends = list_backends_for_kernel('LayerNorm')
# ['brainsmith:LayerNorm_hls']

# Filter by language
hls_only = list_backends_for_kernel('MVAU', language='hls')
```

---

## Source Namespacing

All components use **`source:name`** format to avoid conflicts.

### Built-in Sources

| Source | Location | Description |
|--------|----------|-------------|
| `brainsmith` | Core package | Built-in components |
| `finn` | Via entry points | FINN framework components |
| `project` | `./plugins/` | Project-specific plugins |
| `user` | `~/.brainsmith/plugins/` | User plugins |

### Name Resolution

```python
# Explicit source (always unambiguous)
get_step('brainsmith:streamline')
get_kernel('finn:MVAU')

# Auto-resolution (uses source_priority)
get_step('streamline')  # Searches: project → user → brainsmith → finn
```

**Configure priority** in `.brainsmith/config.yaml`:
```yaml
source_priority:
  - project      # Check project plugins first
  - user         # Then user plugins
  - brainsmith   # Then core
  - finn         # Then FINN
```

---

## Creating Plugins

### Option 1: Project Plugins (Recommended)

**Structure**:
```
my_project/
├── .brainsmith/
│   └── config.yaml
└── plugins/              # Auto-discovered
    ├── __init__.py
    ├── my_step.py
    └── my_kernel/
        ├── __init__.py
        ├── my_kernel.py
        └── my_kernel_hls.py
```

**plugins/__init__.py**:
```python
from brainsmith.component_helpers import create_lazy_module

COMPONENTS = {
    'steps': {
        'my_step': '.my_step',
    },
    'kernels': {
        'MyKernel': '.my_kernel.my_kernel',
    },
    'backends': {
        'MyKernel_hls': '.my_kernel.my_kernel_hls',
    },
}

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

Components auto-register as **`project:name`**.

---

### Option 2: User Plugins

Same structure as project plugins, but in `~/.brainsmith/plugins/`.

Components auto-register as **`user:name`**.

---

### Option 3: Installable Packages (Advanced)

For distributable plugins, use **entry points**.

**pyproject.toml**:
```toml
[project.entry-points."brainsmith.plugins"]
my_plugin = "my_plugin:register_components"
```

**my_plugin/__init__.py**:
```python
def register_components():
    """Return component metadata for brainsmith."""
    return {
        'kernels': [
            {
                'name': 'MyKernel',
                'module': 'my_plugin.my_kernel',
                'class_name': 'MyKernel',
            }
        ],
        'backends': [
            {
                'name': 'MyKernel_hls',
                'module': 'my_plugin.my_kernel_hls',
                'class_name': 'MyKernel_hls',
                'target_kernel': 'my_plugin:MyKernel',
                'language': 'hls',
            }
        ],
        'steps': [
            {
                'name': 'my_step',
                'module': 'my_plugin.steps',
                'func_name': 'my_step',
            }
        ],
    }
```

Components auto-register as **`my_plugin:name`** (entry point name).

---

## Component Discovery

### Automatic Discovery

Discovery happens **automatically** on first component lookup:

```python
from brainsmith import get_kernel

# This triggers discovery if not already done
LayerNorm = get_kernel('LayerNorm')
```

Discovery scans:
1. Core brainsmith components
2. Configured plugin sources (project, user)
3. Entry point plugins (installed packages)

Results are cached in `.brainsmith/component_manifest.json`.

---

### Manual Discovery Control

```python
from brainsmith.loader import discover_plugins

# Force refresh (regenerate cache)
discover_plugins(force_refresh=True)

# Disable cache
discover_plugins(use_cache=False)
```

**When to force refresh**:
- After installing new plugin packages
- After modifying plugin structure
- For debugging

**Cache invalidation**: Cache auto-invalidates when component files are modified (checks mtimes).

---

## Component Lookup API

### Query Functions

```python
from brainsmith import (
    get_step, get_kernel, get_backend,
    has_step,
    list_steps, list_kernels, list_backends,
    get_kernel_infer,
    get_backend_metadata,
    list_backends_for_kernel,
)

# Get component
step = get_step('streamline')
kernel = get_kernel('LayerNorm')
backend = get_backend('LayerNorm_hls')

# Check existence (no exception)
if has_step('custom_step'):
    step = get_step('custom_step')

# List all
all_steps = list_steps()  # All sources
user_steps = list_steps(source='user')  # Filter by source

# Kernel helpers
InferTransform = get_kernel_infer('LayerNorm')  # Get InferTransform

# Backend helpers
meta = get_backend_metadata('LayerNorm_hls')
# Returns: {'class': <class>, 'target_kernel': '...', 'language': 'hls'}

backends = list_backends_for_kernel('MVAU', language='hls')
# Returns: ['finn:MVAU_hls', 'user:MVAU_hls_optimized', ...]
```

---

## Error Handling

```python
from brainsmith import get_kernel

try:
    kernel = get_kernel('NonExistent')
except KeyError as e:
    print(e)
    # Kernel 'project:NonExistent' not found.
    # Available: brainsmith:LayerNorm, brainsmith:Softmax, ...

# Prefer existence checks for optional components
from brainsmith import has_step

if has_step('optional_step'):
    step = get_step('optional_step')
else:
    # Use fallback
    step = get_step('default_step')
```

---

## Configuration

**File**: `.brainsmith/config.yaml` (project root)

```yaml
# Plugin source directories
plugin_sources:
  project: ./plugins          # Project plugins
  user: ~/.brainsmith/plugins # User plugins (default)
  # Custom sources:
  team: /shared/team-plugins

# Source resolution priority
source_priority:
  - project   # Check project first
  - team      # Then team plugins
  - user      # Then user plugins
  - brainsmith
  - finn

# Performance options
cache_plugins: true  # Enable manifest caching (default: true)
```

**Plugin sources** can be:
- Relative paths (resolve to project directory)
- Absolute paths
- `null` for built-in sources (auto-resolved)

---

## Advanced: Explicit Source Context

For plugin packages that need explicit source control:

```python
from brainsmith.registry import source_context

with source_context('my_source'):
    @kernel
    class MyKernel(HWCustomOp):
        """Registered as 'my_source:MyKernel'"""
        pass
```

**Note**: Normally unnecessary - source auto-detected from:
1. Entry point name (for installed packages)
2. Module path (`brainsmith.*` → `brainsmith`)
3. Configuration `source_priority[0]` (fallback)

---

## Best Practices

### 1. Use Lazy Loading Pattern

✅ **DO**:
```python
# plugins/__init__.py
from brainsmith.component_helpers import create_lazy_module

COMPONENTS = {
    'kernels': {'MyKernel': '.my_kernel'},
}

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

❌ **DON'T**:
```python
# plugins/__init__.py
from .my_kernel import MyKernel  # Imports everything at once!
```

### 2. Keep Names Unique

✅ **DO**: Use descriptive, unique names
```python
@step
def project_specific_cleanup(model, cfg):
    pass
```

❌ **DON'T**: Reuse common names without source prefix
```python
@step
def cleanup(model, cfg):  # Conflicts with 'brainsmith:cleanup'
    pass
```

### 3. Provide Metadata

✅ **DO**: Add docstrings and metadata
```python
@kernel(domain='custom.ops')
class MyKernel(HWCustomOp):
    """Hardware implementation of custom operation.

    Attributes:
        Channels: Number of input channels
        SIMD: Parallelism factor
    """
    pass
```

### 4. Test Discovery

```python
# Test that your plugin is discovered
from brainsmith import list_kernels

assert 'project:MyKernel' in list_kernels()
```

---

## Troubleshooting

### Plugin Not Found

```python
KeyError: "Kernel 'MyKernel' not found"
```

**Solutions**:
1. Check plugin source exists: `ls ./plugins/`
2. Verify `__init__.py` has `COMPONENTS` dict
3. Check source priority: `cat .brainsmith/config.yaml`
4. Force refresh: `discover_plugins(force_refresh=True)`
5. List available: `print(list_kernels())`

### Cache Issues

**Symptom**: Changes not reflected after editing plugin

**Solution**:
```python
from brainsmith.loader import discover_plugins
discover_plugins(force_refresh=True)
```

Or delete cache:
```bash
rm .brainsmith/component_manifest.json
```

### Import Errors

**Symptom**: `AttributeError: module has no attribute 'MyKernel'`

**Check**:
1. Component name matches `COMPONENTS` dict key
2. Module path is correct: `'.my_kernel'` → `plugins/my_kernel.py`
3. Class/function name matches dict key

---

## CLI Tools

```bash
# List all components
brainsmith plugins

# Verbose listing (shows metadata)
brainsmith plugins --verbose

# Validate all components (test imports)
brainsmith plugins --validate

# Rebuild cache
brainsmith plugins --refresh
```

---

## Migration from Old API

**Old API** (deprecated):
```python
from brainsmith.registry import get_transform, transform

@transform(name='MyTransform', stage='topology_opt')
class MyTransform(Transformation):
    pass
```

**New API**:
```python
from brainsmith import get_step, step

@step
def my_step(model, cfg):
    return model
```

**Key changes**:
- `@transform` → `@step`
- `get_transform()` → `get_step()`
- Steps are functions, not classes
- Source-prefixed names required for listing

---

## Reference

### Decorators
- `@step` - Register transformation function
- `@kernel` - Register custom operator
- `@backend` - Register code generator

### Lookup Functions
- `get_step(name)` - Get step function
- `get_kernel(name)` - Get kernel class
- `get_backend(name)` - Get backend class
- `get_kernel_infer(name)` - Get kernel's InferTransform
- `get_backend_metadata(name)` - Get backend metadata dict

### Listing Functions
- `list_steps(source=None)` - List all steps
- `list_kernels(source=None)` - List all kernels
- `list_backends(source=None)` - List all backends
- `list_backends_for_kernel(kernel, language=None, sources=None)` - Find backends for kernel

### Existence Checks
- `has_step(name)` - Check if step exists

### Discovery Control
- `discover_plugins(use_cache=True, force_refresh=False)` - Manual discovery

### Helper Functions
- `plugin_helpers.create_lazy_module(components, package_name)` - Create lazy loader

---

## See Also

- [Component Registry Architecture](./plugin_registry.md) - Old implementation details
- [Hardware Kernels Guide](./hardware_kernels.md) - Kernel development
- [CLI Architecture](./cli_architecture.md) - CLI integration

---

**Questions?** File an issue at https://github.com/microsoft/brainsmith
