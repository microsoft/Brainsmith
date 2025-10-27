# Brainsmith Component Registry: User Guide

**A practical guide to registering, discovering, and using components in Brainsmith**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Component Types](#component-types)
4. [Registering Components](#registering-components)
5. [Discovering Components](#discovering-components)
6. [Accessing Components](#accessing-components)
7. [Advanced Patterns](#advanced-patterns)
8. [DSE Flow: Registry in Action](#dse-flow-registry-in-action)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Introduction

### What is the Component Registry?

The **Brainsmith Component Registry** is a centralized system for discovering, loading, and managing FPGA accelerator components. Think of it as a catalog that knows about all available:

- **Steps**: Transformation functions that process neural network models
- **Kernels**: Hardware custom operations (HWCustomOp classes)
- **Backends**: Code generators for specific hardware implementations (HLS/RTL)

### Why Use the Registry?

**Without the registry:**
```python
# ❌ Hard-coded imports
from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.kernels.crop.crop import Crop
from finn.custom.custom_op.channels_last_conv import ChannelsLastConv

# Which package is LayerNorm from? Is Crop from FINN or Brainsmith?
# What if users add custom kernels?
```

**With the registry:**
```python
# ✅ Clean, source-aware lookup
from brainsmith.registry import get_kernel, list_kernels

LayerNorm = get_kernel('LayerNorm')  # Auto-resolves from correct source
all_kernels = list_kernels()          # Discover all available kernels

# Works seamlessly with user plugins!
```

### Key Benefits

✅ **Automatic Discovery**: Components self-register during import
✅ **Source Awareness**: Prioritize project > user > brainsmith > finn
✅ **Lazy Loading**: Import components only when needed
✅ **Extensible**: Add custom components without modifying core code
✅ **Validation**: Catch missing components early with helpful errors

---

## Quick Start

### For Component Users (Most Common)

```python
from brainsmith.registry import get_step, get_kernel, list_backends_for_kernel

# Get a step function
streamline = get_step('streamline')

# Get a kernel class
LayerNorm = get_kernel('LayerNorm')
kernel = LayerNorm(onnx_node)

# Find backends for a kernel
backends = list_backends_for_kernel('LayerNorm')
print(backends)  # ['brainsmith:LayerNorm_hls', 'brainsmith:LayerNorm_rtl']

# List all available steps
all_steps = list_steps()
```

### For Component Authors

```python
from brainsmith.registry import kernel, backend, step

# Register a custom kernel
@kernel
class MyCustomKernel(HWCustomOp):
    op_type = "MyCustomOp"
    # ... implementation

# Register a backend for it
@backend(target_kernel='MyCustomKernel', language='hls')
class MyCustomKernel_hls:
    # ... code generation logic

# Register a step
@step
def my_optimization_step(model, **kwargs):
    # ... transformation logic
    return model
```

---

## Component Types

### 1. Steps

**What**: Functions that transform ONNX models through compilation pipeline
**Examples**: `streamline`, `create_dataflow_partition`, `hw_codegen`
**Signature**: `def step(model, **config) -> model`

```python
@step
def my_step(model, **config):
    """Custom transformation step."""
    # Transform the model
    model = my_transformation(model)
    return model
```

### 2. Kernels

**What**: Hardware custom operations (HWCustomOp classes)
**Examples**: `LayerNorm`, `Softmax`, `MVAU`
**Base Class**: `finn.custom_op.fpgadataflow.HWCustomOp`

```python
@kernel
class MyKernel(HWCustomOp):
    op_type = "MyCustomOperation"

    def make_shape_compatible_op(self, model):
        # Implementation
        pass

    def infer_node_datatype(self, model):
        # Implementation
        pass
```

### 3. Backends

**What**: Code generators that produce HLS/RTL for kernels
**Examples**: `LayerNorm_hls`, `MVAU_rtl`
**Attributes**: `target_kernel`, `language`

```python
@backend(target_kernel='MyKernel', language='hls')
class MyKernel_hls:
    def generate_params(self, model, node):
        # Generate HLS parameters
        pass

    def execute_node(self, context, graph, node):
        # HLS code generation
        pass
```

---

## Registering Components

### Method 1: Decorator Registration (Recommended)

**Best for**: Most use cases, automatic discovery

```python
from brainsmith.registry import kernel, backend, step

# Simple registration
@kernel
class LayerNorm(HWCustomOp):
    op_type = "LayerNorm"
    # ...

# With parameters
@kernel(name='MyKernel', domain='custom.ops')
class CustomKernel(HWCustomOp):
    # ...

@backend(target_kernel='LayerNorm', language='hls')
class LayerNorm_hls:
    # ...

@step(name='my_custom_step')
def my_step(model, **config):
    # ...
```

### Method 2: Lazy Loading Pattern (Advanced)

**Best for**: Large plugin packages with many components

Create a `__init__.py` with a `COMPONENTS` dict:

```python
# plugins/__init__.py
from brainsmith.registry import create_lazy_module

COMPONENTS = {
    'kernels': {
        'MyKernel': '.my_kernel',      # Relative import path
        'AnotherKernel': '.another',
    },
    'backends': {
        'MyKernel_hls': '.my_kernel_hls',
    },
    'steps': {
        'my_step': '.my_step',
    },
}

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

Then in your modules, use decorators as normal:

```python
# plugins/my_kernel.py
from brainsmith.registry import kernel

@kernel
class MyKernel(HWCustomOp):
    # ...
```

**Benefits**:
- Components appear in `dir(plugins)` without importing
- Import only happens on first access
- Faster package initialization

### Method 3: Entry Points (Plugin Packages)

**Best for**: Distributing plugins via pip packages

In your `setup.py` or `pyproject.toml`:

```python
# setup.py
setup(
    name='my-brainsmith-plugin',
    entry_points={
        'brainsmith.plugins': [
            'myplugin = myplugin.registry:register_components',
        ],
    },
)
```

Entry point function returns component metadata:

```python
# myplugin/registry.py
def register_components():
    """Entry point for Brainsmith plugin discovery."""
    return {
        'kernels': [
            {
                'name': 'MyKernel',
                'module': 'myplugin.kernels.my_kernel',
                'class_name': 'MyKernel',
            },
        ],
        'backends': [
            {
                'name': 'MyKernel_hls',
                'module': 'myplugin.backends.my_kernel_hls',
                'class_name': 'MyKernel_hls',
                'target_kernel': 'MyKernel',
                'language': 'hls',
            },
        ],
        'steps': [
            {
                'name': 'my_step',
                'module': 'myplugin.steps.my_step',
                'func_name': 'my_step',
            },
        ],
    }
```

---

## Discovering Components

### Automatic Discovery

Discovery happens automatically on first component access:

```python
from brainsmith.registry import get_step

# First call triggers discovery
streamline = get_step('streamline')  # Discovers all components

# Subsequent calls use cache
cleanup = get_step('cleanup')  # Fast!
```

### Manual Discovery

```python
from brainsmith.registry import discover_components

# Force discovery (useful for CLI commands)
discover_components()

# Force refresh (clear cache and re-discover)
discover_components(force_refresh=True)

# Skip cache (useful during development)
discover_components(use_cache=False)
```

### Discovery Process

```
1. Core Components
   └─ Import brainsmith.kernels, brainsmith.steps
   └─ Decorators auto-register during import

2. Filesystem Sources (project, user)
   └─ Load from configured component_sources paths
   └─ Import __init__.py (triggers decorators)

3. Entry Points (FINN, plugins)
   └─ Scan pip package entry points
   └─ Call register functions, index metadata

4. Backend Linking
   └─ Build kernel → backends relationships

5. Cache Manifest (optional)
   └─ Save to .brainsmith/component_manifest.json
   └─ Speeds up next startup
```

### Configuration

Control discovery via settings:

```python
# In brainsmith.toml or via Settings API
[components]
cache_components = true           # Enable manifest caching
components_strict = false         # Continue on import errors

source_priority = [               # Lookup order
    "project",
    "user",
    "brainsmith",
    "finn"
]

[component_sources]
project = "/path/to/project/plugins"
user = "~/.brainsmith/plugins"
```

---

## Accessing Components

### Steps

```python
from brainsmith.registry import get_step, has_step, list_steps

# Get step callable
streamline = get_step('streamline')
model = streamline(model, **config)

# Qualified name (explicit source)
custom = get_step('user:my_custom_step')

# Check existence (no import)
if has_step('streamline'):
    print("Step available")

# List all steps
all_steps = list_steps()
print(all_steps)
# ['brainsmith:streamline', 'brainsmith:cleanup', 'user:my_step', ...]

# Filter by source
user_steps = list_steps(source='user')
```

### Kernels

```python
from brainsmith.registry import get_kernel, get_kernel_infer, list_kernels

# Get kernel class
LayerNorm = get_kernel('LayerNorm')
kernel_instance = LayerNorm(onnx_node)

# Get InferTransform for a kernel
InferLayerNorm = get_kernel_infer('LayerNorm')
model = model.transform(InferLayerNorm())

# Check existence
if has_kernel('LayerNorm'):
    print("Kernel available")

# List kernels
all_kernels = list_kernels()
brainsmith_kernels = list_kernels(source='brainsmith')
```

### Backends

```python
from brainsmith.registry import (
    get_backend,
    list_backends,
    list_backends_for_kernel,
    get_backend_metadata
)

# Get backend class
backend = get_backend('LayerNorm_hls')

# Get backend metadata
meta = get_backend_metadata('LayerNorm_hls')
print(meta['target_kernel'])  # 'brainsmith:LayerNorm'
print(meta['language'])       # 'hls'

# List all backends
all_backends = list_backends()

# Find backends for specific kernel
ln_backends = list_backends_for_kernel('LayerNorm')
print(ln_backends)
# ['brainsmith:LayerNorm_hls', 'brainsmith:LayerNorm_rtl']

# Filter by language
hls_backends = list_backends_for_kernel('LayerNorm', language='hls')

# Filter by source
user_backends = list_backends_for_kernel('LayerNorm', sources=['user'])
```

### Source Resolution

Short names use **source priority** to resolve ambiguity:

```python
# Default priority: project > user > brainsmith > finn
get_kernel('LayerNorm')  # Searches in order:
# 1. project:LayerNorm (if exists)
# 2. user:LayerNorm (if exists)
# 3. brainsmith:LayerNorm ✓ (found!)

# Override with qualified name
get_kernel('finn:LayerNorm')  # Explicit source
```

---

## Advanced Patterns

### Pattern 1: Validation Before Execution

Validate blueprint components early:

```python
from brainsmith.registry import has_step, has_kernel

def validate_blueprint(blueprint):
    """Check all components exist before building."""
    errors = []

    for step in blueprint['steps']:
        if not has_step(step):
            errors.append(f"Step not found: {step}")

    for kernel in blueprint['kernels']:
        if not has_kernel(kernel):
            errors.append(f"Kernel not found: {kernel}")

    if errors:
        raise ValueError("\n".join(errors))
```

**Why**: Fail fast during parsing, not hours into a build.

### Pattern 2: Dynamic Backend Selection

Choose backends based on criteria:

```python
from brainsmith.registry import list_backends_for_kernel, get_backend

def select_backend(kernel_name, prefer_language='hls'):
    """Select optimal backend for kernel."""
    backends = list_backends_for_kernel(kernel_name, language=prefer_language)

    if not backends:
        # Fallback to any language
        backends = list_backends_for_kernel(kernel_name)

    if not backends:
        raise ValueError(f"No backends for {kernel_name}")

    # Return first backend class
    return get_backend(backends[0])
```

### Pattern 3: Plugin Discovery

List available plugins programmatically:

```python
from brainsmith.registry import list_kernels, list_steps, list_backends

def show_plugin_summary():
    """Display available components by source."""
    sources = ['brainsmith', 'finn', 'user', 'project']

    for source in sources:
        kernels = list_kernels(source=source)
        steps = list_steps(source=source)
        backends = list_backends(source=source)

        print(f"\n{source.upper()}:")
        print(f"  Kernels:  {len(kernels)}")
        print(f"  Steps:    {len(steps)}")
        print(f"  Backends: {len(backends)}")
```

### Pattern 4: Graceful Fallback

Handle missing components gracefully:

```python
from brainsmith.registry import get_step

def get_step_or_fallback(step_name, fallback_fn):
    """Get step from registry, fallback to custom function."""
    try:
        return get_step(step_name)
    except KeyError:
        logger.warning(f"Step '{step_name}' not in registry, using fallback")
        return fallback_fn
```

### Pattern 5: Component Introspection

Examine component metadata without loading:

```python
from brainsmith.registry import get_component_metadata

meta = get_component_metadata('LayerNorm', 'kernel')
print(f"Source: {meta.source}")                    # 'brainsmith'
print(f"Module: {meta.import_spec.module}")        # 'brainsmith.kernels.layernorm.layernorm'
print(f"Domain: {meta.kernel_domain}")             # 'finn.custom'
print(f"Backends: {meta.kernel_backends}")         # ['brainsmith:LayerNorm_hls', ...]
print(f"Loaded: {meta.loaded_obj is not None}")    # False (not imported yet)
```

---

## DSE Flow: Registry in Action

Here's how the registry is used throughout a complete DSE (Design Space Exploration) build:

### Execution Command

```bash
brainsmith dfc model.onnx blueprint.yaml --output-dir ./build
```

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: CLI INVOCATION                                                    │
│ File: brainsmith/cli/commands/dfc.py                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   explore_design_space(model_path, blueprint_path, output_dir)             │
│                                                                             │
│   Registry Access: NONE                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: BLUEPRINT PARSING (Registry Heavy)                                │
│ File: brainsmith/dse/_parser/__init__.py                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: Blueprint YAML                                                    │
│   ┌────────────────────────────────────────────────────────────┐           │
│   │ design_space:                                              │           │
│   │   kernels:                                                 │           │
│   │     - LayerNorm                                            │           │
│   │     - Softmax                                              │           │
│   │   steps:                                                   │           │
│   │     - cleanup                                              │           │
│   │     - streamline                                           │           │
│   │     - infer_kernels                                        │           │
│   └────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────────┐  ┌──────────────────────────────────────┐
│ PHASE 2a: STEP VALIDATION       │  │ PHASE 2b: KERNEL BACKEND RESOLUTION  │
│ File: _parser/steps.py           │  │ File: _parser/kernels.py             │
├─────────────────────────────────┤  ├──────────────────────────────────────┤
│                                 │  │                                      │
│ For each step in blueprint:     │  │ For each kernel in blueprint:        │
│                                 │  │                                      │
│ ┌─────────────────────────────┐ │  │ ┌──────────────────────────────────┐ │
│ │ 🔍 REGISTRY ACCESS #1       │ │  │ │ 🔍 REGISTRY ACCESS #2            │ │
│ │                             │ │  │ │                                  │ │
│ │ has_step('cleanup')         │ │  │ │ list_backends_for_kernel(       │ │
│ │   → True ✓                  │ │  │ │     'LayerNorm'                 │ │
│ │                             │ │  │ │ )                                │ │
│ │ has_step('streamline')      │ │  │ │   → ['brainsmith:LayerNorm_hls',│ │
│ │   → True ✓                  │ │  │ │       'brainsmith:LayerNorm_rtl']│ │
│ │                             │ │  │ └──────────────────────────────────┘ │
│ │ has_step('infer_kernels')   │ │  │                                      │
│ │   → True ✓                  │ │  │ ┌──────────────────────────────────┐ │
│ └─────────────────────────────┘ │  │ │ 🔍 REGISTRY ACCESS #3            │ │
│                                 │  │ │                                  │ │
│ • Validates existence           │  │ │ get_backend(                     │ │
│ • No imports triggered          │  │ │     'brainsmith:LayerNorm_hls'  │ │
│ • Fails fast on missing steps   │  │ │ )                                │ │
│                                 │  │ │   → <class 'LayerNorm_hls'>     │ │
│ Output: Validated step names    │  │ │                                  │ │
│   ['cleanup', 'streamline',     │  │ │ get_backend(                     │ │
│    'infer_kernels']             │  │ │     'brainsmith:LayerNorm_rtl'  │ │
│                                 │  │ │ )                                │ │
│                                 │  │ │   → <class 'LayerNorm_rtl'>     │ │
│                                 │  │ └──────────────────────────────────┘ │
│                                 │  │                                      │
│                                 │  │ • Discovers available backends       │
│                                 │  │ • Loads backend classes              │
│                                 │  │ • Imports modules (cached)           │
│                                 │  │                                      │
│                                 │  │ Output: (kernel, backends) tuples    │
│                                 │  │   [('LayerNorm',                     │
│                                 │  │     [<LayerNorm_hls>,                │
│                                 │  │      <LayerNorm_rtl>]),              │
│                                 │  │    ('Softmax', [...])]               │
└─────────────────────────────────┘  └──────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Output: GlobalDesignSpace                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GlobalDesignSpace(                                                        │
│       model_path='model.onnx',                                              │
│       steps=['cleanup', 'streamline', 'infer_kernels'],  # ← Validated     │
│       kernel_backends=[                                   # ← Loaded classes│
│           ('LayerNorm', [<LayerNorm_hls>, <LayerNorm_rtl>]),                │
│           ('Softmax', [<Softmax_hls>])                                      │
│       ]                                                                     │
│   )                                                                         │
│                                                                             │
│   Registry Access Summary:                                                 │
│   • has_step(): 3 calls (metadata only)                                    │
│   • list_backends_for_kernel(): 2 calls (metadata only)                    │
│   • get_backend(): 4 calls (module imports)                                │
│   • Total time: ~10-50ms (including discovery on first run)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: TREE BUILDING                                                     │
│ File: brainsmith/dse/_builder.py                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DSETreeBuilder().build_tree(design_space, config)                        │
│                                                                             │
│   Registry Access: NONE                                                    │
│                                                                             │
│   • Operates on resolved design space                                      │
│   • Creates segment tree structure                                         │
│   • Embeds backend classes in segments                                     │
│                                                                             │
│   Output: DSETree with segments ready for execution                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: TREE EXECUTION (Per-Segment Registry Access)                      │
│ File: brainsmith/dse/runner.py                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For each segment in tree:                                                │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Segment: root                                                       │  │
│   │ Steps: ['cleanup', 'streamline', 'infer_kernels']                  │  │
│   │                                                                     │  │
│   │ _resolve_steps():                                                  │  │
│   │                                                                     │  │
│   │ ┌─────────────────────────────────────────────────────────────────┐│  │
│   │ │ 🔍 REGISTRY ACCESS #4 (Per Step)                               ││  │
│   │ │                                                                 ││  │
│   │ │ get_step('cleanup')                                             ││  │
│   │ │   → <function cleanup at 0x7f...>                               ││  │
│   │ │                                                                 ││  │
│   │ │ get_step('streamline')                                          ││  │
│   │ │   → <function streamline at 0x7f...>                            ││  │
│   │ │                                                                 ││  │
│   │ │ get_step('infer_kernels')                                       ││  │
│   │ │   → <function infer_kernels at 0x7f...>                         ││  │
│   │ └─────────────────────────────────────────────────────────────────┘│  │
│   │                                                                     │  │
│   │ • Loads step callables on demand                                   │  │
│   │ • Imports modules (cached after first load)                        │  │
│   │ • Graceful fallback to FINN internal steps                         │  │
│   │                                                                     │  │
│   │ Execute segment with loaded steps:                                 │  │
│   │   model = cleanup(model, **config)                                 │  │
│   │   model = streamline(model, **config)                              │  │
│   │   model = infer_kernels(model,                                     │  │
│   │                         kernel_backends=[...])  # From Phase 2     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Registry Access Summary (per segment):                                   │
│   • get_step(): 3 calls (first segment: ~10ms, subsequent: <1ms cached)    │
│   • Backend classes: Already loaded in Phase 2                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: RESULTS & VALIDATION                                              │
│ File: brainsmith/dse/api.py                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   results.validate_success()                                               │
│   result_stats = results.compute_stats()                                   │
│                                                                             │
│   Registry Access: NONE                                                    │
│                                                                             │
│   Output: TreeExecutionResult with build artifacts                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Registry Access Timeline

```
┌────────────────────────────────────────────────────────────────────────┐
│ Timeline of Registry Interactions                                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ Phase 1: CLI Entry                                                    │
│ ├─ Time: 0ms                                                          │
│ └─ Registry: None                                                     │
│                                                                        │
│ Phase 2: Blueprint Parsing                                            │
│ ├─ Time: 10-50ms (includes discovery on first run)                    │
│ ├─ REGISTRY ACCESS #1: has_step() × N steps                           │
│ │  ├─ Triggers: discover_components() on first call                   │
│ │  ├─ Imports: None (metadata lookups only)                           │
│ │  └─ Purpose: Validate steps exist                                   │
│ │                                                                      │
│ ├─ REGISTRY ACCESS #2: list_backends_for_kernel() × N kernels         │
│ │  ├─ Imports: None (metadata lookups only)                           │
│ │  └─ Purpose: Discover available backends                            │
│ │                                                                      │
│ └─ REGISTRY ACCESS #3: get_backend() × N backends                     │
│    ├─ Imports: Backend modules (lazy, cached)                         │
│    └─ Purpose: Load backend classes                                   │
│                                                                        │
│ Phase 3: Tree Building                                                │
│ ├─ Time: <1ms                                                         │
│ └─ Registry: None (operates on resolved objects)                      │
│                                                                        │
│ Phase 4: Tree Execution                                               │
│ ├─ Time: Minutes to hours (actual builds)                             │
│ └─ REGISTRY ACCESS #4: get_step() × N steps × M segments              │
│    ├─ Imports: Step modules (lazy, cached)                            │
│    └─ Purpose: Load step callables                                    │
│                                                                        │
│ Phase 5: Results                                                      │
│ ├─ Time: <1ms                                                         │
│ └─ Registry: None                                                     │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│ Performance Characteristics                                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ First Execution:                                                      │
│   • Component discovery: ~30ms                                        │
│   • Step validation: ~5ms (3 steps)                                   │
│   • Backend loading: ~15ms (4 backends)                               │
│   • Step loading: ~10ms (3 steps, first segment)                      │
│   • Total overhead: ~60ms                                             │
│                                                                        │
│ Subsequent Executions (Cached):                                       │
│   • Component discovery: ~2ms (manifest cache)                        │
│   • Step validation: ~1ms (3 steps)                                   │
│   • Backend loading: ~2ms (cached)                                    │
│   • Step loading: <1ms (cached)                                       │
│   • Total overhead: ~5ms                                              │
│                                                                        │
│ Cache Benefits:                                                       │
│   • Manifest: .brainsmith/component_manifest.json                     │
│   • Component index: In-memory after discovery                        │
│   • Loaded modules: Python's import cache                             │
│   • Speedup: ~12x faster on subsequent runs                           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Insights

1. **Validation Before Execution**
   - Blueprint parsing validates all components exist
   - Fails fast with helpful errors (seconds, not hours)
   - No wasted time on invalid configurations

2. **Lazy Loading Strategy**
   - Phase 2: Load backends (needed for tree structure)
   - Phase 4: Load steps (needed for execution)
   - Import only what's actually used

3. **Caching at Multiple Levels**
   - Manifest cache: Skip discovery on restart
   - Component index: In-memory metadata
   - Python imports: Automatic module caching

4. **Clean Separation**
   - Parse: Registry for validation & backend discovery
   - Build: No registry (operates on resolved objects)
   - Execute: Registry for step loading only

---

## Troubleshooting

### Problem: Component Not Found

**Error:**
```
KeyError: Step 'my_step' not found
```

**Solutions:**

1. **List available components:**
   ```bash
   brainsmith list steps
   brainsmith list kernels
   brainsmith list backends
   ```

2. **Check component exists:**
   ```python
   from brainsmith.registry import has_step
   print(has_step('my_step'))  # False
   ```

3. **Verify registration:**
   ```python
   # In your component file
   from brainsmith.registry import step

   @step  # ← Don't forget the decorator!
   def my_step(model, **config):
       pass
   ```

4. **Check source priority:**
   ```python
   # Explicitly qualify the source
   get_step('user:my_step')  # Not just 'my_step'
   ```

5. **Force refresh:**
   ```python
   from brainsmith.registry import discover_components
   discover_components(force_refresh=True)
   ```

### Problem: Wrong Component Version

**Symptoms:**
- Getting component from unexpected source
- Updates to custom components not reflected

**Solutions:**

1. **Check source priority:**
   ```python
   from brainsmith.settings import get_config
   print(get_config().source_priority)
   # ['project', 'user', 'brainsmith', 'finn']
   ```

2. **Use qualified names:**
   ```python
   # Explicitly specify source
   get_step('user:my_step')      # Force user version
   get_step('brainsmith:cleanup')  # Force brainsmith version
   ```

3. **Inspect component metadata:**
   ```python
   from brainsmith.registry import get_component_metadata
   meta = get_component_metadata('my_step', 'step')
   print(f"Source: {meta.source}")
   print(f"Module: {meta.import_spec.module}")
   ```

### Problem: Import Errors During Discovery

**Error:**
```
Failed to load component source 'user': ImportError: No module named 'foo'
```

**Solutions:**

1. **Enable strict mode (development):**
   ```toml
   # brainsmith.toml
   [components]
   components_strict = true  # Fail fast on import errors
   ```

2. **Lenient mode (production):**
   ```toml
   [components]
   components_strict = false  # Log errors, continue
   ```

3. **Check import paths:**
   ```python
   # Verify module can be imported
   import sys
   print(sys.path)

   import my_plugin  # Test import
   ```

### Problem: Slow Startup

**Symptoms:**
- First `get_*()` call takes seconds
- Frequent re-discovery

**Solutions:**

1. **Enable manifest caching:**
   ```toml
   [components]
   cache_components = true
   ```

2. **Use lazy loading pattern:**
   ```python
   # In __init__.py
   COMPONENTS = {
       'steps': {'my_step': '.my_step'}  # Defer import
   }
   __getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
   ```

3. **Profile discovery:**
   ```bash
   BRAINSMITH_PROFILE=1 brainsmith list steps
   # Shows timing for each component load
   ```

### Problem: Stale Cache

**Symptoms:**
- Changes to components not reflected
- Old components still appearing

**Solutions:**

```bash
# Clear cache manually
rm .brainsmith/component_manifest.json

# Or force refresh in code
discover_components(force_refresh=True)
```

---

## Best Practices

### ✅ DO: Use Public API

```python
# ✅ Good
from brainsmith.registry import get_step, list_backends_for_kernel

# ❌ Bad
from brainsmith.registry._state import _component_index  # Private!
```

### ✅ DO: Validate Early

```python
# ✅ Good - validate during blueprint parsing
from brainsmith.registry import has_step

def parse_blueprint(data):
    for step in data['steps']:
        if not has_step(step):
            raise ValueError(f"Step not found: {step}")
    # ... continue parsing

# ❌ Bad - discover error during execution
def execute(data):
    for step in data['steps']:
        fn = get_step(step)  # May fail hours into build!
```

### ✅ DO: Use Decorators

```python
# ✅ Good - automatic registration
from brainsmith.registry import step

@step
def my_step(model, **config):
    return model

# ❌ Bad - manual registration (fragile)
from brainsmith.registry._decorators import _register_step
_register_step(my_step)  # Don't do this!
```

### ✅ DO: Handle Missing Components

```python
# ✅ Good - graceful degradation
try:
    custom_step = get_step('user:custom_optimization')
except KeyError:
    logger.warning("Custom optimization not available, using default")
    custom_step = get_step('brainsmith:optimization')

# ❌ Bad - assume component exists
custom_step = get_step('user:custom_optimization')  # Crashes if missing
```

### ✅ DO: Use Qualified Names When Needed

```python
# ✅ Good - explicit when overriding
project_step = get_step('project:streamline')  # Custom version
default_step = get_step('brainsmith:streamline')  # Original

# ✅ Also good - let source priority handle it
step = get_step('streamline')  # Uses project version if exists
```

### ✅ DO: Batch Operations

```python
# ✅ Good - single query + batch load
backend_names = list_backends_for_kernel('LayerNorm')
backends = [get_backend(name) for name in backend_names]

# ❌ Bad - repeated queries
for kernel in kernels:
    for backend in all_backends:
        if backend.startswith(kernel):  # Inefficient!
            b = get_backend(backend)
```

### ✅ DO: Document Component Requirements

```python
# ✅ Good - clear dependencies
@step
def my_optimization(model, **config):
    """Custom optimization step.

    Prerequisites:
        - Requires 'streamline' step before this
        - Requires LayerNorm kernel with HLS backend

    Config:
        level (int): Optimization level 1-3
    """
    # ...
```

### ❌ DON'T: Mutate Registry State

```python
# ❌ Bad - direct mutation
from brainsmith.registry._state import _component_index
_component_index['user:my_step'] = my_metadata  # DON'T!

# ✅ Good - use decorators
@step
def my_step(model, **config):
    pass
```

### ❌ DON'T: Import Components at Module Level

```python
# ❌ Bad - eager import, slow startup
from brainsmith.registry import get_step
STREAMLINE = get_step('streamline')  # Runs at import time!

# ✅ Good - lazy import
def run_pipeline():
    streamline = get_step('streamline')  # Load when needed
    # ...
```

### ❌ DON'T: Ignore Discovery Errors (in strict environments)

```python
# ❌ Bad - silent failures
components_strict = false  # Logs errors, continues

# ✅ Good (for development/CI)
components_strict = true   # Fail fast on import errors
```

---

## Summary

### Quick Reference Card

```python
# Registration
from brainsmith.registry import kernel, backend, step

@step
def my_step(model, **config):
    return model

@kernel
class MyKernel(HWCustomOp):
    pass

@backend(target_kernel='MyKernel', language='hls')
class MyKernel_hls:
    pass

# Discovery
from brainsmith.registry import discover_components

discover_components()                    # Auto on first use
discover_components(force_refresh=True)  # Clear cache

# Access
from brainsmith.registry import (
    get_step, has_step, list_steps,
    get_kernel, get_kernel_infer, list_kernels,
    get_backend, list_backends_for_kernel
)

# Steps
streamline = get_step('streamline')
exists = has_step('cleanup')
all_steps = list_steps()
user_steps = list_steps(source='user')

# Kernels
LayerNorm = get_kernel('LayerNorm')
InferLayerNorm = get_kernel_infer('LayerNorm')
kernels = list_kernels()

# Backends
backend = get_backend('LayerNorm_hls')
backends = list_backends_for_kernel('LayerNorm')
hls_only = list_backends_for_kernel('LayerNorm', language='hls')

# Metadata
from brainsmith.registry import get_component_metadata
meta = get_component_metadata('LayerNorm', 'kernel')
print(meta.source, meta.import_spec.module)
```

### Next Steps

1. **New Users**: Start with [Quick Start](#quick-start)
2. **Component Authors**: Read [Registering Components](#registering-components)
3. **Plugin Developers**: Study [Advanced Patterns](#advanced-patterns)
4. **Troubleshooting**: See [Troubleshooting](#troubleshooting) section

### Additional Resources

- **API Reference**: `brainsmith/registry/__init__.py` docstrings
- **Examples**: `examples/custom_components/`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Settings**: `brainsmith/settings/schema.py`

---

**End of User Guide**

For questions or issues, please consult the troubleshooting section or file an issue at the project repository.
