# Component Registry

**Declarative component system for FPGA accelerator construction**

---

## Introduction

The **Component Registry** enables declarative blueprint construction for FPGA accelerators. Instead of hard-coding imports and dependencies, write YAML blueprints that reference components by name:

```yaml
# blueprint.yaml - Declarative configuration
design_space:
  kernels:
    - LayerNorm      # Registry resolves to correct implementation
    - Softmax
  steps:
    - streamline
    - infer_kernels
```

The registry handles three component types:

- **Steps**: Pipeline transformations (`streamline`, `cleanup`, `infer_kernels`)
- **Kernels**: Hardware operators (`LayerNorm`, `MVAU`, `Softmax`)
- **Backends**: Code generators (`LayerNorm_hls`, `MVAU_rtl`)

### Key Capabilities

**Blueprint Validation**: Parse-time validation prevents wasted build time
```python
# Fails immediately if 'custom_step' doesn't exist, not hours into execution
has_step('custom_step')  # Fast metadata check, no imports
```

**Source Priority**: Project components override core implementations
```python
# Resolution order: project > team > brainsmith > finn
get_kernel('LayerNorm')  # Your custom version wins if it exists
```

**Extensibility**: Register custom components without modifying core code
```python
@kernel
class MyCustomKernel(HWCustomOp):
    pass  # Automatically discovered and available in blueprints
```

### Component Source Types

The registry supports four source types:

**1. Core Brainsmith** (`brainsmith`)
- Core framework components loaded via direct import

**2. Entry Point Plugins** (e.g., `finn`)
- External Python packages declaring `brainsmith.plugins` entry points
- Example: FINN registers via `setup.cfg`: `brainsmith.plugins = finn = finn.util.brainsmith_integration:register_all`
- Add your own by declaring entry points in `setup.py` or `pyproject.toml`

**3. Filesystem Sources**
- **`project`** (automatic): Current project's `kernels/` and `steps/` subdirectories
- **Custom paths**: Add via `component_sources` in `brainsmith.yaml`
  ```yaml
  component_sources:
    team: "/shared/components"
    research: "~/experiments/kernels"
  ```
- All filesystem sources cached to manifest for fast startup

**4. Runtime Registration** (`custom`)
- Components registered programmatically at runtime (not via namespace/domain)
- **Never cached** - must re-register each run
- Used for ad-hoc testing and experimentation

**Source Priority**: Configurable via `source_priority` setting (default: `['project', 'brainsmith', 'finn', 'custom']`)

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

# List all available steps (or use: brainsmith registry --verbose)
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

The registry recognizes three component types, each registered via decorators:

- **Steps**: Pipeline transformations ([`@step`](../api/registry.md#brainsmithregistrystep)) - Transform ONNX models through compilation stages
- **Kernels**: Hardware operators ([`@kernel`](../api/registry.md#brainsmithregistrykernel)) - HWCustomOp/KernelOp classes for FPGA operations
- **Backends**: Code generators ([`@backend`](../api/registry.md#brainsmithregistrybackend)) - HLS/RTL implementations targeting specific kernels

See [Component Registry API](../api/registry.md) for decorator parameters and examples.

---

## Registering Components

### Decorator Registration (Recommended)

Apply decorators to automatically register components during import:

```python
from brainsmith.registry import kernel, backend, step

@kernel
class MyKernel(HWCustomOp):
    op_type = "MyKernel"

@backend(target_kernel='MyKernel', language='hls')
class MyKernel_hls:
    pass

@step
def my_step(model, **config):
    return model
```

See [Component Registry API](../api/registry.md) for decorator parameters, examples, and advanced usage.

### Entry Points (Plugin Packages)

**Best for**: Distributing plugins via pip packages

```python
# setup.py or pyproject.toml
entry_points={
    'brainsmith.plugins': ['myplugin = myplugin.registry:register_components']
}

# myplugin/registry.py
def register_components():
    """Return dict mapping component types to metadata lists."""
    return {
        'kernels': [{'name': 'MyKernel', 'module': 'myplugin.kernels', 'class_name': 'MyKernel'}],
        'backends': [{'name': 'MyKernel_hls', 'target_kernel': 'MyKernel', 'language': 'hls', ...}],
        'steps': [{'name': 'my_step', 'module': 'myplugin.steps', 'func_name': 'my_step'}],
    }
```

See [Python Packaging Guide](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/) for entry point packaging details.

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

### Configuration

Control discovery behavior and source priority via [`brainsmith.yaml`](../api/settings.md#brainsmithsettingsSystemConfig):

- **`cache_components`** - Enable manifest caching (default: `true`)
- **`components_strict`** - Fail fast on import errors (default: `true`, recommended for development)
- **`source_priority`** - Resolution order for short names (default: `['project', 'user', 'brainsmith', 'finn']`)
- **`component_sources`** - Filesystem paths for custom sources (e.g., `project = "./"`)

See [Settings API](../api/settings.md) for complete configuration options.

---

## Accessing Components

All component types follow a consistent `get_*/has_*/list_*` API pattern:

- **Get** - Load component (imports module, returns class/function): [`get_step()`](../api/registry.md#brainsmithregistryget_step), [`get_kernel()`](../api/registry.md#brainsmithregistryget_kernel), [`get_backend()`](../api/registry.md#brainsmithregistryget_backend)
- **Has** - Check existence without importing: [`has_step()`](../api/registry.md#brainsmithregistryhas_step), [`has_kernel()`](../api/registry.md#brainsmithregistryhas_kernel)
- **List** - Enumerate available components: [`list_steps()`](../api/registry.md#brainsmithregistrylist_steps), [`list_kernels()`](../api/registry.md#brainsmithregistrylist_kernels), [`list_backends()`](../api/registry.md#brainsmithregistrylist_backends)

**Source Resolution**: Short names (`'LayerNorm'`) resolve via source priority (project > user > brainsmith > finn). Qualified names (`'brainsmith:LayerNorm'`) specify source explicitly.

**Special Functions**:

- [`get_kernel_infer()`](../api/registry.md#brainsmithregistryget_kernel_infer) - Get ONNX → kernel inference transform
- [`list_backends_for_kernel()`](../api/registry.md#brainsmithregistrylist_backends_for_kernel) - Find backends for specific kernel (supports `language=` and `sources=` filters)
- [`get_component_metadata()`](../api/registry.md#brainsmithregistryget_component_metadata) - Introspect component without loading

See [Component Registry API](../api/registry.md) for complete function signatures and examples.

---

## Advanced Patterns

### Dynamic Backend Selection

Choose backends programmatically based on criteria:

```python
from brainsmith.registry import list_backends_for_kernel, get_backend

def select_backend(kernel_name, prefer_language='hls'):
    backends = list_backends_for_kernel(kernel_name, language=prefer_language)
    if not backends:
        backends = list_backends_for_kernel(kernel_name)  # Fallback
    return get_backend(backends[0]) if backends else None
```

### Plugin Inventory

Enumerate available components by source:

```python
from brainsmith.registry import list_kernels, list_steps, list_backends

for source in ['brainsmith', 'finn', 'user', 'project']:
    print(f"{source}: {len(list_kernels(source=source))} kernels, "
          f"{len(list_steps(source=source))} steps")
```

### Component Introspection

Use [`get_component_metadata()`](../api/registry.md#brainsmithregistryget_component_metadata) to inspect components without importing them - useful for CLI tools and discovery.

See also: [`ComponentMetadata`](../api/registry.md#brainsmithregistryComponentMetadata) structure reference


---

## Additional Resources

- [Component Registry API](../api/registry.md) - Complete API reference with examples
- [Settings API](../api/settings.md) - Configuration options and schema
- [Hardware Kernels Guide](hardware-kernels.md) - Kernel architecture and registration details
- [GitHub](https://github.com/microsoft/brainsmith) - Report issues
