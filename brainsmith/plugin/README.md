# BrainSmith Plugin System

The BrainSmith plugin system provides a unified framework for registering and discovering transforms, kernels, backends, and build steps. This document explains how to use the plugin system to extend BrainSmith's capabilities.

## Overview

The plugin system uses decorators to automatically register components, making them discoverable and manageable through a centralized registry. Components are organized by type and purpose, with automatic validation and dependency tracking.

## Component Types

### 1. Transforms

Transforms modify ONNX computational graphs and are organized by compilation stage.

```python
from qonnx.transformation.base import Transformation
from brainsmith.plugin.decorators import transform

@transform(
    name="InferLayerNorm",
    stage="kernel_mapping",
    description="Convert FuncLayerNorm to LayerNorm hardware operations",
    author="your-name",
    version="1.0.0",
    requires=["qonnx>=0.1.0"]
)
class InferLayerNorm(Transformation):
    def apply(self, model):
        # Transform implementation
        return (model, graph_modified)
```

**Valid Stages:**
- `graph_cleanup` - Early graph optimizations (identity removal, cleanup)
- `topology_optimization` - Model-level transforms (norm expansion, etc.)
- `kernel_mapping` - Hardware lowering (e.g., Softmax → HWSoftmax)
- `kernel_optimization` - Operation-specific optimizations
- `graph_optimization` - System-level optimizations
- `metadata` - Metadata extraction transforms
- `model_specific` - Model-specific transforms (e.g., BERT modifications)

### 2. Kernels

Kernels represent hardware operations with associated backends and optimizations.

```python
from brainsmith.plugin.decorators import kernel

@kernel(
    name="MatMul",
    description="Matrix multiplication hardware operation",
    author="your-name",
    version="1.0.0",
    requires=["numpy>=1.20"]
)
class MatMul:
    """Kernel implementation details"""
    pass
```

### 3. Backends

Backends provide hardware implementations for kernels (HLS, RTL, etc.).

```python
from brainsmith.plugin.decorators import backend

@backend(
    name="MatMulHLS",
    kernel="MatMul",
    type="hls",
    description="HLS implementation of MatMul",
    author="your-name",
    version="1.0.0"
)
class MatMulHLS:
    """HLS backend implementation"""
    pass
```

### 4. Hardware Transforms

Hardware-specific optimization transforms for kernels.

```python
from brainsmith.plugin.decorators import hw_transform

@hw_transform(
    name="OptimizeLayerNormDSP",
    kernel="LayerNorm",
    description="DSP optimization for LayerNorm",
    author="your-name",
    version="1.0.0"
)
class OptimizeLayerNormDSP(Transformation):
    """DSP optimization implementation"""
    pass
```

### 5. FINN Build Steps

Build steps orchestrate transforms in FINN's compilation pipeline.

```python
from brainsmith.steps.decorators import finn_step

@finn_step(
    name="infer_hardware",
    category="hardware",
    dependencies=["streamlining"],
    description="Infer hardware layers for operations"
)
def infer_hardware_step(model, cfg):
    # Use transforms from plugin system
    from brainsmith.transforms.kernel_mapping import InferLayerNorm
    model = model.transform(InferLayerNorm())
    return model
```

**Step Categories:**
- `cleanup` - Model cleanup operations
- `conversion` - Format conversions (QONNX → FINN)
- `streamlining` - Graph streamlining
- `hardware` - Hardware inference
- `optimization` - Performance optimizations
- `validation` - Model validation
- `metadata` - Metadata extraction
- `preprocessing` - Input preprocessing
- `bert` - BERT-specific operations

## Directory Structure

```
brainsmith/
├── transforms/                 # Transform plugins by stage
│   ├── graph_cleanup/
│   ├── topology_optimization/
│   ├── kernel_mapping/
│   ├── kernel_optimization/
│   ├── graph_optimization/
│   ├── metadata/
│   └── model_specific/
│
├── kernels/                    # Hardware kernel plugins
│   └── <kernel_name>/
│       ├── <kernel_name>.py    # Kernel definition
│       ├── <kernel_name>_hls.py # HLS backend
│       └── <kernel_name>_rtl.py # RTL backend
│
├── steps/                      # FINN build step plugins
│   ├── decorators.py
│   ├── registry.py
│   └── steps.py               # All step definitions
│
└── plugin/                     # Core plugin infrastructure
    ├── decorators.py
    ├── registry.py
    ├── discovery.py
    └── exceptions.py
```

## Using the Registry

### Accessing Components

```python
from brainsmith.plugin.registry import PluginRegistry

registry = PluginRegistry()

# Get specific components
transform = registry.get_transform("InferLayerNorm")
kernel = registry.get_kernel("MatMul")
backend = registry.get_backend("MatMulHLS")

# List components by type/stage
kernel_mapping_transforms = registry.list_transforms(stage="kernel_mapping")
all_kernels = registry.list_kernels()

# Search across all plugins
results = registry.search_plugins("norm")  # Finds anything with "norm"

# Get plugin metadata
plugin_info = registry.get_plugin_metadata("InferLayerNorm")
print(f"Author: {plugin_info['author']}")
print(f"Version: {plugin_info['version']}")
```

### FINN Steps Registry

```python
from brainsmith.steps import get_step, list_finn_steps

# Get step function
step_fn = get_step("infer_hardware")
model = step_fn(model, cfg)

# List all available steps
steps = list_finn_steps()
```

## Plugin Discovery

Plugins are automatically discovered from:

1. **Built-in locations:**
   - `brainsmith/transforms/<stage>/` - Transforms by stage
   - `brainsmith/kernels/<name>/` - Kernel implementations
   - `brainsmith/steps/` - FINN build steps

2. **User locations:**
   - `~/.brainsmith/plugins/` - User plugins
   - `./brainsmith_plugins/` - Project plugins

3. **Python packages:**
   - Any package starting with `brainsmith-plugin-*`

## Dependency Management

### Specifying Dependencies

```python
@transform(
    name="MyTransform",
    stage="kernel_mapping",
    requires=[
        "qonnx>=0.1.0",           # Python package
        "kernel:MatMul",          # Requires MatMul kernel
        "transform:Streamline"    # Requires Streamline transform
    ]
)
```

### Validation

The registry validates dependencies and will warn about missing requirements:

```python
# Check if all dependencies are satisfied
missing = registry.check_dependencies("MyTransform")
if missing:
    print(f"Missing dependencies: {missing}")
```

## Creating Plugins

### Step 1: Choose Component Type

Determine what type of plugin you're creating:
- **Transform**: Modifies ONNX graphs
- **Kernel**: Hardware operation
- **Backend**: Hardware implementation
- **Build Step**: Pipeline orchestration

### Step 2: Create Plugin File

Place your file in the appropriate directory:
- Transforms: `brainsmith/transforms/<stage>/<name>.py`
- Kernels: `brainsmith/kernels/<name>/<name>.py`
- Build Steps: `brainsmith/steps/steps.py` (add to existing file)

### Step 3: Apply Decorator

Use the appropriate decorator with required metadata:

```python
@transform(
    name="YourTransformName",
    stage="appropriate_stage",
    description="Clear description of what it does",
    author="your-github-username",
    version="1.0.0"
)
```

### Step 4: Test

```python
# Verify registration
from brainsmith.plugin.registry import PluginRegistry
registry = PluginRegistry()
assert "YourTransformName" in [t[0] for t in registry.list_transforms()]
```

## Best Practices

### 1. Naming Conventions
- **Transforms**: `VerbNoun` (e.g., `InferLayerNorm`, `RemoveIdentity`)
- **Kernels**: `OperationName` (e.g., `MatMul`, `LayerNorm`)
- **Backends**: `<Kernel><Type>` (e.g., `MatMulHLS`, `LayerNormRTL`)
- **Steps**: `<action>_<target>_step` (e.g., `infer_hardware_step`)

### 2. Documentation
- Always include a clear `description` in decorators
- Add docstrings to classes and functions
- Document special requirements or limitations

### 3. Dependencies
- Explicitly list all requirements in `requires`
- Use version constraints (e.g., `>=1.0.0,<2.0.0`)
- Prefer optional imports with graceful fallbacks

### 4. Testing
- Test plugins in isolation before integration
- Verify automatic discovery works
- Check dependency validation

## Examples

### Complete Transform Example

```python
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from brainsmith.plugin.decorators import transform

@transform(
    name="ConvertGatherToCrop",
    stage="kernel_mapping",
    description="Convert Gather operations to hardware Crop operations",
    author="jane-doe",
    version="1.0.0",
    requires=["qonnx>=0.1.0", "numpy>=1.20"]
)
class ConvertGatherToCrop(Transformation):
    """Convert eligible Gather nodes to Crop hardware operations."""
    
    def __init__(self, simd=1):
        super().__init__()
        self.simd = simd
    
    def apply(self, model):
        graph = model.graph
        graph_modified = False
        
        for node in graph.node:
            if node.op_type == "Gather":
                # Transformation logic here
                graph_modified = True
        
        if graph_modified:
            model = model.transform(InferShapes())
        
        return (model, graph_modified)
```

### Complete Build Step Example

```python
from brainsmith.steps.decorators import finn_step
import logging

logger = logging.getLogger(__name__)

@finn_step(
    name="optimize_hardware",
    category="optimization",
    dependencies=["infer_hardware"],
    description="Apply hardware-specific optimizations"
)
def optimize_hardware_step(model, cfg):
    """Apply hardware optimizations based on target platform."""
    logger.info("Applying hardware optimizations")
    
    # Import transforms from plugin system
    from brainsmith.plugin.registry import PluginRegistry
    registry = PluginRegistry()
    
    # Get all hardware optimization transforms
    hw_transforms = registry.list_transforms(stage="kernel_optimization")
    
    # Apply relevant transforms
    for transform_name, transform_cls in hw_transforms:
        if should_apply_transform(transform_name, cfg):
            model = model.transform(transform_cls())
    
    return model
```

## Troubleshooting

### Plugin Not Found

If your plugin isn't discovered:
1. Check file location matches expected structure
2. Verify decorator is applied correctly
3. Import the module to trigger registration
4. Check for syntax errors in plugin file

### Dependency Issues

If dependencies fail:
1. Verify package names are correct
2. Check version constraints
3. Ensure kernel/transform dependencies exist
4. Look for circular dependencies

### Registration Errors

Common issues:
- Duplicate plugin names (each must be unique)
- Invalid stage names (use valid stages listed above)
- Missing required decorator parameters
- Class doesn't inherit from expected base class

## Further Information

- See example plugins in `brainsmith/transforms/` and `brainsmith/kernels/`
- Check `brainsmith/plugin/exceptions.py` for error types
- Review tests in `tests/test_plugin_*.py` for usage examples