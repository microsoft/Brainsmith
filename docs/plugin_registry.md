# Plugin Library Registry

## Table of Contents

- [Key Architectural Concepts](#key-architectural-concepts)
- [Plugin Types](#plugin-types)
- [Using Plugins](#using-plugins)
- [Development and Testing](#development-and-testing)
- [Advanced Usage](#advanced-usage)
## Key Architectural Concepts

1. **Singleton Pattern**: A single global registry instance ensures all code sees the same plugins
3. **Namespace Management**: Framework prefixes (`finn:`, `qonnx:`) prevent name collisions between plugins from different sources
4. **Registration Order**: Multiple registrations of the same name result in the last one overwriting previous ones

## Plugin Types

### 1. Transforms

**Purpose**: Modify ONNX graphs for optimization, hardware mapping, or preprocessing

Transforms modify ONNX graphs by pattern matching, node replacement, optimization, or cleanup. Each transform returns a tuple: (modified_model, boolean_indicating_changes). All transforms are subclasses of the [QONNX Transformation pass](https://github.com/fastmachinelearning/qonnx/blob/main/docs/overview.rst#transformation-pass).

**Interface**:
```python
from qonnx.transformation.base import Transformation
from brainsmith.core.plugins import transform

@transform(
    name="MyTransform",   # Defaults to class name if not specified
    stage="topology_opt",
    description="What this transform does",
    author="Your Name"
)
class MyTransform(Transformation):
    def apply(self, model):
        # Modify the model
        graph_modified = False
        # ... transformation logic ...
        return (model, graph_modified)
```

**Example**: `brainsmith/transforms/kernel_opt/set_pumped_compute.py:16`
```python
@transform(
    name="SetPumpedCompute",
    stage="kernel_opt",
    description="Set pumped compute attribute for MVAUs and DynMatMuls"
)
class SetPumpedCompute(Transformation):
    def apply(self, model):
        for node in model.graph.node:
            if node.op_type == "MVAU_rtl":
                inst = registry.getCustomOp(node)
                inst.set_nodeattr("pumpedCompute", 1)
        return (model, False)
```

### 2. Build Steps

**Purpose**: Define reusable sequences of operations in the compilation flow

Steps coordinate sequences of operations in the compilation pipeline. Unlike transforms which modify graphs, steps orchestrate the overall flow and manage shared state through the context dictionary.

**Interface**:
```python
from brainsmith.core.plugins import step

@step(
    name="my_step",  # Required as keyword argument
    category="optimization",
    dependencies=["previous_step"],  # Optional
    description="What this step does"
)
def my_step(blueprint, context):  # Note: signature is (blueprint, context)
    # Apply transforms
    from brainsmith.core.plugins import get_transform
    
    transform = get_transform("SomeTransform")
    model = context.get("model")
    model, _ = transform().apply(model)
    context["model"] = model
    
    # Steps don't return values, they modify context
```

**Example**: `brainsmith/steps/core_steps.py:10`
```python
@step(
    name="qonnx_to_finn",
    category="cleanup",
    description="Convert from QONNX to FINN opset"
)
def qonnx_to_finn_step(blueprint, context):
    model = context["model"]
    # Apply multiple transforms
    model = apply_cleanup_transforms(model)
    context["model"] = model
```

### 3. Kernels

**Purpose**: Define custom hardware operators with specific attributes and behavior

Kernels implement neural network operations in hardware. They define the hardware interface, parameters, and simulation behavior.

**Interface**:
```python
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from brainsmith.core.plugins import kernel

@kernel(
    name="MyKernel",  # Required as keyword argument
    description="Hardware implementation of operation",
    author="Your Name"
)
class MyKernel(HWCustomOp):
    def get_nodeattr_types(self):
        return {
            # Format: (type, required, default)
            # Types: "i"=int, "s"=string, "f"=float
            "NumChannels": ("i", True, ""),  
            "SIMD": ("i", True, 1),
        }
    
    def execute_node(self, context, graph):
        # Simulation logic
        pass
```

**Note**: Functional kernels also require:
- An inference transform to convert ONNX ops to this kernel
- Backend(s) to generate synthesizable code

**Example**: `brainsmith/kernels/layernorm/layernorm.py:10`
```python
@kernel(
    name="LayerNorm",
    description="Hardware implementation of LayerNorm"
)
class LayerNorm(HWCustomOp):
    # Implements layer normalization in hardware
```

### 4. Backends

**Purpose**: Generate synthesizable code (C++ for HLS, Verilog for RTL) from kernel specifications

Backends generate synthesizable code (HLS C++ or RTL Verilog) from kernel specifications. Different backends can optimize for different targets: low latency, high throughput, or minimal resource usage.

**Requirements**: 
- Naming convention: `{KernelName}_{language}` (e.g., `LayerNorm_hls`)
- Multiple inheritance: from both kernel class and backend base class (HLSBackend or RTLBackend)
- First registered backend becomes the default

**Additional Requirements for Functional Backends**:
- Associated RTL or HLS source implementation files
- Wrapper templates for integration with the generated code
- Proper file structure following FINN conventions

**Interface**:
```python
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.core.plugins import backend

@backend(
    name="MyKernel_hls",  # Convention: KernelName_language
    kernel="MyKernel",    # Which kernel this generates code for
    language="hls",       # "hls" or "rtl"
    description="HLS backend for MyKernel"
)
class MyKernel_hls(MyKernel, HLSBackend):  # Multiple inheritance
    def global_includes(self):
        return ['#include "ap_fixed.h"']
    
    def defines(self, var):
        return [f"#define NUM_CHANNELS {self.get_nodeattr('NumChannels')}"]
    
    def docompute(self):
        return """
        for (int i = 0; i < NUM_CHANNELS; i++) {
            output[i] = input[i] * scale[i] + bias[i];
        }
        """
```

## Using Plugins

### Getting Plugins

Retrieve plugins by name, with automatic namespace resolution for unique names.

```python
from brainsmith.core.plugins import (
    get_registry,  # Direct registry access
    get_transform, get_kernel, get_backend, get_step,
    list_transforms, list_kernels, list_backends, list_steps,
    has_transform, has_kernel  # Check existence
)

# Get a specific plugin
transform = get_transform("ExpandNorms")
kernel = get_kernel("LayerNorm")
backend = get_backend("LayerNorm_hls")
step = get_step("qonnx_to_finn")

# Check if plugin exists (won't raise exception)
if has_transform("MyTransform"):
    transform = get_transform("MyTransform")

# List all plugins of a type
all_transforms = list_transforms()  # Returns list of names
all_kernels = list_kernels()
```

### Error Handling

Plugin retrieval raises `KeyError` with helpful messages when plugins aren't found:

```python
try:
    transform = get_transform("NonExistent")
except KeyError as e:
    print(e)
    # KeyError: "Plugin transform:NonExistent not found. Available (162): 
    # ['qonnx:BatchNormToAffine', 'finn:Streamline', ...]"

# Use has_* functions to check without exceptions:
if has_transform("MyTransform"):
    transform = get_transform("MyTransform")
```

### Namespace Resolution

The system automatically tries common framework prefixes when resolving names. For plugins from external frameworks like FINN or QONNX, you can use either the full namespaced name (e.g., "finn:ConvertBipolarMatMulToXnorPopcount") or just the simple name if it's unique.

```python
# These are equivalent if "Streamline" is unique:
transform1 = get_transform("finn:Streamline")
transform2 = get_transform("Streamline")

# For ambiguous names, use explicit namespace:
transform = get_transform("myframework:CommonName")
```

### Finding Plugins by Metadata

Query plugins by their metadata attributes:

```python
from brainsmith.core.plugins import (
    get_transforms_by_metadata,
    get_backends_by_metadata
)

# Find all transforms for a specific stage
topology_transforms = get_transforms_by_metadata(stage="topology_opt")

# Find kernel inference transforms
inference_transforms = get_transforms_by_metadata(kernel_inference=True)

# Find backends by language
hls_backends = get_backends_by_metadata(language="hls")

# Direct registry access for complex queries
registry = get_registry()
custom_transforms = registry.find("transform", author="MyTeam", version="2.0")
```

### Framework Plugins

Brainsmith automatically integrates plugins from FINN and QONNX frameworks on first access:

- **FINN**: ~98 transforms, ~40 kernels/backends
- **QONNX**: ~60 transforms
- **Total**: 200+ pre-registered components

```python
# List plugins from specific framework
registry = get_registry()
finn_transforms = registry.find("transform", framework="finn")
qonnx_transforms = registry.find("transform", framework="qonnx")

# Get framework-specific kernel backends
from brainsmith.core.plugins.registry import list_backends_by_kernel
mvau_backends = list_backends_by_kernel("MVAU")  # Returns ['MVAU_hls', 'MVAU_rtl']
```

## Development and Testing

### Understanding Plugin Registration

Plugins are registered as a side effect of module imports through decorators:

```python
# This happens automatically when the module is imported:
@transform(name="AutoRegistered")
class AutoRegistered(Transformation):
    pass

# The decorator is equivalent to:
# registry.register("transform", "AutoRegistered", AutoRegistered)
```

### Testing with Plugins

Key testing considerations:

1. **Registration is Permanent**: Plugins remain registered for the entire Python session
2. **Import Side Effects**: Decorators execute when modules are imported
3. **The reset() Limitation**: `registry.reset()` clears ALL plugins. Test plugins cannot be reloaded because decorators already executed during import

```python
# DON'T DO THIS in tests:
registry.reset()  # Clears all plugins
# Test plugins won't come back even with re-import!

# DO THIS instead:
# 1. Import test plugins early in your test session
# 2. Use direct registration for test-specific plugins
registry.register("transform", "test_only", TestTransform)
```

### Debugging Plugin Issues

```python
# Check what's registered
registry = get_registry()
print(f"Total transforms: {len(registry._plugins['transform'])}")

# See all registered names
all_transforms = list_transforms()
print(f"Transform names: {all_transforms}")

# Check if lazy loading has occurred
if hasattr(registry, '_discovered'):
    print("Framework plugins have been loaded")

# Force lazy loading
registry._load_plugins()
```

## Advanced Usage

### Direct Registry Access

For testing or debugging, access the registry directly:

```python
from brainsmith.core.plugins import get_registry

registry = get_registry()

# Direct registration (testing only)
registry.register("transform", "test_transform", MyTestTransform, 
                 framework="test", author="tester")

# Inspect registered plugins
transforms = registry._plugins["transform"]  # Dict[str, Tuple[Type, Dict]]

# Check if plugins are loaded
if hasattr(registry, '_discovered'):
    print("Plugins have been loaded")

# Force plugin loading
registry._load_plugins()
```

### Kernel Inference Transforms

Kernel inference transforms are a special category of transform that bridge standard ONNX operations and custom hardware kernels. They analyze the graph to find patterns that can be implemented using specific kernels, then replace those patterns with kernel instances. These transforms typically reside within their kernel's directory rather than the general transforms folder.

```python
from brainsmith.core.plugins import kernel_inference

@kernel_inference(
    kernel="MyKernel",
    description="Infer MyKernel from ONNX patterns"
)
class InferMyKernel(Transformation):
    def apply(self, model):
        # Pattern matching and conversion logic
        graph_modified = False
        # Find patterns and replace with kernel
        return (model, graph_modified)
```

**Note**: The `kernel_inference` decorator is an alias for the `transform` decorator that automatically tags the transform with kernel metadata for discovery.

---
For Plugin Registry implementation details, see `brainsmith/core/plugins/registry.py`. For examples, browse the `brainsmith/transforms/`, `brainsmith/kernels/`, and `brainsmith/steps/` directories.