# BrainSmith Plugin System Guide

The BrainSmith plugin system provides a unified way to extend the framework with new transformations, hardware kernels, code generators, and build steps. All plugins are managed through a central registry using decorator-based registration, accessible via blueprint or direct look-up.

## Overview

The plugin system is the backbone of BrainSmith's extensibility. It allows developers to register new functionality that can be discovered and used dynamically at runtime. The system is built around a singleton registry that maintains a catalog of all available plugins, organized by type and tagged with metadata for easy discovery. When you register a plugin using decorators, it becomes immediately available to the entire BrainSmith ecosystem - from blueprint configurations to programmatic access.

## Plugin Types

### 1. Transforms

**Purpose**: Modify ONNX graphs for optimization, hardware mapping, or preprocessing

Transforms an ONNX model as input, apply specific modifications, and return the transformed model along with a flag indicating whether any changes were made. Transforms can range from simple graph optimizations (like constant folding) to complex pattern matching and replacement operations (like converting high-level operations into hardware-friendly primitives). They form the core of Brainsmith's ability to adapt models for FPGA deployment by progressively lowering high-level operations into hardware-implementable forms.

**Interface**:
```python
from qonnx.transformation.base import Transformation
from brainsmith.core.plugins import transform

@transform(
    name="MyTransform",
    stage="topology_opt",  # Optional: categorize the transform
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

**Example**: `brainsmith/transforms/cleanup/expand_norms.py:10`
```python
@transform(
    name="ExpandNorms",
    stage="topology_opt",
    description="Expand LayerNorms/RMSNorms into functional components"
)
class ExpandNorms(Transformation):
    def apply(self, model):
        # Expands LayerNorm into Div, Sub, Mul operations
```

### 2. Build Steps

**Purpose**: Define reusable sequences of operations in the compilation flow

Build steps are higher-level orchestrators that coordinate multiple transforms and other operations to achieve specific compilation goals. While transforms focus on individual graph modifications, steps represent logical stages in the compilation pipeline. A step might apply several transforms in sequence, perform validation checks, or prepare the model for subsequent processing stages. The brainsmith compiler constructs an execution tree of steps based on the input blueprint.

**Interface**:
```python
from brainsmith.core.plugins import step

@step(
    name="my_step",
    category="optimization",
    dependencies=["previous_step"],  # Optional
    description="What this step does"
)
def my_step(model, cfg):
    # Apply transforms
    from brainsmith.core.plugins import get_transform
    
    transform = get_transform("SomeTransform")
    model, _ = transform().apply(model)
    
    return model
```

**Example**: `brainsmith/steps/core_steps.py:10`
```python
@step(
    name="qonnx_to_finn",
    category="cleanup",
    description="Convert from QONNX to FINN opset"
)
def qonnx_to_finn_step(model, cfg):
    # Applies multiple transforms in sequence
```

### 3. Kernels

**Purpose**: Define custom hardware operators with specific attributes and behavior

Kernels are hardware implementations of neural network operations, and are composed of a variety of files (as described in kernels documentation). The head of each kernel is the HWCustomOp, a custom ONNX operator that models the behavior and resource requirements of FPGA-specific implementations.

***TAFK TODO: Scope description based on Kernel Integrator timeline.***

**Important Note**: While only the HWCustomOp class is required for kernel registration, a fully functional kernel also requires an associated kernel inference transform for pattern matching ONNX operations and converting them to the kernel's HWCustomOp.

**Interface**:
```python
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from brainsmith.core.plugins import kernel

@kernel(
    name="MyKernel",
    description="Hardware implementation of operation",
    author="Your Name"
)
class MyKernel(HWCustomOp):
    def get_nodeattr_types(self):
        return {
            "NumChannels": ("i", True, ""),  # (type, required, default)
            "SIMD": ("i", True, 1),
        }
    
    def execute_node(self, context, graph):
        # Simulation logic
        pass
```

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

Backends separate the logical description of hardware operations (kernels) from their physical implementation (generated code). This separation allows multiple implementation strategies for the same kernel - for instance, different backends might optimize for latency, throughput, or resource usage. Backends must understand both the kernel's semantics and the target synthesis tool's requirements to generate efficient, correct code.

**Important Note**: While only the backend class is required for registration, a fully functional backend also requires:
- Associated RTL or HLS source implementation files
- Wrapper templates for integration with the generated code
- Proper file structure following FINN conventions

**Interface**:
```python
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.core.plugins import backend

@backend(
    name="MyKernelHLS",
    kernel="MyKernel",  # Which kernel this generates code for
    language="hls",     # "hls" or "rtl"
    description="HLS backend for MyKernel"
)
class MyKernel_hls(MyKernel, HLSBackend):
    def global_includes(self):
        return ['#include "ap_fixed.h"']
    
    def defines(self, var):
        return [f"#define NUM_CHANNELS {self.get_nodeattr('NumChannels')}"]
    
    def docompute(self):
        return """
        // HLS computation code
        """
```

**Example**: `brainsmith/kernels/layernorm/layernorm_hls.py:10`
```python
@backend(
    name="LayerNormHLS",
    kernel="LayerNorm",
    language="hls"
)
class LayerNorm_hls(LayerNorm, HLSBackend):
    # Generates HLS C++ code for LayerNorm
```

## Using Plugins

### Getting Plugins

The plugin system provides a straightforward API for retrieving registered plugins. For plugins from external frameworks like FINN or QONNX, you can use either the full namespaced name (e.g., "finn:ConvertBipolarMatMulToXnorPopcount") or just the simple name if it's unique across all frameworks. This flexibility makes it easy to use external plugins while avoiding naming conflicts when necessary.

```python
from brainsmith.core.plugins import (
    get_transform, get_kernel, get_backend, get_step,
    list_transforms, list_kernels, list_backends, list_steps
)

# Get a specific plugin
transform = get_transform("ExpandNorms")
kernel = get_kernel("LayerNorm")
backend = get_backend("LayerNormHLS")
step = get_step("qonnx_to_finn")

# List all plugins of a type
all_transforms = list_transforms()
all_kernels = list_kernels()
```

### Finding Plugins by Metadata

Beyond simple name-based lookup, the plugin system supports metadata-based discovery. For example, you might want all transforms that belong to a specific optimization stage, or all kernel inference transforms for a particular kernel. The metadata system makes plugins self-documenting and discoverable.

```python
from brainsmith.core.plugins import get_transforms_by_metadata

# Find all transforms for a specific stage
topology_transforms = get_transforms_by_metadata(stage="topology_opt")

# Find kernel inference transforms
kernel_transforms = get_transforms_by_metadata(kernel="LayerNorm")
```

### Framework Plugins

BrainSmith integrates plugins from external frameworks like FINN and QONNX, making their transformations and kernels available through the same unified interface. This is currently a highly manual and fragile process on the backend, and will be refactored before release.

```python
# Both work if "MVAU" is unique
kernel1 = get_kernel("finn:MVAU")
kernel2 = get_kernel("MVAU")

# List plugins from specific framework
finn_transforms = [t for t in list_transforms() if t.startswith("finn:")]
```

### Kernel Inference Transforms

Kernel inference transforms are a special category that bridge standard ONNX operations and custom hardware kernels. They analyze the graph to find patterns that can be implemented using specific kernels, then replace those patterns with kernel instances.

```python
from brainsmith.core.plugins import kernel_inference

@kernel_inference(
    kernel="MyKernel",
    description="Infer MyKernel from ONNX patterns"
)
class InferMyKernel(Transformation):
    def apply(self, model):
        # Pattern matching and conversion logic
```
---
For more details, see the plugin registry implementation at `brainsmith/core/plugins/registry.py`.