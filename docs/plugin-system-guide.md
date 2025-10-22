# Brainsmith Plugin System - Complete Guide

A comprehensive guide to creating and using plugins in Brainsmith.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Plugin Structure](#plugin-structure)
4. [Creating Kernels](#creating-kernels)
5. [Creating Steps](#creating-steps)
6. [Creating Backends](#creating-backends)
7. [Plugin Discovery](#plugin-discovery)
8. [Best Practices](#best-practices)
9. [Advanced Topics](#advanced-topics)
10. [Troubleshooting](#troubleshooting)

## Overview

Brainsmith's plugin system allows you to extend the framework with custom:
- **Kernels**: Hardware operations (FPGA custom ops)
- **Steps**: Build pipeline transformations
- **Backends**: Hardware implementations (HLS, RTL)

**Key Features:**
- **Lazy Loading**: Components imported only when used (fast startup)
- **Unified Pattern**: Core brainsmith and user plugins use identical structure
- **Source Isolation**: Plugins organized by source (brainsmith, finn, project, user)
- **Zero Ceremony**: No explicit registration needed with decorators

## Quick Start

### 1. Create Plugin Directory

```bash
# Project-specific plugins
mkdir -p plugins

# Or user-global plugins
mkdir -p ~/.brainsmith/plugins
```

### 2. Create __init__.py with COMPONENTS

```python
# plugins/__init__.py
from brainsmith.plugin_helpers import create_lazy_module

COMPONENTS = {
    'kernels': {
        'MyKernel': '.my_kernel',       # Maps name to module path
    },
    'backends': {
        'MyKernel_hls': '.my_backend',
    },
    'steps': {
        'my_step': '.my_step',
    },
}

# This one line does all the magic
__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

### 3. Create Your Components

```python
# plugins/my_kernel.py
from brainsmith import kernel
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

@kernel
class MyKernel(HWCustomOp):
    op_type = "MyKernel"

    def get_nodeattr_types(self):
        return {"param1": ["i", "required"]}

    def execute_node(self, context, graph):
        # Your hardware operation logic
        pass
```

### 4. Use Your Plugin

```python
from brainsmith import get_kernel

# Lazy loaded on first access
MyKernel = get_kernel('MyKernel')  # or 'project:MyKernel'
```

## Plugin Structure

### Directory Layout

```
plugins/                          # Your plugin root
├── __init__.py                   # COMPONENTS dict + lazy loading setup
├── my_kernel.py                  # Kernel implementation
├── my_backend.py                 # Backend implementation
├── my_step.py                    # Step implementation
└── utils/                        # Optional: shared utilities
    └── helpers.py
```

### The __init__.py Pattern

**Every plugin needs this structure:**

```python
from brainsmith.plugin_helpers import create_lazy_module

COMPONENTS = {
    'kernels': {
        'KernelName': '.module_path',     # Relative import path
    },
    'backends': {
        'BackendName': '.module_path',
    },
    'steps': {
        'step_name': '.module_path',
    },
}

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

**Key Points:**
- Module paths are relative (start with `.`)
- Component names must match class/function names
- Lazy loading is automatic
- No manual imports needed!

## Creating Kernels

Kernels are FPGA hardware operations that extend ONNX with custom ops.

### Basic Kernel

```python
# plugins/layernorm_custom.py
from brainsmith import kernel
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

@kernel
class LayerNormCustom(HWCustomOp):
    """Custom LayerNorm implementation with special optimizations."""

    op_type = "LayerNormCustom"  # ONNX op_type

    def get_nodeattr_types(self):
        """Define node attributes."""
        return {
            "epsilon": ["f", "required", 1e-5],
            "normalized_shape": ["ints", "required"],
        }

    def make_shape_compatible_op(self, model):
        """Handle shape compatibility."""
        pass

    def infer_node_datatype(self, model):
        """Infer output data types."""
        pass

    def execute_node(self, context, graph):
        """Execute during simulation."""
        node = self.onnx_node
        # Your execution logic
        pass

    def verify_node(self):
        """Verify node is valid."""
        pass
```

### Kernel with Infer Transform

Infer transforms convert ONNX ops to your custom kernel.

```python
# plugins/my_kernel.py
from brainsmith import kernel
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.transformation.base import Transformation

@kernel(infer_transform=InferMyKernel)  # Optional: specify transform
class MyKernel(HWCustomOp):
    op_type = "MyKernel"
    # ... implementation

# Define the infer transform
class InferMyKernel(Transformation):
    """Convert ONNX ops to MyKernel."""

    def apply(self, model):
        graph = model.graph
        for node in graph.node:
            if node.op_type == "SomeONNXOp":
                # Check if this node can become MyKernel
                if self._can_convert(node):
                    self._convert_to_my_kernel(model, node)
        return (model, False)
```

### Register in COMPONENTS

```python
# plugins/__init__.py
COMPONENTS = {
    'kernels': {
        'LayerNormCustom': '.layernorm_custom',
        'MyKernel': '.my_kernel',
    }
}
```

## Creating Steps

Steps are build pipeline transformations that modify ONNX models.

### Basic Step

```python
# plugins/optimize_weights.py
from brainsmith import step

@step
def optimize_weights(model, cfg):
    """Optimize model weights for hardware deployment.

    Args:
        model: ModelWrapper (ONNX model)
        cfg: Build configuration

    Returns:
        Modified model
    """
    # Apply your optimization
    for node in model.graph.node:
        if node.op_type == "Conv":
            # Optimize convolution weights
            pass

    return model
```

### Step with Configuration

```python
# plugins/custom_transform.py
from brainsmith import step
import logging

logger = logging.getLogger(__name__)

@step
def custom_transform(model, cfg):
    """Apply custom transformation based on config.

    Config parameters:
        - custom_transform.enabled: bool
        - custom_transform.threshold: float
    """
    if not cfg.get('custom_transform.enabled', True):
        logger.info("Custom transform disabled, skipping")
        return model

    threshold = cfg.get('custom_transform.threshold', 0.5)
    logger.info(f"Applying custom transform with threshold={threshold}")

    # Your transformation logic
    model = model.transform(YourTransformation(threshold))

    return model
```

### Composable Steps

```python
# plugins/preprocessing.py
from brainsmith import step
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_shapes import InferShapes

@step
def preprocess_model(model, cfg):
    """Run standard preprocessing pipeline."""
    transforms = [
        FoldConstants(),
        InferShapes(),
        # Your custom transforms
    ]

    for transform in transforms:
        model = model.transform(transform)

    return model
```

### Register in COMPONENTS

```python
# plugins/__init__.py
COMPONENTS = {
    'steps': {
        'optimize_weights': '.optimize_weights',
        'custom_transform': '.custom_transform',
        'preprocess_model': '.preprocessing',
    }
}
```

## Creating Backends

Backends implement hardware code generation for kernels (HLS or RTL).

### HLS Backend

```python
# plugins/my_kernel_hls.py
from brainsmith import backend
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from .my_kernel import MyKernel

@backend
class MyKernel_hls(MyKernel, HLSBackend):
    """HLS implementation of MyKernel."""

    target_kernel = 'project:MyKernel'  # Which kernel this implements
    language = 'hls'

    def generate_params(self, model, path):
        """Generate HLS parameters."""
        # Extract kernel parameters
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")

        # Write to params.h
        code = f"""
        #define PE {pe}
        #define SIMD {simd}
        """
        with open(f"{path}/params.h", "w") as f:
            f.write(code)

    def execute_node(self, context, graph):
        """Execute in simulation."""
        # Call parent kernel execution
        super().execute_node(context, graph)

    def code_generation_ipi(self):
        """Generate Vivado IPI integration."""
        # Generate Vivado block design code
        pass
```

### RTL Backend

```python
# plugins/my_kernel_rtl.py
from brainsmith import backend
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from .my_kernel import MyKernel

@backend
class MyKernel_rtl(MyKernel, RTLBackend):
    """RTL implementation of MyKernel."""

    target_kernel = 'project:MyKernel'
    language = 'rtl'

    def generate_params(self, model, path):
        """Generate RTL parameters."""
        # Write Verilog/VHDL parameters
        pass
```

### Register in COMPONENTS

```python
# plugins/__init__.py
COMPONENTS = {
    'backends': {
        'MyKernel_hls': '.my_kernel_hls',
        'MyKernel_rtl': '.my_kernel_rtl',
    }
}
```

## Plugin Discovery

### Discovery Sources

Plugins are discovered from multiple sources in order:

1. **Core** (`brainsmith`, `finn`, `qonnx`) - Built-in components
2. **Project** (`{project_dir}/plugins/`) - Project-specific plugins
3. **User** (`~/.brainsmith/plugins/`) - Personal plugins
4. **Custom** - Additional sources from config

### Configuration

```yaml
# brainsmith_config.yaml
plugin_sources:
  project: ./plugins           # Project plugins
  team: /shared/team-plugins   # Shared team plugins
  user: ~/.brainsmith/plugins  # User plugins
```

### Component Naming

Components are namespaced by source:

```python
# Short name (uses default_source from config)
get_kernel('LayerNorm')        # → brainsmith:LayerNorm

# Explicit source
get_kernel('project:MyKernel') # → project:MyKernel
get_kernel('finn:MVAU')        # → finn:MVAU
get_kernel('user:CustomOp')    # → user:CustomOp
```

### Listing Components

```python
from brainsmith import list_kernels, list_steps, list_backends

# List all
kernels = list_kernels()
print(kernels)  # ['brainsmith:LayerNorm', 'finn:MVAU', 'project:MyKernel', ...]

# Filter by source
project_kernels = list_kernels(source='project')
user_steps = list_steps(source='user')
```

## Best Practices

### 1. Use the Unified Pattern

✅ **Do:**
```python
# plugins/__init__.py
from brainsmith.plugin_helpers import create_lazy_module

COMPONENTS = {
    'kernels': {'MyKernel': '.my_kernel'},
}

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

❌ **Don't:**
```python
# Manual imports (eager loading, slow startup)
from .my_kernel import MyKernel
from brainsmith.registry import registry
registry.kernel(MyKernel)
```

### 2. Use Decorators

✅ **Do:**
```python
from brainsmith import kernel

@kernel
class MyKernel(HWCustomOp):
    pass
```

❌ **Don't:**
```python
class MyKernel(HWCustomOp):
    pass

# Manual registration
registry.kernel(MyKernel)
```

### 3. Keep Plugin Structure Flat

✅ **Do:**
```
plugins/
├── __init__.py
├── kernel_a.py
├── kernel_b.py
└── step_x.py
```

❌ **Avoid:**
```
plugins/
├── __init__.py
├── kernels/
│   ├── __init__.py  # Extra complexity
│   └── kernel_a.py
└── steps/
    ├── __init__.py
    └── step_x.py
```

### 4. Name Components Clearly

✅ **Do:**
- `LayerNormOptimized` (descriptive)
- `ConvBatchNormFused` (clear purpose)
- `optimize_memory_layout` (verb for steps)

❌ **Don't:**
- `LNO` (unclear acronym)
- `MyOp` (not descriptive)
- `step1` (meaningless name)

### 5. Document Your Components

```python
@kernel
class LayerNormOptimized(HWCustomOp):
    """Optimized LayerNorm with 2x throughput.

    Optimizations:
    - Uses parallel SIMD lanes
    - Reduces memory bandwidth by 50%
    - Supports epsilon values from 1e-10 to 1e-5

    Attributes:
        epsilon: Normalization epsilon (default: 1e-5)
        normalized_shape: Shape to normalize over

    Example:
        >>> model = model.transform(InferLayerNormOptimized())
        >>> ln = get_kernel('LayerNormOptimized')
    """
```

## Advanced Topics

### Conditional Component Loading

```python
# plugins/__init__.py
COMPONENTS = {
    'kernels': {
        'BasicKernel': '.basic',
    }
}

# Add advanced kernels only if dependencies available
try:
    import specialized_lib
    COMPONENTS['kernels']['AdvancedKernel'] = '.advanced'
except ImportError:
    pass

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

### Custom Domain Registration

For QONNX to find your kernels in ONNX models:

```python
# In your kernel module or __init__.py
from qonnx.custom_op.registry import add_domain_alias

# Map ONNX domain to your plugin
add_domain_alias("my.custom.domain", "plugins")
```

Then in ONNX:
```python
node.domain = "my.custom.domain"
node.op_type = "MyKernel"
```

### Shared Utilities

```python
# plugins/utils.py
def shared_helper():
    """Utility used by multiple components."""
    pass

# plugins/my_kernel.py
from .utils import shared_helper

@kernel
class MyKernel(HWCustomOp):
    def execute_node(self, context, graph):
        result = shared_helper()
```

### Testing Plugins

```python
# tests/test_my_kernel.py
import pytest
from brainsmith import get_kernel

def test_my_kernel_exists():
    """Test kernel can be loaded."""
    MyKernel = get_kernel('project:MyKernel')
    assert MyKernel is not None

def test_my_kernel_execution():
    """Test kernel execution."""
    from qonnx.core.modelwrapper import ModelWrapper

    # Create test model with your kernel
    model = ModelWrapper("test_model.onnx")

    # Execute
    output = model.transform(ExecutionPass())

    # Verify results
    assert output is not None
```

## Troubleshooting

### "Kernel not found" Error

**Problem:**
```python
KeyError: "Kernel 'MyKernel' not found"
```

**Solutions:**
1. Check COMPONENTS dict has correct name
2. Check module path is correct (relative, starts with `.`)
3. Check class name matches COMPONENTS key
4. Verify `@kernel` decorator is present
5. Check plugin directory in config

```python
# Debug: List all kernels
from brainsmith import list_kernels
print(list_kernels())  # Is yours listed?
```

### Slow Startup

**Problem:** Plugin discovery takes several seconds

**Solution:** Use lazy loading pattern!

❌ **Eager (slow):**
```python
# Imports everything immediately
from .kernel1 import Kernel1
from .kernel2 import Kernel2  # Imports torch, numpy, scipy...
```

✅ **Lazy (fast):**
```python
COMPONENTS = {'kernels': {'Kernel1': '.kernel1', 'Kernel2': '.kernel2'}}
__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

### Import Errors

**Problem:**
```python
ImportError: cannot import name 'MyKernel' from 'plugins'
```

**Solution:** Check `__getattr__` and `__dir__` are defined:

```python
# Must have both!
__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
```

### Decorator Not Working

**Problem:** Component not registered despite `@kernel` decorator

**Solution:** Module must be imported for decorator to fire:

```python
# In __init__.py - either:

# Option 1: Lazy (recommended)
COMPONENTS = {'kernels': {'MyKernel': '.my_kernel'}}
__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)

# Option 2: Eager
from . import my_kernel  # Triggers @kernel decorator
```

### QONNX Can't Find Op

**Problem:** QONNX raises "Op 'MyKernel' not found in domain"

**Solution:** Ensure domain matches module path:

```python
# ONNX model
node.domain = "plugins"  # or "your_package_name"
node.op_type = "MyKernel"

# QONNX will call: getattr(plugins, "MyKernel")
# Your __getattr__ must handle this!
```

## Performance Tips

### Fast Listing

```python
# Fast - no imports
kernels = list_kernels()  # <1ms

# Slow - imports everything
from brainsmith.kernels import *
```

### Lazy Dependencies

```python
# plugins/expensive_kernel.py
from brainsmith import kernel
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

@kernel
class ExpensiveKernel(HWCustomOp):
    def execute_node(self, context, graph):
        # Import heavy deps only when actually used
        import tensorflow as tf  # 2s import
        import scipy.signal       # 0.5s import

        # Use them
```

### Benchmark

```python
import time

# Measure listing time
start = time.time()
from brainsmith import list_kernels
kernels = list_kernels()
print(f"Listing: {time.time() - start:.3f}s")  # ~0.001s

# Measure first access
start = time.time()
MyKernel = get_kernel('MyKernel')
print(f"First access: {time.time() - start:.3f}s")  # ~2s with torch

# Measure cached access
start = time.time()
MyKernel = get_kernel('MyKernel')
print(f"Cached: {time.time() - start:.3f}s")  # ~0.001s
```

## Summary

**The Pattern:**
1. Create `plugins/__init__.py` with COMPONENTS dict
2. Use `create_lazy_module(COMPONENTS, __name__)`
3. Add `@kernel`, `@step`, or `@backend` decorators to your components
4. Components loaded lazily on first access

**Key Benefits:**
- ⚡ Fast startup (lazy loading)
- 🎯 Simple pattern (one way to do it)
- 🔌 Extensible (add plugins anywhere)
- 🚀 Performant (only load what you use)

**Next Steps:**
- See `/plugins` directory for working examples
- Check `brainsmith/kernels` for core implementation
- Read FINN docs for HWCustomOp details
- Explore QONNX for transformation patterns

Arete.
