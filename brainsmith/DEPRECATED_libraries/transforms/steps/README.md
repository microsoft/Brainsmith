# BrainSmith Steps Module

**North Star-aligned FINN transformation steps organized by function.**

## Overview

The BrainSmith Steps module provides a simplified, functional approach to FINN model transformations. Steps are organized by **what they do** rather than which models use them, enabling clear composition and reuse across different architectures.

## Philosophy

- **Functions Over Frameworks**: Direct function imports, no complex registry
- **Functional Organization**: Steps grouped by purpose, not model type
- **Docstring Metadata**: Simple parsing, zero decorators
- **Community Ready**: Easy contribution through simple files

## Quick Start

```python
# Direct usage - import what you need
from brainsmith.steps import cleanup_step, qonnx_to_finn_step, streamlining_step

# Simple transformation pipeline
model = cleanup_step(model, cfg)
model = qonnx_to_finn_step(model, cfg)  
model = streamlining_step(model, cfg)

# FINN compatibility - get step by name
from brainsmith.steps import get_step
step_fn = get_step("cleanup")
model = step_fn(model, cfg)
```

## Functional Organization

### 🧹 Cleanup Operations (`cleanup.py`)
- **`cleanup_step()`** - Basic ONNX cleanup (identity removal, input sorting)
- **`cleanup_advanced_step()`** - Advanced cleanup with tensor naming

### 🔄 Conversion Operations (`conversion.py`)
- **`qonnx_to_finn_step()`** - QONNX to FINN conversion with SoftMax handling

### 🌊 Streamlining Operations (`streamlining.py`)
- **`streamlining_step()`** - Absorption and reordering transformations

### ⚡ Hardware Operations (`hardware.py`)
- **`infer_hardware_step()`** - Hardware layer inference for custom operations

### ⚙️ Optimization Operations (`optimizations.py`)
- **`constrain_folding_and_set_pumped_compute_step()`** - Folding and compute optimizations

### ✅ Validation Operations (`validation.py`)
- **`generate_reference_io_step()`** - Reference IO generation for testing

### 📊 Metadata Operations (`metadata.py`)
- **`shell_metadata_handover_step()`** - Shell integration metadata extraction

### 🤖 BERT-Specific Operations (`bert.py`)
- **`remove_head_step()`** - BERT head removal (up to first LayerNorm)
- **`remove_tail_step()`** - BERT tail removal (from global_out_1 back)

## Step Dependencies

Steps automatically validate dependencies through docstring metadata:

```
qonnx_to_finn → streamlining → infer_hardware
```

## Discovery & Validation

```python
from brainsmith.steps import discover_all_steps, validate_step_sequence

# Discover all available steps
steps = discover_all_steps()
print(f"Found {len(steps)} steps")

# Validate step sequence
step_names = ["cleanup", "qonnx_to_finn", "streamlining"]
errors = validate_step_sequence(step_names)
if not errors:
    print("✅ Valid sequence!")
```

## FINN Compatibility

Maintains full compatibility with FINN's DataflowBuildConfig:

```python
# Works with existing FINN builder
from brainsmith.steps import get_step

# BrainSmith steps
step_fn = get_step("cleanup")

# Falls back to FINN built-in steps
step_fn = get_step("some_finn_step")
```

## Community Contributions

Adding custom steps is simple:

1. **Create step file** with descriptive function name
2. **Add docstring metadata** for discovery
3. **Use standard signature** `(model, cfg) -> model`

```python
def my_custom_step(model, cfg):
    """
    My custom transformation step.
    
    Category: custom
    Dependencies: [cleanup]
    Description: Custom transformation for specific use case
    """
    # Your implementation
    return model
```

## Migration from Old Registry

### Before (Enterprise Registry)
```python
from brainsmith.steps import STEP_REGISTRY
registry = STEP_REGISTRY
step_fn = registry.get_step("transformer.streamlining")
```

### After (North Star Functions)
```python
from brainsmith.steps import streamlining_step
model = streamlining_step(model, cfg)
```

## Step Metadata Format

Steps use simple docstring metadata:

```python
def example_step(model, cfg):
    """
    Brief description of what the step does.
    
    Category: functional_category
    Dependencies: [step1, step2]  # or [] for no dependencies
    Description: Detailed description for documentation
    """
    # Implementation...
```

## Transformation Pipeline Example

```python
from brainsmith.steps import (
    cleanup_step, qonnx_to_finn_step, streamlining_step, 
    infer_hardware_step, generate_reference_io_step
)

# Complete transformation pipeline
model = cleanup_step(model, cfg)
model = qonnx_to_finn_step(model, cfg)
model = streamlining_step(model, cfg)
model = infer_hardware_step(model, cfg)
model = generate_reference_io_step(model, cfg)
```

## Testing

Run the test suite to validate functionality:

```bash
python -m pytest tests/test_steps_simplification.py -v
```

## Demo

See the complete demonstration:

```bash
python steps_demo.py
```

---

*This module exemplifies North Star design principles: simple functions over complex frameworks, clear organization by purpose, and zero hidden state.*