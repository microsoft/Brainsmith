############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

# SDIM Architecture Guide

## Overview

The SDIM (Streaming Dimensions) architecture represents a fundamental correction to how Brainsmith models parallelism in hardware kernels. This guide explains the new architecture and how to migrate from the old iPar-based system.

## Key Concepts

### 1. SDIM vs iPar

**Old Model (iPar)**:
- Single scalar value representing "interface parallelism"
- Ambiguous meaning - total elements? per dimension?
- Incorrect assumption that outputs can be independently configured

**New Model (SDIM)**:
- Multi-dimensional representation: one value per tensor dimension
- Clear semantics: elements streamed per cycle per dimension
- Only inputs have configurable SDIM; outputs are computed

### 2. Separate Input/Output Classes

**Old Model**:
```python
# Single InterfaceModel for everything
interface = InterfaceModel(...)
if interface.direction == InterfaceDirection.OUTPUT:
    # Special handling
```

**New Model**:
```python
# Type-safe separate classes
input = InputInterface(...)   # Has sdim property
output = OutputInterface(...)  # Has streaming_rate property
```

### 3. No WEIGHT Interface Type

**Old Model**:
- `InterfaceDirection.WEIGHT` for weights
- `InterfaceDirection.CONFIG` for configuration
- Special handling for different types

**New Model**:
- Everything is either INPUT or OUTPUT
- Weights are just inputs
- Configuration values are just inputs

## Core Classes

### InputInterface

Represents an input interface with configurable SDIM:

```python
class InputInterface:
    tensor_dims: Shape        # Full tensor dimensions
    block_dims: Shape        # Block dimensions
    sdim: Shape             # Streaming dimensions (configurable)
    streaming_bandwidth: int # Total elements per cycle
```

### OutputInterface

Represents an output interface with computed streaming:

```python
class OutputInterface:
    tensor_dims: Shape      # Full tensor dimensions
    block_dims: Shape      # Block dimensions
    streaming_rate: int    # Elements per cycle (computed)
```

### KernelModelV2

Manages kernel with separate input/output lists:

```python
class KernelModelV2:
    input_models: List[InputInterface]
    output_models: List[OutputInterface]
    
    def configure_sdim(self, config: Dict[str, Union[int, Shape]]):
        """Configure SDIM for inputs only"""
    
    def compute_output_rates(self):
        """Compute output streaming rates"""
```

## SDIM Configuration Methods

### 1. Uniform Configuration

All dimensions get the same value:

```python
kernel.configure_sdim({"input": 16})
# Result: sdim = (16, 16, 16) for 3D tensor
```

### 2. Per-Dimension Configuration

Specify each dimension:

```python
kernel.configure_sdim({"input": [8, 16, 32]})
# Result: sdim = (8, 16, 32)
```

### 3. Sparse Configuration

Specify only some dimensions:

```python
kernel.configure_sdim({"input": {0: 8, 2: 32}})
# Result: sdim = (8, 1, 32) - unspecified dims default to 1
```

## Relationship Types

### EQUAL Relationship

All dimensions must match:

```python
kernel_def.add_relationship("A", "B", RelationType.EQUAL)
# If A.sdim = (8, 16), then B.sdim must be (8, 16)
```

### DEPENDENT Relationship

Specific dimension dependencies:

```python
# Copy dependency
kernel_def.add_relationship(
    "A", "B", RelationType.DEPENDENT,
    source_dim=1, target_dim=0,
    dependency_type="copy"
)
# B[0] = A[1]

# Scaled dependency
kernel_def.add_relationship(
    "input", "output", RelationType.DEPENDENT,
    source_dim=0, target_dim=0,
    dependency_type="scaled",
    scale_factor=0.5
)
# output[0] = input[0] * 0.5

# Min dependency
kernel_def.add_relationship(
    "A", "C", RelationType.DEPENDENT,
    source_dim=0, target_dim=0,
    dependency_type="min",
    other_source="B",
    other_source_dim=0
)
# C[0] = min(A[0], B[0])
```

## Migration Guide

### Step 1: Update Imports

```python
# Old
from brainsmith.core.dataflow.core import (
    InterfaceDefinition, InterfaceModel,
    KernelDefinition, KernelModel
)

# New
from brainsmith.core.dataflow.core import (
    InputDefinition, OutputDefinition,
    InputInterface, OutputInterface,
    KernelDefinitionV2, KernelModelV2
)
```

### Step 2: Update Interface Definitions

```python
# Old
kernel_def.add_interface(InterfaceDefinition(
    name="weights",
    direction=InterfaceDirection.WEIGHT,
    ...
))

# New
kernel_def.add_input(InputDefinition(
    name="weights",  # Just an input!
    ...
))
```

### Step 3: Update Kernel Model Creation

```python
# Old
kernel = KernelModel(
    interface_models=[...],  # Mixed list
    ...
)

# New
kernel = KernelModelV2(
    input_models=[...],     # Separate lists
    output_models=[...],
    definition=kernel_def
)
```

### Step 4: Update Parallelism Configuration

```python
# Old
kernel.apply_parallelism({
    "input": 16,
    "weights": 8,
    "output": 16  # Wrong!
})

# New
kernel.configure_sdim({
    "input": 16,
    "weights": 8
    # No output configuration!
})
```

### Step 5: Update SDIM Access

```python
# Old
if interface.direction == InterfaceDirection.OUTPUT:
    rate = interface.streaming_bandwidth
else:
    interface.ipar = 16

# New
# Inputs have sdim
input_model.sdim = (8, 16)

# Outputs have streaming_rate
rate = output_model.streaming_rate
```

## Common Patterns

### Matrix Multiply

```python
kernel_def = KernelDefinitionV2(name="matmul")

# Inputs
kernel_def.add_input(InputDefinition("A", ...))  # [M, K]
kernel_def.add_input(InputDefinition("B", ...))  # [K, N]

# Output
kernel_def.add_output(OutputDefinition("C", ...))  # [M, N]

# K dimensions must match
kernel_def.add_relationship(
    "A", "B", RelationType.DEPENDENT,
    source_dim=1, target_dim=0,
    dependency_type="copy"
)

# Configure
kernel.configure_sdim({
    "A": [8, 16],   # Stream 8x16 patches
    # B[0] will be 16 from relationship
})
```

### Convolution

```python
kernel_def = KernelDefinitionV2(name="conv2d")

# All inputs (no WEIGHT type)
kernel_def.add_input(InputDefinition("ifmap", ...))     # [C, H, W]
kernel_def.add_input(InputDefinition("kernels", ...))   # [OC, IC, K, K]
kernel_def.add_input(InputDefinition("bias", ...))      # [OC]

# Output
kernel_def.add_output(OutputDefinition("ofmap", ...))   # [OC, H, W]

# Configure streaming
kernel.configure_sdim({
    "ifmap": [8, 1, 1],      # 8 channels at a time
    "kernels": [16, 8, 3, 3] # 16 output, 8 input channels
})
```

## Best Practices

1. **Think in dimensions**: SDIM values correspond to tensor dimensions
2. **Start with inputs**: Configure input SDIM first, let relationships propagate
3. **Use DEPENDENT wisely**: For dimension-specific constraints
4. **No output config**: Output rates are always computed
5. **Weights are inputs**: No special handling needed

## Debugging Tips

1. Use `get_sdim_parameters()` to see what can be configured
2. Check `get_sdim_state()` to see current configuration
3. Validation will catch conflicting constraints
4. Output `streaming_rate` is computed after input configuration

## Summary

The SDIM architecture provides:
- **Clarity**: Multi-dimensional streaming is explicit
- **Correctness**: Matches actual hardware behavior
- **Type Safety**: Can't misconfigure outputs
- **Flexibility**: Multiple configuration methods
- **Simplicity**: No special interface types

This clean break from the old model eliminates confusion and provides a solid foundation for hardware kernel modeling.