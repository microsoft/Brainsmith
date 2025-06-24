############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

# Kernel Dataflow Modeling System Architecture

**Version:** 2.0 (SDIM Architecture)  
**Date:** 2025-06-24  
**Module:** `brainsmith.core.dataflow.core`

## Table of Contents

1. [Overview](#overview)
2. [System Purpose](#system-purpose)
3. [Core Concepts](#core-concepts)
4. [Architecture Components](#architecture-components)
5. [Data Hierarchy](#data-hierarchy)
6. [SDIM Parallelism Model](#sdim-parallelism-model)
7. [Relationship System](#relationship-system)
8. [Usage Examples](#usage-examples)
9. [Design Rationale](#design-rationale)

## Overview

The Kernel Dataflow Modeling System provides a high-level abstraction for representing hardware accelerator kernels that execute on FPGAs. It bridges the gap between PyTorch neural network models and low-level RTL (Register Transfer Level) hardware descriptions, enabling automated synthesis of efficient FPGA implementations.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PyTorch Model  │ ──▶ │ Dataflow Model   │ ──▶ │  RTL/Hardware   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              ▲
                              │ This System
```

## System Purpose

### Primary Goals

1. **Abstract Hardware Complexity**: Provide a clean API for defining hardware kernels without RTL knowledge
2. **Enable Optimization**: Support design space exploration through parameterizable architectures
3. **Ensure Correctness**: Validate interface compatibility and data flow constraints
4. **Model Performance**: Predict hardware characteristics before synthesis

### Key Capabilities

- **Interface-Wise Modeling**: Focus on kernel interfaces and their data flow patterns
- **Streaming Architecture**: Model how data streams through hardware pipelines
- **Constraint Validation**: Ensure dimensional compatibility and data type consistency
- **Performance Estimation**: Calculate bandwidth, latency, and resource usage

## Core Concepts

### Definition vs Model Pattern

The system uses a two-layer architecture:

```
┌──────────────────────┐         ┌──────────────────────┐
│   KernelDefinition   │ ──────▶ │     KernelModel      │
│ (Static Schema)      │ create  │ (Runtime Instance)   │
│                      │         │                      │
│ • Interface specs    │         │ • Concrete shapes    │
│ • Relationships      │         │ • SDIM config        │
│ • Constraints        │         │ • Performance metrics│
└──────────────────────┘         └──────────────────────┘
```

**Definition**: Immutable template describing what a kernel CAN do  
**Model**: Concrete instance with specific tensor shapes and configurations

### Interface Types

```
┌─────────────────┐           ┌──────────────────┐
│ InputDefinition │           │ OutputDefinition │
├─────────────────┤           ├──────────────────┤
│ • Data type     │           │ • Data type      │
│ • Block dims    │           │ • Block dims     │
│ • Granularity   │           │ • Granularity    │
│ • ONNX layout   │           │ • ONNX layout    │
└─────────────────┘           └──────────────────┘
         │                             │
         ▼                             ▼
┌─────────────────┐           ┌──────────────────┐
│ InputInterface  │           │ OutputInterface  │
├─────────────────┤           ├──────────────────┤
│ • Tensor shape  │           │ • Tensor shape   │
│ • SDIM config   │           │ • Streaming rate │
│ • Block decomp  │           │ • Block decomp   │
│ • Bandwidth     │           │ • Bandwidth      │
└─────────────────┘           └──────────────────┘
```

## Architecture Components

### 1. Type System (`types.py`)

```python
DataType      # INT8, INT16, UINT8, FIXED, FLOAT32, etc.
Shape         # Multi-dimensional shapes with symbolic support
RelationType  # EQUAL, DEPENDENT, MULTIPLE, DIVISIBLE, COUPLED
```

### 2. Definition Layer

- **InputDefinition**: Schema for input interfaces
- **OutputDefinition**: Schema for output interfaces
- **KernelDefinition**: Container for all interface definitions and relationships

### 3. Model Layer

- **InputInterface**: Runtime model with SDIM configuration
- **OutputInterface**: Runtime model with computed streaming
- **KernelModel**: Orchestrates all interfaces and validates constraints

### 4. Relationships (`relationships.py`)

```python
DimensionRelationship:
  - source/target interfaces
  - dimension indices
  - dependency type (copy, scaled, min)
  - validation logic
```

### 5. Tiling Functions (`tiling_functions.py`)

```python
fixed_tiles(64, 32)                    # Static tiles
parameterized_tiles("TILE_M", "TILE_K") # Runtime parameters
adaptive_tiles()                        # Auto-calculated
```

## Data Hierarchy

The system models data at four granularity levels:

```
┌─────────────────────────────────────────────────────┐
│                     TENSOR                          │
│                  Shape: [512, 256]                  │
│  ┌─────────────────────────────────────────────┐   │
│  │                  BLOCKS                      │   │
│  │              Tile: [64, 32]                  │   │
│  │  ┌───────────────────────────────────────┐  │   │
│  │  │              STREAMS                   │  │   │
│  │  │           SDIM: [8, 16]                │  │   │
│  │  │  ┌─────────────────────────────────┐  │  │   │
│  │  │  │         ELEMENTS                │  │  │   │
│  │  │  │      DataType: INT8             │  │  │   │
│  │  │  └─────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Decomposition Example

For a 512×256 matrix with 64×32 blocks and SDIM=[8,16]:
- **Tensor blocks**: 8×8 grid of blocks
- **Block streams**: Each block streams as 8×2 patches
- **Elements/cycle**: 8×16 = 128 INT8 values
- **Cycles/block**: 8×2 = 16 cycles
- **Total cycles**: 8×8×16 = 1024 cycles

## SDIM Parallelism Model

SDIM (Streaming Dimensions) replaces the ambiguous iPar with explicit per-dimension streaming:

### Configuration Methods

```python
# Method 1: Uniform (all dimensions same)
kernel.configure_sdim({"input": 16})
# Result: [16, 16, 16] for 3D tensor

# Method 2: Per-dimension list
kernel.configure_sdim({"input": [8, 16, 32]})
# Result: [8, 16, 32] exactly

# Method 3: Sparse dictionary
kernel.configure_sdim({"input": {0: 8, 2: 32}})
# Result: [8, 1, 32] (dim 1 defaults to 1)
```

### Streaming Calculation

```
Stream iterations = Block_dim[i] / SDIM[i]
Streaming rate = ∏(SDIM[i]) elements/cycle
Initiation interval = ∏(Block_dim[i] / SDIM[i]) cycles
```

### Output Streaming

Outputs don't have configurable SDIM. Their streaming is computed from:
1. Input SDIM configurations
2. Kernel computation pattern
3. Interface relationships

## Relationship System

Relationships ensure interface compatibility:

### Relationship Types

```
┌────────────┬──────────────────────────────────────┐
│   EQUAL    │ All dimensions must match exactly   │
├────────────┼──────────────────────────────────────┤
│ DEPENDENT  │ Specific dimension dependencies:    │
│            │ • copy: target_dim = source_dim     │
│            │ • scaled: target = source × factor  │
│            │ • min: target = min(src1, src2)     │
├────────────┼──────────────────────────────────────┤
│  MULTIPLE  │ One dimension is multiple of other  │
├────────────┼──────────────────────────────────────┤
│ DIVISIBLE  │ One dimension divides other evenly  │
├────────────┼──────────────────────────────────────┤
│  COUPLED   │ Custom validation function          │
└────────────┴──────────────────────────────────────┘
```

### Example: Matrix Multiply

```python
# A[M,K] @ B[K,N] = C[M,N]

# K dimensions must match
kernel_def.add_relationship(
    "A", "B", RelationType.DEPENDENT,
    source_dim=1, target_dim=0,
    dependency_type="copy"
)

# Output dimensions from inputs
kernel_def.add_relationship(
    "A", "C", RelationType.DEPENDENT,
    source_dim=0, target_dim=0,
    dependency_type="copy"
)
kernel_def.add_relationship(
    "B", "C", RelationType.DEPENDENT,
    source_dim=1, target_dim=1,
    dependency_type="copy"
)
```

## Usage Examples

### Basic Kernel Definition

```python
# Define a simple elementwise operation
kernel_def = KernelDefinition(name="relu")
kernel_def.add_input(InputDefinition(
    name="x",
    dtype=DataType.INT8,
    block_dims=fixed_tiles(64, 64)
))
kernel_def.add_output(OutputDefinition(
    name="y", 
    dtype=DataType.INT8,
    block_dims=fixed_tiles(64, 64)
))

# Shapes must match
kernel_def.add_relationship("x", "y", RelationType.EQUAL)
```

### Complex Kernel with SDIM

```python
# Matrix multiply with streaming
kernel_def = KernelDefinition(name="matmul")

# Inputs with parameterized tiling
kernel_def.add_input(InputDefinition(
    "A", dtype=DataType.INT8,
    block_dims=parameterized_tiles("TILE_M", "TILE_K")
))
kernel_def.add_input(InputDefinition(
    "B", dtype=DataType.INT8,
    block_dims=parameterized_tiles("TILE_K", "TILE_N")
))

# Create runtime model
model = kernel_def.create_model(
    input_shapes={"A": (512, 256), "B": (256, 128)},
    parameters={"TILE_M": 64, "TILE_K": 32, "TILE_N": 64}
)

# Configure streaming
model.configure_sdim({
    "A": [8, 16],  # Stream 8x16 patches
    "B": [16, 8]   # B's first dim=16 due to relationship
})

# Query performance
print(f"A bandwidth: {model.get_input('A').streaming_bandwidth} bits/cycle")
print(f"Initiation interval: {model.compute_initiation_interval()} cycles")
```

## Design Rationale

### Why Definition/Model Split?

1. **Immutability**: Definitions are templates that never change
2. **Reusability**: One definition creates many models with different shapes
3. **Validation**: Constraints defined once, validated at model creation
4. **Clarity**: Clear separation between "what it can do" vs "what it's doing"

### Why SDIM over iPar?

1. **Precision**: iPar was ambiguous - which dimension does it apply to?
2. **Flexibility**: Different streaming rates per dimension
3. **Hardware mapping**: Directly corresponds to AXI stream width configuration
4. **Optimization**: Enables fine-grained streaming control

### Why Separate Input/Output Interfaces?

1. **Semantics**: Inputs have configurable streaming, outputs don't
2. **Clarity**: No need for direction flags or type checks
3. **Type safety**: Can't accidentally configure output SDIM
4. **Simplicity**: Each class does one thing well

### Why Relationships?

1. **Correctness**: Ensure dimensional compatibility
2. **Automation**: Propagate constraints through the graph
3. **Documentation**: Relationships document kernel requirements
4. **Optimization**: Constraint solver can find valid configurations

## Conclusion

The Kernel Dataflow Modeling System provides a powerful abstraction for hardware accelerator design. By separating static definitions from runtime models, supporting flexible streaming configurations, and enforcing relationships, it enables both correctness and optimization in the journey from neural networks to FPGA implementations.

The SDIM architecture represents a significant improvement over previous approaches, providing clear, multi-dimensional parallelism control that maps directly to hardware capabilities while maintaining a clean, intuitive API for users.