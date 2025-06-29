############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

# Kernel Modeling System - Unified Design Document

**Version:** 3.1 (Post-cleanup)  
**Date:** 2025-01-03  
**Status:** Implementation Complete

## Executive Summary

The Kernel Modeling System provides a high-level abstraction for representing hardware accelerator kernels on FPGAs. It bridges PyTorch neural networks to RTL hardware descriptions through a clean, type-safe API. The system has been completely refactored to use QONNX types exclusively and implements the SDIM (Streaming Dimensions) architecture for precise parallelism control.

## System Architecture

### Core Design Principles

1. **Definition/Model Separation**: Static schemas (definitions) vs runtime instances (models)
2. **Type Safety**: Separate InputInterface and OutputInterface classes
3. **QONNX Integration**: Uses QONNX's DataType exclusively, no custom types
4. **Constraint-Based Validation**: Definitions specify constraints, models use concrete types
5. **SDIM Architecture**: Multi-dimensional streaming replacing ambiguous iPar

### Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (PyTorch Models, ONNX Graphs, User Code)                  │
├─────────────────────────────────────────────────────────────┤
│                    Definition Layer                          │
│  InputDefinition    OutputDefinition    KernelDefinition    │
│  (Schemas with constraints, relationships, validation)       │
├─────────────────────────────────────────────────────────────┤
│                      Model Layer                             │
│  InputInterface     OutputInterface     KernelModel          │
│  (Runtime instances with concrete types and SDIM)           │
├─────────────────────────────────────────────────────────────┤
│                     Support Layer                            │
│  Relationships      Tiling Functions    QONNX Types          │
│  (Constraints, block decomposition, type system)             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Type System (`qonnx_types.py`)

Integrates QONNX's type system with constraint validation:

```python
# QONNX types are re-exported
from brainsmith.core.dataflow.qonnx_types import (
    BaseDataType,
    DatatypeConstraintGroup,
    datatype_from_string
)

# Constraint system
@dataclass
class DatatypeConstraintGroup:
    base_type: str      # "INT", "UINT", "FIXED", "FLOAT"
    min_width: int      # Minimum bit width
    max_width: int      # Maximum bit width
```

### 2. Base Classes (`base.py`)

Defines the fundamental patterns:

```python
class BaseDefinition(ABC):
    """Static schema - what CAN be"""
    @abstractmethod
    def create_model(self, **params) -> 'BaseModel':
        pass

class BaseModel(ABC):
    """Runtime instance - what IS"""
    @abstractmethod
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        pass
```

### 3. Definition Layer

#### InputDefinition
- **Purpose**: Schema for input interfaces
- **Key Features**:
  - `datatype_constraints`: List of allowed type groups (required)
  - `block_dims_expr`: Tiling function or expression (optional)
  - `onnx_layout`: Layout hint for ONNX compatibility (optional)
  - `granularity`: Dimension constraints (optional)
  - `optional`: Whether the input is optional (default: False)
  - `rate_pattern`: Streaming rate pattern (optional)
  - `create_model()`: Requires concrete datatype

#### OutputDefinition
- **Purpose**: Schema for output interfaces
- **Key Features**:
  - Similar to InputDefinition (same optional parameters)
  - No SDIM configuration (computed from kernel behavior)
  - `streaming_rate` in models is computed, not configured

#### KernelDefinition
- **Purpose**: Container for interface definitions and relationships
- **Key Features**:
  - Separate `input_definitions` and `output_definitions`
  - `relationships`: Dimensional constraints
  - `create_model()`: Requires concrete type specifications

### 4. Model Layer

#### InputInterface
- **Purpose**: Runtime input with SDIM configuration
- **Key Features**:
  - `datatype`: Concrete QONNX type
  - `sdim`: Configurable streaming dimensions
  - `streaming_bandwidth`: Elements per cycle
  - Performance metrics calculation

#### OutputInterface
- **Purpose**: Runtime output with computed streaming
- **Key Features**:
  - `datatype`: Concrete QONNX type
  - `streaming_rate`: Computed from inputs
  - No configurable SDIM

#### KernelModel
- **Purpose**: Runtime kernel orchestrating all interfaces
- **Key Features**:
  - `configure_sdim()`: Only for inputs
  - `compute_output_rates()`: Derives output streaming
  - Relationship validation and propagation

### 5. Relationships (`relationships.py`)

Ensures interface compatibility:

```python
class RelationType(Enum):
    EQUAL = "equal"           # All dimensions match
    DEPENDENT = "dependent"   # Specific dimension dependency
    MULTIPLE = "multiple"     # Multiple relationship
    DIVISIBLE = "divisible"   # Divisibility constraint
    # ... more types

@dataclass
class DimensionRelationship:
    source_interface: str
    target_interface: str
    relation: RelationType
    source_dim: Optional[int]  # None = total size
    target_dim: Optional[int]  # None = total size
```

### 6. Tiling Functions (`tiling_functions.py`)

Flexible block dimension specification:

```python
# Fixed tiles
fixed_tiles(64, 32)

# Parameter-based
parameterized_tiles("TILE_M", "TILE_K")

# Adaptive
adaptive_tiles("conv_tiles", default=[1, 16, 14, 14])

# Memory-constrained
memory_constrained_tiles(memory_limit_bytes=1024*1024)
```

## Data Hierarchy

The system models data at four granularity levels:

1. **Tensor**: Full data for one inference (e.g., 512×256 matrix)
2. **Block**: Tile processed by kernel (e.g., 64×32 tile)
3. **Stream**: Data per clock cycle (e.g., 8×16 patch)
4. **Element**: Individual data item (e.g., INT8 value)

## SDIM Architecture

### Configuration Methods

```python
# Uniform: same for all dimensions
kernel.configure_sdim({"input": 16})
# Result: sdim = (16, 16, 16) for 3D tensor

# Per-dimension: explicit values
kernel.configure_sdim({"input": [8, 16, 32]})

# Sparse: only specified dimensions
kernel.configure_sdim({"input": {0: 8, 2: 32}})
# Result: sdim = (8, 1, 32)
```

### Key Principles

1. **Input-Only Configuration**: Only inputs have configurable SDIM
2. **Relationship Propagation**: DEPENDENT relationships propagate SDIM
3. **Output Computation**: Output rates derived from kernel behavior
4. **Validation**: Ensures SDIM doesn't exceed block dimensions

## Usage Examples

### Basic Kernel

```python
from brainsmith.core.dataflow import (
    KernelDefinition, InputDefinition, OutputDefinition,
    DatatypeConstraintGroup, RelationType
)
from qonnx.core.datatype import DataType

# Define kernel
kernel_def = KernelDefinition(name="relu")
kernel_def.add_input(InputDefinition(
    name="x",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 16)]
))
kernel_def.add_output(OutputDefinition(
    name="y",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 16)]
))
kernel_def.add_relationship("x", "y", RelationType.EQUAL)

# Create model with concrete types
model = kernel_def.create_model(
    input_specs={"x": ((256, 256), DataType["INT8"])},
    output_specs={"y": ((256, 256), DataType["INT8"])}
)

# Configure streaming
model.configure_sdim({"x": 16})
```

### Matrix Multiply

```python
from brainsmith.core.dataflow import (
    KernelDefinition, InputDefinition, OutputDefinition,
    DatatypeConstraintGroup, RelationType, parameterized_tiles
)
from qonnx.core.datatype import DataType

kernel_def = KernelDefinition(name="matmul")

# Inputs with parameterized tiling
kernel_def.add_input(InputDefinition(
    name="A",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
    block_dims_expr=parameterized_tiles("TILE_M", "TILE_K")
))
kernel_def.add_input(InputDefinition(
    name="B",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
    block_dims_expr=parameterized_tiles("TILE_K", "TILE_N")
))
kernel_def.add_output(OutputDefinition(
    name="C",
    datatype_constraints=[DatatypeConstraintGroup("INT", 32, 32)]
))

# K dimensions must match
kernel_def.add_relationship(
    "A", "B", RelationType.DEPENDENT,
    source_dim=1, target_dim=0,
    dependency_type="copy"
)

# Create and configure
model = kernel_def.create_model(
    input_specs={
        "A": ((512, 256), DataType["INT8"]),
        "B": ((256, 128), DataType["INT8"])
    },
    output_specs={"C": ((512, 128), DataType["INT32"])},
    parameter_binding={"TILE_M": 64, "TILE_K": 32, "TILE_N": 64}
)

model.configure_sdim({
    "A": [8, 16],    # Stream 8×16 patches
    "B": [16, 8]     # B[0]=16 due to relationship
})
```

## Migration from Legacy System

### Key Changes

1. **Types**: Custom DataType → QONNX DataType
2. **Interfaces**: Single Interface class → InputInterface/OutputInterface
3. **Definitions**: InterfaceDefinition → InputDefinition/OutputDefinition
4. **Models**: InterfaceModel → InputInterface/OutputInterface
5. **Parallelism**: iPar → SDIM per dimension
6. **Directions**: InterfaceDirection enum → Separate classes

### Migration Example

```python
# Old
interface = Interface(
    direction=InterfaceDirection.INPUT,
    dtype=DataType.INT8,
    ipar=16
)

# New
from brainsmith.core.dataflow import InputDefinition, DatatypeConstraintGroup
from qonnx.core.datatype import DataType

input_def = InputDefinition(
    name="input",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)]
)
input_model = input_def.create_model(
    tensor_dims=(256, 256),
    datatype=DataType["INT8"]
)
input_model.sdim = (16, 16)  # Explicit per dimension
```

## Performance Modeling

The system provides comprehensive performance metrics:

```python
metrics = model.calculate_performance_metrics()
# Returns:
{
    "inputs": {
        "A": {
            "streaming_bandwidth": 128,     # elements/cycle
            "bandwidth_bits": 1024,         # bits/cycle
            "initiation_interval": 1024     # cycles
        }
    },
    "outputs": {
        "C": {
            "streaming_rate": 64,           # elements/cycle
            "bandwidth_bits": 2048          # bits/cycle
        }
    },
    "aggregate": {
        "throughput_fps": 97656.25      # inferences/second
    }
}
```

## Future Integration

The system is designed for future RTL parser integration:

```
┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│ RTL Parser   │ ──▶ │ KernelDef    │ ──▶ │ HWCustomOp  │
│ (metadata)   │     │ (schema)     │     │ (FINN)      │
└──────────────┘     └──────────────┘     └─────────────┘
```

## Design Rationale

### Why Definition/Model Split?
- **Immutability**: Definitions are reusable templates
- **Flexibility**: One definition → many model configurations
- **Validation**: Constraints checked once at model creation
- **Clarity**: Static "what can be" vs dynamic "what is"

### Why SDIM over iPar?
- **Precision**: Per-dimension control vs ambiguous scalar
- **Hardware Mapping**: Direct correspondence to AXI widths
- **Flexibility**: Different streaming per dimension
- **Correctness**: Outputs computed, not configured

### Why Separate Input/Output?
- **Type Safety**: Can't misconfigure outputs
- **Clarity**: Different semantics and capabilities
- **Simplicity**: Each class has single responsibility
- **Performance**: Optimized for specific use cases

### Why QONNX Types?
- **Standardization**: Industry-standard type system
- **Compatibility**: Direct FINN integration
- **Completeness**: Supports all hardware types
- **Maintenance**: No custom type system to maintain

## Summary

The Kernel Modeling System provides a clean, type-safe abstraction for hardware accelerator design. Through careful separation of concerns, integration with standard types, and precise parallelism control, it enables both correctness and optimization in the path from neural networks to FPGA implementations. The system is production-ready and designed for seamless integration with RTL generation tools.