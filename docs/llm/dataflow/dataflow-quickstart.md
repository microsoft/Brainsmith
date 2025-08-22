# Dataflow Module Quick Start Guide

The Brainsmith Dataflow module provides type-safe abstractions for modeling hardware accelerator kernels on FPGAs. This guide will get you started quickly.

## Core Concepts

### 1. Definition vs Model Pattern
- **Definitions**: Static schemas that define constraints and relationships (what CAN be)
- **Models**: Runtime instances with concrete types and dimensions (what IS)

### 2. Interface Types
- **InputInterface**: Configurable streaming dimensions (SDIM)
- **OutputInterface**: Computed streaming rates based on kernel behavior

### 3. Tiling System
- **Block Tiling**: How tensors are divided into processing blocks
- **Stream Tiling**: How blocks are streamed per clock cycle

## Quick Examples

### Simple Element-wise Kernel (ReLU)

```python
from brainsmith.core.dataflow import (
    KernelDefinition, InputDefinition, OutputDefinition,
    DatatypeConstraintGroup, RelationType
)
from qonnx.core.datatype import DataType

# 1. Define the kernel schema
kernel_def = KernelDefinition(name="relu")

# 2. Add input with constraints
kernel_def.add_input(InputDefinition(
    name="x",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 16)],
    block_tiling=[1, ":"],          # [batch=1, full features]
    stream_tiling=[1, "SIMD"]       # Stream SIMD elements per cycle
))

# 3. Add output (same shape as input)
kernel_def.add_output(OutputDefinition(
    name="y",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 16)],
    block_tiling=[1, ":"]           # Same as input
))

# 4. Define relationships
kernel_def.add_relationship("x", "y", RelationType.EQUAL)

# 5. Create runtime model
model = kernel_def.create_model(
    input_specs={"x": ((256, 256), DataType["INT8"])},
    output_specs={"y": ((256, 256), DataType["INT8"])},
    parameter_binding={"SIMD": 16}  # 16 elements per clock
)

# 6. Configure streaming (optional)
model.configure_sdim({"x": 16})  # Stream 16 elements per cycle

# 7. Get performance metrics
metrics = model.calculate_performance_metrics()
print(f"Throughput: {metrics['aggregate']['throughput_fps']:.0f} fps")
```

### Matrix Multiply Kernel

```python
# Define kernel with multiple inputs
kernel_def = KernelDefinition(name="matmul")

# Matrix A: M×K
kernel_def.add_input(InputDefinition(
    name="A",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
    block_tiling=["TILE_M", "TILE_K"],        # Parameterized tiling
    stream_tiling=["STREAM_M", "STREAM_K"]    # Stream subdivision
))

# Matrix B: K×N
kernel_def.add_input(InputDefinition(
    name="B", 
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
    block_tiling=["TILE_K", "TILE_N"],
    stream_tiling=["STREAM_K", "STREAM_N"]
))

# Output C: M×N
kernel_def.add_output(OutputDefinition(
    name="C",
    datatype_constraints=[DatatypeConstraintGroup("INT", 32, 32)],
    block_tiling=["TILE_M", "TILE_N"]
))

# K dimensions must match
kernel_def.add_relationship(
    "A", "B", RelationType.DEPENDENT,
    source_dim=1, target_dim=0
)

# Create model with concrete parameters
model = kernel_def.create_model(
    input_specs={
        "A": ((512, 256), DataType["INT8"]),
        "B": ((256, 128), DataType["INT8"])
    },
    output_specs={"C": ((512, 128), DataType["INT32"])},
    parameter_binding={
        "TILE_M": 64, "TILE_K": 32, "TILE_N": 64,
        "STREAM_M": 8, "STREAM_K": 16, "STREAM_N": 8
    }
)
```

## Tiling Expression Types

### 1. Singleton (1)
```python
block_tiling=[1, "CHANNELS"]  # First dimension not tiled
```

### 2. Full Dimension (:)
```python
block_tiling=[":", ":"]  # Process full tensor
```

### 3. Fixed Size
```python
block_tiling=[32, 64]  # Fixed 32×64 blocks
```

### 4. Parameters
```python
block_tiling=["BATCH", "CHANNELS"]  # Runtime parameters
```

## Working with Constraints

### Datatype Constraints
```python
# Allow multiple integer widths
DatatypeConstraintGroup("INT", 8, 16)  # INT8 through INT16

# Fixed-point types
DatatypeConstraintGroup("FIXED", 16, 16)  # FIXED16 only

# Multiple constraint groups
datatype_constraints=[
    DatatypeConstraintGroup("INT", 8, 8),
    DatatypeConstraintGroup("UINT", 8, 8)
]  # Allow INT8 or UINT8
```

### Relationships
```python
# Dimension must be equal
kernel_def.add_relationship("input", "output", RelationType.EQUAL)

# Specific dimension dependency
kernel_def.add_relationship(
    "A", "B", RelationType.DEPENDENT,
    source_dim=1, target_dim=0  # A's columns = B's rows
)

# Output is multiple of input
kernel_def.add_relationship(
    "input", "output", RelationType.MULTIPLE,
    factor=4  # output 4x larger
)
```

## SDIM Configuration

### Uniform Configuration
```python
# Same SDIM for all dimensions
model.configure_sdim({"input": 16})
```

### Per-Dimension Configuration
```python
# Different SDIM per dimension
model.configure_sdim({"input": [8, 16, 1, 1]})
```

### Sparse Configuration
```python
# Only configure specific dimensions
model.configure_sdim({"input": {0: 8, 2: 32}})
```

## Performance Analysis

```python
# Get detailed metrics
metrics = model.calculate_performance_metrics(frequency_mhz=200.0)

# Input metrics
for name, inp_metrics in metrics["inputs"].items():
    print(f"{name}: {inp_metrics['streaming_bandwidth']} elem/cycle")

# Output metrics  
for name, out_metrics in metrics["outputs"].items():
    print(f"{name}: {out_metrics['streaming_rate']} elem/cycle")

# Aggregate metrics
print(f"Throughput: {metrics['aggregate']['throughput_fps']:.0f} fps")
print(f"Total bandwidth: {metrics['aggregate']['total_bandwidth_mbps']:.0f} Mbps")
```

## Integration with FINN

The dataflow models integrate directly with FINN HWCustomOp:

```python
# Extract parameters for node attributes
params = kernel_def.get_required_parameters()
# Returns: {"SIMD": "x_stream_tiling", "TILE_M": "A_block_tiling", ...}

# These map to FINN nodeattr_types
for param_name, context in params.items():
    # Generate FINN node attribute
    nodeattr_types[param_name] = ('i', False, 1)
```

## Common Patterns

### Conv2D with Channel Tiling
```python
InputDefinition(
    name="input",
    block_tiling=[1, "CH_TILES", ":", ":"],    # Tile channels only
    stream_tiling=[1, "SIMD", 1, 1]            # Stream within tiles
)
```

### Weights as Static Input
```python
InputDefinition(
    name="weights",
    block_tiling=["OUT_CH", "IN_CH", ":", ":"],
    stream_tiling=["PE", "SIMD", ":", ":"],
    is_weight=True  # Mark as weight for FINN
)
```

### Adaptive Tiling
```python
# Use full dimension when unknown
block_tiling=[1, ":", ":", ":"]  # Adapt to any spatial size

# Or use fixed tiles for known sizes
block_tiling=[1, 64, 14, 14]  # Optimized for specific size
```

## Next Steps

- See [Architecture Guide](dataflow-architecture.md) for design details
- See [Tiling Guide](dataflow-tiling-guide.md) for advanced tiling patterns
- See [API Reference](dataflow-api-reference.md) for complete API documentation
- See [Patterns Guide](dataflow-patterns.md) for more examples