############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

# Brainsmith Kernel Modeling User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Basic Examples](#basic-examples)
4. [Tiling Strategies](#tiling-strategies)
5. [Parallelism and Performance](#parallelism-and-performance)
6. [Advanced Patterns](#advanced-patterns)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Getting Started

The Brainsmith kernel modeling system helps you define and optimize FPGA hardware kernels. It provides a flexible framework for specifying data layouts, tiling strategies, and parallelism patterns.

### Quick Start Example

```python
from brainsmith.core.dataflow.core import *

# 1. Define a simple kernel interface
input_def = InterfaceDefinition(
    name="input",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("UINT8")
)

# 2. Create a runtime model
input_model = input_def.create_model(
    tensor_dims=(128, 256)  # 128x256 matrix
)

# 3. Apply parallelism
input_model.ipar = 16  # 16-way parallel processing

print(f"Block dims: {input_model.block_dims}")    # [(128, 256)]
print(f"Stream dims: {input_model.stream_dims}")  # (16, 1)
```

## Core Concepts

### 1. Definition vs Model

The system uses a two-tier architecture:

- **Definition**: Template or schema (what's possible)
- **Model**: Concrete instance (what's actual)

```python
# Definition - the blueprint
interface_def = InterfaceDefinition(
    name="data",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("INT8"),
    block_dims_expr=parameterized_tiles("TILE_SIZE")
)

# Model - the instance
model = interface_def.create_model(
    tensor_dims=(1024,),
    parameter_binding={"TILE_SIZE": 64}
)
```

### 2. Three-Level Dimension Hierarchy

Every interface has three dimension levels:

1. **Tensor Dimensions**: Full data shape
2. **Block Dimensions**: Tiling for processing
3. **Stream Dimensions**: Parallel execution units

```
Tensor: [1024, 2048]
   ↓ (tiling)
Block:  [64, 128]
   ↓ (parallelism)
Stream: [16, 1]
```

### 3. Relationships

Kernels can define relationships between interfaces:

```python
# Matrix multiply: C = A × B
kernel_def.add_relationship("A", "B", RelationType.EQUAL,
                           source_dim=1, target_dim=0)
# Means: A.shape[1] must equal B.shape[0]
```

## Basic Examples

### Example 1: Simple Vector Addition

```python
# Define vector addition kernel
def create_vector_add():
    # Input vectors
    vec_a = InterfaceDefinition(
        name="vec_a",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("FP32")
    )
    
    vec_b = InterfaceDefinition(
        name="vec_b",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("FP32")
    )
    
    # Output vector
    vec_out = InterfaceDefinition(
        name="vec_out",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("FP32")
    )
    
    # Create kernel
    kernel = KernelDefinition(
        name="vector_add",
        interface_definitions=[vec_a, vec_b, vec_out]
    )
    
    # Add relationships (all vectors same size)
    kernel.add_relationship("vec_a", "vec_out", RelationType.EQUAL)
    kernel.add_relationship("vec_b", "vec_out", RelationType.EQUAL)
    
    return kernel

# Use the kernel
kernel_def = create_vector_add()
models = [
    kernel_def.interface_definitions[0].create_model((1024,)),
    kernel_def.interface_definitions[1].create_model((1024,)),
    kernel_def.interface_definitions[2].create_model((1024,))
]

kernel_model = KernelModel(models, kernel_def)
kernel_model.apply_parallelism({"vec_out": 32})
```

### Example 2: Matrix Multiplication

```python
def create_matmul():
    # A matrix: M×K
    a_def = InterfaceDefinition(
        name="A",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        block_dims_expr=parameterized_tiles("TILE_M", "TILE_K")
    )
    
    # B matrix: K×N
    b_def = InterfaceDefinition(
        name="B",
        direction=InterfaceDirection.WEIGHT,
        dtype=DataType.from_string("INT8"),
        block_dims_expr=parameterized_tiles("TILE_K", "TILE_N")
    )
    
    # C matrix: M×N
    c_def = InterfaceDefinition(
        name="C",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("INT32"),
        block_dims_expr=parameterized_tiles("TILE_M", "TILE_N")
    )
    
    kernel = KernelDefinition(
        name="matmul",
        interface_definitions=[a_def, b_def, c_def]
    )
    
    # Matrix multiply constraints
    kernel.add_relationship("A", "B", RelationType.EQUAL,
                           source_dim=1, target_dim=0)
    kernel.add_relationship("A", "C", RelationType.EQUAL,
                           source_dim=0, target_dim=0)
    kernel.add_relationship("B", "C", RelationType.EQUAL,
                           source_dim=1, target_dim=1)
    
    return kernel

# Create and use
params = {"TILE_M": 64, "TILE_K": 32, "TILE_N": 128}
kernel_def = create_matmul()

a_model = kernel_def.interface_definitions[0].create_model(
    (512, 1024), parameter_binding=params
)
b_model = kernel_def.interface_definitions[1].create_model(
    (1024, 256), parameter_binding=params
)
c_model = kernel_def.interface_definitions[2].create_model(
    (512, 256), parameter_binding=params
)

kernel_model = KernelModel([a_model, b_model, c_model], kernel_def)
```

### Example 3: 2D Convolution

```python
def create_conv2d():
    # Input feature map
    ifmap = InterfaceDefinition(
        name="ifmap",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("UINT8"),
        block_dims_expr=channel_major_tiling(32, "tile"),
        onnx_layout="NCHW"
    )
    
    # Convolution weights
    weights = InterfaceDefinition(
        name="weights",
        direction=InterfaceDirection.WEIGHT,
        dtype=DataType.from_string("INT8"),
        block_dims_expr=channel_major_tiling(32, "full"),
        onnx_layout="OIHW"
    )
    
    # Output feature map
    ofmap = InterfaceDefinition(
        name="ofmap",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("INT16"),
        block_dims_expr=channel_major_tiling(32, "tile"),
        onnx_layout="NCHW"
    )
    
    return KernelDefinition(
        name="conv2d",
        interface_definitions=[ifmap, weights, ofmap]
    )
```

## Tiling Strategies

### Strategy 1: Fixed Tiling

Use when tile sizes are known at definition time:

```python
# Fixed 64x64 tiles
block_dims_expr = fixed_tiles(64, 64)

# Different sizes per dimension
block_dims_expr = fixed_tiles(1, 32, 14, 14)  # NCHW tiling
```

### Strategy 2: Parameterized Tiling

Use for design space exploration:

```python
# Tiles from parameters
block_dims_expr = parameterized_tiles("TILE_H", "TILE_W")

# Usage
model = interface_def.create_model(
    tensor_dims=(224, 224),
    parameter_binding={"TILE_H": 14, "TILE_W": 14}
)
```

### Strategy 3: Adaptive Tiling

Use for runtime configuration:

```python
# Config-driven tiling
block_dims_expr = adaptive_tiles("optimization_mode", 
                                default=[32, 32])

# Usage with different configs
low_latency = interface_def.create_model(
    tensor_dims=(256, 256),
    config={"optimization_mode": [64, 64]}  # Larger tiles
)

high_throughput = interface_def.create_model(
    tensor_dims=(256, 256),
    config={"optimization_mode": [16, 16]}  # Smaller tiles
)
```

### Strategy 4: Channel-Major Tiling

Optimized for CNN operations:

```python
# Channel tiling with spatial modes
block_dims_expr = channel_major_tiling(
    channel_tile=32,      # Process 32 channels at once
    spatial_mode="tile"   # Also tile spatial dimensions
)

# Modes:
# - "full": Keep spatial dimensions intact
# - "tile": Apply standard spatial tiling
# - "line": Process one line at a time
```

### Strategy 5: Memory-Constrained Tiling

Automatically size tiles to fit memory:

```python
# Limit to 256KB with FP32 data
block_dims_expr = memory_constrained_tiles(
    memory_limit_bytes=256 * 1024,
    bytes_per_element=4
)
```

### Strategy 6: Composite Strategies

Chain multiple strategies with fallbacks:

```python
# Try adaptive, then parameterized, then fixed
block_dims_expr = composite_tiling(
    adaptive_tiles("custom_tiles"),       # First choice
    parameterized_tiles("TILE_SIZE"),     # Fallback 1
    fixed_tiles(32, 32)                   # Fallback 2
)
```

## Parallelism and Performance

### Understanding iPar

Interface parallelism (iPar) determines how many parallel processing units handle an interface:

```python
# Set parallelism
model.ipar = 16

# Stream dimensions automatically calculated
# If block_dims = [64, 128], iPar = 16:
# stream_dims = [16, 1] (64÷16=4, so 4 blocks processed per stream)
```

### Parallelism Propagation

The system automatically propagates parallelism through relationships:

```python
# Apply to output
kernel_model.apply_parallelism({"output": 32})

# Automatically propagates to related interfaces:
# - Input interfaces with EQUAL relationships
# - Following dimension mapping rules
```

### Performance Metrics

```python
# Get kernel performance
metrics = kernel_model.calculate_performance_metrics()

print(f"Total bandwidth: {metrics['total_bandwidth_mbps']} MB/s")
print(f"Initiation interval: {metrics['initiation_interval']} cycles")
print(f"Interface parallelisms: {metrics['interface_parallelisms']}")
```

### Optimization Workflow

```python
# Try different configurations
best_config = None
best_performance = 0

for tile_size in [16, 32, 64, 128]:
    for ipar in [1, 4, 8, 16]:
        # Create model with config
        model = interface_def.create_model(
            tensor_dims=(1024, 1024),
            parameter_binding={"TILE": tile_size}
        )
        model.ipar = ipar
        
        # Evaluate
        metrics = model.calculate_performance_metrics()
        performance = metrics["bandwidth_mbps"] / metrics["initiation_interval"]
        
        if performance > best_performance:
            best_performance = performance
            best_config = (tile_size, ipar)

print(f"Best config: tile={best_config[0]}, iPar={best_config[1]}")
```

## Advanced Patterns

### Pattern 1: Multi-Phase CSDF

For cyclo-static dataflow patterns:

```python
# Phase-dependent tiling
phase_tiles = phase_dependent_tiles([
    [128, 128],    # Phase 0: Large blocks
    [64, 64],      # Phase 1: Medium blocks  
    [32, 32],      # Phase 2: Small blocks
])

interface_def = InterfaceDefinition(
    name="csdf_input",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("INT16"),
    block_dims_expr=phase_tiles,
    rate_pattern=[2, 1, 1]  # Consumption pattern
)

# Create models for different phases
for phase in range(3):
    model = interface_def.create_model(
        tensor_dims=(256, 256),
        config={"csdf_phase": phase}
    )
```

### Pattern 2: Hierarchical Tiling

For multi-level memory hierarchies:

```python
def hierarchical_tiling(l2_tile, l1_tile):
    def _tiling(tensor_dims, params, config):
        # L2 cache tiling
        l2_tiles = [min(td, l2_tile) for td in tensor_dims]
        
        # L1 cache sub-tiling
        if config and config.get("enable_l1_tiling"):
            return [min(l2, l1_tile) for l2 in l2_tiles]
        return l2_tiles
    
    return _tiling

# Usage
block_dims_expr = hierarchical_tiling(l2_tile=256, l1_tile=32)
```

### Pattern 3: Dynamic Tiling

For runtime-adaptive tiling:

```python
def dynamic_tiling(tensor_dims, params, config):
    # Analyze tensor shape
    total_elements = 1
    for dim in tensor_dims:
        total_elements *= dim
    
    # Choose tiling based on size
    if total_elements > 1_000_000:
        # Large tensor: aggressive tiling
        return [min(32, d) for d in tensor_dims]
    elif total_elements > 10_000:
        # Medium tensor: moderate tiling
        return [min(64, d) for d in tensor_dims]
    else:
        # Small tensor: process full
        return [":"] * len(tensor_dims)

block_dims_expr = dynamic_tiling
```

### Pattern 4: Sparse Data Handling

For sparse computations:

```python
def sparse_aware_tiling(tensor_dims, params, config):
    sparsity = config.get("sparsity_ratio", 0.0)
    
    if sparsity > 0.8:
        # High sparsity: smaller tiles for better load balancing
        return [min(16, d) for d in tensor_dims]
    else:
        # Dense or low sparsity: standard tiling
        return [min(64, d) for d in tensor_dims]

interface_def = InterfaceDefinition(
    name="sparse_matrix",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("FP32"),
    block_dims_expr=sparse_aware_tiling
)
```

## Best Practices

### 1. Choose Appropriate Tiling

```python
# ❌ Bad: Fixed tiling for all cases
block_dims_expr = fixed_tiles(32, 32)

# ✅ Good: Flexible tiling based on use case
block_dims_expr = composite_tiling(
    adaptive_tiles("user_config"),
    parameterized_tiles("TILE"),
    memory_constrained_tiles(512*1024, 4)
)
```

### 2. Define Clear Relationships

```python
# ❌ Bad: Missing relationships
kernel = KernelDefinition(name="broken", interfaces=[a, b, c])

# ✅ Good: All relationships defined
kernel = KernelDefinition(name="correct", interfaces=[a, b, c])
kernel.add_relationship("input", "output", RelationType.EQUAL)
```

### 3. Use Meaningful Names

```python
# ❌ Bad: Generic names
InterfaceDefinition(name="in1", ...)
InterfaceDefinition(name="in2", ...)

# ✅ Good: Descriptive names
InterfaceDefinition(name="image_input", ...)
InterfaceDefinition(name="conv_weights", ...)
```

### 4. Validate Configurations

```python
# ✅ Good: Check before using
try:
    kernel_model = KernelModel(models, definition)
    kernel_model.apply_parallelism({"output": ipar})
except ValueError as e:
    print(f"Invalid configuration: {e}")
    # Try different config
```

### 5. Profile Performance

```python
# ✅ Good: Measure and optimize
configs = []
for config in design_space:
    model = create_model_with_config(config)
    metrics = model.calculate_performance_metrics()
    configs.append((config, metrics))

# Sort by performance
configs.sort(key=lambda x: x[1]["bandwidth_mbps"], reverse=True)
best_config = configs[0][0]
```

## Troubleshooting

### Common Issues

#### 1. Dimension Mismatch

**Error**: "Expected N dimensions, got M"

**Solution**: Ensure tensor dimensions match block dimension count
```python
# Check dimension counts
assert len(tensor_dims) == len(block_dims_function(tensor_dims, {}, {}))
```

#### 2. Missing Parameters

**Error**: "KeyError: 'TILE_SIZE'"

**Solution**: Provide all required parameters
```python
# List required parameters
required_params = ["TILE_M", "TILE_K", "TILE_N"]
parameter_binding = {p: 64 for p in required_params}
```

#### 3. Invalid iPar

**Error**: "iPar 32 exceeds block dimension 16"

**Solution**: Ensure iPar ≤ smallest block dimension
```python
max_ipar = min(model.block_dims[0])
model.ipar = min(desired_ipar, max_ipar)
```

#### 4. Relationship Violations

**Error**: "Relationship constraint violated"

**Solution**: Check dimension compatibility
```python
# For EQUAL relationship
assert source_model.tensor_dims[source_dim] == target_model.tensor_dims[target_dim]
```

### Debug Helpers

```python
def debug_kernel(kernel_model):
    """Print detailed kernel information"""
    print(f"Kernel: {kernel_model.definition.name}")
    print("\nInterfaces:")
    for iface in kernel_model.interface_models:
        print(f"  {iface.definition.name}:")
        print(f"    Tensor: {iface.tensor_dims}")
        print(f"    Block:  {iface.block_dims}")
        print(f"    Stream: {iface.stream_dims}")
        print(f"    iPar:   {iface.ipar}")
    
    print("\nRelationships:")
    for rel in kernel_model.definition.relationships:
        print(f"  {rel.source_interface} → {rel.target_interface}")
        print(f"    Type: {rel.type.value}")
        print(f"    Dims: {rel.source_dim} → {rel.target_dim}")
    
    print("\nPerformance:")
    metrics = kernel_model.calculate_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

# Use it
debug_kernel(kernel_model)
```

### Performance Debugging

```python
def analyze_bottlenecks(kernel_model):
    """Identify performance bottlenecks"""
    metrics = kernel_model.calculate_performance_metrics()
    
    # Find interface with highest bandwidth
    max_bw_interface = max(
        metrics["interface_bandwidths"].items(),
        key=lambda x: x[1]
    )
    
    print(f"Bandwidth bottleneck: {max_bw_interface[0]}")
    print(f"  Bandwidth: {max_bw_interface[1]:.1f} MB/s")
    
    # Check parallelism utilization
    for name, ipar in metrics["interface_parallelisms"].items():
        model = kernel_model.get_interface_model(name)
        max_ipar = min(model.block_dims[0])
        utilization = (ipar / max_ipar) * 100
        print(f"{name} parallelism: {ipar}/{max_ipar} ({utilization:.0f}%)")
```