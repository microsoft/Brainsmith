# Dataflow Tiling Guide

## Overview

The Brainsmith tiling system provides a declarative approach for specifying how tensors are divided into blocks and how blocks are streamed through hardware. This guide covers the complete tiling system in depth.

## Tiling Hierarchy

```
┌─────────────────────────────────────────────────────┐
│                  TENSOR LEVEL                       │
│            Complete Input/Output Data               │
│              e.g., [32, 256, 224, 224]              │
└──────────────────────┬──────────────────────────────┘
                       │ Block Tiling
┌──────────────────────▼──────────────────────────────┐
│                  BLOCK LEVEL                        │
│            Processing Unit for Kernel               │
│              e.g., [1, 64, 14, 14]                  │
└──────────────────────┬──────────────────────────────┘
                       │ Stream Tiling
┌──────────────────────▼──────────────────────────────┐
│                 STREAM LEVEL                        │
│            Data per Clock Cycle                     │
│              e.g., [1, 8, 1, 1]                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                ELEMENT LEVEL                        │
│            Individual Data Values                   │
│                  e.g., INT8                         │
└─────────────────────────────────────────────────────┘
```

## Expression Types

### 1. Singleton (1)

**Purpose**: No tiling for this dimension

```python
block_tiling=[1, "CHANNELS", ":", ":"]
#             ↑ Batch dimension not tiled
```

**Use Cases**:
- Batch processing one sample at a time
- Dimensions that shouldn't be divided
- Maintaining dimension structure

### 2. Full Dimension (:)

**Purpose**: Use entire tensor dimension

```python
block_tiling=[1, ":", ":", ":"]
#                ↑↑↑ All spatial dimensions processed fully
```

**Use Cases**:
- Small tensors that fit in memory
- Dimensions without locality constraints
- Adaptive to any input size

### 3. Fixed Literal

**Purpose**: Fixed tile size known at design time

```python
block_tiling=[1, 64, 14, 14]
#                ↑↑  ↑↑  ↑↑ All fixed sizes
```

**Use Cases**:
- Hardware-optimized tile sizes
- Memory bandwidth alignment
- Known input dimensions

### 4. Parameters

**Purpose**: Runtime-configurable tile sizes

```python
block_tiling=["BATCH", "CHANNELS", "TILE_H", "TILE_W"]
#              ↑↑↑↑     ↑↑↑↑↑↑↑    ↑↑↑↑↑↑   ↑↑↑↑↑↑
#              All configurable at runtime
```

**Use Cases**:
- Flexible hardware utilization
- Different model configurations
- Performance tuning

## Block vs Stream Tiling

### Block Tiling (BDIM)

Divides tensor into processing units:

```python
# Example: Conv2D with channel tiling
input_def = InputDefinition(
    name="input",
    block_tiling=[1, "CH_TILES", ":", ":"]  # Tile channels only
)

# Tensor [1, 256, 224, 224] with CH_TILES=64
# → Blocks of [1, 64, 224, 224]
# → 4 blocks total (256/64)
```

### Stream Tiling (SDIM)

Subdivides blocks for streaming:

```python
# Continue from above
input_def = InputDefinition(
    name="input",
    block_tiling=[1, "CH_TILES", ":", ":"],
    stream_tiling=[1, "SIMD", 1, 1]  # Stream SIMD channels/cycle
)

# Block [1, 64, 224, 224] with SIMD=8
# → Stream [1, 8, 1, 1] per cycle
# → 8 cycles per spatial position
# → 64/8 * 224 * 224 = 401,408 cycles per block
```

## Common Tiling Patterns

### 1. Element-wise Operations

```python
# Process all spatial dimensions, tile channels
InputDefinition(
    name="input",
    block_tiling=[1, "CH_TILES", ":", ":"],
    stream_tiling=[1, "SIMD", ":", ":"]
)
```

### 2. Matrix Multiplication

```python
# Matrix A: Tile both dimensions
InputDefinition(
    name="A",
    block_tiling=["TILE_M", "TILE_K"],
    stream_tiling=["PE_M", "PE_K"]
)

# Matrix B: Matching K dimension
InputDefinition(
    name="B", 
    block_tiling=["TILE_K", "TILE_N"],
    stream_tiling=["PE_K", "PE_N"]
)
```

### 3. Convolution Patterns

#### Channels-First (Optimized for Channel Parallelism)

```python
InputDefinition(
    name="input",
    block_tiling=[1, "CH_IN_TILES", ":", ":"],
    stream_tiling=[1, "SIMD", 1, 1]
)

InputDefinition(
    name="weights",
    block_tiling=["CH_OUT_TILES", "CH_IN_TILES", ":", ":"],
    stream_tiling=["PE", "SIMD", ":", ":"],
    is_weight=True
)
```

#### Spatial Tiling (Memory-Constrained)

```python
InputDefinition(
    name="input", 
    block_tiling=[1, ":", "TILE_H", "TILE_W"],
    stream_tiling=[1, "SIMD", 1, 1]
)
```

### 4. Depthwise Operations

```python
# Process one channel at a time
InputDefinition(
    name="input",
    block_tiling=[1, 1, ":", ":"],  # Single channel blocks
    stream_tiling=[1, 1, "SY", "SX"]  # Spatial streaming
)
```

## Advanced Tiling Strategies

### Adaptive Tiling

Support different input sizes without recompilation:

```python
# Adaptive to any spatial size
def create_adaptive_conv():
    return InputDefinition(
        name="input",
        block_tiling=[1, "CH_TILES", ":", ":"],  # Full spatial
        stream_tiling=[1, "SIMD", 1, 1]
    )

# Fixed optimal tiling for known size  
def create_optimized_conv():
    return InputDefinition(
        name="input",
        block_tiling=[1, 64, 14, 14],  # Hardware-optimal
        stream_tiling=[1, 8, 1, 1]
    )
```

### Memory-Constrained Tiling

Balance compute and memory:

```python
def create_memory_aware_tiling(memory_kb: int):
    # Calculate tile size based on memory
    elements_per_kb = 1024 // 2  # INT16
    max_tile_elements = memory_kb * elements_per_kb
    
    # Square tile for spatial dimensions
    tile_size = int(math.sqrt(max_tile_elements))
    
    return InputDefinition(
        name="input",
        block_tiling=[1, ":", tile_size, tile_size],
        stream_tiling=[1, "SIMD", 1, 1]
    )
```

### Pipeline-Oriented Tiling

Optimize for kernel pipeline depth:

```python
# Match tiling to pipeline stages
InputDefinition(
    name="input",
    block_tiling=[1, "STAGES", ":", ":"],  # STAGES = pipeline depth
    stream_tiling=[1, 1, ":", ":"]  # Full streaming within stage
)
```

## Parameter Management

### Automatic Parameter Extraction

```python
kernel_def = KernelDefinition(name="conv")
kernel_def.add_input(InputDefinition(
    name="input",
    block_tiling=["BATCH", "CH_IN"],
    stream_tiling=[1, "SIMD"]
))

# Extract all parameters
params = kernel_def.get_required_parameters()
# → {"BATCH": "input_block_tiling",
#    "CH_IN": "input_block_tiling", 
#    "SIMD": "input_stream_tiling"}
```

### Shared Parameters

```python
# Parameters shared across interfaces
input_def = InputDefinition(
    name="input",
    block_tiling=[1, "CHANNELS", ":", ":"]
)

weight_def = InputDefinition(
    name="weights",
    block_tiling=["FILTERS", "CHANNELS", 3, 3],  # Shares CHANNELS
    is_weight=True
)

# Kernel tracks parameter usage
params = kernel_def.get_required_parameters()
# → {"CHANNELS": "input_block_tiling_and_weights_block_tiling", ...}
```

## Tiling Validation

### Compile-Time Validation

```python
# TilingSpec validates expressions
spec = TilingSpec([1, "SIMD", ":", 32])
params = spec.get_parameters()  # → {"SIMD"}

# Validate literal values
errors = spec.validate_against_shape([1, 128, 224, 224])
# Checks: 224 % 32 == 0 ✓
```

### Runtime Validation

```python
# Parameter resolution checks
model = kernel_def.create_model(
    input_specs={...},
    parameter_binding={"SIMD": 16, "CH_TILES": 64}
)
# Validates:
# - All parameters provided
# - Values are positive integers
# - Tiles divide dimensions evenly
```

## Performance Implications

### Bandwidth Calculation

```python
# Streaming bandwidth = product of SDIM
sdim = [1, 8, 1, 1]
bandwidth = 1 * 8 * 1 * 1 = 8 elements/cycle

# Total bandwidth in bits
bandwidth_bits = bandwidth * datatype.bitwidth()
# For INT8: 8 * 8 = 64 bits/cycle
```

### Initiation Interval

```python
# Cycles to process one block
block_elements = prod(block_dims)
stream_elements = prod(sdim)
initiation_interval = block_elements / stream_elements

# Example: [1, 64, 14, 14] block, [1, 8, 1, 1] stream
# II = (1*64*14*14) / (1*8*1*1) = 1,568 cycles
```

### Optimization Guidelines

1. **Maximize Streaming Bandwidth**
   ```python
   # Good: High parallelism
   stream_tiling=[1, 16, 2, 2]  # 64 elements/cycle
   
   # Poor: Low parallelism  
   stream_tiling=[1, 1, 1, 1]   # 1 element/cycle
   ```

2. **Align to Hardware Resources**
   ```python
   # Match DSP/PE counts
   stream_tiling=["PE", "SIMD"]  # PE=8, SIMD=8 → 64 MACs
   ```

3. **Balance Memory and Compute**
   ```python
   # Tile size ∝ on-chip memory
   block_tiling=[1, 64, 32, 32]  # Fits in BRAM
   ```

## Real-World Examples

### ResNet50 First Layer

```python
# 224×224 RGB input, 64 7×7 filters, stride 2
input_def = InputDefinition(
    name="input",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
    block_tiling=[1, 3, ":", ":"],     # Full RGB channels
    stream_tiling=[1, 3, 1, 1]         # Process all 3 channels
)

weight_def = InputDefinition(
    name="weights",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
    block_tiling=[64, 3, 7, 7],        # All weights fit
    stream_tiling=["PE", 3, 1, 1],     # PE parallel filters
    is_weight=True
)

output_def = OutputDefinition(
    name="output",
    datatype_constraints=[DatatypeConstraintGroup("INT", 32, 32)],
    block_tiling=[1, "PE", 112, 112]   # Tiled output channels
)
```

### MobileNet Depthwise

```python
# Depthwise 3×3, 32 channels
input_def = InputDefinition(
    name="input",
    block_tiling=[1, 32, ":", ":"],    # All channels
    stream_tiling=[1, "CH_PAR", 1, 1]  # Parallel channels
)

weight_def = InputDefinition(
    name="weights", 
    block_tiling=[32, 1, 3, 3],        # One filter per channel
    stream_tiling=["CH_PAR", 1, 1, 1],
    is_weight=True
)
```

### Transformer Attention

```python
# Q, K, V projections
q_def = InputDefinition(
    name="queries",
    block_tiling=["SEQ_TILES", "D_MODEL"],
    stream_tiling=["SEQ_PAR", "D_PAR"]
)

k_def = InputDefinition(
    name="keys",
    block_tiling=["SEQ_TILES", "D_MODEL"],
    stream_tiling=["SEQ_PAR", "D_PAR"]
)

scores_def = OutputDefinition(
    name="scores",
    block_tiling=["SEQ_TILES", "SEQ_TILES"]
)
```

## Debugging Tiling Issues

### Common Problems

1. **Dimension Mismatch**
   ```python
   # Error: Tiling spec has wrong dimensions
   tensor_shape = [32, 128, 224, 224]  # 4D
   block_tiling = [32, 128]            # 2D - ERROR!
   
   # Fix: Match dimensions
   block_tiling = [32, 128, ":", ":"]  # 4D
   ```

2. **Indivisible Tiles**
   ```python
   # Error: 224 not divisible by 32
   block_tiling = [1, ":", 32, 32]
   tensor_shape = [1, 3, 224, 224]  # 224 % 32 ≠ 0
   
   # Fix: Use divisible size or full dimension
   block_tiling = [1, ":", 14, 14]  # 224 % 14 = 0 ✓
   ```

3. **Missing Parameters**
   ```python
   # Error: Parameter not provided
   block_tiling = ["BATCH", "CHANNELS"]
   parameter_binding = {"BATCH": 1}  # Missing CHANNELS!
   
   # Fix: Provide all parameters
   parameter_binding = {"BATCH": 1, "CHANNELS": 64}
   ```

### Validation Tools

```python
# Check tiling validity
def validate_tiling(tensor_shape, block_tiling, stream_tiling, params):
    # Create specs
    block_spec = TilingSpec(block_tiling)
    stream_spec = TilingSpec(stream_tiling)
    
    # Validate dimensions
    block_errors = block_spec.validate_against_shape(tensor_shape)
    if block_errors:
        print(f"Block tiling errors: {block_errors}")
        return False
    
    # Resolve block dimensions
    block_dims = block_spec.resolve(tensor_shape, params)
    
    # Validate streaming
    stream_errors = stream_spec.validate_against_shape(block_dims)
    if stream_errors:
        print(f"Stream tiling errors: {stream_errors}")
        return False
    
    return True
```

## Best Practices

1. **Start Simple**: Use full dimensions (":") initially
2. **Profile First**: Measure before optimizing tile sizes
3. **Document Parameters**: Clear names and comments
4. **Validate Early**: Check tiling before hardware generation
5. **Consider Alignment**: Tile sizes often power-of-2
6. **Balance Dimensions**: Avoid very thin/tall tiles
7. **Reuse Parameters**: Share across related interfaces

## Summary

The tiling system provides powerful abstractions for hardware-software co-design:
- **Declarative**: Specify intent, not implementation
- **Flexible**: Mix fixed and parameterized expressions
- **Validated**: Errors caught early
- **Performant**: Direct hardware mapping

Master these concepts to effectively map algorithms to FPGA accelerators.