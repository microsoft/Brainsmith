# Tiling System Architecture Design

## Executive Summary

The Brainsmith Tiling System provides a clean, declarative approach for specifying how tensor dimensions should be divided into computational blocks and streaming patterns for FPGA hardware acceleration. The system enables RTL developers to specify tiling using simple list expressions that are automatically converted to validated tiling specifications and applied at runtime.

**Key Benefits:**
- **Declarative Interface**: Simple list-based specifications (`[1, "BATCH", ":"]`)
- **Parameter Binding**: Runtime resolution of symbolic parameters
- **Type Safety**: Validated expressions with clear error messages
- **FPGA Optimization**: Block and stream tiling for hardware efficiency
- **Template Integration**: Seamless code generation for FINN HWCustomOp

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TILING SYSTEM ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────────┘

User Layer (RTL Developers):
┌─────────────────┐    ┌─────────────────┐
│ SHAPE Pragmas   │    │ Definition APIs │
│ [BATCH, SIMD]   │    │ block_tiling=[] │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
Core Tiling Layer:   │
┌─────────────────────▼─────────────────────┐
│             TilingSpec                   │
│  ┌─────────────────────────────────────┐ │
│  │        TilingExpr[]                 │ │
│  │  ┌───┐ ┌───────┐ ┌─────┐ ┌───────┐ │ │
│  │  │ 1 │ │"BATCH"│ │ ":" │ │ 32    │ │ │
│  │  │   │ │param  │ │full │ │literal│ │ │
│  │  └───┘ └───────┘ └─────┘ └───────┘ │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                     │
Strategy Layer:      │
┌─────────────────────▼─────────────────────┐
│           TilingStrategy                  │
│  ┌─────────────────┐ ┌─────────────────┐ │
│  │   Block Spec    │ │  Stream Spec    │ │
│  │   (BDIM)        │ │   (SDIM)        │ │
│  └─────────────────┘ └─────────────────┘ │
└─────────────────────┬─────────────────────┘
                      │
Runtime Layer:        │
┌─────────────────────▼─────────────────────┐
│          Parameter Resolution             │
│  ┌─────────────────────────────────────┐  │
│  │ {"BATCH": 32, "SIMD": 8}           │  │
│  └─────────────────────────────────────┘  │
└─────────────────────┬─────────────────────┘
                      │
Output:               ▼
┌─────────────────────────────────────────┐
│        Concrete Dimensions              │
│      [32, 32, 14, 14] → [1, 8, 14, 14]  │
│       Tensor Shape    Block Dims        │
└─────────────────────────────────────────┘
```

## Data Hierarchy Model

The tiling system operates on a four-level data hierarchy designed for efficient FPGA computation:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA HIERARCHY                              │
└─────────────────────────────────────────────────────────────────┘

Level 1: TENSOR (Complete Dataset)
┌───────────────────────────────────────────────────────────────┐
│  Tensor Shape: [32, 256, 224, 224]  (Batch×Channel×Height×Width) │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Full Dataset                         │   │
│  │              (Complete Inference)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼ Block Tiling
Level 2: BLOCK (Processing Tile)
┌─────────────────────────────────────────────────────────────────┐
│  Block Shape: [1, 64, 14, 14]    (Kernel Processing Unit)      │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐  │
│  │Block│Block│Block│Block│Block│Block│Block│Block│Block│Block│  │
│  │  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  9  │  │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Stream Tiling
Level 3: STREAM (Clock Cycle Data)
┌─────────────────────────────────────────────────────────────────┐
│  Stream Shape: [1, 8, 14, 14]    (Per-Clock Processing)        │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                             │
│  │S0 │S1 │S2 │S3 │S4 │S5 │S6 │S7 │ ... (8 streams per block)   │
│  └───┴───┴───┴───┴───┴───┴───┴───┘                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Element Level
Level 4: ELEMENT (Individual Data)
┌─────────────────────────────────────────────────────────────────┐
│  Element: INT8 value         (Hardware Data Type)              │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐             │
│  │ 1 │ │127│ │-3 │ │ 42│ │ 0 │ │-18│ │ 91│ │ 7 │             │
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘             │
└─────────────────────────────────────────────────────────────────┘

Tiling Relationships:
• Tensor ÷ Block = Number of blocks to process
• Block ÷ Stream = Number of clock cycles per block  
• Stream ÷ Element = Parallelism factor (SIMD/PE)
```

### Real-World Example: Convolution Layer

```
Convolution: 224×224 RGB → 112×112 Feature Maps

Input Tensor:    [1, 3, 224, 224]     # Batch×Channel×Height×Width
Block Tiling:    [1, 3, 14, 14]       # Process 14×14 patches at a time
Stream Tiling:   [1, 1, 14, 14]       # Stream 1 channel per clock cycle

Hardware Impact:
- Block size determines memory requirements (14×14×3 = 588 elements)
- Stream size determines parallelism (1 channel processed per clock)
- Pipeline depth = Total blocks × Cycles per block
```

## Core Components

### TilingExpr: Expression Building Block

```python
class TilingExprType(Enum):
    SINGLETON = "singleton"   # Value: 1
    FULL = "full"            # Value: ":"  
    LITERAL = "literal"      # Value: 32, 64, etc.
    PARAMETER = "parameter"  # Value: "BATCH", "SIMD", etc.
```

**Expression Type Matrix:**
```
┌─────────────┬─────────────┬─────────────────┬───────────────────┐
│ Type        │ Value       │ Runtime Result  │ Use Case          │
├─────────────┼─────────────┼─────────────────┼───────────────────┤
│ SINGLETON   │ 1           │ Always 1        │ No tiling         │
│ FULL        │ ":"         │ Full dimension  │ Stream all data   │
│ LITERAL     │ 32, 64, etc │ Fixed value     │ Known tile size   │
│ PARAMETER   │ "BATCH"     │ Runtime binding │ Configurable size │
└─────────────┴─────────────┴─────────────────┴───────────────────┘
```

**Visual Expression Examples:**
```
Input: [1, "CHANNELS", ":", 32]

┌─────┐ ┌───────────┐ ┌─────┐ ┌─────┐
│  1  │ │"CHANNELS" │ │ ":" │ │ 32  │
│sing │ │ parameter │ │full │ │lit  │
└─────┘ └───────────┘ └─────┘ └─────┘
   │           │          │       │
   ▼           ▼          ▼       ▼
[1=1] [CHANNELS=64] [H=224] [32=32]
   │           │          │       │
   └───────────┼──────────┼───────┘
               ▼          ▼       
        Block: [1, 64, 224, 32]
```

### TilingSpec: Expression Container

The `TilingSpec` class validates and manages collections of tiling expressions:

```python
# Creation from lists
spec = TilingSpec([1, "SIMD", ":", 32])

# Automatic validation
spec.get_parameters()  # → {"SIMD"}
spec.ndim             # → 4

# Runtime resolution  
resolved = spec.resolve(
    shape=[32, 128, 224, 224],
    parameters={"SIMD": 8}
)
# Result: [1, 8, 224, 32]
```

**TilingSpec Lifecycle:**
```
Creation → Validation → Parameter Collection → Runtime Resolution

List Input:     [1, "SIMD", ":", 32]
     │
     ▼ __init__()
TilingExpr[]:   [SINGLETON(1), PARAMETER("SIMD"), FULL(":"), LITERAL(32)]
     │
     ▼ get_parameters()
Parameters:     {"SIMD"}
     │
     ▼ resolve(shape, params)
Result:         [1, 8, 224, 32]
```

### TilingStrategy: Block + Stream Coordinator

The `TilingStrategy` orchestrates both block-level and stream-level tiling:

```python
class TilingStrategy:
    def __init__(self, block_spec, stream_spec, order):
        self.block_spec = block_spec    # TilingSpec for BDIM
        self.stream_spec = stream_spec  # TilingSpec for SDIM
        self.order = order             # Processing order
```

**Dual-Level Processing:**
```
Input Tensor: [32, 256, 224, 224]

Step 1: Block Tiling (BDIM)
block_spec = TilingSpec([1, "CHANNELS", 14, 14])
parameters = {"CHANNELS": 64}
Result: [1, 64, 14, 14]  ← Block dimensions

Step 2: Stream Tiling (SDIM) 
stream_spec = TilingSpec([1, "SIMD", 1, 1])
parameters = {"SIMD": 8}
Result: [1, 8, 1, 1]     ← Stream dimensions per clock

Hardware Interpretation:
• Process 64 channels in 14×14 spatial blocks
• Stream 8 channels simultaneously per clock cycle
• Total cycles per block: 64÷8 × 14×14 = 8 × 196 = 1,568 cycles
```

## Tiling Pipeline

The complete flow from user specification to runtime execution:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TILING PIPELINE FLOW                         │
└─────────────────────────────────────────────────────────────────┘

1. Definition Phase (Design Time)
┌─────────────────────────────────────────────────────────────────┐
│ RTL Developer Input:                                            │
│                                                                 │
│ InputDefinition(                                                │
│     name="conv_input",                                          │
│     block_tiling=[1, "CHANNELS", 14, 14],    ← User List        │
│     stream_tiling=[1, "SIMD", 1, 1]          ← User List        │
│ )                                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
2. Conversion Phase (__post_init__)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Automatic TilingSpec Creation:                                  │
│                                                                 │
│ _block_tiling_spec = TilingSpec([1, "CHANNELS", 14, 14])       │
│ _stream_tiling_spec = TilingSpec([1, "SIMD", 1, 1])            │
│                                                                 │
│ _tiling_strategy = TilingStrategy(                              │
│     block_spec=_block_tiling_spec,                             │
│     stream_spec=_stream_tiling_spec                            │
│ )                                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
3. Parameter Collection Phase
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Required Parameters Identified:                                 │
│                                                                 │
│ get_required_parameters() → {                                   │
│     "CHANNELS": "block_tiling",                                │
│     "SIMD": "stream_tiling"                                    │
│ }                                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
4. Runtime Resolution Phase
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Parameter Binding + Shape Application:                          │
│                                                                 │
│ tensor_shape = [32, 256, 224, 224]                             │
│ parameters = {"CHANNELS": 64, "SIMD": 8}                       │
│                                                                 │
│ Block Result:                                                   │
│   apply_block_tiling() → [1, 64, 14, 14]                      │
│                                                                 │
│ Stream Result:                                                  │
│   apply_stream_tiling() → [1, 8, 1, 1]                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
5. Hardware Generation Phase
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Generated RTL/FINN Code:                                        │
│                                                                 │
│ // Block processing loop                                        │
│ for(int b_ch = 0; b_ch < 256; b_ch += 64) {                   │
│   // Stream processing (8 channels per clock)                  │
│   for(int s_ch = 0; s_ch < 64; s_ch += 8) {                   │
│     process_streaming_data(input[s_ch:s_ch+8]);               │
│   }                                                             │
│ }                                                               │
│                                                                 │
│ FINN Parameters:                                                │
│   "CHANNELS": 64, "SIMD": 8                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Expression System Deep Dive

### Expression Type Usage Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                  EXPRESSION USAGE GUIDE                         │
└─────────────────────────────────────────────────────────────────┘

Singleton (1):
┌─────────────────────────────────────────────────────────────────┐
│ Use Case: No tiling desired for this dimension                 │
│ Example:  [1, "CHANNELS", ":", ":"]                            │
│           ↑ Batch dimension not tiled                          │
│ Result:   Always produces tile size of 1                       │
│ Common:   Batch dimensions, single-channel processing          │
└─────────────────────────────────────────────────────────────────┘

Full Slice (":"):
┌─────────────────────────────────────────────────────────────────┐
│ Use Case: Process entire dimension at once                     │
│ Example:  [1, "CHANNELS", ":", ":"]                            │
│                            ↑↑ Spatial dims not tiled           │
│ Result:   Tile size equals full tensor dimension               │
│ Common:   Spatial dimensions, when no tiling needed            │
└─────────────────────────────────────────────────────────────────┘

Literal Integer (32, 64, etc.):
┌─────────────────────────────────────────────────────────────────┐
│ Use Case: Fixed tile size known at design time                 │
│ Example:  [1, "CHANNELS", 14, 14]                              │
│                           ↑↑ Fixed 14×14 spatial tiles         │
│ Result:   Always uses the specified tile size                  │
│ Common:   Optimized tile sizes, hardware constraints           │
└─────────────────────────────────────────────────────────────────┘

Parameter ("BATCH", "SIMD", etc.):
┌─────────────────────────────────────────────────────────────────┐
│ Use Case: Runtime-configurable tile size                       │
│ Example:  [1, "CHANNELS", ":", ":"]                            │
│               ↑ Runtime parameter binding                      │
│ Result:   Resolved from parameter binding at runtime           │
│ Common:   User-configurable parallelism, adaptive sizing       │
└─────────────────────────────────────────────────────────────────┘
```

### Common Expression Patterns

```python
# Pattern 1: Batch-Channel-Spatial Layout (NCHW)
block_tiling = [1, "CHANNELS", 14, 14]
stream_tiling = [1, "SIMD", 1, 1]
# → Process fixed 14×14 patches, configurable channel parallelism

# Pattern 2: Full Spatial, Tiled Channels  
block_tiling = [1, "TILE_C", ":", ":"]
stream_tiling = [1, "SIMD", ":", ":"]
# → Process full spatial dimensions, tile channels

# Pattern 3: No Block Tiling, Stream Only
block_tiling = [":", ":", ":", ":"]  
stream_tiling = [1, 1, 1, "SIMD"]
# → Process full tensor, stream last dimension

# Pattern 4: Mixed Fixed and Parameterized
block_tiling = [1, "CHANNELS", 32, 32]
stream_tiling = [1, "SIMD", 4, 4]  
# → Fixed spatial tiles, configurable channel streaming
```

## Usage Patterns and Best Practices

### Integration with InputDefinition/OutputDefinition

```python
# Complete definition with tiling
input_def = InputDefinition(
    name="conv_input",
    datatype_constraints=[
        DatatypeConstraintGroup(base_type="FIXED", min_width=8, max_width=8)
    ],
    block_tiling=[1, "CHANNELS", 14, 14],    # Block-level tiling
    stream_tiling=[1, "SIMD", 1, 1],         # Stream-level tiling
    is_weight=False
)

# Automatic internal conversion:
# input_def._block_tiling_spec  → TilingSpec([1, "CHANNELS", 14, 14])
# input_def._stream_tiling_spec → TilingSpec([1, "SIMD", 1, 1])
# input_def._tiling_strategy    → TilingStrategy(block_spec, stream_spec)
```

### Runtime Model Creation

```python
# Create runtime model with parameter binding
model = input_def.create_model(
    tensor_dims=[32, 256, 224, 224],
    datatype=create_simple_datatype("INT8"),
    parameter_binding={"CHANNELS": 64, "SIMD": 8}
)

# Internal flow:
# 1. Validate datatype against constraints
# 2. Apply block tiling: [32, 256, 224, 224] → [1, 64, 14, 14] 
# 3. Apply stream tiling: [1, 64, 14, 14] → [1, 8, 1, 1]
# 4. Create InputInterface with resolved dimensions
```

### Parameter Collection for Code Generation

```python
# Collect all required parameters across interfaces
kernel_def = KernelDefinition(name="conv_kernel")
kernel_def.add_input(input_def)
kernel_def.add_input(weight_def)  
kernel_def.add_output(output_def)

# Get all parameters needed for this kernel
required_params = kernel_def.get_required_parameters()
# Result: {
#     "CHANNELS": "input_block_tiling_and_weights_block_tiling",
#     "SIMD": "input_stream_tiling",
#     "PE": "weights_stream_tiling"
# }

# Use in template generation for FINN nodeattr_types
for param_name, usage_context in required_params.items():
    generate_nodeattr(param_name, "int", default=1, context=usage_context)
```

## Integration Points

### SHAPE Pragma System Integration

```systemverilog
// RTL file with SHAPE pragmas
// @brainsmith BDIM input [INPUT_H, INPUT_W] SHAPE=[BATCH, CHANNELS]
// @brainsmith SDIM input [INPUT_SIMD] SHAPE=[SIMD]

module conv_kernel #(
    parameter INPUT_H = 224,
    parameter INPUT_W = 224, 
    parameter INPUT_SIMD = 8
)(
    // AXI Stream interface
    input logic [(INPUT_SIMD*8-1):0] input_tdata,
    // ...
);
```

**Parser → Tiling System Flow:**
```python
# RTL Parser extracts SHAPE expressions
pragma_data = {
    "bdim_shape": ["BATCH", "CHANNELS"],  # From SHAPE=[BATCH, CHANNELS]
    "sdim_shape": ["SIMD"]                # From SHAPE=[SIMD]
}

# Converted to InputDefinition
input_def = InputDefinition(
    name="input",
    block_tiling=["BATCH", "CHANNELS"],   # From bdim_shape
    stream_tiling=["SIMD"],               # From sdim_shape
    # ... other fields
)

# Used in template generation  
template_context.shape_nodeattrs = [
    {"name": "BATCH", "source_comment": "BDIM: input"},
    {"name": "CHANNELS", "source_comment": "BDIM: input"}, 
    {"name": "SIMD", "source_comment": "SDIM: input"}
]
```

### FINN HWCustomOp Code Generation

```python
# Generated get_nodeattr_types() method
def get_nodeattr_types(self):
    return {
        # Interface datatypes
        "inputDataType": ('s', False, 'INT8'),
        
        # BDIM/SDIM parameters from tiling system
        "BATCH": ('i', False, 1),     # From block_tiling
        "CHANNELS": ('i', False, 1),   # From block_tiling  
        "SIMD": ('i', False, 1),       # From stream_tiling
        
        # Hardware optimization hints
        "ram_style": ('s', False, 'auto'),
    }

# Generated _create_kernel_definition() method
def _create_kernel_definition(self):
    kernel_def = KernelDefinition(name="conv_kernel")
    
    input_def = InputDefinition(
        name="input",
        block_tiling=["BATCH", "CHANNELS"],    # From tiling system
        stream_tiling=["SIMD"],                # From tiling system
        datatype_constraints=[/*...*/]
    )
    
    kernel_def.add_input(input_def)
    return kernel_def
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Parameter Resolution Errors

**Error:** `KeyError: Parameter 'SIMD' not found in parameter binding`

**Cause:** TilingSpec references a parameter that wasn't provided at runtime

**Solution:**
```python
# Ensure all required parameters are provided
required = input_def.get_tiling_parameters()
print(f"Required parameters: {required}")

# Provide complete parameter binding
model = input_def.create_model(
    tensor_dims=[32, 128, 224, 224],
    parameter_binding={"SIMD": 8, "CHANNELS": 64}  # Include all required
)
```

#### 2. Dimension Mismatch Errors

**Error:** `TilingSpec has 3 dimensions but tensor has 4 dimensions`

**Cause:** Tiling specification dimension count doesn't match tensor

**Solution:**
```python
# Ensure tiling spec matches tensor dimensionality
tensor_dims = [32, 128, 224, 224]  # 4D tensor
block_tiling = [1, "CHANNELS", 14, 14]  # Must be 4D spec
stream_tiling = [1, "SIMD", 1, 1]       # Must be 4D spec
```

#### 3. Validation Errors

**Error:** `Tile size 65 does not evenly divide tensor dimension 128`

**Cause:** Tile size is not a divisor of tensor dimension

**Solution:**
```python
# Use divisible tile sizes or full dimension
block_tiling = [1, 64, 14, 14]  # 64 divides 128 ✓
# OR
block_tiling = [1, ":", 14, 14]  # Full dimension ✓

# Check divisibility at design time
tensor_dim = 128
tile_size = 64
assert tensor_dim % tile_size == 0, f"{tile_size} must divide {tensor_dim}"
```

### Performance Optimization Tips

#### 1. Memory Hierarchy Optimization

```python
# Good: Align block sizes with memory hierarchy
block_tiling = [1, 64, 14, 14]  # 64 channels = good cache line usage
stream_tiling = [1, 8, 1, 1]    # 8 parallel = SIMD width

# Avoid: Very small blocks (overhead) or very large blocks (memory pressure)
block_tiling = [1, 1, 1, 1]     # Too small - high overhead
block_tiling = [1, 1024, ":", ":"]  # Too large - memory pressure
```

#### 2. Hardware Parallelism Alignment

```python
# Good: Stream size matches hardware capabilities  
stream_tiling = [1, 8, 1, 1]    # 8-way SIMD parallelism

# Consider: Hardware constraints
# - DSP slices available
# - Memory bandwidth  
# - Clock frequency targets
```

#### 3. Pipeline Efficiency

```python
# Good: Block and stream sizes that enable pipelining
block_tiling = [1, "CHANNELS", 16, 16]  # Moderate block size
stream_tiling = [1, "SIMD", 4, 4]       # Multiple cycles per block

# Enables: Block-level pipelining while maintaining efficiency
```

## Advanced Usage Scenarios

### Multi-Interface Parameter Sharing

```python
# Shared parameters across interfaces
input_def = InputDefinition(
    name="input",
    block_tiling=["BATCH", "CHANNELS", ":", ":"],
    stream_tiling=[1, "SIMD", 1, 1]
)

weight_def = InputDefinition(
    name="weights", 
    block_tiling=["FILTERS", "CHANNELS", ":", ":"],  # Shares CHANNELS
    stream_tiling=[1, "PE", 1, 1],
    is_weight=True
)

output_def = OutputDefinition(
    name="output",
    block_tiling=["BATCH", "FILTERS", ":", ":"]      # Shares BATCH, FILTERS
)

# Kernel-level parameter collection automatically deduplicates:
kernel_def = KernelDefinition("conv")
kernel_def.add_input(input_def)
kernel_def.add_input(weight_def)
kernel_def.add_output(output_def)

params = kernel_def.get_required_parameters()
# Result: {
#     "BATCH": "input_block_tiling_and_output_block_tiling",
#     "CHANNELS": "input_block_tiling_and_weights_block_tiling", 
#     "FILTERS": "weights_block_tiling_and_output_block_tiling",
#     "SIMD": "input_stream_tiling",
#     "PE": "weights_stream_tiling"
# }
```

### Conditional Tiling Patterns

```python
def create_adaptive_conv_input(spatial_tiling: bool = False):
    """Create input definition with optional spatial tiling."""
    
    if spatial_tiling:
        # Spatial tiling for large feature maps
        block_tiling = [1, "CHANNELS", "TILE_H", "TILE_W"]
        stream_tiling = [1, "SIMD", 1, 1]
    else:
        # Full spatial processing for small feature maps  
        block_tiling = [1, "CHANNELS", ":", ":"]
        stream_tiling = [1, "SIMD", ":", ":"]
    
    return InputDefinition(
        name="adaptive_input",
        block_tiling=block_tiling,
        stream_tiling=stream_tiling
    )

# Usage in different scenarios
small_input = create_adaptive_conv_input(spatial_tiling=False)  # 28×28 inputs
large_input = create_adaptive_conv_input(spatial_tiling=True)   # 224×224 inputs
```

### Complex Pipeline Tiling

```python
def create_pipeline_stage(stage_name: str, input_channels: str, output_channels: str):
    """Create a pipeline stage with consistent tiling."""
    
    return {
        "input": InputDefinition(
            name=f"{stage_name}_input",
            block_tiling=[1, input_channels, "TILE_SIZE", "TILE_SIZE"],
            stream_tiling=[1, "SIMD", 1, 1]
        ),
        "weights": InputDefinition(
            name=f"{stage_name}_weights", 
            block_tiling=[output_channels, input_channels, ":", ":"],
            stream_tiling=["PE", "SIMD", ":", ":"],
            is_weight=True
        ),
        "output": OutputDefinition(
            name=f"{stage_name}_output",
            block_tiling=[1, output_channels, "TILE_SIZE", "TILE_SIZE"]
        )
    }

# Multi-stage pipeline
pipeline = [
    create_pipeline_stage("conv1", "IN_CH", "MID_CH"),
    create_pipeline_stage("conv2", "MID_CH", "OUT_CH"),
    create_pipeline_stage("conv3", "OUT_CH", "FINAL_CH")
]

# Shared parameters: TILE_SIZE, SIMD, PE
# Stage-specific: IN_CH, MID_CH, OUT_CH, FINAL_CH
```

## API Reference

### Core Classes

#### TilingExpr
```python
@dataclass
class TilingExpr:
    expr_type: TilingExprType
    value: Optional[Union[int, str]]
    
    @classmethod
    def from_value(cls, value: Union[int, str]) -> 'TilingExpr'
    
    @property
    def is_static(self) -> bool
    @property  
    def is_parameter(self) -> bool
    @property
    def parameter_name(self) -> Optional[str]
```

#### TilingSpec
```python
@dataclass  
class TilingSpec:
    expressions: List[TilingExpr]
    
    def __init__(self, values: List[Union[int, str]])
    
    @property
    def ndim(self) -> int
    
    def get_parameters(self) -> Set[str]
    def validate_against_shape(self, shape: List[int]) -> List[str]
    def resolve(self, shape: List[int], parameters: dict) -> List[int]
    def to_list(self) -> List[Union[int, str]]
```

#### TilingStrategy
```python
class TilingStrategy:
    def __init__(self, block_spec: Optional[TilingSpec], 
                 stream_spec: Optional[TilingSpec],
                 order: TilingOrder = TilingOrder.ROW_MAJOR)
    
    def get_required_parameters(self) -> Dict[str, str]
    def apply_block_tiling(self, tensor_shape: Shape, 
                          parameters: Dict[str, int]) -> TilingResult
    def apply_stream_tiling(self, block_shape: Shape,
                           parameters: Dict[str, int]) -> TilingResult
    def apply_full_tiling(self, tensor_shape: Shape,
                         parameters: Dict[str, int]) -> Tuple[TilingResult, TilingResult]
    def validate_parameters(self, parameters: Dict[str, int]) -> List[str]
    
    @classmethod
    def from_expressions(cls, block_expr: Optional[List[Union[int, str]]],
                        stream_expr: Optional[List[Union[int, str]]],
                        order: TilingOrder = TilingOrder.ROW_MAJOR) -> 'TilingStrategy'
```

#### TilingResult
```python
@dataclass
class TilingResult:
    block_dims: Shape
    parameters_used: Dict[str, int]
    warnings: List[str] = None
```

### Utility Functions

#### Simple Tiling Functions
```python
def fixed_tiles(*tile_sizes: int) -> Callable
def adaptive_parameterized_tiles(*param_names: str) -> Callable
def parameterized_tiles(*param_names: str) -> Callable  # Deprecated
def adaptive_tiles(config_key: str, default: Optional[List[int]]) -> Callable
def full_tensor() -> Callable
```

### Integration APIs

#### InputDefinition/OutputDefinition Integration
```python
class InputDefinition:
    block_tiling: Optional[List[Union[int, str]]]
    stream_tiling: Optional[List[Union[int, str]]]
    
    # Internal (auto-created)
    _block_tiling_spec: Optional[TilingSpec]
    _stream_tiling_spec: Optional[TilingSpec] 
    _tiling_strategy: Optional[TilingStrategy]
    
    def get_tiling_parameters(self) -> Dict[str, str]
    def derive_block_dims(self, tensor_dims: Shape, 
                         parameter_binding: Optional[ParameterBinding]) -> Shape
    def derive_stream_dims(self, block_dims: Shape,
                          parameter_binding: Optional[ParameterBinding]) -> Shape
```

---

## Conclusion

The Brainsmith Tiling System provides a clean, declarative approach to specifying tensor tiling patterns for FPGA acceleration. By abstracting complex tiling logic behind simple list expressions and automatic validation, it enables RTL developers to focus on hardware optimization while ensuring type safety and runtime correctness.

**Key strengths:**
- **Simple Interface**: List-based specifications are intuitive and readable
- **Type Safety**: Comprehensive validation at both design time and runtime  
- **Flexibility**: Supports fixed, parameterized, and mixed tiling patterns
- **Hardware Oriented**: Designed specifically for FPGA block and stream processing
- **Template Integration**: Seamless code generation for FINN HWCustomOp

The system successfully balances simplicity for users with the flexibility needed for diverse FPGA acceleration scenarios, making it a robust foundation for the Brainsmith hardware kernel generation pipeline.