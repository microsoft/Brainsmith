# Advanced Topics

This chapter covers advanced patterns and techniques for sophisticated kernel implementations.

## Broadcasting Support

Broadcasting allows operations on tensors with different shapes, following ONNX multi-directional broadcasting semantics.

### Understanding Broadcasting

```python
# Example: Channelwise addition
Input:  (1, 224, 224, 64)  # NHWC image
Bias:   (64,)              # Per-channel bias
Output: (1, 224, 224, 64)  # Same as input

# Broadcasting: Bias (64,) broadcasts to (1, 224, 224, 64)
# Each channel value is repeated across batch and spatial dims
```

### BroadcastInfo Helper

Analyzes broadcasting patterns between tensors:

```python
from brainsmith.dataflow.broadcast_helpers import BroadcastInfo

info = BroadcastInfo.compute(lhs_shape=(1, 224, 224, 64), rhs_shape=(64,))
# info.output_shape → (1, 224, 224, 64)
# info.broadcast_dims_rhs → (0, 1, 2)
```

### Using BroadcastInfo in Schemas

```python
from brainsmith.dataflow import KernelSchema, InputSchema, OutputSchema

CHANNELWISE_OP_SCHEMA = KernelSchema(
    name="ChannelwiseOp",
    inputs=[
        InputSchema(
            name="input",
            block_tiling=FULL_SHAPE,
            stream_tiling=["PE"],
        ),
        InputSchema(
            name="param",
            block_tiling=FULL_SHAPE,  # Actual shape: (C,) or (1,C,1,1), etc.
            stream_tiling=[("input", -1)],  # Match input's PE
        ),
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=FULL_SHAPE,  # Broadcasted shape
            stream_tiling=[("input", -1)],
        ),
    ],
)
```

The schema adapts automatically:
- Input: `(1, 224, 224, 64)` → stream: `(1, 1, 1, PE)`
- Param: `(64,)` → stream: `(PE,)` (matches input's PE)
- Output: `(1, 224, 224, 64)` → stream: `(1, 1, 1, PE)`

### BroadcastInfo Utilities

```python
# Buffer shape for HLS
rhs_buffer = info.get_buffer_shape("rhs", pe=32)  # (64,) → (2,) with pe=32

# Index expression for code generation
index_expr = info.get_index_expression("rhs", ("rep", "h", "w", "c"), "pe")

# Conditional reading for loop optimization
condition = info.should_read_new_value("rhs", ("rep", "h", "w", "c"))
```

## Static vs Dynamic Optimization

The system optimizes differently based on tensor types.

### Detecting Static vs Dynamic

```python
# In schema constraint:
from brainsmith.dataflow.constraints import IsStatic, IsDynamic

constraints=[
    IsDynamic("input"),   # Activation (streaming)
    IsStatic("weights")   # Parameters (buffered)
]

# At runtime:
is_static = model.get_initializer(tensor_name) is not None
```

### Value-Optimized Datatypes

Static tensors are optimized based on actual values:

```python
from brainsmith.dataflow import VALUE_OPTIMIZED

InputSchema(name="bias", datatype=VALUE_OPTIMIZED)
# Graph: INT8, Actual: [2, 3, -1, 5] → Optimized: INT3
```

System analyzes min/max values and selects smallest fitting datatype. Falls back to graph datatype for dynamic tensors.

### Context-Aware Datatype Derivation

```python
OutputSchema(datatype=add_datatype("input", "bias"))
# Both dynamic: INT8 + INT8 → INT9 (conservative)
# Bias static [2, 5]: INT8 + [2, 5] → INT8 (optimized)
```

System uses actual values for static tensors, type bounds for dynamic. Computes smallest fitting output datatype.

### Custom Static/Dynamic Logic

```python
def my_output_datatype(interfaces, param_getter, model, tensor_name):
    """Custom datatype that handles static/dynamic differently."""
    input_interface = interfaces["input"]

    # Check if input is static
    input_tensor = getattr(input_interface, 'tensor_name', None)
    is_static = model.get_initializer(input_tensor) is not None

    if is_static:
        # Static path: optimize aggressively
        values = model.get_initializer(input_tensor)
        min_val = float(values.min())
        max_val = float(values.max())

        # Apply operation
        output_min = np.log(min_val) if min_val > 0 else -np.inf
        output_max = np.log(max_val)

        # Find optimal type
        return smallest_datatype_for_range(output_min, output_max)
    else:
        # Dynamic path: conservative
        input_dt = input_interface.datatype
        # Log can increase range for small values
        return DataType["FLOAT32"]  # Conservative
```

## Custom Dimension Derivation

Sometimes you need complex logic to compute dimensions.

### Simple Custom Function

```python
def half_channels(interfaces, param_getter, model, tensor_name):
    """Return half of input's channel dimension."""
    input_shape = interfaces["input"].tensor_shape
    channels = input_shape[-1]
    return channels // 2

# Use in schema:
OutputSchema(
    block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM, half_channels]
)
```

### Context-Dependent Dimension

```python
def adaptive_buffer_size(interfaces, param_getter, model, tensor_name):
    """Compute buffer size based on tensor dimensions and parameters."""
    input_shape = interfaces["input"].block_shape

    # Get spatial dimensions
    spatial_size = input_shape[1] * input_shape[2]  # H × W

    # Get parallelization
    try:
        pe = param_getter("PE")
    except:
        pe = 1  # Default if not configured yet

    # Compute buffer: enough for one spatial row
    buffer_size = input_shape[2] * input_shape[3] // pe  # W × C / PE

    return buffer_size

# Use in internal spec:
internal_dimensions={
    "buffer_size": adaptive_buffer_size
}
```

### Multi-Interface Derivation

```python
def compute_accumulator_depth(interfaces, param_getter, model, tensor_name):
    """Compute accumulator depth for matrix multiply."""
    input_shape = interfaces["input"].tensor_shape
    weight_shape = interfaces["weight"].tensor_shape

    # MatMul: (M, K) × (K, N) → (M, N)
    # Accumulator depth is K dimension
    k_dim = input_shape[-1]

    # Account for SIMD folding
    simd = param_getter("SIMD")
    return k_dim // simd

# Use in internal specs
```

## Multi-Dimensional Parallelization

Some operations parallelize multiple dimensions.

### Schema Definition

```python
MATMUL_SCHEMA = KernelSchema(
    name="MatMul",
    inputs=[
        InputSchema(
            name="input",
            block_tiling=FULL_SHAPE,
            stream_tiling=["SIMD"],  # Parallelize K dimension
        ),
        InputSchema(
            name="weight",
            block_tiling=FULL_SHAPE,
            stream_tiling=["SIMD", "PE"],  # Parallelize K and N dimensions
        ),
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=FULL_SHAPE,
            stream_tiling=["PE"],  # Parallelize N dimension only
        ),
    ],
)
```

### DSE with Multiple Parameters

```python
# Explore 2D parallelization space
base = design_space.configure({"SIMD": 1, "PE": 1})

# Sweep combinations
results = []
for simd in [1, 2, 4, 8, 16, 32]:
    for pe in [1, 2, 4, 8]:
        try:
            point = design_space.configure({"SIMD": simd, "PE": pe})
            results.append({
                "SIMD": simd,
                "PE": pe,
                "cycles": point.initiation_interval,
                "area": simd * pe  # Simplified area model
            })
        except ValueError:
            pass  # Invalid configuration

# Analyze tradeoffs
import pandas as pd
df = pd.DataFrame(results)
print(df.pivot(index="PE", columns="SIMD", values="cycles"))
```

### GCD Constraints

When parameter appears in multiple interfaces:

```python
# Input:  (128, 768) with stream_tiling=["SIMD", 1]
# Weight: (768, 512) with stream_tiling=["SIMD", "PE"]

# SIMD must divide both:
# - input.block_shape[0] = 128
# - weight.block_shape[0] = 768
# Valid SIMD: divisors(gcd(128, 768)) = divisors(128) = {1, 2, 4, 8, 16, 32, 64, 128}

# PE only in weight:
# Valid PE: divisors(512) = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
```

## Custom DSE Dimensions

Add non-parallelization exploration dimensions.

### Resource Allocation

```python
from brainsmith.dataflow.schemas import DSEDimension

KERNEL_SCHEMA = KernelSchema(
    # ...
    dse_dimensions={
        # RAM style: discrete (unordered)
        "ram_style": DSEDimension(
            "ram_style",
            {"distributed", "block", "ultra"},
            default="block"
        ),

        # Resource type: discrete
        "res_type": DSEDimension(
            "res_type",
            {"lut", "dsp"},
            default="dsp"
        ),

        # Buffer depth: ordered
        "buffer_depth": DSEDimension(
            "buffer_depth",
            [128, 256, 512, 1024, 2048],
            default=512
        ),
    }
)
```

### Algorithm Selection

```python
dse_dimensions={
    "algorithm": DSEDimension(
        "algorithm",
        {"iterative", "recursive", "lookup_table"},
        default="iterative"
    )
}

# In backend, choose implementation based on dimension:
def generate_hls(self, model):
    algorithm = self.get_nodeattr("algorithm")

    if algorithm == "iterative":
        return self._generate_iterative_impl()
    elif algorithm == "recursive":
        return self._generate_recursive_impl()
    else:
        return self._generate_lut_impl()
```

### Context-Dependent Dimension Values

```python
def compute_valid_depths(build_ctx):
    """Compute valid buffer depths based on tensor size."""
    tensor_size = np.prod(
        build_ctx.model_w.get_tensor_shape(build_ctx.node_inputs[0])
    )

    # Powers of 2 up to tensor size
    max_power = int(np.log2(tensor_size))
    return [2**i for i in range(7, min(max_power, 13))]  # 128 to 4096

# Use in schema:
dse_dimensions={
    "buffer_depth": DSEDimension("buffer_depth", compute_valid_depths)
}
```

## Internal Datatypes

Define intermediate datatypes for internal buffers/accumulators.

### Static Internal Datatypes

```python
from qonnx.core.datatype import DataType

KERNEL_SCHEMA = KernelSchema(
    # ...
    internal_datatypes={
        "accumulator": DataType["INT32"],  # Fixed type
        "buffer": DataType["INT16"]
    }
)

# Access in backend:
acc_type = self.design_point.internal_datatypes["accumulator"]
```

### Derived Internal Datatypes

```python
def accumulator_datatype(interfaces, param_getter, model, tensor_name):
    """Compute accumulator type for MAC operation."""
    from brainsmith.dataflow.spec_helpers import (
        compute_mul_range,
        smallest_datatype_for_range
    )

    # Get input datatypes
    input_dt = interfaces["input"].datatype
    weight_dt = interfaces["weight"].datatype

    # Multiplication range
    mul_min, mul_max = compute_mul_range(
        input_dt.min(), input_dt.max(),
        weight_dt.min(), weight_dt.max()
    )

    # Accumulation over SIMD elements
    simd = param_getter("SIMD")
    acc_min = mul_min * simd
    acc_max = mul_max * simd

    return smallest_datatype_for_range(acc_min, acc_max)

# Use in schema:
internal_datatypes={
    "accumulator": accumulator_datatype
}
```

### Using Internal Datatypes

```python
# In backend HLS generation:
class MyKernel_hls(HLSBackend):
    def generate_code(self):
        acc_type = self.kernel_op.design_point.internal_datatypes["accumulator"]
        acc_width = acc_type.bitwidth()

        code = f"""
        ap_int<{acc_width}> accumulator = 0;
        for (int i = 0; i < SIMD; i++) {{
            accumulator += input[i] * weight[i];
        }}
        """
        return code
```

## Handling Variable Rank Operations

Some operations work on tensors of any rank.

### Rank-Agnostic Schemas

```python
# Works for any rank
InputSchema(
    name="input",
    block_tiling=FULL_SHAPE,  # Adapts to input rank
    stream_tiling=["PE"],      # Parallelizes last dim regardless
)

# 2D input: (128, 768)
#   block: (128, 768)
#   stream: (128, PE)  # Last dim parallelized

# 4D input: (1, 224, 224, 64)
#   block: (1, 224, 224, 64)
#   stream: (1, 224, 224, PE)  # Last dim parallelized
```

### Rank-Specific Logic

```python
def rank_aware_tiling(interfaces, param_getter, model, tensor_name):
    """Adapt tiling strategy based on input rank."""
    input_shape = interfaces["input"].tensor_shape
    rank = len(input_shape)

    if rank == 2:
        # 2D: Process full rows
        return input_shape[1]
    elif rank == 4:
        # 4D: Process channel groups
        channels = input_shape[3]
        pe = param_getter("PE")
        return channels // pe
    else:
        raise ValueError(f"Unsupported rank: {rank}")

# Use in schema
```

## Best Practices

### DO: Use BroadcastInfo for Elementwise Ops

```python
# Good - handles all broadcasting cases
info = BroadcastInfo.compute(lhs_shape, rhs_shape)
buffer_shape = info.get_buffer_shape("rhs", pe=32)

# Bad - assumes specific shapes
buffer_shape = (rhs_shape[-1] // 32,)  # Breaks for broadcasting!
```

### DO: Optimize Static Tensors

```python
# Good - automatic optimization
OutputSchema(datatype=add_datatype("input", "bias"))

# Acceptable - conservative
OutputSchema(datatype=None)  # Uses graph type

# Bad - hardcoded, may be too wide
OutputSchema(datatype=DataType["INT32"])
```

### DO: Handle Multi-Dimensional Parallelization

```python
# Good - explicit 2D parallelization
stream_tiling=["SIMD", "PE"]

# Also good - 1D parallelization
stream_tiling=["PE"]

# Bad - ambiguous
stream_tiling=[1, "PE"]  # Which dimension is PE?
```

### DON'T: Assume Static or Dynamic

```python
# Bad - assumes static
bias_values = model.get_initializer(bias_name)  # May be None!
min_val = bias_values.min()  # Crashes if dynamic

# Good - check first
bias_values = model.get_initializer(bias_name)
if bias_values is not None:
    min_val = bias_values.min()
else:
    min_val = datatype.min()  # Conservative
```

### DON'T: Hardcode Buffer Sizes

```python
# Bad - breaks for different shapes
buffer_size = 768  # What if channels change?

# Good - derive from shape
def buffer_size(interfaces, param_getter, model, tensor_name):
    channels = interfaces["input"].block_shape[-1]
    pe = param_getter("PE")
    return channels // pe
```

## Next Steps

You now understand advanced topics:

✓ Broadcasting support (BroadcastInfo, buffer shapes)
✓ Static vs dynamic optimization (value-optimized, context-aware)
✓ Custom dimension derivation (callables, multi-interface)
✓ Multi-dimensional parallelization (SIMD + PE, GCD constraints)
✓ Custom DSE dimensions (resource allocation, algorithm selection)
✓ Internal datatypes (derived, accumulator types)
✓ Variable rank operations (rank-agnostic schemas)

**Next chapter:** Best practices, patterns, and troubleshooting.
