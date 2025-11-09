# Schema Design

This chapter explores the art of crafting effective kernel schemas - the declarative blueprints that define your hardware operations.

## Schema Philosophy

A well-designed schema is:

- **Declarative** - Says "what" not "how"
- **Composable** - Combines simple primitives
- **Adaptable** - Works across different tensor sizes
- **Validated** - Catches errors early via constraints

**Anti-pattern:**
```python
# Hard-coded, brittle
block_shape = (1, 784)  # Breaks if input changes!
```

**Good pattern:**
```python
# Template-based, adaptive
block_tiling = FULL_SHAPE  # Adapts to any input size
```

## Template System Deep Dive

Templates express **relationships**, not **values**.

### FULL_SHAPE: Rank-Agnostic Copying

Copies entire tensor shape to block shape, works for any rank. Most operations use `FULL_SHAPE` for `block_tiling`.

### FULL_DIM: Single Dimension Copying

Copies one tensor dimension. Use for partial tiling:

```python
block_tiling = [1, 32, 32, FULL_DIM]  # Tile spatial, keep channels full
# Tensor: (1, 224, 224, 64) → Block: (1, 32, 32, 64)
```

### Tuple Shorthand: Cross-Interface Derivation

Copies dimension from another interface:

```python
OutputSchema(stream_tiling=[("input", -1)])  # Match input's last dim parallelism
```

Syntax: `("interface_name", dim_index)` or `("interface_name", dim_index, hierarchy)`.

### String Parameters: Design Space Variables

```python
# Schema:
stream_tiling = ["SIMD"]

# At configure time:
params = {"SIMD": 64}
stream_shape = (64,)
```

**Use when:** Dimension is a parallelization parameter for DSE.

**Multi-dimensional example:**
```python
# 2D parallelization
stream_tiling = ["MW", "MH"]

# Configure:
params = {"MW": 8, "MH": 16}
stream_shape = (8, 16)
```

### Callable Functions: Custom Derivation

```python
from brainsmith.dataflow.spec_helpers import derive_dim
from brainsmith.dataflow import ShapeHierarchy

# Schema:
def custom_channels(interfaces, param_getter, model, tensor_name):
    """Half of input's channels."""
    input_channels = interfaces["input"].tensor_shape[-1]
    return input_channels // 2

block_tiling = [1, FULL_DIM, FULL_DIM, custom_channels]

# Or use helper:
block_tiling = [1, FULL_DIM, FULL_DIM, derive_dim("input", ShapeHierarchy.TENSOR, -1)]
```

**Use when:** Complex derivation logic not expressible with primitives.

**Signature:**
```python
def my_dimension(
    interfaces: Dict[str, Any],        # Access to input/output interfaces
    param_getter: Callable[[str], Any], # Get nodeattr values
    model: ModelWrapper,                # ONNX graph access
    tensor_name: Optional[str]          # Current tensor name
) -> int:
    # Return computed dimension
    return computed_value
```

## Datatype Derivation

Datatypes can be **fixed**, **derived**, or **optimized**.

### Datatype Options

```python
# Fixed: Always use specific type
InputSchema(datatype=DataType["INT8"])

# Pass-through: Use from ONNX graph
InputSchema(datatype=None)

# Derived: Copy from another interface
OutputSchema(datatype="input")  # String shorthand
```

### Value-Optimized Datatypes

```python
from brainsmith.dataflow import VALUE_OPTIMIZED

InputSchema(
    name="bias",
    datatype=VALUE_OPTIMIZED  # Optimize from actual values
)
```

**Behavior:**
- **Static tensors:** Analyzes min/max values, picks smallest fitting datatype
- **Dynamic tensors:** Falls back to graph datatype

**Example:**
```python
# Bias values: [2.0, 5.5]
# Graph datatype: INT8 (conservative)
# Optimized: INT3 (sufficient for [2, 5])
```

**Use when:** Static parameters with conservative graph types.

### Context-Aware Arithmetic Datatypes

```python
from brainsmith.dataflow.spec_helpers import add_datatype, mul_datatype

# Addition:
OutputSchema(
    datatype=add_datatype("input", "bias")
)

# Multiplication:
OutputSchema(
    datatype=mul_datatype("input", "weights")
)
```

**Behavior:**
- **Both dynamic:** Uses worst-case type bounds (conservative)
- **One static:** Uses type bounds + actual values (optimized)
- **Both static:** Uses actual values for both (fully optimized)

**Example:**
```python
# Both dynamic:
INT8 + INT8 → INT9  # Can overflow to -256

# One static with values [2, 5]:
INT8 [-128, 127] + [2, 5] → INT8  # Max is 132, fits in INT8!
```

**Available helpers:**
- `add_datatype(a, b)` - Addition
- `sub_datatype(a, b)` - Subtraction
- `mul_datatype(a, b)` - Multiplication
- `min_datatype(a, b)` - Minimum
- `max_datatype(a, b)` - Maximum

### Custom Datatype Functions

```python
def accumulator_datatype(interfaces, param_getter, model, tensor_name):
    """Accumulator for dot product: input × weight."""
    from brainsmith.dataflow.spec_helpers import (
        compute_mul_range,
        smallest_datatype_for_range
    )

    input_dt = interfaces["input"].datatype
    weight_dt = interfaces["weight"].datatype

    # Compute output range
    min_val, max_val = compute_mul_range(
        input_dt.min(), input_dt.max(),
        weight_dt.min(), weight_dt.max()
    )

    # Account for accumulation over SIMD elements
    simd = param_getter("SIMD")
    min_val *= simd
    max_val *= simd

    return smallest_datatype_for_range(min_val, max_val)

# Use in schema:
internal_datatypes={
    "accumulator": accumulator_datatype
}
```

## Constraint Patterns

Constraints are **declarative validators** that catch errors early.

### Datatype Constraints

```python
DatatypeInteger(("input", "output"))              # Must be integer
DatatypeFloat(("input", "output"))                # Must be float
DatatypesEqual(("input0", "input1"))              # Must match
DatatypeInRange("input", "INT", 4, 8)             # INT4-INT8
```

### Shape Constraints

```python
ShapesEqual(("input0", "input1"))                 # Shapes match
DimensionEquals("input", 0, 1)                    # Batch = 1
DimensionDivisible("output", -1, "PE")            # Divisible by PE
TensorDimMatches("bias", 0, [1, ("input", -1)])   # Match or broadcast
```

### ONNX-Specific Constraints

```python
IsDynamic("input")                      # No initializer (streaming)
IsStatic("weights")                     # Has initializer (buffered)
HasLayout("input", "NHWC")              # Specific layout
NodeAttributeEquals("axis", -1)         # Node attribute check
```

### Custom Constraints

```python
from brainsmith.dataflow.constraints import CustomConstraint

def check_matmul_compatibility(ctx):
    """Validate matrix multiply dimensions."""
    input_shape = ctx.get_shape("input")
    weight_shape = ctx.get_shape("weight")

    if input_shape[-1] != weight_shape[0]:
        return (
            f"MatMul incompatible: input.shape[-1]={input_shape[-1]} "
            f"vs weight.shape[0]={weight_shape[0]}"
        )
    return None  # Success

# Use in schema:
constraints=[
    CustomConstraint(check_matmul_compatibility, "MatMul dimension compatibility")
]
```

**Use when:** Complex validation not expressible with built-in constraints.

## DSE Dimension Specification

Define explorable resource/implementation dimensions.

### Ordered Dimensions (Navigation)

```python
from brainsmith.dataflow.schemas import DSEDimension

# Explicit values (list/tuple → OrderedDimension)
DSEDimension("depth", [128, 256, 512, 1024], default=256)

# Enables navigation:
point.with_min("depth")           # → 128
point.with_max("depth")           # → 1024
point.with_percentage("depth", 0.5)  # → 512
point.with_step_up("depth", 2)    # → 1024 (from 512)
```

**Use when:** Dimension has natural ordering (depth, num_layers, etc.).

### Discrete Dimensions (Membership)

```python
# Unordered choices (set/frozenset → discrete)
DSEDimension("ram_style", {"distributed", "block"}, default="distributed")
DSEDimension("res_type", {"lut", "dsp"})

# Only supports:
point.with_dimension("ram_style", "block")  # Direct assignment
point.sweep_dimension("ram_style")          # Iterate all values
```

**Use when:** Dimension is categorical (RAM style, resource type, algorithm choice).

### Context-Dependent Dimensions

```python
def compute_valid_depths(build_ctx):
    """Compute valid depths based on tensor size."""
    tensor_size = build_ctx.model_w.get_tensor_shape(build_ctx.node_inputs[0])
    max_depth = min(tensor_size[-1], 1024)  # Cap at tensor size or 1024

    # Return powers of 2 up to max
    return {2**i for i in range(int(np.log2(max_depth)) + 1)}

DSEDimension("depth", compute_valid_depths)
```

**Use when:** Valid values depend on graph context (tensor sizes, etc.).

## Layout Requirements

Specify expected tensor layouts for correct operation.

### Input Layout Requirements

```python
InputSchema(
    name="input",
    required_layout="NHWC"  # Operation expects NHWC
)
```

**Effect:** Transformation pipeline will insert layout conversions if needed.

**Use when:** HLS code assumes specific layout (e.g., channel-last).

### Output Layout Behavior

```python
OutputSchema(
    name="output",
    required_layout="NHWC",         # Force specific output layout
    preserves_input_layout=True      # Or preserve whatever input has
)
```

**Options:**
- `required_layout="NHWC"` - Force specific layout
- `preserves_input_layout=True` - Keep input's layout (default)
- Both None - No layout requirement

## Complete Examples

### Complete Schema Examples

See Chapter 3 for a complete ChannelwiseAdd example. Additional patterns:

```python
# Elementwise: Both inputs, same parallelization
ADD_SCHEMA = KernelSchema(
    inputs=[InputSchema(stream_tiling=["PE"]),
            InputSchema(stream_tiling=[("input0", -1)])],
    constraints=[ShapesEqual(("input0", "input1"))]
)

# Reduction: Output dimension reduces to 1
REDUCE_SCHEMA = KernelSchema(
    inputs=[InputSchema(stream_tiling=[1, 1, 1, "SIMD"])],
    outputs=[OutputSchema(block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM, 1])]
)

# Matrix: 2D parallelization with internal types
MATMUL_SCHEMA = KernelSchema(
    inputs=[InputSchema(stream_tiling=["SIMD"]),
            InputSchema(stream_tiling=["SIMD", "PE"])],
    internal_datatypes={"accumulator": accumulator_datatype},
    dse_dimensions={"ram_style": DSEDimension("ram_style", {"distributed", "block"})}
)
```

## Best Practices

### DO: Use Templates for Flexibility

```python
# Good - adapts to any size
block_tiling = FULL_SHAPE

# Bad - breaks for different sizes
block_tiling = [1, 784]
```

### DO: Derive Datatypes When Possible

```python
# Good - optimizes based on actual values
OutputSchema(datatype=add_datatype("input", "bias"))

# Acceptable - conservative
OutputSchema(datatype=None)  # Uses graph datatype
```

### DO: Validate Early with Constraints

```python
# Good - catches errors at build time
constraints=[
    ShapesEqual(("input0", "input1")),
    IsStatic("weights")
]

# Bad - fails during RTL generation
# (no constraints, errors caught late)
```

### DON'T: Hardcode Dimensions

```python
# Bad
stream_tiling = [64]  # Breaks if block shape changes!

# Good
stream_tiling = ["PE"]  # Adapts to any block size
```

### DON'T: Over-Constrain

```python
# Bad - too restrictive
constraints=[
    DimensionEquals("input", 0, 1),  # Batch = 1
    DimensionEquals("input", 1, 224),  # Height = 224
    DimensionEquals("input", 2, 224),  # Width = 224
    # Won't work for any other size!
]

# Good - only essential constraints
constraints=[
    DimensionEquals("input", 0, 1),  # Batch = 1 (if truly required)
    # Let other dimensions vary
]
```

## Next Steps

You now understand schema design:

✓ Template system (FULL_SHAPE, FULL_DIM, tuples, callables)
✓ Datatype derivation (fixed, derived, optimized, arithmetic)
✓ Constraint patterns (datatype, shape, ONNX, custom)
✓ DSE dimension specification (ordered vs discrete)
✓ Layout requirements

**Next chapter:** Design space exploration - navigating and optimizing configurations.
