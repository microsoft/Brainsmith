# Kernel Schema Reference

*[Hardware Kernels](index.md) > Schema Reference*

Complete API reference for kernel schema components: inputs, outputs, parameters, datatypes, and constraints.


## Schema Components Overview

A `KernelSchema` consists of five main components:

```python
df.KernelSchema(
    name="MyKernel",
    inputs=[...],           # Input interface definitions
    outputs=[...],          # Output interface definitions
    kernel_params={...},    # Algorithm configuration parameters
    dse_parameters={...},   # Implementation choice parameters (optional)
    constraints=[...],      # Validation constraints (optional)
)
```


## Input and Output Definitions

Interfaces define how the kernel relates to tensor data:

```python
df.InputSchema(
    name="input",
    block_tiling=[FULL_DIM, FULL_DIM, "K"],  # Processing quantum template
    stream_tiling=[1, 1, "SIMD"],             # Parallelization template
    datatype=DataType["INT8"],                # Datatype specification
    required_layout="NHWC",                   # Layout requirement
)
```

### Field Reference

#### `name` (required)
Interface identifier used for:
- Accessing interface properties: `point.inputs["input"]`
- Cross-referencing in derivation: `datatype="input"`
- Backend code generation variable names

#### `block_tiling` (required)
**Template for block shapes**: Defines kernel's processing quantum

- Resolved in Phase 1 (build) using tensor shapes from ModelWrapper
- Determines valid parallelization ranges (stream dims can't exceed block dims)
- Example: `[FULL_DIM, FULL_DIM, "K"]` → `[224, 224, 64]` for specific tensor

**Template elements:**
- `FULL_DIM` - Process entire dimension (use specific positions)
- `FULL_SHAPE` - Process entire tensor (rank-agnostic, expands to match tensor)
- Integer literals - Fixed block size (e.g., `8` for 8-element blocks)
- String parameters - DSE parameter from schema (e.g., `"K"` for kernel size)
- Callables - Custom logic: `lambda ctx: ctx.get_tensor_shape()[0] // 2`

#### `stream_tiling` (required)
**Template for stream shapes**: Defines per-cycle parallelism

- Resolved in Phase 2 (configure) using DSE parameters
- String elements become DSE parameters (e.g., `"SIMD"` → parameter with valid range)
- Example: `[1, 1, "SIMD"]` → `[1, 1, 64]` when configured with `SIMD=64`

**Template elements:**
- Integer literals - Fixed parallelization (e.g., `1` for no parallelization)
- String parameters - Becomes OrderedParameter (e.g., `"SIMD"`, `"PE"`)
- Callables - Derivation from other interfaces (use `derive_dim()` helper)

#### `datatype` (optional)
**Datatype specification**: Fixed or derived from other interfaces

Union type accepts:
- `None` - Infer from ONNX graph tensor (default)
- `DataType` - Fixed datatype: `DataType["FLOAT32"]`, `DataType["INT8"]`
- `str` - Derive from interface: `"input"` (shorthand, recommended)
- `VALUE_OPTIMIZED` - Optimize based on actual tensor values
- Callable - Custom derivation (4 parameters, see below)

Resolved in Phase 1 (build), synced to nodeattrs for persistence.

#### `required_layout` (optional)
**Layout requirement**: Ensures correct data format

- Validation only, doesn't affect shape computation
- Use for kernels sensitive to memory layout (NCHW vs NHWC)
- Example: `"NHWC"` for channel-last operations

### Block vs Stream Tiling

**Why these exist:** Hardware kernels have two dimensions of parallelism:

- **Block Tiling** - How much data the kernel needs loaded for one computation cycle
- **Stream Tiling** - How many elements are processed per clock cycle

**For most simple kernels:**
- Use `FULL_SHAPE` for block_tiling (process entire tensor, rank-agnostic)
- Use parameter names like `"SIMD"` or `"PE"` for stream_tiling

**Example - Element-wise operation:**
```python
inputs=[
    df.InputSchema(
        name="input",
        block_tiling=df.FULL_SHAPE,   # Process entire tensor (rank-agnostic)
        stream_tiling=["SIMD"],        # SIMD elements per cycle
    )
]
```

**Example - Matrix-vector operation:**
```python
inputs=[
    df.InputSchema(
        name="weights",
        block_tiling=[FULL_DIM, "K"],  # One row at a time
        stream_tiling=[1, "SIMD"],     # SIMD elements per cycle
    )
]
```

👉 See [Kernel Architecture: Data Hierarchy](kernel-architecture.md#data-hierarchy-deep-dive) for complete theory.


## Parameters and Constraints

### Parameter Types

Schemas support three parameter categories:

#### 1. Kernel Parameters
**Algorithm configuration** - Affects computation behavior:

```python
kernel_params={
    "epsilon": ("f", True, 1e-5),  # (type, required, default)
    "axis": ("i", False, -1),
}
```

**Format:** `name: (type_code, required, default_value)`

**Type codes:**
- `"f"` - Float (32-bit)
- `"i"` - Integer (signed 64-bit)
- `"s"` - String

**Purpose:** Define WHAT the kernel computes (epsilon for LayerNorm, axis for reduction, etc.)

#### 2. Parallelization Parameters
**Auto-extracted from stream_tiling** - DSE dimensions for throughput:

```python
stream_tiling=["Width", "Height"]  # Parameter names are arbitrary
# Valid ranges computed by factoring block dimensions
# Navigate via interface indices: point.with_input_stream(0, 64)
```

**Characteristics:**
- Names like "SIMD" or "PE" are conventions, not requirements
- Become `OrderedParameter` instances (support navigation)
- Valid values computed as divisors of block dimensions
- Accessed via: `design_space.get_ordered_parameter("SIMD")`

#### 3. Implementation Parameters
**Explicit DSE dimensions** - Affect performance/resources, not correctness:

```python
dse_parameters={
    "ram_style": df.ParameterSpec("ram_style", {"distributed", "block"}),
    "fifo_depth": df.ParameterSpec("fifo_depth", [128, 256, 512]),
}
```

**Purpose:** Memory type, buffer depth, algorithm variant selection

**Container types determine exploration:**
- `list/tuple` → Ordered navigation (supports min/max, step up/down)
- `set/frozenset` → Categorical choices (membership only)

### Parameter Type Distinctions

#### Ordered Parameters (OrderedParameter)

**Characteristics:**
- Values have natural ordering (1 < 2 < 4 < 8)
- Support navigation (step up/down, percentage-based access)
- Used for resource allocation (SIMD, PE, buffer depth)

**Creation:**
- Tiling parameters: Auto-extracted from `stream_tiling`, values computed as divisors
- DSE parameters: Declared with list/tuple container type

**Navigation:**
```python
# Tiling parameters (automatically ordered)
stream_tiling=["SIMD"]  # SIMD becomes OrderedParameter

# DSE parameters (explicitly ordered)
dse_parameters={
    "depth": df.ParameterSpec("depth", [128, 256, 512, 1024])
}

# Navigation
dim = design_space.get_ordered_parameter("SIMD")
print(dim.min(), dim.max())  # 1, 64
print(dim.at_percentage(0.5))  # Middle value

point = point.with_step_up("SIMD", 2)  # Step up 2 positions
```

#### Discrete Parameters (frozenset)

**Characteristics:**
- Values are categorical choices (no ordering)
- Membership testing only (no navigation)
- Used for implementation choices (algorithm variant, memory type)

**Creation:**
- DSE parameters: Declared with set/frozenset container type

**Usage:**
```python
# DSE parameters (discrete)
dse_parameters={
    "ram_style": df.ParameterSpec("ram_style", {"distributed", "block", "ultra"})
}

# Access (no navigation)
for value in sorted(design_space.parameters["ram_style"]):
    point = base.with_dimension("ram_style", value)
```

**Why the Distinction?**

Ordered parameters enable sophisticated DSE strategies (binary search, gradient descent, percentage-based sampling). Discrete parameters require exhaustive enumeration. The type system makes this explicit.

### Constraints

Validate configurations during design space construction:

```python
constraints=[
    df.AttrCompare("epsilon", ">", 0),
    df.DimensionCompare("Width", "<=", "tensor_width"),
]
```

**Constraint types:**
- `AttrCompare` - Validate kernel parameters
- `DimensionCompare` - Validate DSE parameters against tensor shapes
- `DimensionEquals` - Enforce matching dimensions across interfaces


## Datatype Specifications

### Fixed Datatypes

Explicit datatype for interface:

```python
from qonnx.core.datatype import DataType

datatype=DataType["FLOAT32"]
datatype=DataType["INT8"]
datatype=DataType["UINT16"]
```

**Use when:** Interface always has specific datatype regardless of graph.

### Derived from Inputs

String shorthand copies datatype from another interface:

```python
datatype="input"  # Copy from interface named "input" (recommended)
```

**Use when:** Output datatype matches input datatype (most common).

### Custom Derivation

Callable with 4 parameters for complex logic:

```python
def custom_dtype(interfaces, param_getter, model, tensor_name):
    """Custom datatype derivation.

    Args:
        interfaces: Dict of InterfaceDesignSpace instances
        param_getter: Function to retrieve nodeattr values
        model: ModelWrapper for graph access
        tensor_name: ONNX tensor name (optional, may be None)
    """
    input_dtype = interfaces["input"].datatype
    bitwidth = input_dtype.bitwidth()
    signed = input_dtype.signed()
    return DataType[f"INT{bitwidth * 2}" if signed else f"UINT{bitwidth * 2}"]

# Use in schema
datatype=custom_dtype
```

**Use when:** Datatype depends on complex logic (bit-width expansion, mixed precision, etc.)

**Function signature requirements:**
1. **interfaces** - Dict mapping interface names to `InterfaceDesignSpace` objects
2. **param_getter** - Function `(param_name: str) → value` for retrieving nodeattrs
3. **model** - `ModelWrapper` for ONNX graph queries
4. **tensor_name** - Optional string, ONNX tensor name (may be None)

**Common patterns:**
```python
# Bit-width expansion
def double_bitwidth(interfaces, param_getter, model, tensor_name):
    input_dtype = interfaces["input"].datatype
    new_bitwidth = input_dtype.bitwidth() * 2
    return DataType[f"INT{new_bitwidth}"]

# Conditional datatype based on parameter
def conditional_dtype(interfaces, param_getter, model, tensor_name):
    use_float = param_getter("use_float")
    return DataType["FLOAT32"] if use_float else DataType["INT8"]

# Mixed precision based on position
def output_dtype(interfaces, param_getter, model, tensor_name):
    # Different outputs can have different datatypes
    if tensor_name and "accumulator" in tensor_name:
        return DataType["INT32"]
    return interfaces["input"].datatype
```

### VALUE_OPTIMIZED

Special sentinel for value-based optimization:

```python
from brainsmith.dataflow import VALUE_OPTIMIZED

datatype=VALUE_OPTIMIZED
```

**Use when:** System should analyze tensor values and optimize bitwidth accordingly (rare, advanced use case).


## Dimension Derivation

Copy dimensions from other interfaces to ensure compatible parallelization:

```python
from brainsmith.dataflow.spec_helpers import derive_dim
from brainsmith.dataflow.types import ShapeHierarchy

stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)]
```

**Parameters:**
- `interface_name` - Name of interface to copy from
- `hierarchy` - Which shape to copy: `ShapeHierarchy.STREAM`, `ShapeHierarchy.BLOCK`, `ShapeHierarchy.TENSOR`
- `dim_index` - Dimension index (-1 for last dimension)

**Use when:** Output parallelization must match input parallelization.

**Example - Matching input/output parallelization:**
```python
outputs=[
    df.OutputSchema(
        name="output",
        block_tiling=FULL_SHAPE,
        stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)],
        datatype="input",
    )
]
```

This copies the last stream dimension from the input, ensuring compatible parallelization.


## Quick Reference Tables

### Template Element Types

| Element | Usage | Example | Resolved |
|---------|-------|---------|----------|
| `FULL_DIM` | Process entire dimension (specific position) | `block_tiling=[FULL_DIM, FULL_DIM, "K"]` | Phase 1 (build) |
| `FULL_SHAPE` | Process entire tensor (rank-agnostic) | `block_tiling=FULL_SHAPE` | Phase 1 (build) |
| Integer | Fixed size | `stream_tiling=[1, 1, 16]` | Immediately |
| String | DSE parameter | `stream_tiling=["SIMD"]` | Phase 2 (configure) |
| Callable | Custom logic | `block_tiling=[lambda ctx: ctx.shape[0]//2]` | Phase 1 (build) |

### Datatype Specification Types

| Type | Example | Use Case |
|------|---------|----------|
| `None` | `datatype=None` | Infer from ONNX graph (default) |
| `DataType` | `datatype=DataType["FLOAT32"]` | Fixed datatype |
| `str` | `datatype="input"` | Copy from interface (recommended) |
| `VALUE_OPTIMIZED` | `datatype=VALUE_OPTIMIZED` | Value-based optimization (advanced) |
| Callable | `datatype=custom_dtype` | Custom derivation logic |

### Parameter Types

| Type | Declaration | Navigation | Use Case |
|------|-------------|------------|----------|
| Kernel params | `kernel_params={"epsilon": ("f", True, 1e-5)}` | `get_nodeattr("epsilon")` | Algorithm configuration |
| Tiling params | `stream_tiling=["SIMD"]` | `get_ordered_parameter("SIMD")` | Auto-generated from schema |
| Ordered DSE | `dse_parameters={"depth": ParameterSpec("depth", [128, 256])}` | `get_ordered_parameter("depth")` | Ordered implementation choices |
| Discrete DSE | `dse_parameters={"style": ParameterSpec("style", {"a", "b"})}` | `parameters["style"]` (set) | Categorical choices |


## See Also

- **[Kernel Architecture](kernel-architecture.md)** - Understanding how schemas enable two-phase construction
- **[Kernel Tutorial](kernel-tutorial.md)** - Examples demonstrating schema usage
- **[Dataflow API Reference](../api/dataflow.md)** - Complete programmatic API documentation
