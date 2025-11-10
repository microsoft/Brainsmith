# Building Your First Kernel

This chapter walks through creating a complete kernel from scratch. We'll build a **ChannelwiseAdd** operation that adds a per-channel bias to an input tensor.

## The Problem

**Operation:** Elementwise addition with broadcasting

```python
Input:  (1, 224, 224, 64)  # NHWC: Batch, Height, Width, Channels
Bias:   (64,)              # Per-channel bias
Output: (1, 224, 224, 64)  # Same as input

Semantics: output[b,h,w,c] = input[b,h,w,c] + bias[c]
```

**Hardware challenges:**
- Input is dynamic (streaming from DRAM)
- Bias is static (stored in on-chip BRAM)
- Need to broadcast bias across spatial dimensions
- Parallelization on channel dimension

## Step 1: Define the Schema

Start by declaring **what** your kernel needs:

```python
from brainsmith.dataflow import (
    KernelSchema,
    InputSchema,
    OutputSchema,
    FULL_SHAPE,
    DatatypesEqual,
    ShapesEqual,
    ShapeHierarchy,
    TensorDimMatches,
    IsStatic,
    IsDynamic,
)
from brainsmith.dataflow.spec_helpers import add_datatype

CHANNELWISE_ADD_SCHEMA = KernelSchema(
    name="ChannelwiseAdd",

    # === INPUTS ===
    inputs=[
        # Input 0: Activation data (dynamic)
        InputSchema(
            name="input",
            block_tiling=FULL_SHAPE,        # Copy full tensor dimensions
            stream_tiling=["PE"],            # Parallelize on last dimension
            datatype=None,                   # Use from ONNX graph
            required_layout="NHWC"           # Expect NHWC layout
        ),
        # Input 1: Per-channel bias (static)
        InputSchema(
            name="bias",
            block_tiling=FULL_SHAPE,        # Copy bias tensor dims
            stream_tiling=[("input", -1)],   # Match input's parallelism
            datatype=None,                   # Use from ONNX graph
        ),
    ],

    # === OUTPUTS ===
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=FULL_SHAPE,        # Copy output tensor dims
            stream_tiling=[("input", -1)],   # Match input's parallelism
            datatype=add_datatype("input", "bias"),  # Optimized result type
            preserves_input_layout=True      # Output has same layout as input
        ),
    ],

    # === CONSTRAINTS ===
    constraints=[
        # Bias must be static (initializer)
        IsStatic("bias"),

        # Input must be dynamic (activation)
        IsDynamic("input"),

        # Bias must be 1D, matching input's last dimension
        TensorDimMatches("bias", 0, [("input", -1)]),

        # Integer datatypes only (for this simple example)
        # DatatypeInteger(("input", "bias", "output")),
    ],
)
```

**What we've declared:**

1. **Structure:**
   - Two inputs (input, bias)
   - One output
   - All use FULL_SHAPE (adapt to any size)

2. **Parallelization:**
   - Input streams by PE (last dimension)
   - Bias matches input's PE (tuple shorthand)
   - Output matches input's PE

3. **Datatypes:**
   - Input/bias: from ONNX graph
   - Output: optimized addition result (considers actual bias values!)

4. **Requirements:**
   - Bias must be static (stored in BRAM)
   - Input must be dynamic (streaming)
   - Bias shape must match input's channel dimension

## Step 2: Create the Kernel Class

Implement the kernel op that uses this schema:

```python
from brainsmith.dataflow import KernelOp, KernelSchema
from qonnx.core.modelwrapper import ModelWrapper
from onnx import NodeProto

class ChannelwiseAdd(KernelOp):
    """Channelwise addition with broadcasting.

    Adds a per-channel bias to an input tensor:
        output[b,h,w,c] = input[b,h,w,c] + bias[c]

    Hardware implementation:
    - Input streams from DRAM
    - Bias stored in on-chip BRAM
    - Parallel processing of PE channels per cycle
    """

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper) -> KernelSchema:
        """Return the kernel schema.

        For static schemas (same structure for all instances),
        just return the constant.
        """
        return CHANNELWISE_ADD_SCHEMA
```

**That's it!** The base class handles:
- Design space construction (via builder)
- FINN API implementation (get_folded_shape, get_exp_cycles, etc.)
- Caching and invalidation
- Nodeattr registry generation

## Step 3: Register the Kernel

Make it discoverable by the registry:

```python
from brainsmith.registry import kernel

@kernel
class ChannelwiseAdd(KernelOp):
    # ... (same implementation as above)
```

Now you can use: `get_kernel("ChannelwiseAdd")`

## Step 4: Test the Kernel

### Basic Construction

```python
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from onnx import helper, TensorProto

# Create test ONNX graph
input_shape = (1, 224, 224, 64)
bias_shape = (64,)

input_tensor = helper.make_tensor_value_info(
    "input", TensorProto.INT8, input_shape
)
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.INT8, input_shape
)

# Create bias initializer
bias_values = np.random.randint(-10, 10, size=bias_shape, dtype=np.int8)
bias_init = helper.make_tensor(
    "bias", TensorProto.INT8, bias_shape, bias_values.flatten().tolist()
)

# Create node
node = helper.make_node(
    "ChannelwiseAdd",
    inputs=["input", "bias"],
    outputs=["output"],
)

# Create graph
graph = helper.make_graph(
    nodes=[node],
    name="test_channelwise",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[bias_init]
)

model = helper.make_model(graph)
model_wrapper = ModelWrapper(model)

# Instantiate kernel
from brainsmith.kernels.channelwise.channelwise_add import ChannelwiseAdd
op = ChannelwiseAdd(node)

# Initialize design space
op.build_design_space(model_wrapper)

print(f"Design space built!")
print(f"Valid PE values: {op.design_space.get_dimension('PE').values}")
```

**Output:**
```
Design space built!
Valid PE values: (1, 2, 4, 8, 16, 32, 64)
```

### Explore Configurations

```python
# Try different configurations
for pe in [1, 8, 64]:
    point = op.design_space.configure({"PE": pe})
    print(f"PE={pe}: {point.initiation_interval} cycles")

# Output:
# PE=1: 3211264 cycles
# PE=8: 401408 cycles
# PE=64: 50176 cycles
```

### Navigate Design Space

```python
base = op.design_space.configure({"PE": 1})

# Navigate
faster = base.with_max("PE")           # PE=64
balanced = base.with_percentage("PE", 0.5)  # PE=8

# Sweep all values
for point in base.sweep_dimension("PE"):
    speedup = base.initiation_interval / point.initiation_interval
    print(f"PE={point.config['PE']}: {speedup:4.1f}x speedup")
```

Perfect scaling: Doubling PE halves cycles (for this operation).

## Understanding the Results

**Valid PE values** are divisors of the channel dimension (64): {1, 2, 4, 8, 16, 32, 64}. Non-divisors like 5 are invalid (can't process 0.8 of an element per cycle).

**Cycle calculation**: total_elements / elements_per_cycle = 3,211,264 / 8 = 401,408 cycles (for PE=8).

**Datatype optimization**: `add_datatype("input", "bias")` analyzes actual bias values [-10, 10] and determines INT8 output suffices. If bias values were [-100, 100], output would widen to INT9.

## Step 5: Add Inference Support (Optional)

To automatically convert ONNX Add nodes to ChannelwiseAdd:

```python
from brainsmith.dataflow import TransformationResult
from onnx import helper

class ChannelwiseAdd(KernelOp):
    # ... (previous code)

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if this ONNX Add node qualifies for ChannelwiseAdd.

        Requirements:
        - Op type: Add
        - Second input is static (initializer)
        - Second input is 1D
        - Second input size matches first input's last dimension
        """
        if node.op_type != "Add":
            return False

        # Check second input is static
        input1_name = node.input[1]
        if model.get_initializer(input1_name) is None:
            return False

        # Check shapes
        input0_shape = model.get_tensor_shape(node.input[0])
        input1_shape = model.get_tensor_shape(node.input[1])

        # Bias must be 1D and match last dim of input
        if len(input1_shape) != 1:
            return False
        if input1_shape[0] != input0_shape[-1]:
            return False

        return True

    @classmethod
    def infer_from(cls, node: NodeProto, model: ModelWrapper, insert_index: int) -> TransformationResult:
        """Convert ONNX Add to ChannelwiseAdd."""
        # Create new node with same inputs/outputs
        hw_node = helper.make_node(
            "ChannelwiseAdd",
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name or f"ChannelwiseAdd_{insert_index}"
        )

        return TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node]
        )
```

Now use with InferKernelList:

```python
from brainsmith.steps import InferKernelList

model = ... # ONNX model with Add nodes
model = InferKernelList().apply(model)
# Add nodes meeting requirements → ChannelwiseAdd nodes
```

## Common Patterns

### Pattern 1: Same Input/Output Shapes

```python
# Schema:
inputs=[
    InputSchema(name="input", block_tiling=FULL_SHAPE, ...)
],
outputs=[
    OutputSchema(name="output", block_tiling=FULL_SHAPE, ...)
],
constraints=[
    ShapesEqual(("input", "output"))
]
```

### Pattern 2: Matching Parallelization

```python
# Output matches input's stream:
outputs=[
    OutputSchema(
        name="output",
        stream_tiling=[("input", -1)]  # Copy input's last dim parallelism
    )
]
```

### Pattern 3: Static + Dynamic Inputs

```python
inputs=[
    InputSchema(name="input", ...),   # Dynamic (streaming)
    InputSchema(name="weights", ...),  # Static (buffered)
],
constraints=[
    IsDynamic("input"),
    IsStatic("weights")
]
```

### Pattern 4: Context-Aware Datatypes

```python
from brainsmith.dataflow.spec_helpers import add_datatype, mul_datatype

# Addition output:
OutputSchema(
    datatype=add_datatype("input", "bias")  # Optimized for actual values
)

# Multiplication output:
OutputSchema(
    datatype=mul_datatype("input", "weights")
)
```

## Troubleshooting

### Issue: "Parameter 'PE' not found in nodeattrs"

**Cause:** Forgot to initialize design space before accessing.

**Fix:**
```python
# Before:
op = ChannelwiseAdd(node)
op.design_space  # ERROR: not initialized

# After:
op = ChannelwiseAdd(node)
op.build_design_space(model_wrapper)  # Initialize!
op.design_space  # Works
```

Or access via a method that auto-initializes:
```python
op.get_valid_ranges(model_wrapper)  # Auto-initializes
```

### Issue: "Invalid PE=5. Valid range: [1, 64]"

**Cause:** Trying to set PE to a value that doesn't divide the dimension.

**Fix:**
```python
# Check valid values first:
valid = op.design_space.get_dimension("PE")
print(f"Valid PE: {valid.values}")

# Only use valid values:
point = op.design_space.configure({"PE": 8})  # ✓
point = op.design_space.configure({"PE": 5})  # ✗ Invalid
```

### Issue: "ShapesEqual constraint failed"

**Cause:** Schema constraint violated (shapes don't match).

**Fix:**
Check the ONNX graph shapes match your expectations:
```python
input_shape = model_wrapper.get_tensor_shape(node.input[0])
output_shape = model_wrapper.get_tensor_shape(node.output[0])
print(f"Input: {input_shape}, Output: {output_shape}")

# If they don't match, your schema constraint is wrong
# or the ONNX graph is incorrect
```

### Issue: "Design space invalidated after set_nodeattr"

**Cause:** Changed a structural parameter (datatype, shape).

**Expected behavior:** This is correct! Structural changes require rebuild.

**Workaround:**
```python
# If doing DSE, configure point instead of setting nodeattr:
point = design_space.configure({"PE": 64})  # Doesn't affect op nodeattrs

# When ready to commit:
op.apply_design_point(point)  # Syncs to nodeattrs without invalidation
```

## Next Steps

You've built a complete kernel! You now know:

✓ How to define a KernelSchema
✓ How to create a KernelOp class
✓ How to test and explore configurations
✓ How to navigate design space
✓ Common patterns and troubleshooting

**Next chapter:** Deep dive into schema design - templates, constraints, and advanced patterns.
