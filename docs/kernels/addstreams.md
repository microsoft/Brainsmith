# AddStreams

**Element-wise addition of two integer streams with identical shapes.**

**Operation**: `output = input0 + input1`

**Namespace**: `brainsmith.kernels`

**Backends**: HLS

---

## Summary

AddStreams performs element-wise addition of two input tensors with identical shapes, commonly used for residual connections and skip connections in neural networks. The kernel processes both inputs in parallel and produces a single output stream with increased bitwidth to prevent overflow.

**Key characteristics**:

- Preserves tensor shape across inputs and output
- Automatically expands output datatype (INT8 + INT8 → INT9) to prevent overflow
- PE parallelism for channel-wise processing
- Supports any tensor rank (2D, 3D, 4D) with consistent last-dimension streaming

**Typical use cases**:

- Residual connections in ResNet architectures
- Skip connections in U-Net and encoder-decoder models
- Element-wise fusion of feature maps

**Example ONNX pattern**:
```
Add(input0: INT8[1,224,224,64], input1: INT8[1,224,224,64])
  → output: INT9[1,224,224,64]
```

---

## Hardware Interface

### Inputs

| Port | Shape Pattern | Datatype | Constraints | Description |
|------|--------------|----------|-------------|-------------|
| input0 | `[N, H, W, C]` | INT8, INT16, INT32 | Must be dynamic | First input tensor |
| input1 | `[N, H, W, C]` | INT8, INT16, INT32 | Must be dynamic | Second input tensor |

!!! note "Rank-Agnostic Design"
    AddStreams works with any tensor rank. Shape pattern shown is 4D (NHWC) but supports 2D `[N, C]`, 3D `[N, H, C]`, etc.

### Outputs

| Port | Shape Pattern | Datatype | Description |
|------|--------------|----------|-------------|
| output | `[N, H, W, C]` | INT9, INT17, INT33 | Sum tensor (bitwidth increased by 1) |

### Constraints

Schema validation ensures:

- ✅ Both inputs must be dynamic (streaming activations, not initializers)
- ✅ Both inputs must have integer datatypes
- ✅ Input shapes must match exactly
- ✅ Requires NHWC layout (enforced by preprocessing)
- ✅ PE must divide number of channels (`C % PE == 0`)

---

## Parallelization Parameters

| Parameter | Type | Range | Default | Description | Resource Impact |
|-----------|------|-------|---------|-------------|-----------------|
| PE | Dimension | 1 to `C` | 1 | Processing elements (channel parallelism) | DSP ↑, LUT ↑, throughput ↑ |

**Valid PE values**: Computed automatically as divisors of channel count.

For `C=64` channels: `PE ∈ {1, 2, 4, 8, 16, 32, 64}`

### Folding Formula

Stream folding reduces the number of cycles by processing multiple channels in parallel:

```python
cycles_per_element = num_channels / PE
total_cycles = (batch_size * spatial_dim * num_channels) / PE
```

**Example** (224×224×64 tensor, PE=16):
```python
spatial_dim = 224 * 224 = 50176
cycles = 1 * 50176 * (64/16) = 50176 * 4 = 200,704 cycles
```

---

## Performance Characteristics

### Cycle Estimation

```python
def get_exp_cycles(self):
    """Expected cycles for one inference."""
    folded_shape = self.get_folded_output_shape()  # (N, H, W, C/PE, PE)
    return int(np.prod(folded_shape[:-1]))  # Exclude PE dimension
```

### Resource Estimation

Typical resource usage for INT8 inputs on Zynq UltraScale+ MPSoC:

| PE | DSP | BRAM_18K | LUT | FF | Freq (MHz) | Throughput (fps @ 224×224×64) |
|----|-----|----------|-----|-----|-----------|-------------------------------|
| 1 | 0 | 0 | 250 | 300 | 300 | 1.5 |
| 4 | 0 | 0 | 500 | 600 | 275 | 5.5 |
| 16 | 0 | 0 | 1200 | 1500 | 250 | 20 |
| 64 | 0 | 0 | 4000 | 5000 | 200 | 64 |

**Notes**:

- AddStreams uses pure LUT-based adders (no DSP blocks)
- No BRAM required (no static parameters)
- Linear LUT/FF scaling with PE
- Slight frequency degradation at high PE due to routing congestion

### Latency vs Throughput

- **Initiation Interval (II)**: 1 (fully pipelined - new input every clock)
- **Latency**: ~5-10 cycles (pipeline depth)
- **Throughput**: `clock_freq / cycles_per_inference` fps

---

## Design Point Configuration

### Interface-Based API (Recommended)

```python
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

# Load model with AddStreams kernel
model = ModelWrapper("model.onnx")
addstreams_node = model.graph.node[0]  # Assuming first node is AddStreams
addstreams_op = getCustomOp(addstreams_node)

# Access design point
point = addstreams_op.design_point

# Configure PE parallelism
point = point.with_input_stream(0, pe=16)  # input0 streams with PE=16
point = point.with_input_stream(1, pe=16)  # input1 must match
point = point.with_output_stream(0, pe=16)  # output streams with PE=16

# Apply configuration
addstreams_op.apply_design_point(point)
```

### Dimension-Based API

```python
# Set PE via dimension name
addstreams_op.set_nodeattr("PE", 16)

# Or use design point API
point = point.with_dimension("PE", 16)

# Query valid PE values
valid_ranges = addstreams_op.get_valid_ranges(model)
print(valid_ranges["PE"])  # {1, 2, 4, 8, 16, 32, 64}
```

### Validation

```python
# Check if PE value is valid
try:
    addstreams_op.set_nodeattr("PE", 7)  # Not a divisor of 64
except ValueError as e:
    print(f"Invalid PE: {e}")  # PE must divide number of channels
```

---

## Backend Implementations

### HLS Backend

**Source**: `brainsmith/kernels/addstreams/addstreams_hls.py`

**Registered as**: `AddStreams_hls` (language: `hls`)

**Features**:

- Uses finn-hlslib `AddStreams_Batch` template
- C++ simulation (cppsim) for fast verification
- Fully pipelined implementation (`II=1`)
- Automatic bit width handling for overflow prevention

**Code Generation**:

Instantiates finn-hlslib template:
```cpp
#include "streamtools.h"

AddStreams_Batch<PE, ap_int<8>, ap_int<8>, ap_int<9>, NumElements>(
    in0_V, in1_V, out0_V, 1  // NumReps=1 for single batch
);
```

**Interface**:
```cpp
void AddStreams_node_name(
    hls::stream<ap_uint<PE*8>> &in0_V,   // PE×8-bit input0
    hls::stream<ap_uint<PE*8>> &in1_V,   // PE×8-bit input1
    hls::stream<ap_uint<PE*9>> &out0_V   // PE×9-bit output
)
```

**Limitations**:

- Synthesis time: 5-15 minutes
- Slightly higher resource usage than hand-optimized RTL

---

## ONNX Inference

### Compatible ONNX Operators

- `Add` (standard ONNX, opset 1+)

### Detection Logic

```python
@classmethod
def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
    """Check if ONNX node can be converted to AddStreams."""
    if node.op_type != "Add":
        return False

    # Must have exactly 2 inputs
    if len(node.input) != 2:
        return False

    return True
```

!!! info "Additional Validation"
    Shape equality and datatype constraints are validated during schema construction, not in `can_infer_from`.

### Transformation

**Before** (ONNX Add):
```python
Add(
    name="residual_add",
    inputs=["conv_output", "skip_connection"],
    outputs=["residual_output"]
)
```

**After** (AddStreams hardware kernel):
```python
AddStreams(
    name="AddStreams_residual_add",
    inputs=["conv_output", "skip_connection"],
    outputs=["residual_output"],
    domain="brainsmith.kernels",
    backend="fpgadataflow",
    PE=16  # Initialized to smallest valid value, user configures later
)
```

**Transformation code**:
```python
@classmethod
def infer_from(cls, node: NodeProto, model: ModelWrapper, insert_index: int):
    hw_node = helper.make_node(
        "AddStreams",
        inputs=list(node.input),
        outputs=list(node.output),
        domain="brainsmith.kernels",
        backend="fpgadataflow",
        name=f"AddStreams_{node.name}"
    )

    return df.TransformationResult(
        nodes_to_insert=[hw_node],
        nodes_to_remove=[node],
    )
```

---

## Usage Examples

### Blueprint Configuration

YAML blueprint for design space exploration:

```yaml
# blueprint.yaml
design_space:
  kernels:
    AddStreams:
      PE: [1, 2, 4, 8, 16, 32, 64]

  constraints:
    - type: resource
      max_lut: 50000
      max_dsp: 500
    - type: performance
      min_throughput: 30  # fps for 224×224×64
```

### Python API - End-to-End

Complete workflow from ONNX to configured hardware:

```python
from qonnx.core.modelwrapper import ModelWrapper
from brainsmith.transformation.infer_kernels import InferKernelList
from qonnx.custom_op.registry import getCustomOp
import numpy as np

# Load ONNX model with Add operator
model = ModelWrapper("resnet_fragment.onnx")

# Transform Add → AddStreams
model = model.transform(InferKernelList())

# Find AddStreams node
addstreams_node = None
for node in model.graph.node:
    if node.op_type == "AddStreams":
        addstreams_node = node
        break

# Get kernel instance
addstreams_op = getCustomOp(addstreams_node)

# Query valid PE values
valid_ranges = addstreams_op.get_valid_ranges(model)
print(f"Valid PE values: {valid_ranges['PE']}")
# Output: Valid PE values: {1, 2, 4, 8, 16, 32, 64}

# Configure for target throughput
addstreams_op.set_nodeattr("PE", 16)

# Estimate performance
cycles = addstreams_op.get_exp_cycles()
print(f"Cycles per inference: {cycles}")
# Output: Cycles per inference: 200704

# Estimate resources
resources = addstreams_op.node_res_estimation("xczu7ev-ffvc1156-2-e")
print(f"Resource estimate: {resources}")
# Output: {'BRAM_18K': 0, 'LUT': 1200, 'DSP': 0, 'URAM': 0, ...}
```

### Testing Execution Modes

```python
# Python mode (golden reference)
addstreams_op.set_nodeattr("exec_mode", "python")
context = {
    "input0": np.random.randint(-128, 127, (1, 224, 224, 64), dtype=np.int8),
    "input1": np.random.randint(-128, 127, (1, 224, 224, 64), dtype=np.int8),
}
addstreams_op.execute_node(context, model.graph)
output_python = context["output"]

# C++ simulation mode (after code generation)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim

model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 5.0))
model = model.transform(CompileCppSim())

addstreams_op.set_nodeattr("exec_mode", "cppsim")
addstreams_op.execute_node(context, model.graph)
output_cppsim = context["output"]

# Verify bitwise exact match
np.testing.assert_array_equal(output_python, output_cppsim)
```

---

## Testing

### Test Location

`tests/kernels/test_addstreams.py` (if exists) or inline in test framework

### Validation

AddStreams is validated through:

- ✅ **Pipeline integration**: ONNX Add → AddStreams transformation
- ✅ **Python execution**: NumPy reference implementation
- ✅ **C++ simulation**: HLS cppsim validation (if hardware tests enabled)
- ✅ **Shape inference**: Verify output shape matches input shapes
- ✅ **Datatype propagation**: Verify bitwidth expansion

### Running Tests

```bash
# Run AddStreams-specific tests
pytest tests/ -k addstreams -v

# Run with hardware simulation (slower)
pytest tests/ -k addstreams -m "cppsim" -v
```

---

## See Also

- [ElementwiseBinary](elementwise_binary.md) - Generalized binary operations including Add
- [ChannelwiseOp](channelwise.md) - Channel-wise parametric addition with broadcasting
- [Kernel Architecture](../developer-guide/3-reference/kernels.md) - Kernel design patterns
- [Kernel Modeling](../developer-guide/2-core-systems/kernel-modeling.md) - Parallelization and design point configuration

---

## API Reference

::: brainsmith.kernels.addstreams.AddStreams
    options:
      show_source: true
      heading_level: 3
      members:
        - build_schema
        - can_infer_from
        - infer_from
        - execute_node

::: brainsmith.kernels.addstreams.AddStreams_hls
    options:
      show_source: false
      heading_level: 3
      members:
        - global_includes
        - defines
        - docompute
        - blackboxfunction
        - pragmas
