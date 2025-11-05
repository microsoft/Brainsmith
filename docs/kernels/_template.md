# KernelName

<!-- TEMPLATE FILE - DO NOT PUBLISH -->
<!-- This template provides a consistent structure for kernel documentation -->
<!-- Copy and customize for each kernel -->

**Brief one-sentence description of what this kernel does.**

**Operation**: Mathematical formula or algorithmic description

**Namespace**: `brainsmith.kernels` | `finn` | `project`

**Backends**: HLS | RTL | Python

---

## Summary

Detailed description of the kernel's purpose, typical use cases in neural networks, and key characteristics.

Example ONNX pattern this kernel handles:
```
OperatorName(input: DTYPE[SHAPE]) -> output: DTYPE[SHAPE]
```

---

## Hardware Interface

### Inputs

| Port | Shape Pattern | Datatype | Constraints | Description |
|------|--------------|----------|-------------|-------------|
| input0 | `[N, H, W, C]` | INT8, INT16 | Must be dynamic | Primary input tensor |
| parameters | `[C]` or `[1]` | INT8, INT16 | Must be static | Static parameter tensor |

### Outputs

| Port | Shape Pattern | Datatype | Description |
|------|--------------|----------|-------------|
| output | `[N, H, W, C]` | INT8, INT16 | Result tensor |

### Constraints

List all schema constraints:

- Input datatype must be integer
- Shapes must match exactly
- Requires NHWC layout
- PE must divide number of channels

---

## Parallelization Parameters

| Parameter | Type | Range | Default | Description | Resource Impact |
|-----------|------|-------|---------|-------------|-----------------|
| PE | Dimension | 1-1024 | 1 | Processing elements (channel parallelism) | DSP, LUT, FF ↑ |
| SIMD | Dimension | 1-256 | 1 | SIMD width for operations | Multipliers ↑ |
| ram_style | DSE Param | {auto, block, distributed, ultra} | auto | Memory implementation | BRAM vs LUT tradeoff |

### Folding Formula

Mathematical relationship between parallelization and execution time:

```python
cycles_per_inference = (num_channels / PE) * (spatial_dim) * batch_size
throughput = clock_freq / cycles_per_inference
```

---

## Performance Characteristics

### Cycle Estimation

Describe the cycle count computation:

```python
def get_exp_cycles(self):
    """Expected cycles for one inference."""
    return (num_elements / PE) * batch_size
```

### Resource Estimation

Typical resource usage for common configurations:

| PE | SIMD | DSP | BRAM_18K | URAM | LUT | FF | Freq (MHz) |
|----|------|-----|----------|------|-----|-----|-----------|
| 1 | 1 | 1 | 2 | 0 | 500 | 1000 | 250 |
| 8 | 8 | 64 | 16 | 0 | 2000 | 4000 | 200 |
| 16 | 16 | 256 | 32 | 0 | 4000 | 8000 | 175 |

**Notes**:
- DSP usage scales linearly with PE×SIMD for multiply operations
- BRAM usage depends on ram_style selection
- Frequency typically decreases with higher parallelism

### Latency vs Throughput

- **Initiation Interval (II)**: Cycles between processing new inputs (typically 1 for pipelined designs)
- **Latency**: Cycles from input to output for single inference
- **Throughput**: Inferences per second = `clock_freq / (II * cycles_per_inference)`

---

## Design Point Configuration

### Interface-Based API (Recommended)

Configure stream parallelism using the dataflow modeling interface:

```python
from brainsmith.registry import get_kernel

# Get kernel class
KernelClass = get_kernel('KernelName')

# Access design point
kernel_op = get_custom_op(node)
point = kernel_op.design_point

# Configure input stream parallelism
point = point.with_input_stream(0, pe=16)

# Configure output stream parallelism
point = point.with_output_stream(0, pe=16)

# Apply configuration
kernel_op.apply_design_point(point)
```

### Dimension-Based API (Alternative)

Configure by dimension name for DSE:

```python
# Set specific dimension values
point = point.with_dimension("PE", 16)
point = point.with_dimension("SIMD", 8)
point = point.with_dimension("ram_style", "block")

# Increase/decrease dimensions
point = point.increase_dimension("PE", factor=2)  # 16 → 32
point = point.decrease_dimension("SIMD", factor=2)  # 8 → 4
```

### Validation

```python
# Query valid ranges
valid_ranges = kernel_op.get_valid_ranges(model)
print(valid_ranges)
# {'PE': {1, 2, 4, 8, 16, 32}, 'SIMD': {1, 2, 4}, 'ram_style': {'auto', 'block', 'distributed'}}

# Check if configuration is valid
try:
    point = point.with_dimension("PE", 128)
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

---

## Backend Implementations

### HLS Backend

**Source**: `brainsmith/kernels/kernelname/kernelname_hls.py`

**Registered as**: `KernelName_hls` (language: `hls`)

**Features**:

- C++ simulation (cppsim) for fast verification
- Automatic pipelining with `#pragma HLS pipeline II=1`
- DSP inference from operations
- Stream-based interfaces (`hls::stream<ap_uint<>>`)

**Code Generation**:

Uses finn-hlslib templates:
```cpp
#include "streamtools.h"

// Template instantiation
KernelOperation<PE, SIMD, InputType, OutputType>(
    in0_V, out0_V, parameters
);
```

**Limitations**:

- Higher resource usage than hand-optimized RTL
- Less control over timing closure
- Synthesis time 10-30 minutes

### RTL Backend (if available)

**Source**: `finn-rtllib/kernelname/hdl/kernelname.sv`

**Registered as**: `KernelName_rtl` (language: `rtl`)

**Features**:

- Hand-optimized for minimal resource usage
- Precise timing control
- Pre-verified IP blocks
- Fast simulation with compiled .so libraries

**Architecture**:

Describe RTL-specific implementation details (e.g., binary search tree, pipelined datapath, etc.)

**Limitations**:

- Longer development time
- Harder to modify or extend
- Simulation slower than C++ (cppsim)

### Backend Comparison

| Aspect | HLS | RTL |
|--------|-----|-----|
| **Development Time** | Hours | Days-Weeks |
| **Resource Efficiency** | Moderate | High |
| **Timing Predictability** | Low | High |
| **Simulation Speed** | Fast (cppsim) | Slow (rtlsim) |
| **Flexibility** | Easy to modify | Requires HDL expertise |
| **Use Case** | Rapid prototyping | Production optimization |

---

## ONNX Inference

### Compatible ONNX Operators

List ONNX operators that can be transformed into this kernel:

- `OperatorName` (standard ONNX)
- `FuncOperatorName` (custom domain)

### Detection Logic

```python
@classmethod
def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
    """Check if ONNX node matches this kernel's pattern."""
    if node.op_type != "TargetOp":
        return False

    # Check structural constraints
    # Example: verify input count
    if len(node.input) != 2:
        return False

    # Check shape compatibility
    # Example: ensure inputs have same shape
    in0_shape = model.get_tensor_shape(node.input[0])
    in1_shape = model.get_tensor_shape(node.input[1])
    if in0_shape != in1_shape:
        return False

    return True
```

### Transformation

Example showing ONNX graph before and after transformation:

**Before** (ONNX standard operator):
```
Add(
    input0: INT8[1,224,224,64],
    input1: INT8[1,224,224,64]
) -> output: INT8[1,224,224,64]
```

**After** (Hardware kernel):
```
KernelName(
    input0: INT8[1,224,224,64],
    input1: INT8[1,224,224,64],
    domain="brainsmith.kernels",
    backend="fpgadataflow",
    PE=16
) -> output: INT8[1,224,224,64]
```

**Transformation code**:
```python
@classmethod
def infer_from(cls, node: NodeProto, model: ModelWrapper, insert_index: int):
    """Transform ONNX node to hardware kernel node."""
    hw_node = helper.make_node(
        "KernelName",
        inputs=list(node.input),
        outputs=list(node.output),
        domain="brainsmith.kernels",
        backend="fpgadataflow",
        name=f"KernelName_{node.name}",
        # Kernel-specific parameters
        param1=value1,
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
    KernelName:
      PE: [1, 2, 4, 8, 16]
      SIMD: [1, 2, 4]
      ram_style: ["distributed", "block"]

  constraints:
    - type: resource
      max_lut: 100000
      max_dsp: 500
    - type: performance
      min_throughput: 1000  # inferences/sec
```

### Python API

Complete example from ONNX model to configured hardware:

```python
from qonnx.core.modelwrapper import ModelWrapper
from brainsmith.dse import explore_design_space
from brainsmith.registry import get_kernel

# Load ONNX model
model = ModelWrapper("model.onnx")

# Run kernel inference transformation
from brainsmith.transformation.infer_kernels import InferKernelList
model = model.transform(InferKernelList())

# Get kernel node
kernel_node = model.graph.node[0]  # Assuming first node
kernel_op = get_custom_op(kernel_node)

# Query valid configurations
valid_ranges = kernel_op.get_valid_ranges(model)
print(f"Valid PE values: {valid_ranges['PE']}")

# Configure manually
kernel_op.set_nodeattr("PE", 16)
kernel_op.set_nodeattr("ram_style", "block")

# Or explore design space automatically
results = explore_design_space(model, "blueprint.yaml")
best_config = results.get_pareto_optimal()[0]
```

### Testing Kernel Behavior

```python
# Execute in Python mode (golden reference)
kernel_op.set_nodeattr("exec_mode", "python")
context = {
    "input0": np.random.randn(1, 224, 224, 64).astype(np.float32),
}
kernel_op.execute_node(context, model.graph)
output_python = context["output0"]

# Execute in C++ simulation mode
kernel_op.set_nodeattr("exec_mode", "cppsim")
# ... code generation and compilation ...
kernel_op.execute_node(context, model.graph)
output_cppsim = context["output0"]

# Verify match
np.testing.assert_allclose(output_python, output_cppsim, rtol=1e-5)
```

---

## Testing

### Test Location

`tests/kernels/test_kernelname.py` or `tests/kernels/test_kernelname_backend.py`

### Test Framework

Uses composition-based test framework:

```python
from tests.frameworks.single_kernel_test import SingleKernelTest

class TestKernelName(SingleKernelTest):
    """Test KernelName kernel with all validation modes."""

    def make_test_model(self):
        """Create test ONNX model."""
        # ... model creation ...
        return model, node_name

    def get_kernel_inference_transform(self):
        """Return inference transformation."""
        from brainsmith.kernels.kernelname import InferKernelName
        return InferKernelName

    def compute_golden_reference(self, inputs):
        """Compute expected outputs."""
        # ... golden reference computation ...
        return {"output": expected}
```

### Validation Modes

Inherited tests automatically validate:

- ✅ **Pipeline integration**: Kernel inference and graph transformation
- ✅ **Python execution**: Reference implementation correctness
- ✅ **C++ simulation** (if `get_backend_fpgapart()` defined): HLS cppsim validation
- ✅ **RTL simulation** (if backend supports): RTL rtlsim validation
- ✅ **Resource estimation**: BRAM/LUT/DSP calculation accuracy
- ✅ **Performance**: Cycle count validation

### Running Tests

```bash
# Run all kernel tests
pytest tests/kernels/test_kernelname.py -v

# Run only fast tests (skip cppsim/rtlsim)
pytest tests/kernels/test_kernelname.py -m "not slow" -v

# Run with coverage
pytest tests/kernels/test_kernelname.py --cov=brainsmith.kernels.kernelname -v
```

---

## Related Kernels

- [RelatedKernel1](relatedkernel1.md) - Similar operation or complementary function
- [RelatedKernel2](relatedkernel2.md) - Alternative implementation

---

## See Also

- [Kernel Architecture](../developer-guide/3-reference/kernel-architecture.md) - Conceptual overview
- [Design Space Exploration](../developer-guide/2-core-systems/design-space-exploration.md) - DSE guide
- [Parallelization Guide](../developer-guide/2-core-systems/parallelization.md) - PE/SIMD patterns
- [HLS Backend Development](../developer-guide/backends/hls-backend.md) - Creating HLS backends

---

## API Reference

::: brainsmith.kernels.kernelname.KernelName
    options:
      show_source: true
      heading_level: 3
      members:
        - build_schema
        - can_infer_from
        - infer_from
        - execute_node
        - get_exp_cycles

::: brainsmith.kernels.kernelname.KernelName_hls
    options:
      show_source: false
      heading_level: 3
      members:
        - generate_params
        - global_includes
        - defines
        - docompute
        - blackboxfunction
        - pragmas
