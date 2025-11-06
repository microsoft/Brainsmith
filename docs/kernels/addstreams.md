# AddStreams

Element-wise addition of two integer streams with identical shapes.

**Operation**: `output = input0 + input1`

**Namespace**: `brainsmith.kernels`

**ONNX Inference**: `Add` (both inputs dynamic, no broadcasting)

---

## Inputs

| Port | Datatype | Block Tiling | Stream Tiling | Dynamic/Static | Constraints |
|------|----------|--------------|---------------|----------------|-------------|
| input0 | INT8, INT16, INT32 | `FULL_SHAPE` | `["PE"]` | Dynamic | Integer, dynamic, NHWC layout |
| input1 | INT8, INT16, INT32 | `FULL_SHAPE` | `["PE"]` | Dynamic | Integer, dynamic, NHWC layout, `shape == input0`, `dtype == input0` |

!!! note "Rank-Agnostic Design"
    `FULL_SHAPE` expands to match tensor rank: 4D → `[FULL_DIM, FULL_DIM, FULL_DIM, FULL_DIM]`, 3D → `[FULL_DIM, FULL_DIM, FULL_DIM]`, etc.
    Stream tiling `["PE"]` auto-pads to match block rank: 4D → `[1, 1, 1, "PE"]`

## Outputs

| Port | Datatype | Block Tiling | Stream Tiling | Derivation Rule |
|------|----------|--------------|---------------|-----------------|
| output | INT9, INT17, INT33 | `FULL_SHAPE` | `[("input0", -1)]` | `add_datatype("input0", "input1")` |

**Datatype Derivation**: `smallest_datatype_for_range(input0_min + input1_min, input0_max + input1_max)`

**Stream Tiling**: `[("input0", -1)]` derives last dimension from input0's stream shape (effectively `["PE"]` after normalization)

## Parallelization

| Dimension | Type | Valid Values | Derivation |
|-----------|------|--------------|------------|
| PE | Tiling | `divisors(C)` | Channel parallelism: `stream_tiling = [1, 1, 1, "PE"]` |

**Folding**: `cycles = (N × H × W × C) / PE`

**Example**: For `C=64` channels: `PE ∈ {1, 2, 4, 8, 16, 32, 64}`

## Backends

| Backend | Language | Status |
|---------|----------|--------|
| [HLS](addstreams-hls.md) | C++/HLS | ✅ Stable |

---

## Schema Definition

### Constraints

Schema validation enforces:

- ✅ Both inputs must be dynamic (streaming activations, not initializers)
- ✅ Both inputs must have integer datatypes
- ✅ Input shapes must match exactly
- ✅ Input datatypes must match exactly
- ✅ Requires NHWC layout (enforced by preprocessing)
- ✅ PE must divide number of channels (`C % PE == 0`)

---

## Dataflow Interface

### Stream Configuration

| Interface | Direction | Parallelization | Description |
|-----------|-----------|-----------------|-------------|
| input0 | Input | PE channels | First input stream with PE-parallel channels |
| input1 | Input | PE channels | Second input stream (must match input0 PE) |
| output | Output | PE channels | Result stream with PE-parallel channels |

**PE consistency**: All three streams must use the same PE value for proper alignment.

### Folding Semantics

```python
# Cycle calculation
cycles_per_inference = (N * H * W * C) / PE
```

**Explanation**: The kernel processes `PE` channels in parallel per clock cycle. Higher PE reduces the number of cycles proportionally but increases hardware resources. The batch dimension (N), spatial dimensions (H, W), and channel folding factor (C/PE) determine total cycle count.

**Example** (224×224×64 tensor, PE=16):
```python
spatial_elements = 224 * 224 = 50,176
cycles = 1 * 50,176 * (64/16) = 200,704 cycles
```

---

## ONNX Inference

### Compatible ONNX Operators

- `Add` (standard ONNX, opset 1+)

**Additional requirements**:

- Both inputs must be dynamic tensors (not initializers)
- Input shapes must be identical (no broadcasting)
- Integer datatypes only

### Detection Logic

This kernel is inferred from ONNX when:

- Node op_type is `Add`
- Exactly 2 inputs
- Both inputs are dynamic (streaming)
- Input shapes match exactly

!!! info "Shape and Datatype Validation"
    Shape equality and datatype constraints are validated during schema construction, not in initial detection.

### Transformation

**Before** (ONNX):
```python
Add(
    name="residual_add",
    inputs=["conv_output", "skip_connection"],
    outputs=["residual_output"]
)
```

**After** (AddStreams KernelOp):
```python
AddStreams(
    name="AddStreams_residual_add",
    inputs=["conv_output", "skip_connection"],
    outputs=["residual_output"],
    domain="brainsmith.kernels",
    backend="fpgadataflow",
    PE=1  # Initialized to smallest valid value
)
```

**Initialization**: The PE parameter is initialized to 1 (minimum parallelism). Users configure the design point during design space exploration or manual optimization.

---

## Available Backends

| Backend | Language | Status | Documentation |
|---------|----------|--------|---------------|
| HLS | C++/HLS | ✅ Stable | [AddStreams_hls](addstreams-hls.md) |

---

## See Also

- [AddStreams HLS Backend](addstreams-hls.md) - Hardware implementation details
- [ElementwiseBinary](elementwise_binary.md) - Generalized binary operations including Add
- [ChannelwiseOp](channelwise.md) - Channel-wise parametric addition with broadcasting
- [Kernel Architecture](../developer-guide/3-reference/kernels.md) - Kernel design patterns
- [Kernel Modeling](../developer-guide/2-core-systems/kernel-modeling.md) - Design point configuration

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
