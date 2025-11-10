# Hardware Kernels

## Introduction

**Hardware kernels are the fundamental building blocks of dataflow accelerators.** Each kernel implements one or more neural network operations (LayerNorm, MatMul, Attention) as a self-contained streaming circuit. Brainsmith constructs dataflow accelerators by iteratively applying graph transformations to lower ONNX nodes to matching Kernels, connected via streaming interfaces. During this process, **kernels are modeled by the relationship between their input and output streams** with the internal architecture largely abstracted away.

<div align="center" markdown>
![Kernel Interfaces](../images/dataflow_kernel.png){ width="400" }
</div>

This separation allows for hand-crafted kernel implementations to exploit hardware expertise (fused kernels, systolic arrays), while automated design space exploration optimizes both kernel parameterization (parallelism, memory depth, bit-widths) and dataflow infrastructure for the target graph.

This architectural choice raises a fundamental design question: **at what level of granularity should we segment the computational graph?** A MatMul operation could be decomposed into thousands of individual multiply-add primitives, or preserved as a single layer-level kernel. Brainsmith uses layer-level granularity rather than fine-grained primitives:

1. **Preserve Design Space** - Neural networks are expressed as layers in PyTorch/ONNX. Preserving this granularity enables natural extension from AI frameworks while maintaining semantic meaning through the compilation pipeline.

2. **Prevent Exponential Explosion** - Decomposing layers into individual operations (add, multiply, shift) creates thousands of tiny blocks, making design space exploration computationally intractable.

3. **Enable Hand Optimization** - Hardware engineers can optimize kernels at the layer scale without deep AI model knowledge, achieving performance that auto-generated designs cannot match while maintaining flexibility through automated composition.

---

## Data Hierarchy


The stream relationships that define Kernels in the compiler require systematic refinement to connect ONNX's model-level abstractions to RTL's cycle-accurate streaming implementation. ONNX describes computation in terms of complete tensors (a (1, 224, 224, 64) activation map), while RTL executes on bit-streams cycling through hardware (16 elements per clock).

Brainsmith bridges this semantic gap through a three-tier hierarchy, each level refining the data representation for hardware realization:

<div align="center" markdown>
![Input Chunking](../images/input_chunking.png){ width="600" }
</div>

### Tensor / Block / Stream

**TENSOR** - Complete dimensions from ONNX graph (e.g., `(1, 224, 224, 64)`)

- Defines functional correctness and accuracy requirements

**BLOCK** - Kernel's atomic computation unit (e.g., `(1, 7, 7, 64)`)

- Data required for one cycle of the kernel's computation state
- Controls memory footprint, pipeline depth, and latency

**STREAM** - Elements processed per clock cycle (e.g., 16 8-bit Integers)

- Determines throughput and resource usage
- Constrained by BLOCK shape (cannot exceed block dimensions)

### Kernel-Specific Block Semantics

The BLOCK shape adapts to kernel computation characteristics:

**Simple kernels** (elementwise operations like Add, Multiply):

- BLOCK = TENSOR (entire input processed as one unit)
- No intermediate computation state between elements
- STREAM parallelization limited only by resource availability

**Complex kernels** (reduction operations like MatMul, LayerNorm):

- BLOCK = one quantum of the calculation state
- Example: Dot product BLOCK = one input vector
- Kernel must accumulate/reduce across BLOCK before producing output
- STREAM parallelization cannot exceed BLOCK dimensions

This constraint exists because **kernels process one BLOCK at a time**. A MatMul with input vectors of length 64 can parallelize up to STREAM=64, but STREAM=128 is invalid—the kernel's computation state operates on 64-element vectors. Higher throughput requires instantiating multiple parallel kernel instances, each processing independent BLOCKs.

### Lowering from ONNX to RTL

The hierarchy enables automatic derivation of hardware execution characteristics:

```python
# Spatial decomposition
tensor_blocks = ceil(tensor_dim / block_dim)

# Temporal execution
stream_cycles = ceil(block_dim / stream_dim)

# Total latency
total_cycles = prod(tensor_blocks) × prod(stream_cycles)
```

**Example:** For tensor `(100, 64)`, block `(32, 16)`, stream `(8, 4)`:

- Tensor blocks: `(4, 4)` → 16 blocks cover the full tensor
- Stream cycles: `(4, 4)` → 16 cycles stream each block
- Total cycles: 256

### Inter-Kernel Dataflow

<div align="center" markdown>
![Dataflow Chunking with FIFO](../images/dataflow_chunking_fifo.png){ width="700" }
</div>

Kernels communicate via **streaming interfaces**, producing and consuming data cycle-by-cycle. Elastic FIFOs between kernels accumulate these streams as **data blocks** for buffering, then stream them out to downstream consumers. This infrastructure automatically adapts to different kernel semantics through shape-driven buffering.

#### Composing Kernels with Different Block Semantics

Consider a simple pipeline: `Add → LayerNorm → Softmax`

```
Add (elementwise)           LayerNorm (reduction)       Softmax (reduction)
BLOCK = TENSOR (1,224,64)   BLOCK = (1,1,64)           BLOCK = (1,1,N_classes)
STREAM = (1,1,16)           STREAM = (1,1,16)          STREAM = (1,1,8)
```

**What happens at kernel boundaries:**

1. **Add → LayerNorm**: Producer outputs (1,224,64) blocks, consumer expects (1,1,64) blocks

   - FIFO buffers shape transformation
   - Add streams 14 blocks × 16 cycles each = 224 cycles
   - LayerNorm consumes in (1,1,64) chunks, computing normalization per spatial position

2. **LayerNorm → Softmax**: Block shapes may differ based on computation semantics

   - Each kernel's BLOCK reflects its reduction domain
   - FIFOs provide elastic buffering for rate adaptation

#### Automatic Infrastructure Derivation

Block and stream shapes drive hardware generation:

**FIFO Depths** - Determined by producer/consumer rate mismatch:
```python
producer_rate = prod(block_shape) / prod(stream_shape)  # cycles per block
consumer_rate = # depends on consumer's internal pipeline depth
fifo_depth = max(producer_burst, consumer_backpressure_tolerance)
```

!!! note "Implementation Reality"
    The formula above describes the theoretical basis for FIFO sizing. In practice, the current implementation uses **iterative cycle-accurate RTL simulation** to converge on efficient FIFO depths. This will be expanded on in future releases.

**Stream Width Matching** - Interface widths must align:

- Producer STREAM=(1,1,16) × INT8 → 128-bit AXI-Stream
- Consumer expects matching width or automatic width conversion
- Datatype changes (INT8 → FLOAT32) insert conversion logic

**Rate Mismatch Handling** - Kernels may have different throughputs:

- Elementwise: 1 output per cycle (after initial latency)
- Reduction: Multiple cycles per output (accumulation phase)
- FIFOs absorb transient rate differences, prevent pipeline stalls

#### Schema-Driven Interface Resolution

Schemas declare interfaces using templates that adapt to runtime shapes:

```python
# Producer (Add kernel)
OutputSchema(
    block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM],  # Process entire spatial dims
    stream_tiling=[1, 1, "PE"]                      # Parallelize over channels
)

# Consumer (LayerNorm kernel)
InputSchema(
    block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM],  # Same spatial processing
    stream_tiling=[1, 1, "SIMD"]                    # Match channel parallelism
)
```

At compile time:

1. ONNX tensor shapes resolve `FULL_DIM` → actual dimensions
2. DSE parameters (PE=16, SIMD=16) resolve stream tiling
3. Infrastructure generates FIFOs matching computed shapes
4. Validation ensures producer/consumer compatibility

This declarative approach separates *what* (tensor semantics) from *how* (hardware implementation), enabling design space exploration while maintaining correctness by construction. The compiler automatically inserts width converters, reshaping logic, and elastic buffering as needed.

---

## Compiler Integration

Brainsmith kernels use a schema-driven architecture with three files:

```
brainsmith/kernels/layernorm/
├── layernorm.py          # Schema + KernelOp + Inference transform
├── layernorm_hls.py      # HLS backend (code generation)
└── layernorm.hpp         # HLS implementation (C++)
```

- **KernelSchema** defines interface structure, validation constraints, and design space parameters. The schema drives automatic design space construction and validation.
- **KernelOp** provides state management and transformation from ONNX to hardware nodes via `can_infer_from()` and `infer_from()` class methods.
- **Backend** inherits from both KernelOp (for schema access) and HLSBackend/RTLBackend (for code generation), using multiple inheritance to combine interface structure with build infrastructure.

Components register via `@kernel` and `@backend` decorators for automatic discovery. See [Component Registry](registry.md) for registration details.

---

## Example: LayerNorm Kernel

A complete kernel implementation showing schema definition, transformation, and backend code generation.

### Schema Definition

`brainsmith/kernels/layernorm/layernorm.py` defines structure declaratively:

```python
import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM
from brainsmith.dataflow.spec_helpers import derive_dim
from brainsmith.dataflow.types import ShapeHierarchy

LAYERNORM_SCHEMA = df.KernelSchema(
    name="LayerNorm",

    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],         # Process full spatial dimensions
            stream_tiling=["SIMD"],          # Parallelize over channels
            required_layout="NHWC",
        )
    ],

    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],
            stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)],
            datatype=df.constant_datatype("FLOAT32"),
            required_layout="NHWC",
        )
    ],

    kernel_params={
        "epsilon": ("f", True, 1e-5),
    },

    constraints=[
        df.AttrCompare("epsilon", ">", 0),
    ],
)
```

The schema automatically generates:

- `SIMD` parallelization parameter from `stream_tiling`
- Input/output datatype attributes
- Validation constraints
- Design space for exploration

### Kernel Operator

```python
from brainsmith.dataflow import KernelOp
from brainsmith.registry import kernel

@kernel(description="Hardware LayerNorm", author="Shane Fleming")
class LayerNorm(KernelOp):
    """LayerNorm kernel with schema-driven design."""

    @classmethod
    def build_schema(cls, node, model):
        return LAYERNORM_SCHEMA

    @classmethod
    def can_infer_from(cls, node, model):
        """Check if ONNX node can convert to this kernel."""
        if node.op_type != "FuncLayerNorm":
            return False

        axis_attr = get_by_name(node.attribute, "axis")
        return axis_attr is None or axis_attr.i == -1

    @classmethod
    def infer_from(cls, node, model, insert_index):
        """Transform ONNX node to hardware kernel."""
        epsilon_attr = get_by_name(node.attribute, "epsilon")
        epsilon = epsilon_attr.f if epsilon_attr else 1e-5

        hw_node = helper.make_node(
            "LayerNorm",
            inputs=list(node.input),
            outputs=list(node.output),
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            epsilon=epsilon,
        )

        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node]
        )

    def execute_node(self, context, graph):
        """Reference numpy implementation for validation."""
        in_values = context[self.onnx_node.input[0]]
        epsilon = self.get_nodeattr("epsilon")

        mean = np.mean(in_values, axis=-1, keepdims=True)
        var = np.var(in_values, axis=-1, keepdims=True)
        normalized = (in_values - mean) / np.sqrt(var + epsilon)

        context[self.onnx_node.output[0]] = normalized.astype(np.float32)
```

### Backend Implementation

`brainsmith/kernels/layernorm/layernorm_hls.py` provides code generation:

```python
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.registry import backend

@backend(
    target_kernel="brainsmith:LayerNorm",
    language="hls",
    description="HLS backend for LayerNorm",
    author="Shane Fleming",
)
class LayerNorm_hls(LayerNorm, HLSBackend):
    """HLS backend combining kernel schema with HLS infrastructure."""

    def global_includes(self):
        """C++ includes for generated code."""
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_vector.h>",
            '#include "layernorm.hpp"',
            '#include "bs_utils.hpp"'
        ]

    def defines(self, var):
        """C++ constants extracted from design_point."""
        point = self.design_point
        simd = point.inputs["input"].stream_shape[-1]
        width = point.inputs["input"].tensor_shape[-1]
        epsilon = self.get_nodeattr("epsilon")

        self.code_gen_dict["$DEFINES$"] = [
            f"#define SIMD {simd}",
            f"#define W {width}",
            f"#define epsilon {epsilon}",
        ]

    def docompute(self):
        """Computation kernel invocation."""
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "layernorm<SIMD, W, TI, TO>(in0, out, epsilon);"
        ]
```

Multiple inheritance provides:

- `LayerNorm` → kernel_schema, design_space, design_point
- `HLSBackend` → cppsim, rtlsim, code generation infrastructure

### Hardware Implementation

`brainsmith/kernels/layernorm/layernorm.hpp` contains the HLS C++ implementation:

```cpp
template<unsigned SIMD, unsigned W, typename TI, typename TO>
void layernorm(
    hls::stream<hls::vector<TI, SIMD>>& in,
    hls::stream<hls::vector<TO, SIMD>>& out,
    float epsilon
) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE s_axilite port=epsilon
#pragma HLS INTERFACE ap_ctrl_none port=return

    // Hardware implementation
    // ...
}
```

Template parameters (SIMD, width, datatypes) come from backend `defines()`. AXI-Stream pragmas create standardized interfaces. Runtime parameters use AXI-Lite.

---

## Creating Your Own Kernel

**1. Define the schema** (`my_kernel.py`):

```python
import brainsmith.dataflow as df
from brainsmith.registry import kernel

MY_KERNEL_SCHEMA = df.KernelSchema(
    name="MyKernel",
    inputs=[...],
    outputs=[...],
    kernel_params={...},
    constraints=[...],
)

@kernel(description="Custom kernel", author="Your Name")
class MyKernel(KernelOp):
    @classmethod
    def build_schema(cls, node, model):
        return MY_KERNEL_SCHEMA

    @classmethod
    def can_infer_from(cls, node, model):
        return node.op_type == "MyOnnxOp"

    @classmethod
    def infer_from(cls, node, model, insert_index):
        hw_node = helper.make_node("MyKernel", ...)
        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node]
        )

    def execute_node(self, context, graph):
        # Numpy reference implementation
        pass
```

**2. Implement the backend** (`my_kernel_hls.py` or `my_kernel_rtl.py`):

```python
from brainsmith.registry import backend
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

@backend(target_kernel="brainsmith:MyKernel", language="hls")
class MyKernel_hls(MyKernel, HLSBackend):
    def global_includes(self):
        # Include headers
        pass

    def defines(self, var):
        # Extract parameters from self.design_point
        pass

    def docompute(self):
        # Kernel invocation
        pass
```

For RTL backends, inherit from `RTLBackend` and reference Verilog modules.

**3. Write hardware implementation** (`my_kernel.hpp` or `my_kernel.v`) with standard interfaces.

**4. Register in `__init__.py`**:

```python
from .my_kernel import MyKernel
from .my_kernel_hls import MyKernel_hls

__all__ = ["MyKernel", "MyKernel_hls"]
```

Components are automatically discovered on import via the registry system.

---

## Design Space Exploration

KernelSchema automatically generates design space from interface specifications:

```python
op._ensure_ready(model)
design_space = op.design_space

# Navigate configurations
for point in design_space.sweep_dimension("SIMD"):
    cycles = point.initiation_interval
    # Evaluate performance

# Apply chosen configuration
op.apply_design_point(best_point)
```

Design points are immutable snapshots with navigation APIs:

- `point.with_input_stream(0, 32)` - Configure stream parallelization
- `point.with_dimension("SIMD", 64)` - Generic DSE parameter setting

See [Dataflow API Reference](../api/dataflow.md) for complete design space API.

---

## See Also

- **[Dataflow API Reference](../api/dataflow.md)** - KernelSchema, KernelOp, DesignSpace details
- **[Component Registry](registry.md)** - @kernel and @backend decorator usage
- **[Blueprint Schema](blueprint-schema.md)** - Design space configuration in YAML
