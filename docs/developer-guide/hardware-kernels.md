# Hardware Kernels

**Hardware kernels are layer-level operations (LayerNorm, MatMul, Attention) implemented as self-contained streaming circuits for FPGAs.**

<div align="center" markdown>
![Kernel Interfaces](../images/dataflow_kernel.png){ width="400" }
</div>

Brainsmith constructs dataflow accelerators by iteratively applying graph transformations to lower ONNX nodes to matching kernels, connected via streaming interfaces. During this process, **kernels are modeled by the relationship between their input and output streams** with the internal architecture largely abstracted away.

---

## Design for Agility

Brainsmith's kernel system enables rapid DSE and custom hardware integration through three architectural capabilities:

**1. Efficient Design Space Exploration** → *Two-Phase Construction*
: Build design space once (tensor shapes, valid ranges), navigate configurations thousands of times cheaply. Explore parallelization factors without rebuilding structure. Immutable design points ensure consistency.

**2. Schema-Driven Automation** → *Declarative Definitions*
: Define interfaces once, everything else auto-derives. Schemas generate nodeattrs, DSE parameters, validation constraints, and FINN integration. Hardware engineers specify structure, not storage or integration logic.

**3. Seamless ONNX Integration** → *Lazy Initialization + Shape Extraction*
: Transform ONNX nodes via pattern matching. Shapes extracted from model graph (never stored). Lazy initialization minimizes overhead. Compatible with FINN's execution and transformation pipeline.

The rest of this document shows how to use these features.

---

## Quick Start: Minimum Kernel

A kernel requires three files in your project:

```
project/kernels/my_kernel/
├── my_kernel.py          # Schema + transformation logic
├── my_kernel_hls.py      # Code generation backend
└── my_kernel.hpp         # Hardware implementation
```

**my_kernel.py** - Schema + KernelOp:
```python
import brainsmith.dataflow as df
from brainsmith.registry import kernel

MY_KERNEL_SCHEMA = df.KernelSchema(
    name="MyKernel",
    inputs=[df.InputSchema(name="input", block_tiling=[df.FULL_DIM], stream_tiling=["SIMD"])],
    outputs=[df.OutputSchema(name="output", block_tiling=[df.FULL_DIM], stream_tiling=["PE"])],
)

@kernel(description="Custom kernel", author="Your Name")
class MyKernel(df.KernelOp):
    @classmethod
    def build_schema(cls, node, model):
        return MY_KERNEL_SCHEMA

    @classmethod
    def can_infer_from(cls, node, model):
        return node.op_type == "MyOnnxOp"  # Pattern match ONNX nodes

    @classmethod
    def infer_from(cls, node, model, insert_index):
        hw_node = helper.make_node("MyKernel", inputs=list(node.input),
                                    outputs=list(node.output), domain="brainsmith.kernels")
        return df.TransformationResult(nodes_to_insert=[hw_node], nodes_to_remove=[node])

    def execute_node(self, context, graph):
        # Reference numpy implementation for validation
        pass
```

**my_kernel_hls.py** - Backend:
```python
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.registry import backend

@backend(target_kernel="brainsmith:MyKernel", language="hls")
class MyKernel_hls(MyKernel, HLSBackend):
    def defines(self, var):
        # Extract parameters from self.design_point
        point = self.design_point
        simd = point.inputs["input"].stream_shape[-1]
        self.code_gen_dict["$DEFINES$"] = [f"#define SIMD {simd}"]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = ["my_kernel<SIMD>(in0, out);"]
```

**my_kernel.hpp** - Hardware:
```cpp
template<unsigned SIMD, typename TI, typename TO>
void my_kernel(hls::stream<hls::vector<TI, SIMD>>& in,
               hls::stream<hls::vector<TO, SIMD>>& out) {
    // Your hardware implementation
}
```

**Register in `__init__.py`:**
```python
from .my_kernel import MyKernel
from .my_kernel_hls import MyKernel_hls
__all__ = ["MyKernel", "MyKernel_hls"]
```

**Use in blueprint:**
```yaml
design_space:
  kernels:
    - MyKernel
```

Sections below explain each piece in detail.

---

## Kernel Architecture

### Layer-Level Granularity

Brainsmith uses **layer-level kernels** rather than primitive operations. A MatMul operation remains a MatMul kernel, not decomposed into thousands of individual multiply-add primitives.

**Why layer-level granularity?**

1. **Preserve Design Space** - Neural networks are expressed as layers in PyTorch/ONNX. Preserving this granularity enables natural extension from AI frameworks while maintaining semantic meaning through the compilation pipeline.

2. **Prevent Exponential Explosion** - Decomposing layers into individual operations (add, multiply, shift) creates thousands of tiny blocks, making design space exploration computationally intractable.

3. **Enable Hand Optimization** - Hardware engineers can optimize kernels at the layer scale without deep AI model knowledge, achieving performance that auto-generated designs cannot match while maintaining flexibility through automated composition.

### Three-File Structure

Each kernel consists of:

1. **Schema + KernelOp** (`kernel.py`) - Interface definitions, validation constraints, and ONNX-to-hardware transformation logic
2. **Backend** (`kernel_hls.py` or `kernel_rtl.py`) - Code generation that bridges schema to implementation
3. **Hardware** (`kernel.hpp` or `kernel.v`) - Actual RTL/HLS implementation with standard interfaces

This separation enables:
- Multiple backends for the same kernel (HLS vs RTL, vendor-specific optimizations)
- Schema-driven automation (DSE parameter derivation, validation, interface generation)
- Hardware expertise isolation (implement kernels without compiler knowledge)

---

## Complete Example: LayerNorm Kernel

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

**What this generates automatically:**
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

**Multiple inheritance provides:**
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

## Understanding Schemas

**Schemas separate kernel structure from execution context.**

A schema defines the kernel's **invariant properties**—how it processes data, what parallelization it supports, what constraints it enforces. These properties remain constant whether the kernel processes a `(1, 224, 224, 64)` tensor or a `(1, 56, 56, 128)` tensor.

```python
InputSchema(
    name="input",
    block_tiling=[FULL_DIM],           # Relationship: "process entire dimension"
    stream_tiling=["SIMD"],            # Capability: "parallelize by SIMD"
    datatype=derive_datatype("input"), # Constraint: "match input datatype"
    required_layout="NHWC",            # Requirement: "needs NHWC layout"
)
```

**This separation enables:**

**Reusability** - Same LayerNorm kernel works on any tensor shape
: Schema defines processing rules, not specific dimensions

**Single Source of Truth** - ModelWrapper owns graph state
: Tensor shapes and datatypes queried from ONNX graph, never duplicated

**Two-Phase Construction** - Build structure once, explore configurations many times
: Schema defines what varies (SIMD parameter), builder computes valid ranges (divisors)

**Clear Ownership** - Each component has distinct responsibility
: Schema: structure and constraints | ModelWrapper: current graph state | Builder: runtime models

**Example: A LayerNorm kernel processes `(1, 224, 224, 64)` and `(1, 56, 56, 128)` with the same schema, but different runtime models built from different graph contexts.**

---

## Schema Components

### Input and Output Definitions

Interfaces define how the kernel relates to tensor data:

```python
df.InputSchema(
    name="input",
    block_tiling=[FULL_DIM, FULL_DIM, "K"],  # Processing quantum template
    stream_tiling=[1, 1, "SIMD"],             # Parallelization template
    datatype=df.constant_datatype("INT8"),    # Datatype specification
    required_layout="NHWC",                   # Layout requirement
)
```

**Field purposes:**

- `block_tiling` - **Template for block shapes**: Defines kernel's processing quantum
  - Resolved in Phase 1 (build) using tensor shapes from ModelWrapper
  - Determines valid parallelization ranges (stream dims can't exceed block dims)
  - Example: `[FULL_DIM, FULL_DIM, "K"]` → `[224, 224, 64]` for specific tensor

- `stream_tiling` - **Template for stream shapes**: Defines per-cycle parallelism
  - Resolved in Phase 2 (configure) using DSE parameters
  - String elements become DSE parameters (e.g., `"SIMD"` → parameter with valid range)
  - Example: `[1, 1, "SIMD"]` → `[1, 1, 64]` when configured with `SIMD=64`

- `datatype` - **Datatype specification**: Fixed or derived from other interfaces
  - Union type: `None` (from graph), `DataType` (fixed), `str` (derive from interface), callable (custom)
  - Resolved in Phase 1 (build), synced to nodeattrs for persistence

- `required_layout` - **Layout requirement**: Ensures correct data format
  - Validation only, doesn't affect shape computation
  - Use for kernels sensitive to memory layout (NCHW vs NHWC)

### Block vs Stream Tiling

**Why these exist:** Hardware kernels have two dimensions of parallelism:

- **Block Tiling** - How much data the kernel needs loaded for one computation cycle
- **Stream Tiling** - How many elements are processed per clock cycle

**For most simple kernels:**
- Use `FULL_DIM` for block_tiling (process entire dimension)
- Use parameter names like `"SIMD"` or `"PE"` for stream_tiling

**Example - Element-wise operation:**
```python
inputs=[
    df.InputSchema(
        name="input",
        block_tiling=[FULL_DIM],      # Process entire tensor
        stream_tiling=["SIMD"],       # SIMD elements per cycle
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

👉 See [Data Hierarchy Deep Dive](#data-hierarchy-deep-dive) for complete theory.

### Parameters and Constraints

**Three parameter types with distinct roles:**

**1. Kernel Parameters** - Algorithm configuration:
```python
kernel_params={
    "epsilon": ("f", True, 1e-5),  # Affects computation behavior
    "axis": ("i", False, -1),
}
# Define WHAT the kernel computes
```

**2. Parallelization Parameters** - Auto-extracted from stream_tiling:
```python
stream_tiling=["Width", "Height"]  # Parameter names are arbitrary
# Valid ranges computed by factoring block dimensions
# Navigate via interface indices: point.with_input_stream(0, 64)
```
Names like "SIMD" or "PE" are conventions, not requirements. Choose names that make sense for your kernel.

**3. Implementation Parameters** - Explicit DSE dimensions:
```python
dse_parameters={
    "ram_style": df.ParameterSpec("ram_style", {"distributed", "block"}),
    "fifo_depth": df.ParameterSpec("fifo_depth", [128, 256, 512]),
}
# Affect performance/resources, not correctness
# Examples: memory type, buffer depth, algorithm variant
```

**Container types determine exploration:**
- `list/tuple` → Ordered navigation (supports min/max, step up/down)
- `set/frozenset` → Categorical choices (membership only)

**Constraints** validate configurations:
```python
constraints=[
    df.AttrCompare("epsilon", ">", 0),
    df.DimensionCompare("Width", "<=", "tensor_width"),
]
```

### Datatype Specifications

**Fixed datatypes:**
```python
datatype=df.constant_datatype("FLOAT32")
```

**Derived from inputs:**
```python
datatype=df.derive_datatype("input")  # Copy from input
```

**Custom derivation:**
```python
def custom_dtype(ctx):
    input_dtype = ctx.get_datatype("input")
    return DataType("INT", input_dtype.bitwidth * 2)

datatype=custom_dtype
```

### Dimension Derivation

**Copy dimensions from other interfaces:**
```python
stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)]
```

This copies the last stream dimension from the input, ensuring compatible parallelization.

---

## Data Hierarchy Deep Dive

!!! info "Deep Dive Section"
    This section provides theoretical foundation for the TENSOR/BLOCK/STREAM hierarchy.

    **Required reading for:** Complex kernels with reduction operations, custom DSE logic, understanding performance characteristics

    **Safe to skip if:** Using simple element-wise operations with FULL_DIM blocks

The stream relationships that define kernels require systematic refinement to connect ONNX's model-level abstractions to RTL's cycle-accurate streaming implementation. ONNX describes computation in terms of complete tensors (e.g., `(1, 224, 224, 64)` activation map), while RTL executes on bit-streams cycling through hardware (16 elements per clock).

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

!!! info "Deep Dive Section"
    Understanding how kernels connect through streaming interfaces and elastic buffering.

    **Required for:** Custom infrastructure transforms, debugging FIFO depth issues, understanding dataflow composition

    **Safe to skip if:** Standard pipeline usage with automatic FIFO sizing

<div align="center" markdown>
![Dataflow Chunking with FIFO](../images/dataflow_chunking_fifo.png){ width="700" }
</div>

Kernels communicate via **streaming interfaces**, producing and consuming data cycle-by-cycle. Elastic FIFOs between kernels accumulate these streams as **data blocks** for buffering, then stream them out to downstream consumers. This infrastructure automatically adapts to different kernel semantics through shape-driven buffering.

**👉 The compiler handles this automatically via schemas. You rarely need to think about it.**

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

## Creating Your Own Kernel

### Step-by-Step Workflow

Follow this process to add a custom kernel to your project:

**1. Define the schema** - Start with interface declarations

Create `project/kernels/my_kernel/my_kernel.py`:

```python
import brainsmith.dataflow as df
from brainsmith.registry import kernel
from onnx import helper

MY_KERNEL_SCHEMA = df.KernelSchema(
    name="MyKernel",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[df.FULL_DIM],
            stream_tiling=["SIMD"],
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[df.FULL_DIM],
            stream_tiling=["PE"],
        )
    ],
    kernel_params={
        "scale": ("f", True, 1.0),
    },
    constraints=[
        df.AttrCompare("scale", ">", 0),
    ],
)

@kernel(description="Custom kernel", author="Your Name")
class MyKernel(df.KernelOp):
    @classmethod
    def build_schema(cls, node, model):
        return MY_KERNEL_SCHEMA

    @classmethod
    def can_infer_from(cls, node, model):
        """Pattern match ONNX nodes that this kernel can implement."""
        return node.op_type == "MyOnnxOp"

    @classmethod
    def infer_from(cls, node, model, insert_index):
        """Transform ONNX node to hardware kernel."""
        hw_node = helper.make_node(
            "MyKernel",
            inputs=list(node.input),
            outputs=list(node.output),
            domain="brainsmith.kernels",
            scale=1.0,  # Extract from node attributes
        )
        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node]
        )

    def execute_node(self, context, graph):
        """Reference numpy implementation for validation."""
        import numpy as np
        in_values = context[self.onnx_node.input[0]]
        scale = self.get_nodeattr("scale")
        # Implement operation
        result = in_values * scale
        context[self.onnx_node.output[0]] = result
```

**2. Create backend** - Code generation

Create `project/kernels/my_kernel/my_kernel_hls.py`:

```python
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.registry import backend
from .my_kernel import MyKernel

@backend(target_kernel="brainsmith:MyKernel", language="hls")
class MyKernel_hls(MyKernel, HLSBackend):
    """HLS backend for MyKernel."""

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_vector.h>",
            '#include "my_kernel.hpp"',
        ]

    def defines(self, var):
        """Extract parameters from design_point."""
        point = self.design_point
        simd = point.inputs["input"].stream_shape[-1]
        pe = point.outputs["output"].stream_shape[-1]
        scale = self.get_nodeattr("scale")

        self.code_gen_dict["$DEFINES$"] = [
            f"#define SIMD {simd}",
            f"#define PE {pe}",
            f"#define SCALE {scale}",
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "my_kernel<SIMD, PE, TI, TO>(in0, out, SCALE);"
        ]
```

**3. Write hardware** - RTL/HLS implementation

Create `project/kernels/my_kernel/my_kernel.hpp`:

```cpp
#include <hls_vector.h>
#include <hls_stream.h>

template<unsigned SIMD, unsigned PE, typename TI, typename TO>
void my_kernel(
    hls::stream<hls::vector<TI, SIMD>>& in,
    hls::stream<hls::vector<TO, PE>>& out,
    float scale
) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE s_axilite port=scale
#pragma HLS INTERFACE ap_ctrl_none port=return

    // Your hardware implementation here
    hls::vector<TI, SIMD> input_vec = in.read();
    hls::vector<TO, PE> output_vec;

    for (unsigned i = 0; i < PE; i++) {
#pragma HLS UNROLL
        output_vec[i] = static_cast<TO>(input_vec[i] * scale);
    }

    out.write(output_vec);
}
```

**4. Register** - Make discoverable

Create `project/kernels/my_kernel/__init__.py`:

```python
from .my_kernel import MyKernel
from .my_kernel_hls import MyKernel_hls

__all__ = ["MyKernel", "MyKernel_hls"]
```

**5. Test** - Validate with cppsim

```bash
# Use in your blueprint
brainsmith registry  # Verify MyKernel appears

# Run cppsim validation
smith model.onnx blueprint.yaml --stop-step folded_hls_cppsim
```

### Common Patterns

**Element-wise operations:**
```python
# Simple passthrough or arithmetic
inputs=[df.InputSchema(block_tiling=[FULL_DIM], stream_tiling=["SIMD"])]
outputs=[df.OutputSchema(block_tiling=[FULL_DIM], stream_tiling=["SIMD"])]
```

**Reduction operations:**
```python
# Reduce over a dimension (e.g., sum, mean)
inputs=[df.InputSchema(
    block_tiling=[FULL_DIM, FULL_DIM, "K"],  # Reduction quantum
    stream_tiling=[1, 1, "SIMD"]
)]
outputs=[df.OutputSchema(
    block_tiling=[FULL_DIM, FULL_DIM, 1],    # Reduced dimension
    stream_tiling=[1, 1, 1]
)]
```

**Multi-input kernels:**
```python
# Ensure compatible shapes
inputs=[
    df.InputSchema(name="a", stream_tiling=["PE"]),
    df.InputSchema(name="b", stream_tiling=[derive_dim("a", ShapeHierarchy.STREAM, -1)]),
]
```

**Memory-based kernels:**
```python
# Add DSE parameters for memory configuration
dse_parameters={
    "ram_style": df.ParameterSpec("ram_style", {"distributed", "block"}, default="distributed"),
    "depth": df.ParameterSpec("depth", [256, 512, 1024], default=512),
}
```

### Common Pitfalls

❌ **Forgetting to call `_ensure_ready()`** before accessing design_space
```python
# Wrong
design_space = op.design_space  # Error if schema not initialized

# Right
op._ensure_ready(model)
design_space = op.design_space
```

❌ **Mismatched stream shapes** between producer/consumer
```python
# Will cause FIFO issues
Producer: stream_tiling=["PE"]  where PE=16
Consumer: stream_tiling=["SIMD"] where SIMD=32
# Fix: Use derive_dim() to match shapes
```

❌ **Not providing `execute_node()`** for validation
```python
# Without this, cppsim can't validate correctness
def execute_node(self, context, graph):
    raise NotImplementedError()  # Will break validation
```

❌ **Hardcoding parameters** instead of using design_point
```python
# Wrong - not explorable
def defines(self, var):
    self.code_gen_dict["$DEFINES$"] = ["#define SIMD 16"]

# Right - follows design_point
def defines(self, var):
    simd = self.design_point.inputs["input"].stream_shape[-1]
    self.code_gen_dict["$DEFINES$"] = [f"#define SIMD {simd}"]
```

### RTL Backend Pattern

For Verilog/VHDL implementations, inherit from `RTLBackend`:

```python
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

@backend(target_kernel="brainsmith:MyKernel", language="rtl")
class MyKernel_rtl(MyKernel, RTLBackend):
    def get_verilog_top_module_name(self):
        return "my_kernel_top"

    def get_verilog_top_module_intf_names(self):
        return {
            "clk": "ap_clk",
            "rst": "ap_rst_n",
            "in0": ("in0_V_TDATA", "in0_V_TVALID", "in0_V_TREADY"),
            "out": ("out_V_TDATA", "out_V_TVALID", "out_V_TREADY"),
        }
```

---

## Design Space Exploration

Schemas automatically generate design spaces from interface specifications:

```python
# Initialize kernel with model context
op._ensure_ready(model)

# Access design space
design_space = op.design_space
print(f"Parameters: {design_space.dimensions.keys()}")
# Output: Parameters: dict_keys(['SIMD', 'PE', 'ram_style'])

# Navigate configurations
for point in design_space.sweep_dimension("SIMD"):
    cycles = point.initiation_interval
    resources = estimate_resources(point)
    # Evaluate performance vs resource tradeoff

# Apply chosen configuration
op.apply_design_point(best_point)
```

### Design Point Navigation

Design points are immutable snapshots with fluent navigation APIs:

**Interface-based API** (for stream parameters):
```python
point = point.with_input_stream(0, 32)   # Set first input PE=32
point = point.with_output_stream(0, 16)  # Set first output PE=16
```

**Dimension-based API** (for generic DSE parameters):
```python
point = point.with_dimension("SIMD", 64)
point = point.with_dimension("ram_style", "distributed")
```

**Accessing configuration:**
```python
simd = point.get_dimension("SIMD")
input_shape = point.inputs["input"].stream_shape
tensor_shape = point.inputs["input"].tensor_shape
```

### Integration with Blueprint DSE

Brainsmith's segment-based DSE automatically explores kernel design spaces:

1. **Schema generates design space** - Valid parameter ranges from stream_tiling
2. **Blueprint specifies steps** - Transformation pipeline including DSE
3. **DSE explores configurations** - Evaluates performance vs resources
4. **Best points applied** - Optimal configurations used for RTL generation

**Blueprint example:**
```yaml
design_space:
  kernels:
    - LayerNorm
  steps:
    - "infer_kernels"
    - "target_fps_parallelization"  # DSE step
    - "apply_folding_config"
```

The DSE step uses kernel schemas to:
- Determine valid parallelization factors
- Estimate resource usage
- Calculate throughput
- Find Pareto-optimal configurations

See [Design Space Exploration API](../api/dse.md) for complete DSE workflow and [Dataflow API Reference](../api/dataflow.md) for design space navigation details.

---

## See Also

- **[Dataflow API Reference](../api/dataflow.md)** - Complete KernelSchema, KernelOp, and DesignSpace documentation
- **[Component Registry](registry.md)** - Component registration and discovery patterns
- **[Component Registry User Guide](registry-user-guide.md)** - Detailed registry usage
- **[Blueprint Schema](blueprint-schema.md)** - YAML configuration for design space exploration
- **[Design Space Exploration API](../api/dse.md)** - Programmatic DSE control
