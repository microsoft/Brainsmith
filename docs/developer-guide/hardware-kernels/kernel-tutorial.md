# Kernel Tutorial

*[Hardware Kernels](index.md) > Tutorial*

Learn kernel development through three progressively complex examples: minimum viable kernel, element-wise operation, and complete layer implementation.


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
from onnx import helper

MY_KERNEL_SCHEMA = df.KernelSchema(
    name="MyKernel",
    inputs=[df.InputSchema(name="input", block_tiling=df.FULL_SHAPE, stream_tiling=["SIMD"])],
    outputs=[df.OutputSchema(name="output", block_tiling=df.FULL_SHAPE, stream_tiling=["PE"])],
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


## Intermediate Example: Element-wise Multiply

Before diving into complex kernels, let's implement a simple element-wise operation to understand core patterns.

### Schema (multiply.py)

```python
import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_SHAPE
from brainsmith.registry import kernel
from onnx import helper
import numpy as np

# Simple schema - same parallelization for input and output
MULTIPLY_SCHEMA = df.KernelSchema(
    name="Multiply",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=FULL_SHAPE,   # Process entire tensor
            stream_tiling=["SIMD"],    # Parallelize over last dimension
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=FULL_SHAPE,
            stream_tiling=["SIMD"],    # Match input parallelization
            datatype="input",          # Copy input datatype
        )
    ],
    kernel_params={
        "scale": ("f", True, 1.0),     # Multiplication factor
    },
)

@kernel(description="Element-wise multiply by constant", author="Tutorial")
class Multiply(df.KernelOp):
    @classmethod
    def build_schema(cls, node, model):
        return MULTIPLY_SCHEMA

    @classmethod
    def can_infer_from(cls, node, model):
        # Match ONNX Mul nodes with scalar constant as second input
        if node.op_type != "Mul":
            return False
        # Check if second input is a scalar constant (implementation detail)
        return is_scalar_constant(node.input[1], model)

    @classmethod
    def infer_from(cls, node, model, insert_index):
        # Extract scalar value from constant tensor
        scale = get_scalar_value(node.input[1], model)

        hw_node = helper.make_node(
            "Multiply",
            inputs=[node.input[0]],  # Only dynamic input
            outputs=list(node.output),
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            scale=scale,
        )
        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node]
        )

    def execute_node(self, context, graph):
        # Reference implementation for validation
        in_values = context[self.onnx_node.input[0]]
        scale = self.get_nodeattr("scale")
        context[self.onnx_node.output[0]] = (in_values * scale).astype(in_values.dtype)
```

### Key Patterns Demonstrated

1. **FULL_SHAPE:** Rank-agnostic tiling (works for any tensor rank)
2. **Datatype derivation:** String shorthand `"input"` copies from interface
3. **Pattern matching:** `can_infer_from()` checks ONNX structure
4. **Transformation:** `infer_from()` creates hardware node from ONNX node
5. **Validation:** `execute_node()` provides reference implementation

### Backend Implementation (multiply_hls.py)

```python
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.registry import backend
from .multiply import Multiply

@backend(target_kernel="brainsmith:Multiply", language="hls")
class Multiply_hls(Multiply, HLSBackend):
    """HLS backend for Multiply kernel."""

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_vector.h>",
            '#include "multiply.hpp"',
        ]

    def defines(self, var):
        """Extract parameters from design_point."""
        point = self.design_point
        simd = point.inputs["input"].stream_shape[-1]
        scale = self.get_nodeattr("scale")

        self.code_gen_dict["$DEFINES$"] = [
            f"#define SIMD {simd}",
            f"#define SCALE {scale}f",
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "multiply<SIMD, TI, TO>(in0, out, SCALE);"
        ]
```

### Hardware Implementation (multiply.hpp)

```cpp
#include <hls_vector.h>
#include <hls_stream.h>

template<unsigned SIMD, typename TI, typename TO>
void multiply(
    hls::stream<hls::vector<TI, SIMD>>& in,
    hls::stream<hls::vector<TO, SIMD>>& out,
    float scale
) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE s_axilite port=scale
#pragma HLS INTERFACE ap_ctrl_none port=return

    hls::vector<TI, SIMD> input_vec = in.read();
    hls::vector<TO, SIMD> output_vec;

    for (unsigned i = 0; i < SIMD; i++) {
#pragma HLS UNROLL
        output_vec[i] = static_cast<TO>(input_vec[i] * scale);
    }

    out.write(output_vec);
}
```

### Testing

```bash
# Create test model with Mul operation
python create_test_model.py

# Verify kernel appears in registry
brainsmith registry | grep Multiply

# Run through pipeline with cppsim validation
smith test_mul.onnx blueprint.yaml --stop-step folded_hls_cppsim
```


## Complete Example: LayerNorm Kernel

A complete kernel implementation showing schema definition, transformation, and backend code generation.

### Schema Definition

`brainsmith/kernels/layernorm/layernorm.py` defines structure declaratively:

```python
import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM
from brainsmith.dataflow.spec_helpers import derive_dim
from brainsmith.dataflow.types import ShapeHierarchy
from qonnx.core.datatype import DataType

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
            datatype=DataType["FLOAT32"],
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
from qonnx.util.basic import get_by_name
import numpy as np

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


## What You've Learned

**Quick Start** showed you:
- Minimum file structure for a kernel
- Basic schema with FULL_SHAPE tiling
- Pattern matching and transformation
- Backend code generation

**Intermediate Example** demonstrated:
- Element-wise operation patterns
- Datatype derivation with string shorthand
- Testing workflow with cppsim

**Complete Example** illustrated:
- Complex schema with constraints
- Dimension derivation for matching parallelization
- Multiple inheritance for backend composition
- Complete hardware implementation


## Next Steps

### Understand the architecture deeply
**→ [Kernel Architecture](kernel-architecture.md)** - Learn about two-phase construction, lazy initialization, data hierarchy

### Reference while coding
**→ [Kernel Schema Reference](kernel-schema-reference.md)** - Complete API reference for all schema components

### Deploy and optimize
**→ [Kernel Workflows](../design-space-exploration/workflows.md)** - ONNX-to-bitstream pipeline, DSE, troubleshooting
