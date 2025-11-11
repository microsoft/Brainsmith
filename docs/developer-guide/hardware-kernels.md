# Hardware Kernels

**Hardware kernels are layer-level operations (LayerNorm, MatMul, Attention) implemented as self-contained streaming circuits for FPGAs.**

<div align="center" markdown>
![Kernel Interfaces](../images/dataflow_kernel.png){ width="400" }
</div>

Brainsmith constructs dataflow accelerators by iteratively applying graph transformations to lower ONNX nodes to matching kernels, connected via streaming interfaces. During this process, **kernels are modeled by the relationship between their input and output streams** with the internal architecture largely abstracted away.


## Designed for Agility

The kernel system is the foundation of Brainsmith's agility, enabling key architectural capabilities:

**1. Schema-Driven Automation** - All integration code, constraint validation, and DSE parameters are auto-derived from the kernel's schema, allowing hardware engineers to create efficient kernels without onerous compiler integration work.

**2. Standardized DSE Interface** - Structural and optimization design parameters construct a *kernel design space* with unified functions for efficient exploration and testing.

**3. Automated Testing Framework** - Validate parity against target ONNX operators at various levels of abstraction with a simple testbench.

See [Dataflow Modeling](dataflow-modeling.md) for theoretical foundations.


## Layer-Level Granularity

Unlike MLIR-based or fully HLS toolchains that fully decompose the model into primitive operations, Brainsmith maintains **layer-level** design abstraction throughout the compiler. This has several key advantages that further serve the goals of agility and separation of concerns:

1. **Preserve Design Space** - Neural networks are expressed as layers in PyTorch/ONNX. Preserving this granularity enables natural extension from AI frameworks while maintaining semantic meaning through the compilation pipeline.

2. **Prevent Exponential Explosion** - Decomposing layers into individual operations (add, multiply, shift) creates thousands of tiny blocks, making design space exploration computationally intractable.

3. **Enable Hand Optimization** - Hardware engineers can optimize kernels at the layer scale without deep AI model knowledge, achieving performance that auto-generated designs cannot match while maintaining flexibility through automated composition.


## Three-File Structure

Each kernel consists of:

1. **Schema + KernelOp** (`kernel.py`) - Interface definitions, validation constraints, and ONNX-to-hardware transformation logic
2. **Backend** (`kernel_hls.py` or `kernel_rtl.py`) - Code generation that bridges schema to implementation
3. **Hardware** (`kernel.hpp` or `kernel.v`) - Actual RTL/HLS implementation with standard interfaces

This separation enables:

- Multiple backends for the same kernel (HLS vs RTL, vendor-specific optimizations)
- Schema-driven automation (DSE parameter derivation, validation, interface generation)
- Hardware expertise isolation (implement kernels without compiler knowledge)

## Quick Start: Minimum Kernel

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

## Complete Example: LayerNorm Kernel

The LayerNorm kernel demonstrates a complete production implementation with schema definition, ONNX transformation, HLS backend, and hardware code.

**Implementation files:**

- **[`layernorm.py`](https://github.com/microsoft/brainsmith/blob/main/brainsmith/kernels/layernorm/layernorm.py)** - Schema definition with `FULL_DIM` tiling, `derive_dim()` for matching parallelization, epsilon parameter, and ONNX-to-hardware transformation
- **[`layernorm_hls.py`](https://github.com/microsoft/brainsmith/blob/main/brainsmith/kernels/layernorm/layernorm_hls.py)** - HLS backend extracting parameters from design_point and generating C++ defines/docompute
- **[`layernorm.hpp`](https://github.com/microsoft/brainsmith/blob/main/brainsmith/kernels/layernorm/layernorm.hpp)** - HLS C++ implementation with AXI-Stream interfaces and normalization logic

**Key patterns demonstrated:**

- Schema-driven automation (SIMD parameter auto-extracted from `stream_tiling`)
- Dimension derivation (`derive_dim()`) to match input/output parallelization
- Validation constraints (`AttrCompare` for epsilon > 0)
- Pattern matching (`can_infer_from()` checks for FuncLayerNorm with axis=-1)
- Multiple inheritance (LayerNorm + HLSBackend) for code generation
- Reference implementation (`execute_node()`) for cppsim validation
