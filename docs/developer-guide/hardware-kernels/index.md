# Hardware Kernels

**Hardware kernels are layer-level operations (LayerNorm, MatMul, Attention) implemented as self-contained streaming circuits for FPGAs.**

<div align="center" markdown>
![Kernel Interfaces](../../images/dataflow_kernel.png){ width="400" }
</div>

Brainsmith constructs dataflow accelerators by iteratively applying graph transformations to lower ONNX nodes to matching kernels, connected via streaming interfaces. During this process, **kernels are modeled by the relationship between their input and output streams** with the internal architecture largely abstracted away.


## Designed for Agility

Brainsmith's kernel system enables rapid DSE and custom hardware integration through three architectural capabilities:

**1. Efficient Design Space Exploration** → *Two-Phase Construction*
: Build design space once (tensor shapes, valid ranges), navigate configurations thousands of times cheaply. Explore parallelization factors without rebuilding structure. Immutable design points ensure consistency.

**2. Schema-Driven Automation** → *Declarative Definitions*
: Define interfaces once, everything else auto-derives. Schemas generate nodeattrs, DSE parameters, validation constraints, and FINN integration. Hardware engineers specify structure, not storage or integration logic.

**3. Seamless ONNX Integration** → *Lazy Initialization + Shape Extraction*
: Transform ONNX nodes via pattern matching. Shapes extracted from model graph (never stored). Lazy initialization minimizes overhead. Compatible with FINN's execution and transformation pipeline.

See [Kernel Architecture](kernel-architecture.md) for deep dive into these architectural patterns.


## Layer-Level Granularity

Brainsmith uses **layer-level kernels** rather than primitive operations. A MatMul operation remains a MatMul kernel, not decomposed into thousands of individual multiply-add primitives.

**Why layer-level granularity?**

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

**File structure example:**
```
project/kernels/my_kernel/
├── my_kernel.py          # Schema + transformation logic
├── my_kernel_hls.py      # Code generation backend
└── my_kernel.hpp         # Hardware implementation
```

**[Start with the tutorial →](kernel-tutorial.md)** to build your first kernel step-by-step.


## Expanding Kernel Coverage

Current kernel coverage is limited to foundational operations. The kernel system is designed to be extended—contributions of new kernels are strongly encouraged. Whether you're implementing standard operations (Softmax, attention mechanisms) or domain-specific accelerators, the framework provides the patterns and infrastructure to integrate cleanly.

Check available kernels: `brainsmith registry`


## Component Integration

Kernels work with other Brainsmith components:

- **[Component Registry](registry.md)** - Kernel registration and discovery via decorators
- **[Blueprint Schema](../design-space-exploration/blueprint-schema.md)** - Specifying kernels in design space configuration
- **[Multi-Layer Offload](multi-layer-offload.md)** - Using kernels with weight streaming for large models


## See Also

- **[Dataflow API Reference](../api/dataflow.md)** - Complete KernelSchema, KernelOp, and DesignSpace API documentation
- **[Design Space Exploration API](../api/dse.md)** - Programmatic DSE control
