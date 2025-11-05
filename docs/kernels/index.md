# Kernel Reference

Hardware kernels are the fundamental building blocks of Brainsmith dataflow accelerators. Each kernel implements a specific neural network operation (e.g., convolution, activation, normalization) optimized for FPGA execution with configurable parallelism and resource usage.

This reference provides detailed specifications for all available kernels, organized by namespace and maturity level.

## Organization

Kernels are organized into three categories:

- **Core Kernels**: Production-ready kernels maintained by the Brainsmith team for common operations
- **FINN Legacy**: FINN-compatible kernels for migration and backward compatibility
- **Experimental**: Preview kernels under active development

## Kernel Index

=== "Core Kernels"

    Production-ready kernels for common neural network operations.

    | Kernel | Backends | Operation | Status |
    |--------|----------|-----------|--------|
    | [AddStreams](addstreams.md) | HLS | Element-wise stream addition | ✅ Stable |
    | [Crop](crop.md) | HLS | Spatial tensor cropping | ✅ Stable |
    | [DuplicateStreams](duplicate_streams.md) | HLS | Stream fanout routing | ✅ Stable |
    | [ElementwiseBinary](elementwise_binary.md) | HLS | Polymorphic binary operations (17 ops) | ✅ Stable |
    | [LayerNorm](layernorm.md) | HLS | Layer normalization | ✅ Stable |
    | [Softmax](softmax.md) | HLS | Softmax activation | ✅ Stable |

=== "FINN Legacy"

    FINN-compatible kernels for migration and backward compatibility.

    | Kernel | Backends | Operation | Status |
    |--------|----------|-----------|--------|
    | [ChannelwiseOp](channelwise.md) | HLS | Channel-wise parametric operations | ✅ Stable |
    | [Thresholding](thresholding.md) | HLS, RTL | Multi-threshold activation | ✅ Stable |

=== "Experimental"

    Preview kernels under active development. APIs may change.

    | Kernel | Backends | Operation | Status |
    |--------|----------|-----------|--------|
    | [RotaryEmbedding](rotaryembedding.md) | Python | Rotary position encoding | 🔬 Experimental |
    | [Shuffle](shuffle.md) | HLS | Channel/dimension shuffling | ⚠️ Beta |

## Understanding Kernel Documentation

Each kernel page includes:

- **Summary**: Brief description and mathematical operation
- **Hardware Interface**: Input/output specifications and constraints
- **Parallelization Parameters**: Configurable dimensions (PE, SIMD, etc.)
- **Performance Characteristics**: Cycle counts and resource estimates
- **Design Point Configuration**: API examples for configuration
- **Backend Implementations**: HLS and/or RTL backend details
- **ONNX Inference**: How the kernel is inferred from ONNX operators
- **Usage Examples**: Blueprint YAML and Python API examples
- **API Reference**: Auto-generated class documentation

## Key Concepts

### Backends

Kernels can have multiple backend implementations:

- **HLS (High-Level Synthesis)**: C++ to RTL compilation via Vitis HLS
    - Faster development and iteration
    - C++ simulation (cppsim) for quick verification
    - Automatic optimization and pipelining
- **RTL (Register Transfer Level)**: Hand-written Verilog/SystemVerilog
    - Fine-grained control over timing and resources
    - Often more resource-efficient than HLS
    - Direct IP reuse from finn-rtllib

### Parallelization

Hardware kernels expose parallelization parameters for performance/resource tradeoffs:

- **PE (Processing Elements)**: Number of parallel compute units
- **SIMD**: Single Instruction Multiple Data width
- **ram_style**: Memory implementation (block, distributed, ultra)

### Design Space Exploration (DSE)

Brainsmith automatically computes valid parameter ranges based on tensor shapes and constraints:

```python
# Query valid configurations
valid_ranges = kernel.get_valid_ranges(model)
# Returns: {"PE": {1, 2, 4, 8, 16}, "ram_style": {"block", "distributed"}}

# Explore design space
for pe in valid_ranges["PE"]:
    kernel.set_nodeattr("PE", pe)
    cycles = kernel.get_exp_cycles()
    resources = kernel.node_res_estimation(fpgapart)
```

## Status Badges

- ✅ **Stable**: Production-ready, API stable, thoroughly tested
- ⚠️ **Beta**: Feature-complete but API may change, testing in progress
- 🔬 **Experimental**: Under development, breaking changes expected

## See Also

- [Kernel Architecture](../developer-guide/3-reference/kernels.md) - Conceptual overview and patterns
- [Design Space Exploration](../developer-guide/2-core-systems/design-space-exploration.md) - DSE system guide
- [Component Registry](../developer-guide/2-core-systems/component-registry.md) - Kernel discovery and registration
- [API Reference: KernelOp](../api/kernel_op.md) - Base class documentation
