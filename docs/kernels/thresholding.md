# Thresholding

**Multi-threshold activation function with quantization.**

**Operation**: Piecewise-constant activation via threshold lookup

**Namespace**: `brainsmith.kernels` (migrated from FINN)

**Backends**: HLS, RTL

!!! info "Documentation In Progress"
    This page is under development. See [AddStreams](addstreams.md) for an example of complete kernel documentation.

---

## Summary

Thresholding implements multi-threshold activation functions by comparing input values against a set of learned thresholds to produce quantized outputs. This is the primary quantization mechanism in quantized neural networks.

**Key Features**:

- Supports arbitrary quantization levels (1-bit to N-bit)
- PE-parallelized threshold lookup
- Both HLS and RTL backends (RTL uses binary search optimization)
- Runtime-writable thresholds (internal_decoupled mode)
- Efficient memory packing for threshold storage

**Typical Use Cases**:

- Activation quantization in BNNs (Binary Neural Networks)
- Post-training quantization
- QAT (Quantization-Aware Training) deployment

---

## Hardware Interface

### Inputs

| Port | Pattern | Datatype | Description |
|------|---------|----------|-------------|
| input | `[N, H, W, C]` | INT8/INT16/INT32 | Input activations |
| thresholds | `[C, num_steps]` | INT8/INT16/INT32 | Threshold tensor (static) |

### Outputs

| Port | Pattern | Datatype | Description |
|------|---------|----------|-------------|
| output | `[N, H, W, C]` | INT2/INT4/INT8 | Quantized activations |

### Memory Modes

- **internal_embedded**: Thresholds in BRAM/LUTRAM (fixed at synthesis)
- **internal_decoupled**: Runtime-writable via AXI-Lite interface

---

## Parallelization Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| PE | 1 to C | Processing elements (parallel channels) |
| ram_style | {auto, block, distributed, ultra} | Memory implementation |
| runtime_writeable_weights | {0, 1} | Enable AXI-Lite threshold updates |

---

## Backend Implementations

### HLS Backend

Standard finn-hlslib implementation using parallel comparators.

### RTL Backend

Hand-optimized implementation using binary search tree for reduced latency and resources.

**Advantages**:
- 50% lower LUT usage vs HLS
- Logarithmic search complexity: O(log num_steps)
- More predictable timing

---

## ONNX Inference

Compatible with QONNX `MultiThreshold` operator:

```python
MultiThreshold(
    input: INT8[1,224,224,64],
    thresholds: INT8[64, 255],  # 255 thresholds = 8-bit output
    out_bias=0,
    out_scale=1.0
) → output: INT8[1,224,224,64]
```

---

## See Also

- [ChannelwiseOp](channelwise.md) - Channel-wise operations
- [Kernel Architecture](../developer-guide/3-reference/kernels.md)

---

## API Reference

::: brainsmith.kernels.thresholding.Thresholding
    options:
      show_source: true
      heading_level: 3

::: brainsmith.kernels.thresholding.Thresholding_hls
    options:
      show_source: false
      heading_level: 3

::: brainsmith.kernels.thresholding.Thresholding_rtl
    options:
      show_source: false
      heading_level: 3
