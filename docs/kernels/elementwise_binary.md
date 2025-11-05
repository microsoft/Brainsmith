# ElementwiseBinary

**Polymorphic hardware kernel for 17 binary operations (arithmetic, logical, comparison, bitwise).**

**Operations**: Add, Sub, Mul, Div, And, Or, Xor, Equal, Less, LessOrEqual, Greater, GreaterOrEqual, BitwiseAnd, BitwiseOr, BitwiseXor, BitShift

**Namespace**: `brainsmith.kernels`

**Backends**: HLS

!!! info "Documentation In Progress"
    This page is under development. See [AddStreams](addstreams.md) for an example of complete kernel documentation.

---

## Summary

ElementwiseBinaryOp is a polymorphic kernel that handles 17 different binary operations through a single implementation. The operation type is specified via the `func` nodeattr parameter, enabling code reuse while maintaining operation-specific optimizations.

**Supported Input Patterns**:

- **Phase 1** (`dynamic_static`): One streaming input + one static parameter
- **Phase 2** (`dynamic_dynamic`): Both inputs streaming with ONNX broadcasting support

**Key Features**:

- Polymorphic datatype resolution (operation-specific bitwidth rules)
- Broadcasting support for Phase 2 pattern
- PE parallelism for channel processing
- Configurable memory style for static parameters

---

## Hardware Interface

### Inputs

| Port | Pattern | Datatype | Description |
|------|---------|----------|-------------|
| lhs | `[N, H, W, C]` | INT8/INT16/FLOAT32 | Left-hand side operand |
| rhs | `[N, H, W, C]` or `[C]` | INT8/INT16/FLOAT32 | Right-hand side operand (streaming or static) |

### Outputs

| Port | Pattern | Datatype | Description |
|------|---------|----------|-------------|
| output | `[N, H, W, C]` | Operation-dependent | Result (bitwidth depends on `func`) |

### Supported Operations

| Category | Operations | Output Datatype Rule |
|----------|------------|---------------------|
| Arithmetic | Add, Sub, Mul, Div | Bitwidth expansion (e.g., INT8+INT8→INT9) |
| Logical | And, Or, Xor | BINARY (0 or 1) |
| Comparison | Equal, Less, LessOrEqual, Greater, GreaterOrEqual | BINARY (0 or 1) |
| Bitwise | BitwiseAnd, BitwiseOr, BitwiseXor | Max(lhs_width, rhs_width) |
| BitShift | BitShift | LHS datatype (shift doesn't change type) |

---

## Parallelization Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| PE | 1 to C | Processing elements (channel parallelism) |
| ram_style | {auto, block, distributed} | Memory implementation for static RHS |
| input_pattern | {dynamic_static, dynamic_dynamic} | Input streaming pattern |

---

## Design Point Configuration

```python
# Set operation type
op.set_nodeattr("func", "Mul")

# Configure parallelism
op.set_nodeattr("PE", 16)

# Select memory style for parameters
op.set_nodeattr("ram_style", "block")
```

---

## ONNX Inference

Compatible with standard ONNX operators:
- `Add`, `Sub`, `Mul`, `Div`
- `And`, `Or`, `Xor`
- `Equal`, `Less`, `LessOrEqual`, `Greater`, `GreaterOrEqual`

Detection distinguishes between:
- ChannelwiseOp pattern (1 dynamic + 1 static parameter)
- AddStreams pattern (2 dynamic, same shape)
- ElementwiseBinary pattern (2 dynamic with broadcasting, or polymorphic operation)

---

## See Also

- [AddStreams](addstreams.md) - Specialized element-wise addition
- [ChannelwiseOp](channelwise.md) - Channel-wise parametric operations
- [Kernel Architecture](../developer-guide/3-reference/kernels.md)

---

## API Reference

::: brainsmith.kernels.elementwise_binary.ElementwiseBinaryOp
    options:
      show_source: true
      heading_level: 3

::: brainsmith.kernels.elementwise_binary.ElementwiseBinaryOp_hls
    options:
      show_source: false
      heading_level: 3
