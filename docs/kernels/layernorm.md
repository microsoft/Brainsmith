# LayerNorm

**Layer normalization over channel dimension.**

**Operation**: `(x - mean) / sqrt(var + epsilon)`

**Namespace**: `brainsmith.kernels`

**Backends**: HLS

!!! info "Stub Page"
    This page will be expanded with full documentation.

## Summary

Performs layer normalization across the channel axis (last dimension).

**Parameter**: `epsilon` (numerical stability constant, default 1e-5)

## API Reference

::: brainsmith.kernels.layernorm.LayerNorm
    options:
      show_source: true
      heading_level: 3
