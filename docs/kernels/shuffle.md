# Shuffle

**Tensor dimension rearrangement via permutation.**

**Operation**: Transpose with optional reshape

**Namespace**: `brainsmith.kernels`

**Backends**: HLS

!!! info "Stub Page"
    This page will be expanded with full documentation.

## Summary

Rearranges tensor dimensions according to a permutation array. Implemented using perfect loop nest generation for efficient streaming.

**Parameter**: `perm` (permutation tuple, e.g., `[0, 2, 1, 3]` swaps H and W)

## API Reference

::: brainsmith.kernels.shuffle.Shuffle
    options:
      show_source: true
      heading_level: 3
