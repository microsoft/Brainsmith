# DuplicateStreams

**Stream fanout routing (1 input → N identical outputs).**

**Namespace**: `brainsmith.kernels`

**Backends**: HLS

!!! info "Stub Page"
    This page will be expanded with full documentation.

## Summary

Infrastructure kernel for duplicating a single input stream to multiple output streams. Used for graph topology routing when a tensor is consumed by multiple operations.

**Dynamic Schema**: Output count determined from ONNX graph structure.

## API Reference

::: brainsmith.kernels.duplicate_streams.DuplicateStreams
    options:
      show_source: true
      heading_level: 3
