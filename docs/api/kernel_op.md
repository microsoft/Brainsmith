# Kernel Base Class

Base class for all Brainsmith hardware kernel implementations.

## Overview

`KernelOp` extends FINN's `HWCustomOp` with schema-based design space modeling:

- Build design space once, configure many times
- Automatic caching of design spaces
- Immutable design points prevent configuration errors

## Usage

To implement a custom kernel:

1. Subclass `KernelOp` and register with `@kernel`
2. Implement `build_schema()` to define kernel structure
3. Use `design_point` property to access configuration
4. Call `apply_design_point()` to update configuration

## API Reference

::: brainsmith.dataflow.kernel_op.KernelOp
    options:
      members:
        - build_schema
        - build_design_space
        - design_point
        - apply_design_point
        - can_infer_from

::: brainsmith.dataflow.kernel_op.KernelOpError

## See Also

- [Kernels Architecture](../developer-guide/3-reference/kernels.md) - Kernel implementation guide
- [Kernel Modeling](../developer-guide/2-core-systems/kernel-modeling.md) - Schema-based modeling concepts
- [Component Registry](registry.md) - How to register custom kernels
