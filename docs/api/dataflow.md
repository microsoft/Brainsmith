# Dataflow Modeling

Core abstractions for modeling hardware kernels using schema-based design spaces.

## Overview

Two-phase construction separates expensive setup from fast configuration:

1. **Design Space** - Built once, defines valid parameter ranges
2. **Design Point** - Configured many times, represents a specific hardware instance

This enables efficient exploration by avoiding redundant computation.

## Design Space

::: brainsmith.dataflow.KernelDesignSpace

::: brainsmith.dataflow.InterfaceDesignSpace

## Design Point

::: brainsmith.dataflow.KernelDesignPoint

::: brainsmith.dataflow.InterfaceDesignPoint

## Schema Definitions

::: brainsmith.dataflow.KernelSchema

::: brainsmith.dataflow.InputSchema

::: brainsmith.dataflow.OutputSchema

## Builder

::: brainsmith.dataflow.DesignSpaceBuilder

::: brainsmith.dataflow.BuildContext

## Navigation

::: brainsmith.dataflow.OrderedDimension

## Validation

::: brainsmith.dataflow.Constraint

::: brainsmith.dataflow.ValidationError

::: brainsmith.dataflow.DesignSpaceValidationContext

::: brainsmith.dataflow.ConfigurationValidationContext

## Transformation

::: brainsmith.dataflow.TransformationResult

## Types

::: brainsmith.dataflow.Shape

::: brainsmith.dataflow.ShapeHierarchy

::: brainsmith.dataflow.FULL_DIM

::: brainsmith.dataflow.FULL_SHAPE

## See Also

- [Kernel Modeling Architecture](../developer-guide/2-core-systems/kernel-modeling.md) - Schema-based modeling concepts
- [Understanding Kernels](../developer-guide/1-foundations/understanding-kernels.md) - Hardware concepts and design tradeoffs
- [Kernel Base Class](kernel_op.md) - Base class for implementing kernels
