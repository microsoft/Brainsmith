---
hide:
  - toc
---

# Developer Guide

Technical documentation for extending Brainsmith and understanding its architecture.

## Core Documentation

**[Component Registry](registry.md)** - Plugin system for registering custom kernels, backends, and pipeline steps.

**[Multi-Layer Offload](multi-layer-offload.md)** - Using weight streaming to implement large models (experimental).

## Hardware Kernel Documentation

**[Hardware Kernels](hardware-kernels/index.md)** - What kernels are, design principles, and when to create custom kernels.

**[Kernel Tutorial](hardware-kernels/kernel-tutorial.md)** - Learn by building: minimum → intermediate → complete examples.

**[Kernel Architecture](hardware-kernels/kernel-architecture.md)** - Deep dive: schemas, two-phase construction, data hierarchy, inter-kernel dataflow.

**[Kernel Schema Reference](hardware-kernels/kernel-schema-reference.md)** - Complete API reference for schema components: inputs, outputs, parameters, datatypes.

## Design Space Exploration

**[Blueprint Schema Reference](design-space-exploration/blueprint-schema.md)** - Complete YAML schema for design space configuration files.

**[Workflows](design-space-exploration/workflows.md)** - ONNX-to-bitstream pipeline, DSE navigation, troubleshooting, debugging, performance optimization.
