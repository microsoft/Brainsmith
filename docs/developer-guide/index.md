# Developer Guide

Technical documentation for extending Brainsmith and understanding its architecture.

## Core Documentation

**[CLI Architecture](cli-architecture.md)** - Dual CLI design (`brainsmith` vs `smith`), configuration hierarchy, and extension points.

**[Component Registry](registry.md)** - Plugin system for registering custom kernels, backends, and pipeline steps.

**[Hardware Kernels](hardware-kernels.md)** - Layer-level operations on FPGA: interfaces, architecture, and implementation patterns.

**[Blueprint Schema Reference](blueprint-schema.md)** - Complete YAML schema for design space configuration files.

**[Multi-Layer Offload](multi-layer-offload.md)** - Graph partitioning and heterogeneous execution strategies.

## Additional Resources

**Experimental Docs** - `experimental/` contains older comprehensive docs with conceptual depth (outdated APIs, older terminology).

**See Also** - [API Reference](../api/index.md) · [Getting Started](../getting-started.md) · [Test Framework](../../tests/README.md)
