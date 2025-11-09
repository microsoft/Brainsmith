# Multi-Layer Offload

**Graph partitioning and heterogeneous execution strategies**

---

## Overview

Multi-layer offload enables selective acceleration by partitioning neural network graphs between FPGA hardware and CPU/GPU execution. This approach targets performance-critical subgraphs for hardware acceleration while maintaining flexibility for operations better suited to general-purpose processors.

## Key Concepts

**Graph Partitioning** - Identify and extract subgraphs suitable for FPGA acceleration based on:

- Operation types (hardware-accelerated kernels available)
- Data movement costs (minimize host-device transfers)
- Compute intensity (maximize hardware utilization)

**Heterogeneous Execution** - Coordinate execution across:

- FPGA dataflow accelerators (custom kernels)
- CPU fallback (unsupported operations)
- GPU acceleration (where applicable)

**Interface Boundaries** - Manage data transfers at partition boundaries:

- Host memory ↔ FPGA memory
- Tensor serialization/deserialization
- Synchronization and scheduling

## Design Considerations

- **Partition Granularity** - Balance between offload overhead and acceleration benefit
- **Memory Hierarchy** - Minimize data movement through strategic buffering
- **Fallback Strategies** - Graceful degradation when hardware acceleration unavailable

---

> **Status:** Documentation in development. This page will be expanded with implementation patterns, API examples, and case studies demonstrating effective partitioning strategies.

## See Also

- [Component Registry](registry.md) - Kernel availability and discovery
- [Hardware Kernels](hardware-kernels.md) - Available accelerated operations
- [Blueprint Schema](blueprint-schema.md) - Configuring accelerator pipelines
