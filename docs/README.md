# Getting Started with Brainsmith

## Brainsmith Compiler

Brainsmith builds on the [FINN compiler](https://finn.readthedocs.io/en/latest/index.html) which creates accelerators by replacing each node in the model's ONNX graph with a *Hardware Kernel* that implements the operator in RTL/HLS. This constructs a dataflow accelerator tailored to the model's compute pattern with tunable parameters to balance throughput, latency, and resource utilization.

Brainsmith automates this process by defining a *Design Space* of potential Kernels and optimization strategies, exploring that space to converge on the optimal design for your design goals and resource budget. Declarative `.yaml` Blueprints define reusable design space exploration (DSE) pipelines for different model types.

***PRE-RELEASE NOTE***: Truly automated DSE (evaluating different paths to determine the best one) doesn't exist yet, just exhaustive exploration.

*Read more: [Design Space Exploration](docs/design_space_exploration.md), [Blueprint Schema](docs/blueprint_schema.md), [Multilayer Offload](docs/multilayer_offload.md)*

## Brainsmith Library

The Brainsmith compiler relies on a rich library of kernels, graph transforms, and build steps for effective design space exploration. This library is managed through a **singleton plugin registry**, allowing new components to be easily registered with relevant metadata via function decorators. Registered plugins can be referenced by name in DSE blueprints or directly through a variety of plugin access functions.

*Read more: [Plugin Registry](./plugin_registry.md)*

## Advanced Features

### Multilayer Offload (MLO)

For large neural networks that exceed on-chip memory capacity, Brainsmith supports **Multilayer Offload** - a technique that implements a single repeating layer (like a transformer encoder) in hardware and cycles weights through external high-bandwidth memory. This enables acceleration of much larger models.

MLO is particularly effective for transformer-based models (e.g. BERT) where the same layer structure repeats many times. Instead of implementing all layers in hardware, MLO implements just one layer and reuses it sequentially with different weights streamed from DRAM/HBM.

*Read more: [Multilayer Offload](docs/multilayer_offload.md)*

## Building Components

Hardware kernels are the synthesizable building blocks that implement neural network operations in RTL or HLS, requiring complex integration across multiple files (operators, backends, templates, and tests). Brainsmith provides the **Kernel Integrator** tool to automatically generate the Python wrapper and integration code from annotated SystemVerilog, dramatically simplifying the process of adding custom hardware implementations to your design space. This enables hardware engineers to focus on optimizing kernel implementations while the framework handles the integration complexity.

***PRE-RELEASE NOTE***: The Kernel Integrator currently supports RTL kernels only. Vitis HLS support is planned for a future release.

*Read more: [Hardware Kernels](docs/hardware_kernels.md), [Kernel Integrator](docs/kernel-integrator-user-guide.md), [Pragma Reference](docs/kernel-integrator-pragma-reference.md)*
