# !!!DISLAIMER!!! - These docs are under active rework on this branch, none should be taken as finished.

# Getting Started with Brainsmith

## Brainsmith Compiler

Brainsmith builds on the [FINN compiler](https://finn.readthedocs.io/en/latest/index.html) which creates accelerators by replacing each node in the model's ONNX graph with a *Hardware Kernel* that implements the operator in RTL/HLS. This constructs a dataflow accelerator tailored to the model's compute pattern with tunable parameters to balance throughput, latency, and resource utilization.

Brainsmith automates this process by defining a *Design Space* of potential Kernels and optimization strategies, exploring that space to converge on the optimal design for your design goals and resource budget. Declarative `.yaml` Blueprints define reusable design space exploration (DSE) pipelines for different model types.

***PRE-RELEASE NOTE***: Truly automated DSE (evaluating different paths to determine the best one) doesn't exist yet, just exhaustive exploration.

*Read more: [Design Space Exploration](docs/design_space_exploration.md), [Blueprint Schema](docs/blueprint_schema.md)*

## Brainsmith Library

The Brainsmith compiler relies on a rich library of kernels, graph transforms, and build steps for effective design space exploration. This library is managed through a **singleton plugin registry**, allowing new components to be easily registered with relevant metadata via function decorators. Registered plugins can be referenced by name in DSE blueprints.
```python
from brainsmith.core.plugins import step

@step(
    name="my_step",
    category="optimization",
    author="your name or organization"
)
def my_optimization_step(blueprint, context):
    # Apply transforms to model in place
```

Registered plugins can then be accessed by name in blueprints or a variety of plugin

and immediately accessed via blueprints.


The Brainsmith compiler relies on a rich library of kernels, graph transforms, and build steps for effective design exploration.

To enable easy addition and access to this library, the Plugin Registry maintains a singleton catalog

singleton registry of plugins


The plugin registry enables extensibility by allowing developers to register new functionality that can be discovered and used dynamically at runtime. The system uses a **singleton registry** that maintains a catalog of all available plugins, organized by type and tagged with metadata. Plugins registered via decorators become immediately available for use in Brainsmith's blueprint system (see [Blueprint Guide](./blueprints.md)) references components by name in YAML.

All plugins are managed through a central registry using decorator-based registration, accessible via blueprint or direct look-up.

**Example

*Read more: [Hardware Kernels](docs/hardware_kernels.md), [Plugin Registry](./plugin_registry.md)

## Building Components 



*Read more: [Hardware Kernels](docs/hardware_kernels.md), [Kernel Integrator](docs/kernel-integrator-user-guide.md)*
