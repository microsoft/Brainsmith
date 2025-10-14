# Plugin System API Reference

The plugin system is the core of Brainsmith's extensibility.

## Registry

::: brainsmith.core.plugins.registry.Registry
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - register
        - get
        - find
        - all
        - reset

## Registration Decorators

::: brainsmith.core.plugins.registry.transform
    options:
      show_root_heading: true
      heading_level: 3

::: brainsmith.core.plugins.registry.kernel
    options:
      show_root_heading: true
      heading_level: 3

::: brainsmith.core.plugins.registry.backend
    options:
      show_root_heading: true
      heading_level: 3

::: brainsmith.core.plugins.registry.step
    options:
      show_root_heading: true
      heading_level: 3

## Access Functions

::: brainsmith.core.plugins.registry.get_transform
    options:
      show_root_heading: true
      heading_level: 3

::: brainsmith.core.plugins.registry.get_kernel
    options:
      show_root_heading: true
      heading_level: 3

::: brainsmith.core.plugins.registry.get_backend
    options:
      show_root_heading: true
      heading_level: 3

::: brainsmith.core.plugins.registry.get_step
    options:
      show_root_heading: true
      heading_level: 3

## Query Functions

::: brainsmith.core.plugins.registry.list_transforms
    options:
      show_root_heading: true
      heading_level: 3

::: brainsmith.core.plugins.registry.list_kernels
    options:
      show_root_heading: true
      heading_level: 3

::: brainsmith.core.plugins.registry.has_transform
    options:
      show_root_heading: true
      heading_level: 3

::: brainsmith.core.plugins.registry.has_kernel
    options:
      show_root_heading: true
      heading_level: 3
