# Component Registry

Plugin system for extending Brainsmith with custom kernels, backends, and pipeline steps.

## Overview

Register components using decorators:

- Custom hardware kernels
- Backend implementations (HLS/RTL)
- Pipeline transformation steps

Components are discovered automatically from brainsmith, FINN, your project, and custom plugins.

## Registration Decorators

::: brainsmith.registry.kernel

::: brainsmith.registry.backend

::: brainsmith.registry.step

::: brainsmith.registry.source_context

## Discovery and Lifecycle

::: brainsmith.registry.discover_components

::: brainsmith.registry.reset_registry

::: brainsmith.registry.is_initialized

## Lookup Functions - Kernels

::: brainsmith.registry.get_kernel

::: brainsmith.registry.get_kernel_infer

::: brainsmith.registry.has_kernel

::: brainsmith.registry.list_kernels

## Lookup Functions - Backends

::: brainsmith.registry.get_backend

::: brainsmith.registry.get_backend_metadata

::: brainsmith.registry.list_backends

::: brainsmith.registry.list_backends_for_kernel

## Lookup Functions - Steps

::: brainsmith.registry.get_step

::: brainsmith.registry.has_step

::: brainsmith.registry.list_steps

## Metadata Access

::: brainsmith.registry.get_component_metadata

::: brainsmith.registry.get_all_component_metadata

## Domain Resolution

::: brainsmith.registry.get_domain_for_backend

## Metadata Structures

::: brainsmith.registry.ComponentMetadata

::: brainsmith.registry.ComponentType

::: brainsmith.registry.ImportSpec

## See Also

- [Component Registry Architecture](../developer-guide/2-core-systems/component-registry.md) - Plugin system concepts and design
- [CLI Reference](../developer-guide/3-reference/cli.md) - Using `brainsmith list` commands
