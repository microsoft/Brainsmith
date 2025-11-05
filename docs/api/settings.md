# Settings

Configuration management for Brainsmith.

## Overview

Hierarchical configuration with type-safe access:

1. **System defaults** - Built-in defaults
2. **Environment variables** - `FINN_ROOT`, `XILINX_PATH`, etc.
3. **Project configuration** - `.brainsmith/config.yaml`

## Configuration Access

::: brainsmith.settings.get_config

::: brainsmith.settings.load_config

::: brainsmith.settings.get_default_config

## Configuration Schema

::: brainsmith.settings.SystemConfig

## Environment Export

::: brainsmith.settings.EnvironmentExporter

## See Also

- [Configuration Guide](../getting-started/configuration.md) - How to configure Brainsmith
- [CLI Reference](../developer-guide/3-reference/cli.md) - Using `brainsmith project` commands
