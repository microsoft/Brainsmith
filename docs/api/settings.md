# Settings

Configuration management for Brainsmith projects with hierarchical loading and type-safe validation using Pydantic.

Supports loading from CLI arguments, environment variables (BSMITH_* prefix), project config files (brainsmith.yaml), and built-in defaults.

---

::: brainsmith.settings.load_config

**Example:**

```python
from brainsmith.settings import load_config

# Load with default project config
config = load_config()

# Load with custom project file
config = load_config(project_file="custom.yaml")

# Load with CLI overrides
config = load_config(
    build_dir="./custom-build",
    vivado_path="/tools/Xilinx/Vivado/2024.2"
)
```

---

::: brainsmith.settings.get_config

**Example:**

```python
from brainsmith.settings import get_config

# Get cached config instance
config = get_config()

# Access configuration values
print(config.build_dir)
print(config.vivado_path)
print(config.logging.level)
```

---

::: brainsmith.settings.get_default_config

**Example:**

```python
from brainsmith.settings import get_default_config

# Get default config without loading from files/env
default_config = get_default_config()
```

---

::: brainsmith.settings.SystemConfig

**Example:**

```python
from brainsmith.settings import SystemConfig

# Create config with custom values
config = SystemConfig(
    build_dir="./build",
    vivado_path="/tools/Xilinx/Vivado/2024.2"
)

# Access nested configuration
print(config.logging.level)
print(config.netron_port)
```

---

::: brainsmith.settings.EnvironmentExporter

**Example:**

```python
from brainsmith.settings import get_config, EnvironmentExporter

config = get_config()
exporter = EnvironmentExporter(config)

# Export environment variables for external tools
env_dict = exporter.to_external_dict()
print(env_dict['FINN_ROOT'])
print(env_dict['VIVADO_PATH'])
```


## See Also

- [Getting Started](../getting-started.md) - Installation and project setup
- [GitHub](https://github.com/microsoft/brainsmith) - Issues and questions
