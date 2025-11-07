# Configuration


Configuration management for Brainsmith projects with hierarchical loading and type-safe validation using Pydantic.

## Configuration Hierarchy

Settings are loaded from multiple sources with the following priority (highest to lowest):

1. **CLI arguments** - Passed to `load_config()` or command-line tools
2. **Environment variables** - `BSMITH_*` prefix (e.g., `BSMITH_BUILD_DIR`)
3. **Project config file** - `.brainsmith/config.yaml` in project root
4. **Built-in defaults** - Field defaults in `SystemConfig`

**First source with a value wins** - higher priority overrides lower priority.

---

## Environment Variables

All settings can be configured via environment variables using the `BSMITH_` prefix:

### Naming Convention

```bash
# Basic fields
BSMITH_{FIELD_NAME}={value}

# Nested objects (using double underscore)
BSMITH_LOGGING__LEVEL=debug
BSMITH_LOGGING__MAX_LOG_SIZE_MB=100

# Shortcuts
BSMITH_LOG_LEVEL=debug  # Shorthand for BSMITH_LOGGING__LEVEL
```

### Examples

```bash
# Core paths
export BSMITH_BUILD_DIR=/custom/build
export BSMITH_PROJECT_DIR=/path/to/project

# Xilinx tools
export BSMITH_XILINX_PATH=/tools/Xilinx
export BSMITH_XILINX_VERSION=2024.2

# Runtime
export BSMITH_DEFAULT_WORKERS=16
export BSMITH_COMPONENTS_STRICT=false

# Logging
export BSMITH_LOGGING__LEVEL=verbose
```

---

## Configuration Fields

### Core Paths

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `build_dir` | `Path` | `"build"` | Build directory for compilation artifacts. Relative paths resolve to `project_dir`. |
| `project_dir` | `Path` | *auto-detected* | Project root directory (parent of `.brainsmith/`). Auto-detected via upward directory walk or `BSMITH_PROJECT_DIR`. |
| `bsmith_dir` | `Path` | *auto-detected* | Brainsmith installation root (cached property). |
| `deps_dir` | `Path` | `{bsmith_dir}/deps` | Dependencies directory. **Always internal to brainsmith installation** (user input ignored). |

!!! note "Path Resolution"
    - **Absolute paths**: Used as-is
    - **Relative paths from CLI**: Resolve to current working directory
    - **Relative paths from YAML/env**: Resolve to `project_dir`

### Xilinx Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `xilinx_path` | `Path` | `"/tools/Xilinx"` | Xilinx root installation path. |
| `xilinx_version` | `str` | `"2024.2"` | Xilinx tool version (e.g., `"2024.2"`, `"2025.1"`). |
| `vivado_path` | `Path \| None` | *auto-detected* | Path to Vivado. Auto-detected from `{xilinx_path}/Vivado/{xilinx_version}`. |
| `vitis_path` | `Path \| None` | *auto-detected* | Path to Vitis. Auto-detected from `{xilinx_path}/Vitis/{xilinx_version}`. |
| `vitis_hls_path` | `Path \| None` | *auto-detected* | Path to Vitis HLS. Auto-detected from `{xilinx_path}/Vitis_HLS/{xilinx_version}`. |
| `vivado_ip_cache` | `Path \| None` | `{build_dir}/vivado_ip_cache` | Vivado IP cache directory for faster builds. |
| `vendor_platform_paths` | `str` | `"/opt/xilinx/platforms"` | Colon-separated vendor platform repository paths. |

!!! info "Tool Auto-Detection"
    If `vivado_path`, `vitis_path`, or `vitis_hls_path` are not explicitly set, they are automatically detected using the `xilinx_path` and `xilinx_version` settings.

### Component Registry

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `component_sources` | `Dict[str, Path \| None]` | `{'project': None}` | Filesystem-based component source paths. `'project'` defaults to `project_dir` (supports `kernels/` and `steps/` subdirectories). Core namespace (`'brainsmith'`) and entry points (`'finn'`) are loaded automatically. |
| `source_priority` | `List[str]` | `['project', 'brainsmith', 'finn', 'custom']` | Component source resolution priority (first match wins). Custom sources are auto-appended if not listed. |
| `source_module_prefixes` | `Dict[str, str]` | `{'brainsmith.': 'brainsmith', 'finn.': 'finn'}` | Module prefix → source name mapping for component classification. |
| `components_strict` | `bool` | `True` | Enable strict component loading (fail on errors vs. warn). Set to `false` for development. |
| `cache_components` | `bool` | `True` | Enable manifest caching for component discovery (auto-invalidates on file changes). |

### FINN Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `finn_root` | `Path \| None` | `{deps_dir}/finn` | FINN root directory. Defaults to `deps/finn` in brainsmith installation. |
| `finn_build_dir` | `Path \| None` | `{build_dir}` | FINN build directory. Defaults to project `build_dir`. |
| `finn_deps_dir` | `Path \| None` | `{deps_dir}` | FINN dependencies directory. Defaults to brainsmith `deps_dir`. |

### Runtime Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_workers` | `int` | `4` | Default number of workers for parallel operations. Exported as `NUM_DEFAULT_WORKERS`. |
| `netron_port` | `int` | `8080` | Port for Netron neural network visualization server. |

### Logging Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `logging.level` | `str` | `"normal"` | Console verbosity: `quiet` \| `normal` \| `verbose` \| `debug`. |
| `logging.finn_tools` | `Dict[str, str] \| None` | `None` | Per-tool log levels for FINN tools (e.g., `{'vivado': 'WARNING', 'hls': 'INFO'}`). |
| `logging.suppress_patterns` | `List[str] \| None` | `None` | Regex patterns to suppress from console output (file logs unaffected). |
| `logging.max_log_size_mb` | `int` | `0` | Maximum log file size in MB (0 = no rotation). |
| `logging.keep_backups` | `int` | `3` | Number of rotated log backups to keep. |

---

## Configuration File Format

### Location

`.brainsmith/config.yaml` in project root directory.

### Example Configuration

```yaml
# Build output directory
build_dir: build

# Xilinx tools (adjust paths for your installation)
xilinx_path: /tools/Xilinx
xilinx_version: "2024.2"

# Runtime options
default_workers: 14
netron_port: 8080
components_strict: false
vendor_platform_paths: /opt/xilinx/platforms

# Logging (advanced)
logging:
  level: verbose
  finn_tools:
    vivado: WARNING
    hls: INFO
  suppress_patterns:
    - "^DEBUG:"
  max_log_size_mb: 100
  keep_backups: 5

# Component sources (advanced)
component_sources:
  project: null  # defaults to project_dir
  my_custom: /path/to/custom/components

source_priority:
  - project
  - brainsmith
  - finn
  - my_custom
  - custom
```

### Environment Variable Expansion

YAML files support environment variable expansion:

```yaml
build_dir: ${HOME}/brainsmith-builds
xilinx_path: ${XILINX_ROOT}
default_workers: ${NUM_WORKERS:-8}  # Default to 8 if not set
```

---

## Common Configuration Patterns

### Minimal Configuration

Quick start with essential settings only:

```yaml
# .brainsmith/config.yaml
xilinx_path: /tools/Xilinx
xilinx_version: "2024.2"
```

### Development Configuration

Developer-friendly settings with verbose logging and relaxed validation:

```yaml
xilinx_path: /tools/Xilinx
xilinx_version: "2024.2"
build_dir: build
default_workers: 8
logging:
  level: verbose
components_strict: false
```

### Production Configuration

Optimized for CI/CD and production builds:

```yaml
xilinx_path: /opt/Xilinx
xilinx_version: "2024.2"
build_dir: /data/builds
default_workers: 32
logging:
  level: normal
  max_log_size_mb: 500
  keep_backups: 10
components_strict: true
cache_components: true
```

### Custom Component Sources

For teams with internal kernel libraries:

```yaml
component_sources:
  project: null  # defaults to project_dir
  internal: /company/fpga-lib
  experimental: ./custom-kernels

source_priority:
  - project
  - internal
  - brainsmith
  - finn
  - experimental
  - custom
```

---
