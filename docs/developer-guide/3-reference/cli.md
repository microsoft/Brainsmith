# CLI Reference

This guide covers the Brainsmith command-line interfaces: `brainsmith` and `smith`.

## Overview

Brainsmith provides a dual CLI system designed to separate administrative tasks from operational workflows:

- **`brainsmith`** - Application-level configuration and setup commands
- **`smith`** - Streamlined operational commands for hardware design generation

## CLI Architecture

### Dual Entry Points

The system uses two distinct entry points to provide different user experiences:

**Administrative CLI (`brainsmith`)**:
- Configuration management (`config`)
- Setup utilities (`setup`)
- Project initialization (`project`)
- Can invoke operational commands via subcommands
- Intended for system administrators and initial setup

**Operational CLI (`smith`)**:
- Focused on core workflows (dataflow core creation, kernel generation)
- Simplified interface for daily use
- Inherits configuration from brainsmith setup
- Intended for design engineers and regular users

### Configuration Hierarchy

Brainsmith follows a clear precedence order for configuration:

1. **Command-line arguments** (highest priority)
2. **Environment variables** (`BSMITH_*` prefix)
3. **Project configuration** (`.brainsmith/config.yaml`)
4. **User configuration** (`~/.brainsmith/config.yaml`)
5. **Built-in defaults** (lowest priority)

See: [Configuration Guide](../../getting-started/configuration.md)

---

## Command Reference

### `brainsmith` - Application Configuration

```bash
brainsmith [OPTIONS] COMMAND [ARGS]...
```

**Global Options:**
- `--config, -c PATH` - Use specific configuration file
- `--build-dir PATH` - Override build directory
- `--debug` - Enable debug mode with detailed logging
- `--version` - Show Brainsmith version
- `--help` - Show help message

---

## Configuration Management

### `brainsmith config show`

Display current configuration with all active settings.

```bash
brainsmith config show [OPTIONS]
```

**Options:**
- `--format, -f [table|yaml|json|env]` - Output format (default: table)
- `--verbose, -v` - Include source information and path validation
- `--external-only` - For env format, show only external tool variables

**Examples:**
```bash
# Show configuration as table
brainsmith config show

# Show as YAML with source information
brainsmith config show --format yaml --verbose

# Show environment variables
brainsmith config show --format env
```

### `brainsmith config init`

Initialize a new configuration file.

```bash
brainsmith config init [OPTIONS]
```

**Options:**
- `--user` - Create user-level config (~/.brainsmith/config.yaml)
- `--project` - Create project-level config (./brainsmith_config.yaml)
- `--force, -f` - Overwrite existing file
- `--full` - Include all possible configuration fields

**Examples:**
```bash
# Create project configuration
brainsmith config init --project

# Create user defaults
brainsmith config init --user --force

# Create full configuration with all fields
brainsmith config init --full
```

### `brainsmith config export`

Export configuration as shell environment script.

```bash
brainsmith config export [OPTIONS]
```

**Options:**
- `--shell [bash|zsh|fish|powershell]` - Shell format (default: bash)

**Examples:**
```bash
# Export for current shell
eval $(brainsmith config export)

# Export for fish shell
eval (brainsmith config export --shell fish)
```

---

## Project Management

### `brainsmith project init`

Initialize a new project directory with configuration.

```bash
brainsmith project init [PATH]
```

**Arguments:**
- `PATH` - Directory to initialize (default: current directory)

**Examples:**
```bash
# Initialize current directory
brainsmith project init

# Create and initialize new project
brainsmith project init ~/my-fpga-project
```

### `brainsmith project info`

Display current project configuration.

```bash
brainsmith project info
```

Shows effective configuration for the current project, including:
- Xilinx tool paths
- Build directories
- Active settings

### `brainsmith project allow-direnv`

Enable direnv integration for automatic environment activation.

```bash
brainsmith project allow-direnv
```

This configures direnv to automatically load `.brainsmith/env.sh` when you cd into the directory.

---

## Setup and Dependencies

### `brainsmith setup all`

Install all dependencies (C++ simulation, Xilinx simulation, board files).

```bash
brainsmith setup all [OPTIONS]
```

**Options:**
- `--force, -f` - Force reinstallation even if already present

### `brainsmith setup cppsim`

Setup C++ simulation dependencies (cnpy, finn-hlslib).

```bash
brainsmith setup cppsim [OPTIONS]
```

**Options:**
- `--force, -f` - Force reinstallation

### `brainsmith setup xsim`

Setup Xilinx simulation (build finn-xsim with Vivado).

```bash
brainsmith setup xsim [OPTIONS]
```

**Options:**
- `--force, -f` - Force rebuild

### `brainsmith setup boards`

Download FPGA board definition files.

```bash
brainsmith setup boards [OPTIONS]
```

**Options:**
- `--force, -f` - Force redownload
- `--repo, -r [xilinx|avnet|realdigital]` - Specific repository to download
- `--verbose, -v` - Show detailed list of all board files

### `brainsmith setup check`

Check the status of all setup components.

```bash
brainsmith setup check [OPTIONS]
```

**Options:**
- `--verbose, -v` - Show detailed information

**Example output:**
```
Setup Status:
✓ cnpy (C++ NPY support)
✓ finn-hlslib headers
✓ finn-xsim
✓ Vivado 2024.2
✓ Vitis HLS 2024.2
✓ Board files (47 boards)
```

---

## Operational Commands

### `smith` - Streamlined Operations

```bash
smith [COMMAND] [OPTIONS]
```

**Global Options:**
- `--version` - Show version
- `--help` - Show help message

---

## Dataflow Core Creation

### `smith dfc` (default command)

Create a dataflow core accelerator for neural network acceleration.

```bash
smith dfc MODEL BLUEPRINT [OPTIONS]
```

**Arguments:**
- `MODEL` - Path to ONNX model file
- `BLUEPRINT` - Path to Blueprint YAML file defining the dataflow architecture

**Options:**
- `--output-dir, -o PATH` - Output directory (defaults to build dir with timestamp)
- `--start-step STEP` - Start execution from this step (inclusive)
- `--stop-step STEP` - Stop execution at this step (inclusive)

**Examples:**
```bash
# Basic dataflow core creation
smith dfc model.onnx blueprint.yaml

# Shorthand (dfc is the default command)
smith model.onnx blueprint.yaml

# With custom output directory
smith dfc model.onnx blueprint.yaml --output-dir ./results

# Using brainsmith context with debug mode
brainsmith --debug smith dfc model.onnx blueprint.yaml

# Run specific step range
smith dfc model.onnx blueprint.yaml --start-step streamline --stop-step specialize_layers
```

---

## Hardware Kernel Generation

### `smith kernel`

Generate hardware kernel from RTL for FINN integration.

```bash
smith kernel RTL_FILE [OPTIONS]
```

**Arguments:**
- `RTL_FILE` - Path to SystemVerilog RTL source file with embedded pragmas

**Options:**
- `--output-dir, -o PATH` - Directory for generated files (default: RTL file location)
- `--validate` - Validate RTL only without generating files
- `--info` - Display parsed kernel metadata and exit
- `--artifacts [kernelop|rtlbackend|wrapper]` - Generate specific files only
- `--no-strict` - Disable strict validation
- `--include-rtl PATH` - Additional RTL files to include (can specify multiple)
- `--rtl-path PATHS` - Colon-separated paths to search for RTL files
- `--verbose, -v` - Enable verbose output

**Examples:**
```bash
# Generate kernel files
smith kernel my_accelerator.sv

# Validate RTL only
smith kernel my_accelerator.sv --validate

# Generate specific artifacts
smith kernel my_accelerator.sv --artifacts kernelop --artifacts wrapper

# With additional RTL files
smith kernel top.sv --include-rtl helper.sv --include-rtl memory.sv
```

---

## Configuration Settings

### Available Settings

#### Core Settings
- `debug` (bool) - Enable DEBUG-level logging and detailed error traces
- `build_dir` (path) - Build directory for artifacts
- `deps_dir` (path) - Dependencies directory

#### Xilinx Tools
- `xilinx_path` (path) - Xilinx root installation path
- `xilinx_version` (string) - Xilinx tool version (e.g., "2024.2")
- `vivado_path` (path) - Path to Vivado (auto-detected)
- `vitis_path` (path) - Path to Vitis (auto-detected)
- `vitis_hls_path` (path) - Path to Vitis HLS (auto-detected)

#### Tool Settings
- `platform_repo_paths` (string) - Platform repository paths
- `plugins_strict` (bool) - Strict plugin loading
- `vivado_ip_cache` (path) - Vivado IP cache directory
- `netron_port` (int) - Port for Netron visualization

#### FINN Configuration
- `finn.finn_root` (path) - FINN root directory
- `finn.finn_build_dir` (path) - FINN build directory
- `finn.finn_deps_dir` (path) - FINN dependencies directory
- `finn.num_default_workers` (int) - Default number of workers

### Environment Variables

All settings can be overridden using environment variables with the `BSMITH_` prefix:

```bash
export BSMITH_DEBUG=true
export BSMITH_BUILD_DIR=/tmp/builds
export BSMITH_XILINX_VERSION=2024.2
export BSMITH_FINN__NUM_DEFAULT_WORKERS=8
```

!!! note "Nested Settings"
    Nested settings use double underscore (`__`) as delimiter.

---

## Usage Patterns

### Initial Setup
```bash
# Install all dependencies
brainsmith setup all

# Initialize user configuration
brainsmith config init --user

# Edit ~/.brainsmith/config.yaml to configure Xilinx tools as needed

# Verify setup
brainsmith setup check
```

### Daily Workflow
```bash
# Create dataflow core with user defaults
smith model.onnx blueprint.yaml

# Generate hardware kernel
smith kernel accelerator.sv

# Override settings temporarily
brainsmith --build-dir /tmp/test smith model.onnx blueprint.yaml
```

### Team Collaboration
```bash
# Create project configuration
brainsmith config init --project

# Team members use project settings
smith model.onnx blueprint.yaml

# Check active configuration
brainsmith config show --verbose
```

### CI/CD Integration
```bash
# Export configuration for CI scripts
eval $(brainsmith config export)

# All tools now have correct paths
vivado -version
vitis_hls -version

# Run automated builds
smith model.onnx blueprint.yaml --output-dir $CI_ARTIFACTS_DIR
```

---

## Troubleshooting

### Check Configuration
```bash
# See all active settings
brainsmith config show --verbose

# Verify tool detection
brainsmith setup check --verbose
```

### Debug Mode
```bash
# Enable DEBUG-level logging and detailed error information
brainsmith --debug smith model.onnx blueprint.yaml

# Or edit config to set debug: true as default
```

### Reset Configuration
```bash
# Remove user configuration
rm ~/.brainsmith/config.yaml

# Remove project configuration
rm .brainsmith/config.yaml

# Reinitialize
brainsmith config init --user
```

---

## Next Steps

- [Blueprints](blueprints.md) - Learn the YAML configuration format
- [Design Space Exploration](../2-core-systems/design-space-exploration.md) - Understand DSE concepts
- [Configuration Guide](../../getting-started/configuration.md) - Deep dive on configuration
