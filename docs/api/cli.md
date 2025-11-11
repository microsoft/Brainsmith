# CLI Reference

Dual command-line interface: `brainsmith` for project management (setup, configuration), and `smith` for hardware design generation (DFC creation).


## Global Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-b, --build-dir` | Path | `build/` | Override build directory |
| `-c, --config` | Path | `brainsmith.yaml` | Override configuration file |
| `-l, --log-level` | Choice | `normal` | Set log verbosity (`quiet`, `normal`, `verbose`, `debug`) |
| `--no-progress` | Flag | - | Disable progress spinners and animations |
| `--version` | Flag | - | Show version and exit |
| `-h, --help` | Flag | - | Show help message and exit |


## Operational Commands

### dfc

Create a dataflow core accelerator for neural network acceleration.

**Syntax:**

```bash
smith MODEL BLUEPRINT [OPTIONS]
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `MODEL` | Path | Path to ONNX model file |
| `BLUEPRINT` | Path | Path to Blueprint YAML file defining the dataflow architecture |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output-dir` | Path | `build/{timestamp}` | Output directory for generated files |
| `--start-step` | Text | - | Override blueprint start_step (start execution from this step, inclusive) |
| `--stop-step` | Text | - | Override blueprint stop_step (stop execution at this step, inclusive) |

**Example:**

```bash
# Basic usage
smith model.onnx blueprint.yaml

# Custom output directory
smith model.onnx blueprint.yaml --output-dir ./results

# Run specific step range
smith model.onnx blueprint.yaml \
  --start-step streamline \
  --stop-step specialize_layers
```

See also: [Blueprint Schema](../developer-guide/blueprint-schema.md), [Design Space Exploration](dse.md)


## Administrative Commands

### project

Manage Brainsmith projects and configuration.

**Syntax:**

```bash
brainsmith project <SUBCOMMAND> [OPTIONS]
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `init` | Initialize project with configuration and environment scripts |
| `info` | Display current project configuration |
| `allow-direnv` | Enable direnv integration for automatic environment activation |

#### project init

Initialize a Brainsmith project with configuration file and environment scripts.

**Syntax:**

```bash
brainsmith project init [PATH] [OPTIONS]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `PATH` | Path | `.` | Directory to initialize (current directory if not specified) |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-f, --force` | Flag | Overwrite existing `brainsmith.yaml` if present |

**Behavior:**

- Creates `brainsmith.yaml` configuration file
- Creates `.brainsmith/` directory for project metadata
- Generates `env.sh` activation script
- Generates `.envrc` for direnv support

**Example:**

```bash
# Initialize current directory
brainsmith project init

# Create and initialize new project
brainsmith project init ./my-fpga-project

# Overwrite existing configuration
brainsmith project init --force
```

#### project info

Display current project configuration with source information.

**Syntax:**

```bash
brainsmith project info [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--finn` | Flag | Include FINN-specific configuration settings |

**Output:**

- Configuration metadata (project directory, environment status)
- Core paths (build directory, dependencies directory)
- Component registry settings
- Toolchain configuration
- Xilinx tools paths
- FINN configuration (with `--finn` flag)

**Example:**

```bash
# Show project configuration
brainsmith project info

# Include FINN-specific settings
brainsmith project info --finn
```

#### project allow-direnv

Enable direnv integration for automatic environment activation when entering project directory.

**Syntax:**

```bash
brainsmith project allow-direnv
```

**Behavior:**

- Verifies direnv is installed
- Generates `.envrc` file if needed
- Executes `direnv allow` to trust the configuration
- Validates shell hook is configured

**Example:**

```bash
brainsmith project allow-direnv
```


### registry

Display registered components (kernels, backends, pipeline steps) organized by source.

**Syntax:**

```bash
brainsmith registry [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-v, --verbose` | Flag | Show detailed component information with full listings |
| `-r, --rebuild` | Flag | Rebuild component cache and validate all entries (slower but thorough) |

**Output:**

- Component sources table (source, type, path, status)
- Component summary by source (steps, kernels, backends counts)
- With `--verbose`: Detailed listings organized by component type
- With `--rebuild`: Validation results for all discovered components

**Example:**

```bash
# Quick listing using cached registry
brainsmith registry

# Show detailed component information
brainsmith registry --verbose

# Rebuild cache and validate all components
brainsmith registry --rebuild

# Verbose output with cache rebuild
brainsmith registry -v -r
```

See also: [Component Registry](registry.md) - Programmatic access to registered components


### setup

Install and configure dependencies for Brainsmith development and testing.

**Syntax:**

```bash
brainsmith setup <SUBCOMMAND> [OPTIONS]
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `all` | Install all dependencies (cppsim, xsim, boards) |
| `cppsim` | Setup C++ simulation for fast functional testing |
| `xsim` | Setup Xilinx RTL simulation for cycle-accurate validation (requires Vivado) |
| `boards` | Download FPGA board definition files for deployment |
| `check` | Check installation status of all setup components |

#### setup all

Install all dependencies at once.

**Syntax:**

```bash
brainsmith setup all [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-f, --force` | Flag | Force reinstallation even if already installed |
| `-r, --remove` | Flag | Remove all dependencies |
| `-y, --yes` | Flag | Skip confirmation prompts |

**Example:**

```bash
# Install all dependencies
brainsmith setup all

# Remove all dependencies
brainsmith setup all --remove --yes
```

#### setup cppsim

Setup C++ simulation dependencies (cnpy for NPY file support, finn-hlslib headers).

**Syntax:**

```bash
brainsmith setup cppsim [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-f, --force` | Flag | Force reinstallation |
| `-r, --remove` | Flag | Remove C++ simulation dependencies |
| `-y, --yes` | Flag | Skip confirmation prompts |

**Example:**

```bash
# Install C++ simulation dependencies
brainsmith setup cppsim

# Force reinstallation
brainsmith setup cppsim --force

# Remove dependencies
brainsmith setup cppsim --remove --yes
```

#### setup xsim

Setup Xilinx simulation by building finn-xsim with Vivado.

**Syntax:**

```bash
brainsmith setup xsim [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-f, --force` | Flag | Force rebuild even if already built |
| `-r, --remove` | Flag | Remove Xilinx simulation dependencies |
| `-y, --yes` | Flag | Skip confirmation prompts |

**Requirements:**

- Vivado must be configured in `brainsmith.yaml` or via environment variables

**Example:**

```bash
# Build finn-xsim
brainsmith setup xsim

# Force rebuild
brainsmith setup xsim --force

# Remove xsim
brainsmith setup xsim --remove --yes
```

#### setup boards

Download FPGA board definition files from supported repositories.

**Syntax:**

```bash
brainsmith setup boards [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-f, --force` | Flag | Force redownload even if already present |
| `--remove` | Flag | Remove board definition files |
| `-r, --repo` | Text | Specific repository to download (multiple allowed) |
| `-v, --verbose` | Flag | Show detailed list of all board definitions by repository |
| `-y, --yes` | Flag | Skip confirmation prompts |

**Available Repositories:**

- `xilinx` - Official Xilinx board files
- `avnet` - Avnet board files
- `rfsoc4x2` - RFSoC 4x2 board files
- `kv260` - Kria KV260 board files
- `aupzu3` - AU+PZU3 board files
- `pynq-z1` - PYNQ-Z1 board files
- `pynq-z2` - PYNQ-Z2 board files

**Example:**

```bash
# Download all board repositories
brainsmith setup boards

# Download specific repositories
brainsmith setup boards --repo xilinx --repo avnet

# Show detailed board list
brainsmith setup boards --verbose

# Remove all board files
brainsmith setup boards --remove --yes
```

#### setup check

Check the installation status of all setup components.

**Syntax:**

```bash
brainsmith setup check
```

**Output:**

Displays installation status table showing:

- cnpy (C++ NPY file support)
- finn-hlslib headers
- finn-xsim
- Vivado (with version and sourcing status)
- Vitis HLS (with version and sourcing status)
- Board files (repository count)

**Example:**

```bash
brainsmith setup check
```


## See Also

- [Design Space Exploration](dse.md) - DSE API for exploring hardware configurations
- [Settings](settings.md) - Configuration management API
- [Component Registry](registry.md) - Programmatic access to registered components
- [Getting Started](../getting-started.md) - Installation and quickstart guide
- [GitHub](https://github.com/microsoft/brainsmith) - Issues and questions
