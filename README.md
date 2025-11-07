## Brainsmith

Brainsmith automates the creation of dataflow core accelerators and implementation of neural networks on FPGA, from PyTorch to RTL.

## Pre-Release

**This repository is in a pre-release state and under active co-development by Microsoft and AMD.**

### Pre-release features:
- **Component registry** - Extensible architecture for registering custom kernels, transforms, and build steps
- **Blueprint interface** - YAML-based declarative configuration with inheritance support for defining design spaces
- **Segment-based execution** - Efficient DSE through intelligent computation reuse between exploration branches
- **BERT demo** - Example end-to-end acceleration (PyTorch to stitched-IP RTL accelerator)

Please follow the [Brainsmith Feature Roadmap](https://github.com/orgs/microsoft/projects/2017) for planned and upcoming features.

## Quick Start

### Dependencies
1. Ubuntu 22.04+
2. Vivado Design Suite 2024.2 (migration to 2025.1 in process)
3. Python 3.10+ and [Poetry](https://python-poetry.org/docs/#installation)
4. [Optional] [direnv](https://direnv.net/) for automatic environment activation
5. [Optional] Docker with [non-root permissions](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)

### 1. Installation Options

```bash
# Clone and setup brainsmith
git clone https://github.com/microsoft/brainsmith.git ./brainsmith
cd brainsmith
```

#### Option A: Local Development with Poetry

```bash
# Run automated setup (creates .venv, initializes project, configures direnv if installed)
./setup-venv.sh

# Edit configuration for your Xilinx installation
vim .brainsmith/config.yaml  # Set xilinx_path, xilinx_version

# Activate environment
# Option 1: For direnv users
cd .
# Option 2: Manual (must be run from brainsmith root)
source .venv/bin/activate && source .brainsmith/env.sh

# Verify setup
brainsmith project info
```

#### Option B: Docker-based Development
```bash
# Customize key environment variables in ctl-docker.sh as needed
export BSMITH_XILINX_PATH=/opt/Xilinx/Vivado/2024.2
export BSMITH_XILINX_VERSION=2024.2
export XILINXD_LICENSE_FILE=/path/to/your/license.lic

# Start container with automatic setup
./ctl-docker.sh start

# Open interactive shell and verify setup
./ctl-docker.sh shell
brainsmith project info

# OR run one-off command
./ctl-docker.sh "brainsmith project info"
```

### 2. Working with Multiple Projects

Create separate project directories to isolate work from the brainsmith repository:

```bash
# Activate brainsmith venv (always required)
source /path/to/brainsmith/.venv/bin/activate

# Create a new project directory
brainsmith project init ~/my-fpga-project
cd ~/my-fpga-project

# Edit project-specific configuration
vim .brainsmith/config.yaml

# Enable direnv (auto-reloads on config changes) or manually activate
brainsmith project allow-direnv && cd .  # direnv option
source .brainsmith/env.sh                # manual option (re-run after config edits)
```

### 3. Validate installation with simple example

```bash
./examples/bert/quicktest.sh
```

### 4. Create your own Dataflow Core accelerator

```bash
# Create dataflow core with default command
smith model.onnx blueprint.yaml

# Or specify output directory
smith model.onnx blueprint.yaml --output-dir ./results
```

## CLI Overview

Brainsmith provides two complementary CLI commands:

- **`brainsmith`** - Application configuration, setup, and environment management
- **`smith`** - Operational commands for dataflow core creation and kernel generation

For detailed command reference, see the [CLI API documentation](docs/cli_api_reference.md).

## Documentation

For detailed documentation and guides, see the `prerelease-docs` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Brainsmith is developed through a collaboration between Microsoft and AMD.

The project builds upon:
- [FINN](https://github.com/Xilinx/finn) - Dataflow compiler for quantized neural networks on FPGAs
- [QONNX](https://github.com/fastmachinelearning/qonnx) - Quantized ONNX model representation
- [Brevitas](https://github.com/Xilinx/brevitas) - PyTorch quantization library
