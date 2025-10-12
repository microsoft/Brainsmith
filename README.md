## Brainsmith

Brainsmith automates design space exploration (DSE) and implementation of neural networks on FPGA, from PyTorch to RTL.

## Pre-Release

**This repository is in a pre-release state and under active co-development by Microsoft and AMD.**

### Pre-release features:
- **Plugin system** - Extensible architecture for registering custom kernels, transforms, and build steps
- **Blueprint interface** - YAML-based declarative configuration with inheritance support for defining design spaces
- **Segment-based execution** - Efficient DSE through intelligent computation reuse between exploration branches
- **BERT demo** - Example end-to-end acceleration (PyTorch to stitched-IP RTL accelerator)

### Planned major features:
- **Multi-Layer Offload** - Implement a repeating slice of a model (e.g. 1 transformer encoder) and cycle weights through DRAM/HBM, enabling drastically larger model support.
- **Automated Design Space Exploration (DSE)** - Iteratively run builds across a design space, evaluating performance to converge on the optimal design for given search objectives and constraints
- **Parallelized tree execution** - Execute multiple builds in parallel, intelligently re-using build artifacts
- **Automated Kernel Integrator** - Easy integration of new hardware kernels, generate full compiler integration python code from RTL or HLS code alone
- **FINN Kernel backend rework** - Flexible backends for FINN kernels, currently you can only select between HLS or RTL backend, in the future releases multiple RTL or HLS backends will be supported to allow for more optimization
- **Accelerated FIFO sizing** - The FIFO sizing phase of Brainsmith builds currently represents >90% of runtime (not including Vivado Synthesis + Implementation). This will be significantly accelerated in future releases.

## Quick Start

### Dependencies
1. Ubuntu 22.04+
2. Vivado Design Suite 2024.2 (migration to 2025.1 in process)
3. For Docker setup: Docker with [non-root permissions](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)
4. For local setup: Python 3.10+ and [Poetry](https://python-poetry.org/docs/#installation)

### 1. Installation Options

#### Option A: Docker-based Development

```bash
# Start container with automatic setup
./ctl-docker.sh start

# Open interactive shell
./ctl-docker.sh shell

# Run commands directly
./ctl-docker.sh "smith model.onnx blueprint.yaml"

# Optional: Enable persistent user configuration
BSMITH_DOCKER_USER_MOUNT=1 ./ctl-docker.sh start
```

#### Option B: Local Development with Poetry

```bash
# Run automated setup script
./setup-venv.sh

# Activate the virtual environment
source .venv/bin/activate

# Check installation status
brainsmith setup check
```

### 2. Configure Brainsmith

```bash
# Initialize user configuration
brainsmith config init --user

# Edit ~/.brainsmith/config.yaml to set xilinx_path and xilinx_version as needed

# View current configuration
brainsmith config show
```

### 3. Run Design Space Exploration

```bash
# Run DSE with default command
smith model.onnx blueprint.yaml

# Or specify output directory
smith model.onnx blueprint.yaml --output-dir ./results
```

### 4. Validate Installation

```bash
# For Docker setup:
./ctl-docker.sh ./examples/bert/quicktest.sh

# For local setup:
./examples/bert/quicktest.sh
```

## CLI Overview

Brainsmith provides two complementary CLI commands:

- **`brainsmith`** - Application configuration, setup, and environment management
- **`smith`** - Operational commands for DSE and kernel generation

For detailed command reference, see the [CLI API documentation](docs/cli_api_reference.md).

### Quick Examples

```bash
# Setup and configuration
brainsmith setup all              # Install all dependencies
eval $(brainsmith config export)  # Export environment for Xilinx tools

# Operations
smith model.onnx blueprint.yaml   # Run design space exploration
smith kernel accelerator.sv       # Generate hardware kernel from RTL
```

## Documentation

For detailed documentation and guides, see the [documentation overview](docs/README.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Brainsmith is developed through a collaboration between Microsoft and AMD.

The project builds upon:
- [FINN](https://github.com/Xilinx/finn) - Dataflow compiler for quantized neural networks on FPGAs
- [QONNX](https://github.com/fastmachinelearning/qonnx) - Quantized ONNX model representation
- [Brevitas](https://github.com/Xilinx/brevitas) - PyTorch quantization library
