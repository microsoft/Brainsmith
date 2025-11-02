## Brainsmith

Brainsmith automates the creation of dataflow core accelerators and implementation of neural networks on FPGA, from PyTorch to RTL.

## Pre-Release

**This repository is in a pre-release state and under active co-development by Microsoft and AMD.**

### Pre-release features:
- **Component registry** - Extensible architecture for registering custom kernels, transforms, and build steps
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
# Run automated setup (creates .venv and initializes project)
./setup-venv.sh

# Activate Python virtual environment
source .venv/bin/activate

# Edit configuration for your Xilinx installation
vim .brainsmith/config.yaml  # Set xilinx_path, xilinx_version

# [Optional] Enable direnv for automatic environment activation
brainsmith project allow-direnv
# Otherwise, activate project environment
source .brainsmith/env.sh

# Verify setup
brainsmith project show
```

#### Option B: Docker-based Development
```bash
# Customize key environment variables in ctl-docker.sh as needed
export BSMITH_XILINX_PATH=/opt/Xilinx/Vivado/2024.2
export BSMITH_XILINX_VERSION=2024.2
export XILINXD_LICENSE_FILE=/path/to/your/license.lic

# Start container with automatic setup
./ctl-docker.sh start

# Open interactive shell
./ctl-docker.sh shell

# Verify setup
brainsmith project show

# OR run commands directly
./ctl-docker.sh "smith model.onnx blueprint.yaml"
```

### 2. Working with Multiple Projects

If you want to isolate work from the brainsmith repository, you can create separate project directories:

```bash
# Activate brainsmith venv (always required)
source /path/to/brainsmith/.venv/bin/activate

# Create a new project directory
brainsmith project init ~/my-fpga-project
cd ~/my-fpga-project

# Edit project-specific configuration
vim .brainsmith/config.yaml

# [Optional] Enable direnv for automatic environment activation
brainsmith project allow-direnv
# Otherwise, activate project environment
source .brainsmith/env.sh
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

**Advanced Docker Integration:** After running `brainsmith project init`, the generated `.brainsmith/.env` file can be used with Docker:

```bash
# Use .env file for Docker environment variables
docker run --env-file .brainsmith/.env microsoft/brainsmith:latest smith model.onnx blueprint.yaml

# Or with docker-compose (add to docker-compose.yml):
services:
  brainsmith:
    env_file: .brainsmith/.env
```

This provides unified environment management across local development and containerized builds.


## CLI Overview

Brainsmith provides two complementary CLI commands:

- **`brainsmith`** - Application configuration, setup, and environment management
- **`smith`** - Operational commands for dataflow core creation and kernel generation

For detailed command reference, see the [CLI API documentation](docs/cli_api_reference.md).

### Quick Examples

```bash
# Setup and configuration
brainsmith setup all               # Install all dependencies
source .brainsmith/env.sh          # Skip if using direnv

# Operations
smith model.onnx blueprint.yaml    # Create dataflow core accelerator
smith kernel accelerator.sv        # Generate hardware kernel from RTL
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
