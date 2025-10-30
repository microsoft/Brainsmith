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
3. For Docker setup: Docker with [non-root permissions](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)
4. For local setup: Python 3.10+ and [Poetry](https://python-poetry.org/docs/#installation)

### 1. Installation Options

#### Option A: Docker-based Development

```bash
# Customize key environment variables in ctl-docker.sh as needed
export BSMITH_XILINX_PATH=/opt/Xilinx/Vivado/2024.2
export BSMITH_XILINX_VERSION=2024.2
export XILINXD_LICENSE_FILE=/path/to/your/license.lic

# Start container with automatic setup
./ctl-docker.sh start

# Open interactive shell
./ctl-docker.sh shell

# OR run commands directly
./ctl-docker.sh "smith model.onnx blueprint.yaml"
```

#### Option B: Local Development with Poetry

```bash
# 1. Run automated setup (creates .venv + .brainsmith/config.yaml)
./setup-venv.sh

# 2. Configure settings & tools for your system
vim .brainsmith/config.yaml

# 3. Choose environment activation method (see below)
```

**Choose your environment method:**

**Option 1: direnv (Recommended)**
```bash
# One-time setup
sudo apt install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc

# Enable for brainsmith repo (applies to CWD)
brainsmith project allow-direnv

# Daily use - automatic on cd
cd ~/brainsmith  # Environment auto-loads
pytest tests/
```

**Option 2: Manual activation**
```bash
# Each shell session - two commands
source .venv/bin/activate       # Python packages
source .brainsmith/env.sh       # Xilinx environment

# Daily use
pytest tests/
smith model.onnx blueprint.yaml
```

**Note:** The brainsmith repository itself becomes a project during development. To create additional projects for your designs:

```bash
brainsmith project init ~/my-fpga-project
cd ~/my-fpga-project
brainsmith project allow-direnv  # or manual activation
```

### 3. Validate installation with simple example

```bash
# For Docker setup:
./ctl-docker.sh ./examples/bert/quicktest.sh

# For local setup:
./examples/bert/quicktest.sh
```

### 4. Create your own Dataflow Core accelerator

```bash
# Create dataflow core with default command
smith model.onnx blueprint.yaml

# Or specify output directory
smith model.onnx blueprint.yaml --output-dir ./results
```

## Environment Management

Brainsmith requires Xilinx tools (Vivado, Vitis HLS) to be available before Python starts. We provide two approaches:

### Option 1: direnv (Recommended)

Automatic environment switching when you cd between projects:

```bash
# One-time setup
sudo apt install direnv  # or: brew install direnv (macOS)
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc

# Enable for this project
brainsmith project allow-direnv

# Daily use - just cd
cd ~/work/project        # Auto-loads environment
smith model.onnx blueprint.yaml
pytest tests/

# Multi-project support
cd ~/work/project-vivado-2024-1   # Auto-loads Vivado 2024.1
cd ~/work/project-vivado-2024-2   # Auto-switches to Vivado 2024.2
```

**Benefits:**
- Automatic activation on cd
- Auto-regenerates when config changes
- Supports multiple projects with different configs
- Clean unloading when leaving directory

### Option 2: Manual Activation

Simple, explicit control with two-step activation:

```bash
# Activate once per session (two commands)
source .venv/bin/activate       # 1. Python packages (pytest, brainsmith CLI, etc.)
source .brainsmith/env.sh       # 2. Brainsmith config (paths, Xilinx tools, build settings)

# Work normally
smith model.onnx blueprint.yaml
pytest tests/

# Deactivate when done
source .brainsmith/deactivate.sh  # Clear Brainsmith environment
deactivate                         # Deactivate Python venv
```

**Benefits:**
- No additional tools required
- Explicit, predictable
- Works in any shell
- Clear separation: Python packages vs project configuration

### Common Commands

```bash
# Initialize a new project (creates config + env scripts)
brainsmith project init
brainsmith project init new-project  # Create in new directory

# Regenerate environment scripts after config changes
brainsmith project init  # Won't overwrite config.yaml

# View current configuration
brainsmith project show
```

## CLI Overview

Brainsmith provides two complementary CLI commands:

- **`brainsmith`** - Application configuration, setup, and environment management
- **`smith`** - Operational commands for dataflow core creation and kernel generation

For detailed command reference, see the [CLI API documentation](docs/cli_api_reference.md).

### Quick Examples

```bash
# Setup and configuration
brainsmith setup all              # Install all dependencies
brainsmith project init           # Initialize project configuration
brainsmith project allow-direnv   # Enable automatic environment

# Operations (after environment is loaded)
smith model.onnx blueprint.yaml   # Create dataflow core accelerator
smith kernel accelerator.sv       # Generate hardware kernel from RTL
pytest tests/                     # Run tests
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
