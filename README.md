## Brainsmith

Brainsmith automates the creation of dataflow core accelerators and implementation of neural networks on FPGA, from PyTorch to RTL.

## Pre-Release

## Quick Start

### Dependencies
1. Ubuntu 22.04+
2. Python 3.11+ and [Poetry](https://python-poetry.org/docs/#installation)
3. [Optional] Vivado Design Suite 2024.2 for synthesis and simluation
4. [Optional] [direnv](https://direnv.net/) for automatic environment activation

### Installation Steps

```bash
# Run automated setup (creates .venv, initializes project, configures direnv if installed)
./setup-venv.sh

# Edit configuration for your Xilinx installation
vim brainsmith.yaml  # Set xilinx_path, xilinx_version

# Activate environment
# Option 1: For direnv users
cd .
# Option 2: Manual (must be run from brainsmith root)
source .venv/bin/activate && source .brainsmith/env.sh

# Verify setup
brainsmith project info
```

See Getting Started documentation for detailed install instruction.

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
