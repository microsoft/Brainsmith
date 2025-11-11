# Brainsmith

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Brainsmith transforms ONNX neural networks into optimized dataflow accelerators for FPGAs through automated design space exploration.

### Key Features

- **Design Space Exploration** - Automated search across hardware configurations
- **Blueprint System** - YAML-based design space definition with inheritance
- **Schema-Driven Kernels** - Declarative hardware semantics with automatic validation
- **Component Registry** - Plugin architecture for custom kernels and pipeline steps

**[Documentation](https://microsoft.github.io/brainsmith/)** • **[Discussions](https://github.com/microsoft/brainsmith/discussions)** • **[Roadmap](https://github.com/orgs/microsoft/projects/2017)**


## Quick Start

### Prerequisites

- **Ubuntu 22.04+** (primary development platform)
- **Python 3.11+** and [Poetry](https://python-poetry.org/docs/#installation)
- **Vivado 2024.2** (optional - required for RTL synthesis)
- **direnv** (optional - for auto environment activation)

### Installation

```bash
# Clone and setup
git clone https://github.com/microsoft/brainsmith.git
cd brainsmith

# Automated setup (creates venv, initializes project)
./setup-venv.sh

# Configure Xilinx tools path
vim brainsmith.yaml  # Edit xilinx_path and xilinx_version

# Activate environment
source .venv/bin/activate && source .brainsmith/env.sh
# Or with direnv: cd .

# Verify installation
brainsmith project info
```

[Detailed installation options and alternative Docker setup →](https://microsoft.github.io/brainsmith/getting-started/)

### Run Your First Build

```bash
# Navigate to BERT example
cd examples/bert

# Run quicktest (30-60 minutes)
./quicktest.sh
```

This generates a single-layer BERT accelerator with RTL simulation. [See example walkthrough →](https://microsoft.github.io/brainsmith/getting-started/#run-your-first-dse)

### Create Custom Accelerators

```bash
# Basic usage
smith model.onnx blueprint.yaml

# With custom output directory
smith model.onnx blueprint.yaml --output-dir ./my-design

# Run specific pipeline steps
smith model.onnx blueprint.yaml \
  --start-step streamline \
  --stop-step specialize_layers
```

**Blueprint Example:**
```yaml
name: "My Accelerator"
clock_ns: 5.0  # 200MHz

design_space:
  kernels:
    - MVAU
    - LayerNorm
    - Softmax

  steps:
    - "streamline"
    - "infer_kernels"
    - "specialize_layers"
    - "dataflow_partition"
```

[Blueprint schema reference →](https://microsoft.github.io/brainsmith/developer-guide/blueprint-schema/)


## CLI Overview

Two commands for different workflows:

- **smith** - Create accelerators (shown in Quick Start above)
- **brainsmith** - Manage projects and development environment

Common brainsmith commands:

```bash
brainsmith project init ~/my-project    # Initialize new project
brainsmith registry -v                  # List available components
brainsmith setup cppsim                 # Setup C++ simulation
brainsmith project info                 # Show configuration
```

[Complete CLI reference →](https://microsoft.github.io/brainsmith/api/cli/)

## Acknowledgments

Brainsmith is developed through collaboration between **Microsoft** and **AMD**.

Built on open-source foundations:
- [FINN](https://github.com/Xilinx/finn) - Dataflow compiler for quantized neural networks
- [QONNX](https://github.com/fastmachinelearning/qonnx) - Quantized ONNX representation
- [Brevitas](https://github.com/Xilinx/brevitas) - PyTorch quantization library
