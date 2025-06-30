# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brainsmith is an open-source platform for FPGA AI accelerators developed collaboratively by Microsoft and AMD. It converts PyTorch models to RTL implementations for FPGA deployment using Interface-Wise Dataflow Modeling.

## Development Commands

All development happens inside Docker containers managed by the `smithy` script:

```bash
# Build Docker image (required after dependency updates)
./smithy build

# Start persistent daemon container
./smithy daemon

# Execute commands in container
./smithy exec "python script.py"

# Open interactive shell
./smithy shell

# Check container status
./smithy status

# Stop container
./smithy stop

# Clean up container
./smithy cleanup
```

### Running Tests

```bash
# Run all tests from project root
./smithy exec "cd tests && pytest ./"

# Run specific test directories
./smithy exec "pytest brainsmith/core/dataflow/tests/"

# Run hardware kernel generator E2E tests
./smithy exec "./brainsmith/tools/hw_kernel_gen/tests/run_e2e_test.sh"

# Run BERT demo tests (multi-hour builds)
./smithy exec "cd demos/bert && make single_layer"
./smithy exec "cd demos/bert && ./quicktest.sh"  # Faster test

# Run specific test file
./smithy exec "python brainsmith/tools/hw_kernel_gen/tests/test_e2e_generation.py"

# Run single test function
./smithy exec "pytest brainsmith/core/dataflow/tests/test_kernel_model.py::TestKernelModel::test_simple_kernel"

# Run tests with verbose output
./smithy exec "pytest -v brainsmith/core/dataflow/tests/"

# Run tests with output capture disabled (see print statements)
./smithy exec "pytest -s brainsmith/core/dataflow/tests/"
```

### Hardware Kernel Generator

Convert SystemVerilog RTL to FINN HWCustomOp:

```bash
# Basic usage
./smithy exec "python -m brainsmith.tools.hw_kernel_gen <rtl_file> -o <output_dir>"

# With module name (if multiple modules in file)
./smithy exec "python -m brainsmith.tools.hw_kernel_gen <rtl_file> -o <output_dir> -m <module_name>"

# Example
./smithy exec "python -m brainsmith.tools.hw_kernel_gen brainsmith/hw_kernels/thresholding/thresholding_axi.sv -o output/"
```

## High-Level Architecture

### Core Components

1. **Interface-Wise Dataflow Modeling** (`brainsmith/core/dataflow/`):
   - Data hierarchy: Tensor → Block → Stream → Element
   - `KernelDefinition`: Static kernel metadata and constraints
   - `KernelModel`: Runtime instance with actual dimensions
   - `InputDefinition`/`OutputDefinition`: Interface schemas (replaced InterfaceDefinition)
   - `InputInterface`/`OutputInterface`: Runtime models (replaced InterfaceModel)
   - Stream dimension (SDIM) architecture for parallelism modeling
   - Note: Components now directly in `dataflow/`, not `dataflow/core/`

2. **Hardware Kernels** (`brainsmith/hw_kernels/`):
   - SystemVerilog RTL implementations
   - FINN integration modules (`*/finn/` subdirectories)
   - Generated wrapper code for FINN HWCustomOp

3. **RTL Parser** (`brainsmith/tools/hw_kernel_gen/rtl_parser/`):
   - Parses SystemVerilog with pragma annotations
   - Extracts interface definitions and parameters
   - Generates FINN-compatible Python wrappers

4. **Template System** (`brainsmith/tools/hw_kernel_gen/templates/`):
   - Jinja2 templates for code generation
   - `rtl_backend.py.j2`: FINN RTL backend class
   - `rtl_wrapper_minimal.v.j2`: Verilog wrapper

### Pragma System

The RTL parser uses pragmas to annotate SystemVerilog code:

```systemverilog
// @brainsmith BDIM <interface> <param> [SHAPE=<shape>] [RINDEX=<idx>]
// @brainsmith SDIM <interface> <param>
// @brainsmith DATATYPE <interface> <type> <min_bits> <max_bits>
// @brainsmith DATATYPE_PARAM <interface> <property> <rtl_param>
// @brainsmith WEIGHT <interface>
// @brainsmith ALIAS <rtl_param> <python_name>
// @brainsmith DERIVED_PARAMETER <param> <python_expression>
// @brainsmith TOP_MODULE <module_name>
```

### Data Hierarchy

The system models data at four granularity levels:

1. **Tensor**: Full data for one inference (e.g., 512×256 matrix)
2. **Block**: Tile processed by kernel (e.g., 64×32 tile)
3. **Stream**: Data per clock cycle (e.g., 8×16 patch)
4. **Element**: Individual data item (e.g., INT8 value)

### SDIM (Streaming Dimensions) Architecture

SDIM replaces the legacy `iPar` parallelism model with per-dimension streaming control:

```python
# Configure streaming for each dimension
kernel.configure_sdim({
    "input": [8, 16, 32]  # Stream 8×16×32 per cycle
})

# Or uniform streaming
kernel.configure_sdim({
    "input": 16  # Stream 16 elements in each dimension
})
```

Key principles:
- Only inputs have configurable SDIM
- Outputs streaming rates are computed from kernel behavior
- SDIM values cannot exceed block dimensions
- Relationships propagate SDIM between interfaces

### Key Workflows

1. **Model Conversion**: PyTorch → ONNX → Dataflow Model → RTL
2. **Hardware Generation**: Uses FINN framework for synthesis
3. **Testing**: E2E tests validate RTL generation and functionality

## Current Development Focus

The project is actively developing on the `experimental/hwkg` branch with focus on:
- Native relationship modeling system (replacing pragma-based approach)
- Stream dimension (SDIM) architecture implementation
- Clean separation between definition (static) and model (runtime) layers
- Function-based tiling system for parallelism

## Code Header Convention

When creating substantial files in the project, use the following header:

```
############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
```

## Environment Setup

### Required Environment Variables

Set these before building the Docker container:

```bash
export BSMITH_ROOT="~/brainsmith"
export BSMITH_BUILD_DIR="~/builds/brainsmith"
export BSMITH_XILINX_PATH="/tools/Xilinx"
export BSMITH_XILINX_VERSION="2024.2"
export BSMITH_DOCKER_EXTRA=" -v /opt/Xilinx/licenses:/opt/Xilinx/licenses -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
```

### Updating Dependencies

Dependencies are managed in `docker/fetch-repos.sh`. To update FINN:

```bash
# Edit docker/fetch-repos.sh
# Change FINN_COMMIT variable to new commit hash or branch

# Rebuild container to fetch updated dependencies
./smithy cleanup
./smithy build
```

## Important Notes

- **No linting/formatting tools configured**: The project currently lacks Python linting or formatting configuration
- **All commands must run in Docker**: Use `./smithy exec` prefix for all Python commands
- **Dependencies auto-fetched**: FINN and other deps are fetched during container build via `docker/fetch-repos.sh`
- **FINN integration**: The project extends FINN's HWCustomOp framework for custom hardware kernels
- **Breaking changes preferred**: Per user preferences, prefer breaking refactors over compatibility layers
- **Python package installed in dev mode**: Changes to code are immediately reflected without rebuilding container

## Development Guidelines

- **License and File Notes**:
  - Don't add the license header to markdown files

## Common File Locations

- **Kernel Modeling Tests**: `brainsmith/core/dataflow/tests/`
- **RTL Parser Tests**: `brainsmith/tools/hw_kernel_gen/tests/`
- **Example Hardware Kernels**: `brainsmith/hw_kernels/`
- **FINN Integration**: Look for `*/finn/` subdirectories
- **Build Outputs**: Check `$BSMITH_BUILD_DIR` environment variable

## Debugging Tips

```bash
# View generated RTL wrapper
./smithy exec "cat output/rtl_wrapper.v"

# Check pragma parsing results
./smithy exec "python -m brainsmith.tools.hw_kernel_gen.rtl_parser.parser <rtl_file>"

# Run with debug logging
./smithy exec "PYTHONPATH=/workspace python -m brainsmith.tools.hw_kernel_gen --debug <rtl_file>"

# Interactive debugging
./smithy shell
# Then use ipdb or pdb in your code
```

## Memories

- **Always run python commands with smithy**
- **User preferences from .claude/CLAUDE.md include**:
  - **Break Fearlessly (PD-1)**: Assume zero external users, prefer breaking refactors
  - **Visual Clarity (PD-2)**: Use diagrams when helpful
  - **Concrete Tests (PD-3)**: Test against real implementations, avoid mocks
  - **Gate-Kept Commits (PD-4)**: Get explicit approval before git commits