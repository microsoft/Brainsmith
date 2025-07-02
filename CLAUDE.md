# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**⚠️ DEVELOPMENT BRANCH**: This is the `experimental/hwkg` branch dedicated exclusively to **Kernel Integrator** (Hardware Kernel Generator) development. Many core Brainsmith features have been removed for streamlined development.

This branch focuses on developing the **Hardware Kernel Generator (HKG)** component of Brainsmith, which converts SystemVerilog RTL modules into FINN-compatible HWCustomOp implementations. The full Brainsmith platform (with complete AI model compilation) is available on the main branch.

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

# Run AutoHWCustomOp tests (Kernel Modeling integration)
./smithy exec "pytest brainsmith/tools/hw_kernel_gen/tests/test_auto_hw_custom_op_v2.py"

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

2. **FINN Integration** (`brainsmith/core/finn/`):
   - `AutoHWCustomOp`: Base class for FINN HWCustomOp implementations using Kernel Modeling
   - `AutoRTLBackend`: Base class for FINN RTLBackend implementations with template support
   - Bridges Kernel Modeling system with FINN's execution framework

3. **Hardware Kernels** (`brainsmith/hw_kernels/`):
   - SystemVerilog RTL implementations
   - FINN integration modules (`*/finn/` subdirectories)
   - Generated wrapper code for FINN HWCustomOp

4. **RTL Parser** (`brainsmith/tools/hw_kernel_gen/rtl_parser/`):
   - Parses SystemVerilog with pragma annotations
   - Extracts interface definitions and parameters
   - Generates FINN-compatible Python wrappers

5. **Template System** (`brainsmith/tools/hw_kernel_gen/templates/`):
   - Jinja2 templates for code generation
   - `hw_custom_op.py.j2`: AutoHWCustomOp subclass generation
   - `rtl_backend.py.j2`: AutoRTLBackend subclass generation

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
// @brainsmith RELATIONSHIP <source> <target> <type> [args...]
// @brainsmith AXILITE_PARAM <parameter> <interface>
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

1. **Hardware Kernel Integration**: SystemVerilog RTL → Pragma Analysis → FINN HWCustomOp
   - Parse SystemVerilog with `@brainsmith` pragmas
   - Generate KernelDefinition from interface analysis
   - Create AutoHWCustomOp and AutoRTLBackend subclasses
   - Integrate with FINN compilation pipeline

2. **Kernel Modeling Development**: Definition → Model → Configuration
   - Define static interface schemas with KernelDefinition
   - Create runtime instances with KernelModel
   - Configure streaming parallelism with SDIM
   - Validate constraints and relationships

3. **Testing**: E2E validation of RTL generation and FINN integration

## Current Development Focus

This `experimental/hwkg` branch is a streamlined development environment focused on:
- **Hardware Kernel Generator**: Complete SystemVerilog → FINN HWCustomOp pipeline
- **Kernel Modeling System**: BDIM/SDIM architecture with clean definition/runtime separation
- **FINN Integration**: AutoHWCustomOp and AutoRTLBackend base classes (moved to `brainsmith.core.finn`)
- **Template-Based Generation**: Automated HWCustomOp and RTLBackend class generation
- **Pragma System**: Enhanced RTL parsing with RELATIONSHIP and AXILITE_PARAM support

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

### Container Management Notes

- **Persistent containers**: `smithy daemon` creates long-running containers for faster operations
- **Container naming**: Containers are named based on directory hash for persistence across sessions
- **Build requirements**: Container builds require ~15GB disk space
- **Initialization monitoring**: Container startup includes automatic dependency fetching and can take several minutes

## Development Guidelines

- **License and File Notes**:
  - Don't add the license header to markdown files

## Common File Locations

- **Kernel Modeling Tests**: `brainsmith/core/dataflow/tests/`
- **RTL Parser Tests**: `brainsmith/tools/hw_kernel_gen/tests/`
- **FINN Integration Classes**: `brainsmith/core/finn/` (AutoHWCustomOp, AutoRTLBackend)
- **Example Hardware Kernels**: `brainsmith/hw_kernels/`
- **Generated Code Examples**: `examples/auto_hw_custom_op/`
- **Template Files**: `brainsmith/tools/hw_kernel_gen/templates/`
- **Build Outputs**: Check `$BSMITH_BUILD_DIR` environment variable

## Key Import Paths

```python
# FINN integration base classes
from brainsmith.core.finn import AutoHWCustomOp, AutoRTLBackend

# Kernel Modeling system
from brainsmith.core.dataflow import (
    KernelDefinition, KernelModel,
    InputDefinition, OutputDefinition,
    InputInterface, OutputInterface
)

# Hardware Kernel Generator
from brainsmith.tools.hw_kernel_gen import RTLParser
```

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

# Container troubleshooting
./smithy logs              # View container logs
./smithy status           # Check container state
BSMITH_SHOW_INIT_LOGS=true ./smithy daemon  # Debug container startup
```

## Memories

- **Always run python commands with smithy**
- **User preferences from .claude/CLAUDE.md include**:
  - **Break Fearlessly (PD-1)**: Assume zero external users, prefer breaking refactors
  - **Visual Clarity (PD-2)**: Use diagrams when helpful
  - **Concrete Tests (PD-3)**: Test against real implementations, avoid mocks
  - **Gate-Kept Commits (PD-4)**: Get explicit approval before git commits