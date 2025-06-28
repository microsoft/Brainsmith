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
```

### Running Tests

```bash
# Run all tests from project root
./smithy exec "cd tests && pytest ./"

# Run specific test directories
./smithy exec "pytest brainsmith/core/dataflow/tests/"
./smithy exec "pytest brainsmith/core/dataflow/core/tests/"

# Run hardware kernel generator E2E tests
./smithy exec "./brainsmith/tools/hw_kernel_gen/tests/run_e2e_test.sh"

# Run BERT demo tests (multi-hour builds)
./smithy exec "cd demos/bert && make single_layer"
./smithy exec "cd demos/bert && ./quicktest.sh"  # Faster test

# Run specific test file
./smithy exec "python brainsmith/tools/hw_kernel_gen/tests/test_e2e_generation.py"
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
   - `InterfaceDefinition`/`InterfaceModel`: Interface specifications
   - Stream dimension (SDIM) architecture for parallelism modeling

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

## Important Notes

- **No linting/formatting tools configured**: The project currently lacks Python linting or formatting configuration
- **All commands must run in Docker**: Use `./smithy exec` prefix for all Python commands
- **Dependencies auto-fetched**: FINN and other deps are fetched during container build via `docker/fetch-repos.sh`
- **FINN integration**: The project extends FINN's HWCustomOp framework for custom hardware kernels
- **Breaking changes preferred**: Per user preferences, prefer breaking refactors over compatibility layers

## Development Guidelines

- **License and File Notes**:
  - Don't add the license header to markdown files

## Memories

- **Always run python commands with smithy**