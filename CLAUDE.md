# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brainsmith is an open-source platform for FPGA AI accelerators developed collaboratively by Microsoft and AMD. It provides a framework for converting PyTorch models to RTL implementations for FPGA deployment using Interface-Wise Dataflow Modeling.

## Development Environment

**Docker-based development is required.** Set these environment variables before starting:

```bash
export BSMITH_ROOT="~/brainsmith"
export BSMITH_BUILD_DIR="~/builds/brainsmith"
export BSMITH_XILINX_PATH="/tools/Xilinx"
export BSMITH_XILINX_VERSION="2024.2"
export BSMITH_DOCKER_EXTRA=" -v /opt/Xilinx/licenses:/opt/Xilinx/licenses -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
```

Launch development container with `./run-docker.sh`

## Key Commands

### Testing
- `cd tests && pytest ./` - Run comprehensive test suite
- `python -m pytest tests/dataflow/ -v` - Test dataflow components
- `python -m pytest tests/tools/hw_kernel_gen/ -v` - Test hardware kernel generation
- `./run-docker.sh e2e` - End-to-end validation
- `./run-docker.sh pytest` - Run pytest in container

### BERT Demo
- `cd demos/bert && make single_layer` - Quick single layer test
- `python gen_initial_folding.py --simd 12 --pe 8 --num_layers 1 -t 1 -o ./configs/l1_simd12_pe8.json` - Generate folding config
- `python end2end_bert.py -o l1_simd12_pe8 -n 12 -l 1 -z 384 -i 1536 --run_fifo_sizing -p ./configs/l1_simd12_pe8.json` - End-to-end BERT compilation

### Hardware Kernel Generation
- `python -m brainsmith.tools.hw_kernel_gen.hkg <rtl_file> <compiler_data> -o <output_dir>` - Generate RTL wrapper templates

## Architecture

### Core Pipeline
```
PyTorch Model → Brevitas Quantization → ONNX → FINN → RTL Synthesis
SystemVerilog RTL → RTL Parser → Interface Analysis → Template Generation → FINN Integration
```

### Key Modules
- **`brainsmith/dataflow/`** - Interface-wise dataflow modeling framework with automatic hardware custom operation generation
- **`brainsmith/tools/hw_kernel_gen/`** - Hardware Kernel Generator (HKG) for automating RTL integration into FINN compiler
- **`brainsmith/custom_op/fpgadataflow/`** - FPGA dataflow operations and HLS kernel implementations
- **`brainsmith/hw_kernels/`** - Hardware kernel implementations (HLS headers and RTL modules)
- **`brainsmith/transformation/`** - Model transformation utilities for converting to hardware layers

### Template System
Uses Jinja2 templates in `brainsmith/tools/hw_kernel_gen/templates/` for generating:
- HWCustomOp Python classes
- RTLBackend implementations  
- Verilog wrapper modules
- Test suites
- Documentation

### RTL Parser
Located in `brainsmith/tools/hw_kernel_gen/rtl_parser/` - SystemVerilog analysis engine that extracts:
- Interface definitions
- Port specifications
- Protocol validation
- Width parsing for tensor dimensions

## Development Patterns

### When working with dataflow models:
- Check `brainsmith/dataflow/core/` for interface abstractions
- Use `DataflowInterface` and `DataflowModel` classes for hardware modeling
- Tensor chunking strategies are in `tensor_chunking.py`

### When adding custom operations:
- Extend `brainsmith/custom_op/fpgadataflow/` for FPGA-specific ops
- HLS implementations go in `brainsmith/hw_kernels/hls/`
- RTL modules go in `brainsmith/hw_kernels/rtl/`

### When working with templates:
- Templates are in `brainsmith/tools/hw_kernel_gen/templates/`
- Template context is managed by `template_context.py`
- Use existing patterns from golden reference files in `tests/tools/hw_kernel_gen/golden/`

### Testing approach:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Golden reference comparisons for generated code
- BERT demo serves as comprehensive validation

## Key Dependencies
- PyTorch 2.7.0 with CUDA 12.1
- FINN (AMD's quantized neural network framework)
- Brevitas (quantization framework)
- ONNX ecosystem (onnx, onnxruntime, onnxsim)
- tree-sitter for SystemVerilog parsing
- Jinja2 for template generation