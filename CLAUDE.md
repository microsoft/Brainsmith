# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brainsmith is an open-source platform for FPGA AI accelerators developed collaboratively by Microsoft and AMD. It converts PyTorch models to RTL implementations for FPGA deployment using Interface-Wise Dataflow Modeling.

## Development Environment

**Docker-based development is required.** 

### Environment Setup
```bash
export BSMITH_ROOT="~/brainsmith"
export BSMITH_BUILD_DIR="~/builds/brainsmith"
export BSMITH_XILINX_PATH="/tools/Xilinx"
export BSMITH_XILINX_VERSION="2024.2"
export BSMITH_DOCKER_EXTRA=" -v /opt/Xilinx/licenses:/opt/Xilinx/licenses -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
```

### Container Management
```bash
# Start persistent container (one-time setup)
./smithy daemon

# Get instant shell access anytime  
./smithy shell

# Execute commands quickly
./smithy exec "python script.py"

# Other useful commands
./smithy build        # Build/rebuild Docker image
./smithy status       # Check container status
./smithy logs [-f]    # View container logs
./smithy restart      # Restart container
./smithy stop         # Stop container
./smithy cleanup      # Remove container completely
```

## Key Commands

### Testing
- `cd tests && pytest ./` - Run comprehensive test suite
- `python -m pytest tests/dataflow/ -v` - Test dataflow components
- `python -m pytest tests/tools/hw_kernel_gen/ -v` - Test hardware kernel generation
- `python -m pytest tests/dataflow/unit/test_dataflow_interface.py::TestDataflowInterface::test_creation` - Run single test
- `./smithy exec "cd tests && pytest ./"` - Run tests in container

### BERT Demo (Primary Validation)
- `cd tests/end2end/bert && make single_layer` - Full validation test (generates DCP, multi-hour build)
- `cd demos/bert && ./quicktest.sh` - Quick test without DCP generation
- Alternative quick test from `demos/bert/`:
  - `python gen_initial_folding.py --simd 12 --pe 8 --num_layers 1 -t 1 -o ./configs/l1_simd12_pe8.json` - Generate folding config
  - `python end2end_bert.py -o l1_simd12_pe8 -n 12 -l 1 -z 384 -i 1536 --run_fifo_sizing -p ./configs/l1_simd12_pe8.json` - Full compilation
  - `python end2end_bert.py -o l1_simd12_pe8 -n 12 -l 1 -z 384 -i 1536 -x True -p ./configs/l1_simd12_pe8.json -d False` - Skip DCP generation

### Hardware Kernel Generation
- `python -m brainsmith.tools.hw_kernel_gen.hkg <rtl_file> <compiler_data> -o <output_dir>` - Generate RTL wrapper templates
- `python -m brainsmith.tools.hw_kernel_gen.cli parse <rtl_file>` - Parse RTL and show interfaces
- `python -m brainsmith.tools.hw_kernel_gen.cli generate <rtl_file> <compiler_data> -o <output_dir>` - Full generation pipeline

### Linting and Code Style
The project uses `.editorconfig` for consistent formatting:
- 4-space indentation (2 for YAML)
- Max line length: 80 characters
- UTF-8 encoding, LF line endings
- Trim trailing whitespace, final newline required

## Architecture

### Core Pipeline
```
PyTorch Model → Brevitas Quantization → ONNX → FINN → RTL Synthesis
SystemVerilog RTL → RTL Parser → Interface Analysis → Template Generation → FINN Integration
```

### Unified Interface Type System
The codebase uses a unified interface type system defined in `brainsmith/dataflow/core/interface_types.py`:
- **INPUT/OUTPUT/WEIGHT** - AXI-Stream dataflow interfaces
- **CONFIG** - AXI-Lite configuration interface
- **CONTROL** - Global control signals (clk, rst)

Interface roles are inherently tied to protocols (e.g., INPUT is always AXI-Stream).

### Key Architectural Components

**Interface-Wise Dataflow Modeling** (`brainsmith/dataflow/core/`)
- `DataflowInterface` - Represents hardware interfaces with tensor dimensions and chunking
- `DataflowModel` - Unified computational model for hardware generation
- `AutoHWCustomOp`/`AutoRTLBackend` - Automatic generation from dataflow models
- Tensor chunking follows: tensor_dims → block_dims → stream_dims → element

**Hardware Kernel Generator (HKG)** (`brainsmith/tools/hw_kernel_gen/`)
- RTL Parser uses tree-sitter for SystemVerilog parsing
- Pragma system for annotations: `@brainsmith BDIM`, `@brainsmith DATATYPE`, etc.
- Template-based code generation for FINN integration
- Direct RTL → Template pipeline bypassing DataflowModel for performance

**Template System**
- Jinja2 templates in `brainsmith/tools/hw_kernel_gen/templates/`
- Generates HWCustomOp, RTLBackend, Verilog wrappers, test suites
- Template context built directly from EnhancedRTLParsingResult
- Minimal instantiation templates that use dataflow models

## Development Patterns

### Adding Custom Operations
1. Create HLS implementation in `brainsmith/hw_kernels/hls/`
2. Add RTL module in `brainsmith/hw_kernels/rtl/` with proper pragmas
3. Extend `brainsmith/custom_op/fpgadataflow/` for FPGA-specific operations
4. Use AutoHWCustomOp/AutoRTLBackend for automatic generation

### RTL Integration with Pragmas
```systemverilog
// @brainsmith BDIM in0 -1 [16]           // Block dimension chunking
// @brainsmith DATATYPE weights FIXED 8 8  // Datatype constraints
// @brainsmith WEIGHT weights_V            // Mark as weight interface
module my_accelerator(...);
```

### Testing Strategy
- Unit tests for components in `tests/dataflow/unit/`
- Integration tests in `tests/integration/`
- Golden reference comparisons in `tests/tools/hw_kernel_gen/golden/`
- BERT demo as comprehensive validation
- Parameter sweep testing: `demos/bert/tests/param_sweep.sh`

### AI Cache Directory
Use `ai_cache/` for development artifacts (plans, analysis, checklists) that should not be part of the codebase. This keeps development documentation separate from production code.

## Dependency Management

Dependencies are fetched automatically during container initialization:
- FINN from `custom/transformer` branch → `deps/finn/`
- Other dependencies via `docker/fetch-repos.sh`
- To update FINN: edit `FINN_COMMIT` in `docker/fetch-repos.sh` and rebuild container

## Key Dependencies
- PyTorch 2.7.0 with CUDA 12.1
- FINN (AMD's quantized neural network framework)
- Brevitas (quantization framework)
- ONNX ecosystem (onnx, onnxruntime, onnxsim)
- tree-sitter 0.24.0 for SystemVerilog parsing
- Jinja2 for template generation