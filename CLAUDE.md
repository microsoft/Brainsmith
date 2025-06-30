# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brainsmith is an open-source platform for FPGA AI accelerators developed collaboratively by Microsoft and AMD. It converts PyTorch models to RTL implementations for FPGA deployment using a Blueprint-based design space exploration approach.

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

### Execution Guidelines
- **Always** run python commands with `./smithy exec <command>` to execute in the docker environment
- **ALWAYS** run python commands with `./smithy exec "python <command>"` to ensure proper environment execution

### Testing
- `cd tests && pytest ./` - Run comprehensive test suite
- `python -m pytest tests/unit/ -v` - Run unit tests
- `python -m pytest tests/integration/ -v` - Run integration tests
- `./smithy exec "cd tests && pytest ./"` - Run tests in container

### BERT Demo (Primary Validation)
- `cd demos/bert && ./quicktest.sh` - Quick test without DCP generation
- Alternative quick test from `demos/bert/`:
  - `python gen_initial_folding.py --simd 12 --pe 8 --num_layers 1 -t 1 -o ./configs/l1_simd12_pe8.json` - Generate folding config
  - `python end2end_bert.py -o l1_simd12_pe8 -n 12 -l 1 -z 384 -i 1536 --run_fifo_sizing -p ./configs/l1_simd12_pe8.json` - Full compilation
  - `python end2end_bert.py -o l1_simd12_pe8 -n 12 -l 1 -z 384 -i 1536 -x True -p ./configs/l1_simd12_pe8.json -d False` - Skip DCP generation

### Hardware Kernel Generation
- `python -m brainsmith.libraries.analysis.tools.hw_kernel_gen.hkg <rtl_file> <compiler_data> -o <output_dir>` - Generate RTL wrapper templates
- `python -m brainsmith.libraries.analysis.tools.hw_kernel_gen.cli parse <rtl_file>` - Parse RTL and show interfaces
- `python -m brainsmith.libraries.analysis.tools.hw_kernel_gen.cli generate <rtl_file> <compiler_data> -o <output_dir>` - Full generation pipeline

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
Blueprint V2 Design → DSE → Hardware Implementation → FPGA Deployment
```

### V2 Architecture (Blueprint-based DSE)
The codebase follows a unified V2 architecture focused on design space exploration:

**Core Components** (`brainsmith/core/`)
- `Blueprint` - Configuration format for hardware designs
- `DSE` (Design Space Exploration) - Optimization engine for finding optimal hardware configurations
- `API` - High-level interface for model compilation and deployment

**Libraries System** (`brainsmith/libraries/`)
- `kernels/` - Hardware kernel implementations (RTL, HLS)
- `transforms/` - Model transformation utilities
- `analysis/` - Analysis tools including the hardware kernel generator
- `memory/` - Memory management utilities
- `common/` - Shared utilities and helpers

### Function-Focused Design
The project emphasizes single-purpose functions and modules:
```python
# Example from brainsmith/libraries/common/util.py
def load_json(filepath):
    """Load JSON file from filepath."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
```

### Key Architectural Components

**Blueprint V2 System**
- JSON-based configuration format for hardware designs
- Hierarchical structure supporting layers, operations, and parameters
- Integration with DSE for automatic optimization

**Hardware Kernel Generator (HKG)** (`brainsmith/libraries/analysis/tools/hw_kernel_gen/`)
- RTL Parser uses tree-sitter for SystemVerilog parsing
- Pragma system for annotations: `@brainsmith BDIM`, `@brainsmith DATATYPE`, etc.
- Template-based code generation for FINN integration
- Support for both HLS and RTL kernel types

**Template System**
- Jinja2 templates in `brainsmith/libraries/analysis/tools/hw_kernel_gen/templates/`
- Generates HWCustomOp, RTLBackend, Verilog wrappers, test suites
- Minimal instantiation templates that integrate with FINN

## Development Patterns

### Adding Hardware Kernels
1. Create RTL implementation in `brainsmith/libraries/kernels/rtl/` with proper pragmas
2. Or create HLS implementation in `brainsmith/libraries/kernels/hls/`
3. Use hardware kernel generator to create FINN integration code
4. Register kernel in the appropriate registry for discovery

### RTL Integration with Pragmas
```systemverilog
// @brainsmith BDIM in0 -1 [16]           // Block dimension chunking
// @brainsmith DATATYPE weights FIXED 8 8  // Datatype constraints
// @brainsmith WEIGHT weights_V            // Mark as weight interface
module my_accelerator(...);
```

### Testing Strategy
- Unit tests for components in `tests/unit/`
- Integration tests in `tests/integration/`
- Golden reference comparisons in `tests/libraries/analysis/tools/hw_kernel_gen/golden/`
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