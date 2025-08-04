# Brainsmith - Kernel Integrator Branch

**⚠️ DEVELOPMENT BRANCH**: This is the `experimental/hwkg` branch focused on **Kernel Integrator** development. For the full Brainsmith platform, use the main branch.

## Overview

The Kernel Integrator (KI) converts SystemVerilog RTL modules into FINN-compatible Python operators. It enables hardware engineers to integrate custom RTL implementations into ML compiler flows.

**Key Features:**
- 🔍 **RTL Parser** - Tree-sitter based SystemVerilog analysis with pragma support
- 📐 **Kernel Modeling** - Interface-wise dataflow modeling with BDIM/SDIM architecture
- 🔌 **FINN Integration** - AutoHWCustomOp and AutoRTLBackend base classes
- 🏗️ **Template Generation** - Automated Python class generation from RTL

## Documentation

- **[Kernel Integrator Architecture](brainsmith/tools/kernel_integrator/ARCHITECTURE.md)** - System design and components
- **[RTL Parser Design](brainsmith/tools/kernel_integrator/rtl_parser/README.md)** - Comprehensive parser documentation
- **[Pragma Guide](brainsmith/tools/kernel_integrator/rtl_parser/PRAGMA_GUIDE.md)** - RTL annotation reference
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines and project structure

## Quick Start

### 1. Environment Setup

```bash
export BSMITH_ROOT="~/brainsmith"
export BSMITH_BUILD_DIR="~/builds/brainsmith"
export BSMITH_XILINX_PATH="/tools/Xilinx"
export BSMITH_XILINX_VERSION="2024.2"
export BSMITH_DOCKER_EXTRA=" -v /opt/Xilinx/licenses:/opt/Xilinx/licenses -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
```

### 2. Clone and Build

```bash
git clone -b experimental/hwkg git@github.com:microsoft/Brainsmith.git
cd Brainsmith
./smithy build  # Builds Docker container with dependencies
```

### 3. Start Development Environment

```bash
# Start persistent container
./smithy daemon

# Open shell for interactive work
./smithy shell

# Or run commands directly
./smithy exec "python -m brainsmith.tools.kernel_integrator --help"
```

## Usage Examples

### Basic RTL Conversion

```bash
# Convert SystemVerilog to FINN HWCustomOp
./smithy exec "python -m brainsmith.tools.kernel_integrator mykernel.sv -o output/"

# With specific module selection
./smithy exec "python -m brainsmith.tools.kernel_integrator complex.sv -m target_module -o output/"
```

### Running Tests

```bash
# Kernel Integrator tests
./smithy exec "pytest brainsmith/tools/kernel_integrator/tests/"

# RTL Parser tests
./smithy exec "pytest brainsmith/tools/kernel_integrator/rtl_parser/tests/"

# Kernel Modeling tests
./smithy exec "pytest brainsmith/core/dataflow/tests/"

# End-to-end test
./smithy exec "./brainsmith/tools/kernel_integrator/tests/run_e2e_test.sh"
```

### Working with Examples

```bash
# Thresholding example
./smithy exec "python -m brainsmith.tools.kernel_integrator brainsmith/kernels/thresholding/thresholding_axi.sv -o output/"

# View generated files
./smithy exec "ls -la output/"
./smithy exec "cat output/thresholding_axi_hw_custom_op.py"
```

## Key Concepts

### RTL Pragmas

Annotate SystemVerilog with semantic information:

```systemverilog
// @brainsmith DATATYPE input0 UINT 8 32
// @brainsmith BDIM input0 [TILE_H, TILE_W]
// @brainsmith ALIAS PE parallelism_factor
// @brainsmith DERIVED_PARAMETER MEM_DEPTH self.calc_memory_depth()
```

### Automatic Parameter Linking

The parser automatically links parameters based on naming conventions:
- `{interface}_WIDTH` → Interface bit width
- `{interface}_SIGNED` → Signedness flag
- `{interface}_BDIM` → Block dimensions
- `{interface}_SDIM` → Stream dimensions

### Generated Outputs

Each RTL module generates:
- `*_hw_custom_op.py` - FINN operator class
- `*_rtl.py` - RTL compilation backend
- `*_wrapper.v` - SystemVerilog wrapper
- `generation_metadata.json` - Build information

## Development Workflow

1. **Design RTL** - Create SystemVerilog module with standard interfaces
2. **Add Pragmas** - Annotate with `@brainsmith` directives
3. **Generate** - Run KI to create FINN integration files
4. **Test** - Validate in FINN compilation flow
5. **Iterate** - Refine pragmas and parameters as needed

## Container Management

The `smithy` tool provides efficient Docker container management:

```bash
./smithy daemon   # Start persistent container
./smithy status   # Check container status
./smithy shell    # Interactive shell
./smithy exec     # Run commands
./smithy logs     # View container logs
./smithy stop     # Stop container
./smithy cleanup  # Remove container
```

## Updating Dependencies

Dependencies are fetched during container build. To update FINN:

```bash
# Edit docker/fetch-repos.sh
# Change FINN_COMMIT variable

# Rebuild container
./smithy cleanup
./smithy build
```

## License

Copyright (c) Microsoft Corporation. Licensed under the MIT License.