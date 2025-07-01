## Brainsmith - Kernel Integrator Development Branch

**⚠️ DEVELOPMENT BRANCH**: This is the `experimental/hwkg` branch dedicated exclusively to **Kernel Integrator** (Hardware Kernel Generator) development. Many core Brainsmith features have been removed for streamlined development.

### About This Branch

This branch focuses on developing the **Hardware Kernel Generator (HKG)** component of Brainsmith, which converts SystemVerilog RTL modules into FINN-compatible HWCustomOp implementations.

**Available Features:**
- RTL Parser with pragma system for SystemVerilog analysis
- Kernel Modeling system with BDIM/SDIM architecture for parallelism
- FINN integration via AutoHWCustomOp and AutoRTLBackend base classes
- Template-based code generation for HWCustomOp and RTLBackend classes
- Complete SystemVerilog → FINN HWCustomOp conversion pipeline

**Removed for Streamlined Development:**
- End-to-end BERT compilation demos
- QuantSoftMax and Shuffle hardware operations
- Full AI model compilation pipeline
- Production deployment and optimization tools

### Relationship to Main Brainsmith

This development branch will be merged back into the main Brainsmith platform once the Kernel Integrator reaches maturity. For the full Brainsmith platform with complete AI model compilation capabilities, use the main branch.

### Quick start

1. Set environment variables (separate from FINN variables), example below:
```bash
export BSMITH_ROOT="~/brainsmith"
export BSMITH_BUILD_DIR="~/builds/brainsmith"
export BSMITH_XILINX_PATH="/tools/Xilinx"
export BSMITH_XILINX_VERSION="2024.2"
export BSMITH_DOCKER_EXTRA=" -v /opt/Xilinx/licenses:/opt/Xilinx/licenses -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
```

2. Clone this repository:
```bash
git clone git@github.com:microsoft/Brainsmith.git
```

3. **Dependencies**: Dependencies are automatically fetched during Docker container initialization:
   - **FINN**: Fetched from `custom/transformer` branch to `deps/finn/`
   - **Other dependencies**: Managed via `docker/fetch-repos.sh`
   
   To update FINN to a newer commit, edit `docker/fetch-repos.sh` and change the `FINN_COMMIT` variable:
```bash
# Edit docker/fetch-repos.sh
FINN_COMMIT="new-commit-hash-or-branch"

# Rebuild container to fetch updated dependencies
./smithy cleanup
./smithy build
```

4. Launch the docker container. Since the Python repo is installed in developer mode in the docker container, you can edit the files, push to git, etc. and run the changes in docker without rebuilding the container.

```bash
# Start persistent container (one-time setup)
./smithy daemon

# Get instant shell access anytime
./smithy shell

# Or execute commands quickly
./smithy exec "python script.py"

# Check status
./smithy status

# Stop when done
./smithy stop
```

> **Note for existing users**: If you previously used `./run-docker.sh`, it now automatically redirects to `smithy` for compatibility. The new `smithy` tool provides 73% faster container operations with persistent containers. See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for details.

5. **Hardware Kernel Generator Usage**:

Convert SystemVerilog RTL to FINN HWCustomOp:
```bash
# Basic conversion
./smithy exec "python -m brainsmith.tools.hw_kernel_gen <rtl_file> -o <output_dir>"

# Example with thresholding kernel
./smithy exec "python -m brainsmith.tools.hw_kernel_gen brainsmith/hw_kernels/thresholding/thresholding_axi.sv -o output/"
```

6. **Run Hardware Kernel Generator Tests**:
```bash
# Run HKG-specific tests
./smithy exec "pytest brainsmith/tools/hw_kernel_gen/tests/"

# Run end-to-end generation test
./smithy exec "./brainsmith/tools/hw_kernel_gen/tests/run_e2e_test.sh"

# Test Kernel Modeling system
./smithy exec "pytest brainsmith/core/dataflow/tests/"
```

7. **Explore Examples**:
```bash
# View example implementations
./smithy exec "python examples/auto_hw_custom_op/thresholding_km.py"

# Check RTL parser capabilities  
./smithy exec "python -m brainsmith.tools.hw_kernel_gen.rtl_parser.parser --help"
```
