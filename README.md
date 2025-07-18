## Brainsmith

Brainsmith is an open-source platform for FPGA AI accelerators.
This repository is in a pre-release state and under active co-development by Microsoft and AMD.

## Pre-Release 

This repository is in a ***pre-release state*** with many features under active development.

TAFK TODO: Put "works" vs "doesn't work" orientation guide for collaborators.

## Overview

Brainsmith uses Blueprints that define a Design Space to explore different configurations for a given neural network definition to be implemented on FPGAs. A Blueprint is a yaml file which can configure the following:
- Model optimization and network surgery by specifying combinations of graph transformations, such as expanding / fusing multiple ONNX-level operations
- Hardware configuration parameters, e.g., different FPGA targets
- FPGA compiler step variations, e.g., assembly of FINN compiler build steps
- Future support for multiple kernel implementations for the same layer, e.g. optimized RTL or HLS for specific network settings

The system currently uses the Legacy FINN backend for compilation, as FINN does not yet support the new entrypoint-based plugin system. The architecture is designed to support future backends with kernel-level customization.

## Core Pipeline

```
PyTorch Model → Brevitas Quantization → quantized ONNX (e.g., QONNX) → Blueprint YAML → Brainsmith DSE (uses FINN under the hood) → Hardware Implementation (selected by Brainsmith DSE) → FPGA
                                                        ↓
                                            Systematic exploration of:
                                            • Kernel implementations
                                            • Graph transformations  
                                            • Compiler build configurations
                                            • Hardware parameters
```

## Architecture

The DSE system consists of three phases:

### Phase 1: Design Space Constructor
- Parses Blueprint YAML specifications
- Validates model compatibility
- Generates valid configuration combinations

### Phase 2: Design Space Explorer  
- Iterates through configurations
- Manages build execution
- Collects and analyzes results

### Phase 3: Build Runner
- Executes individual builds
- Applies preprocessing and postprocessing transformsls
- Primary backend: Legacy FINN (current FINN toolchain)
- Extensible for future backends

For detailed documentation, see:
- [Phase 1 Architecture](docs/PHASE1_ARCHITECTURE.md)
- [Phase 2 Architecture](docs/PHASE2_ARCHITECTURE.md)  
- [Phase 3 Architecture](docs/PHASE3_ARCHITECTURE.md)
- [Plugin System Architecture](brainsmith/core/plugins/ARCHITECTURE.md)

## Quick Start

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
   - **FINN**: Current FINN compiler (fetched from `custom/brainsmith-patch` branch to `deps/finn/`)
   - **QONNX**: Quantized ONNX framework (fetched from `custom/brainsmith` branch to `deps/qonnx/`)
   - **Brevitas**: PyTorch quantization library (fetched to `deps/brevitas/`)
   - **Other dependencies**: Managed via `docker/fetch-repos.sh`
   
   To update dependencies to newer commits, edit `docker/fetch-repos.sh` and change the relevant commit variables:
```bash
# Edit docker/fetch-repos.sh
FINN_COMMIT="new-commit-hash-or-branch"

# Rebuild container to fetch updated dependencies
./smithy clean  # Removes container and build artifacts
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

# Stop when done (container persists for quick restart)
./smithy stop

# Clean up when needed
./smithy clean       # Remove container and build artifacts
./smithy clean --deep  # Full reset: removes images and optionally deps/
```

5. Validate with a 1 layer end-to-end build (generates DCP image, multi-hour build):
```bash
# TAFK TODO: Update with new demo instructions
```

6. Alternatively, run a simplified test skipping DCP gen:
```bash
# TAFK TODO: Update with new demo instructions
```

## Usage Examples

### Import Structure

Brainsmith provides a clean public API through the top-level module:

```python
# User-facing imports (recommended)
from brainsmith import forge, explore, create_build_runner_factory, BuildStatus

# Internal imports (for advanced use only)
from brainsmith.core.phase1 import ForgeAPI  # Direct access to forge internals
from brainsmith.core.phase2 import ExplorerEngine  # Direct access to explorer
```

### Blueprint-Based DSE (Recommended)

Create a Blueprint YAML to define your design space. Note that the configuration differs based on the backend:

#### For Legacy FINN Backend (Current Default)

```yaml
# bert_blueprint_legacy.yaml
blueprint:
  name: "BERT Layer Optimization"
  description: "Explore BERT configurations with legacy FINN"
  
  model: "bert_layer.onnx"
  
  # For legacy FINN, use stages to organize transforms
  stages:
    - stage: "streamline"
      transforms:
        - "ConvertDivToMul"
        - "BatchNormToAffine"
    - stage: "convert_to_hw"
      transforms:
        - "ConvertBipolarMatMulToXnorPopcount"
    - stage: "specialize_layers"
      transforms:
        - "SpecializeLayers"
      
  build_steps:
    - "step_tidy_up"
    - "step_streamline"
    - "step_convert_to_hw"
    - "step_specialize_layers"
```

#### For Future Backends (Experimental)

```yaml
# bert_blueprint_future.yaml
blueprint:
  name: "BERT Layer Optimization"
  description: "Future backend configuration"
  
  model: "bert_layer.onnx"
  
  # For future backends, specify kernels and transforms separately
  kernels:
    - kernel: "LayerNorm"
      backends: ["finn_hls", "finn_rtl"]
    - kernel: "MVAU" 
      backends: ["finn_hls"]
      
  transforms:
    pre_proc:
      - "ConvertDivToMul"
      - "BatchNormToAffine"
    cleanup:
      - "RemoveIdentityOps"
      
  build_steps:
    - "SpecializeLayers"
    - "MinimizeBitWidth"
```

Then explore the design space:

```python
from brainsmith import forge, explore, create_build_runner_factory

# Parse blueprint and create design space
design_space = forge("bert_layer.onnx", "bert_blueprint.yaml")

# Create build runner for the backend
build_runner = create_build_runner_factory()

# Explore all configurations
results = explore(design_space, build_runner)

# Get best configuration
best = results.get_best()
print(f"Best config: {best.config_id}")
print(f"Status: {best.status}")
if best.metrics:
    print(f"LUTs: {best.metrics.resources.lut}")
    print(f"Throughput: {best.metrics.performance.throughput_ips} inf/sec")

# Get Pareto optimal configurations
pareto = results.get_pareto_frontier()
print(f"Pareto optimal configs: {len(pareto)}")
```

### Legacy Direct Usage

For backward compatibility, you can still use the direct approach:

```bash
cd demos/bert
python gen_initial_folding.py --simd 12 --pe 8 --num_layers 1 -t 1 -o ./configs/l1_simd12_pe8.json
python end2end_bert.py -o l1_simd12_pe8 -n 12 -l 1 -z 384 -i 1536 -x True -p ./configs/l1_simd12_pe8.json -d False
```

### Running Tests

Run the comprehensive test suite:
TAFK TODO: Verify, plans

```bash
cd tests
pytest ./
```

This validates:
- Plugin system functionality
- Transform and kernel correctness
- End-to-end compilation flow
- DSE exploration capabilities

## Plugin System

Brainsmith uses a plugin architecture to manage transforms, kernels, backends, and build steps.

### Features

- Plugins register at decoration time via decorators
- Direct class access through collections
- Integration with QONNX and FINN transforms
- Pre-computed indexes for fast lookups

### Usage Patterns

```python
# Create plugin collections
from brainsmith.core.plugins import create_collections
collections = create_collections()
tfm = collections['transforms']
kn = collections['kernels']
bk = collections['backends']
steps = collections['steps']

# Direct attribute access
model = model.transform(tfm.ConvertDivToMul())
model = model.transform(tfm.qonnx.RemoveIdentityOps())  # Framework-qualified

# Dictionary-style access (for dynamic plugin names)
transform_class = tfm['BatchNormToAffine']
model = model.transform(transform_class())

# Framework-specific collections
finn_kernels = kn.finn.all()  # All FINN kernels
qonnx_transforms = tfm.qonnx.all()  # All QONNX transforms

# Query backends for specific kernels
backends = bk.find(kernel='LayerNorm', language='hls')

# Category-based step access
estimates = steps.estimates.GenerateEstimateReports()
```

### Creating Custom Plugins

```python
from brainsmith.core.plugins import transform, kernel, backend, step

@transform(stage="optimization")
class MyCustomTransform:
    def apply(self, model):
        # Transform implementation
        return model

@kernel(operation="MyOp", frameworks=["pytorch", "onnx"])
class MyCustomKernel:
    def infer(self, node):
        # Kernel implementation
        pass

@backend(kernel="MyOp", language="hls")
class MyOpHLSBackend:
    def generate(self, node):
        # Backend implementation
        pass
```

For more details, see [Plugin System Architecture](brainsmith/core/plugins/ARCHITECTURE.md).

## Project Structure

```
brainsmith/
├── brainsmith/                 # Main source code
│   ├── core/                   # Core DSE v3 implementation
│   │   ├── phase1/            # Design Space Constructor
│   │   ├── phase2/            # Design Space Explorer
│   │   ├── phase3/            # Build Runner
│   │   └── plugins/           # Plugin registry system
│   ├── transforms/            # Transform implementations
│   ├── kernels/              # Kernel implementations
│   ├── backends/             # Backend implementations
│   └── steps/                # Build step implementations
├── docs/                      # Architecture and design documentation
├── demos/                     # Example applications
├── tests/                     # Test suite
├── docker/                    # Container configuration
└── deps/                      # External dependencies (auto-fetched)
```

## Documentation

### Architecture Documents
- [Phase 1: Design Space Constructor](docs/PHASE1_ARCHITECTURE.md) - Blueprint parsing and validation
- [Phase 2: Design Space Explorer](docs/PHASE2_ARCHITECTURE.md) - Systematic exploration engine
- [Phase 3: Build Runner](docs/PHASE3_ARCHITECTURE.md) - Build execution and backends
- [Plugin System Architecture](brainsmith/core/plugins/ARCHITECTURE.md) - Plugin registry design

### Additional Resources
- Blueprint YAML examples in `demos/` directory
- Test examples in `tests/` directory
- Plugin implementations in `brainsmith/transforms/`, `brainsmith/kernels/`, etc.

## Contributing

We welcome contributions! Please follow these guidelines:
- Code style and standards
- Development workflow
- Testing requirements
- Pull request process

Also review our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Brainsmith is developed through a collaboration between Microsoft and AMD.

The project builds upon:
- [FINN](https://github.com/Xilinx/finn) - Dataflow compiler for quantized neural networks on FPGAs
- [QONNX](https://github.com/fastmachinelearning/qonnx) - Quantized ONNX model representation
- [Brevitas](https://github.com/Xilinx/brevitas) - PyTorch quantization library
