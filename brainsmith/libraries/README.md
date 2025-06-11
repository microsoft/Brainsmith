# Brainsmith Libraries

This directory contains the rich component libraries that form the extensible foundation of the Brainsmith FPGA accelerator toolchain.

## Overview

The libraries layer provides specialized implementations organized into logical families, each with clear extension points for stakeholder contributions.

## Library Structure

### 🔧 Kernels (`kernels/`)
Hardware kernel implementations, custom operations, and hardware sources.
- **Core**: Kernel functions, types, and performance modeling
- **Hardware**: HLS/RTL sources and custom FINN operations
- **Extensions**: `contrib/` for stakeholder kernel implementations

### 🔄 Transforms (`transforms/`)
Model transformation and compilation pipeline components.
- **Pipeline Steps**: Compilation workflow transformations
- **Operations**: Direct model manipulation functions
- **Extensions**: `contrib/` for custom transformation implementations

### 📊 Analysis (`analysis/`)
Performance analysis, profiling, and benchmarking tools.
- **Profiling**: Roofline analysis and performance measurement
- **Tools**: Hardware kernel generation and analysis utilities
- **Extensions**: `contrib/` for custom analysis tools

### 🤖 Automation (`automation/`)
Batch processing, parameter sweeps, and workflow orchestration.
- **Batch Processing**: Large-scale parallel execution
- **Parameter Sweeps**: Design space exploration automation
- **Extensions**: `contrib/` for workflow automation tools

## Usage Patterns

### Direct Library Access
```python
from brainsmith.libraries.kernels import functions
from brainsmith.libraries.transforms.steps import optimizations
from brainsmith.libraries.analysis.profiling import roofline_analysis
from brainsmith.libraries.automation import batch
```

### Integration with Core API
```python
from brainsmith.core.api import forge

# Libraries are automatically integrated during compilation
result = forge(
    model=model,
    blueprint="efficient_inference",
    kernels=["conv2d_hls", "matmul_rtl"],
    transforms=["cleanup", "optimizations"],
    analysis=["roofline"],
    automation={"batch_size": 10}
)
```

## Extension Points

Each library provides a `contrib/` directory for stakeholder extensions:
- **Clear interfaces**: Follow existing patterns and APIs
- **Automatic discovery**: Registry systems detect new components
- **Documentation**: Comprehensive README files in each contrib directory
- **Testing**: Include tests for contributed components

## Design Principles

1. **Modularity**: Each library is self-contained with clear interfaces
2. **Extensibility**: Rich contribution points for stakeholders
3. **Compatibility**: Maintains backward compatibility through import aliases
4. **Integration**: Seamless integration with core API and infrastructure
5. **Documentation**: Comprehensive documentation and examples

## Contributing

See individual library README files for specific contribution guidelines:
- [`kernels/README.md`](kernels/README.md) - Kernel implementation guide
- [`transforms/README.md`](transforms/README.md) - Transform development guide
- [`analysis/README.md`](analysis/README.md) - Analysis tool creation guide
- [`automation/README.md`](automation/README.md) - Automation tool development guide

Each `contrib/` directory contains detailed guidelines for adding new components.