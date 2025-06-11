# Kernel Libraries

This directory contains the core kernel implementations and hardware sources for the Brainsmith FPGA accelerator toolchain.

## Structure

### Core Components
- **`functions.py`** - Core kernel function implementations
- **`types.py`** - Kernel type definitions and interfaces
- **`performance.py`** - Performance modeling and analysis

### Hardware Sources
- **`custom_ops/`** - FINN custom operation implementations
  - `fpgadataflow/` - FPGA dataflow operations (crop, softmax, layernorm, shuffle)
  - `general/` - General-purpose operations (norms)
- **`hw_sources/`** - Hardware source files
  - `hls/` - High-Level Synthesis headers and implementations
  - `rtl/` - RTL source files and documentation

### Kernel Definitions
- **`conv2d_hls/`** - Convolution 2D HLS kernel implementation
- **`matmul_rtl/`** - Matrix multiplication RTL kernel implementation

### Extension Points
- **`contrib/`** - Stakeholder-contributed kernel implementations

## Usage

### Using Existing Kernels
```python
from brainsmith.libraries.kernels import functions
from brainsmith.libraries.kernels import types

# Access kernel functions
kernel_func = functions.get_kernel("conv2d_hls")

# Work with kernel types
kernel_config = types.KernelConfiguration(...)
```

### Adding New Kernels
1. Create kernel YAML definition following existing patterns
2. Implement custom operations in `custom_ops/`
3. Add hardware sources in `hw_sources/`
4. Update kernel registries for automatic discovery

## Integration
- Kernels are automatically discovered and registered
- Compatible with FINN compilation flow
- Support for both HLS and RTL implementations
- Performance modeling integration available