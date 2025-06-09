# Interface-Wise Dataflow Modeling Framework

A unified abstraction layer for hardware kernel design that simplifies the complexity of integrating custom RTL implementations into the FINN/Brainsmith ecosystem through standardized interface-based modeling and automated code generation.

## Overview

The Interface-Wise Dataflow Modeling Framework provides a comprehensive solution for modeling, validating, and optimizing hardware kernels through standardized interface abstractions. It bridges the gap between RTL implementations and high-level neural network frameworks by providing:

- **Unified Interface Abstraction**: Standardized representation of hardware interfaces with mathematical relationships
- **Automatic Code Generation**: Base classes that eliminate boilerplate code for FINN integration
- **Constraint-Based Validation**: Flexible datatype and dimension constraint system
- **Performance Optimization**: Integrated parallelism optimization for FINN workflows
- **RTL Integration**: Seamless conversion from RTL Parser outputs to dataflow models

## Architecture Overview

### Core Components

```
brainsmith/dataflow/
├── core/                          # Core framework components
│   ├── dataflow_interface.py      # Interface abstraction layer
│   ├── dataflow_model.py          # Computational model
│   ├── auto_hw_custom_op.py       # Auto-generated HWCustomOp base
│   ├── auto_rtl_backend.py        # Auto-generated RTLBackend base
│   ├── validation.py              # Validation framework
│   ├── tensor_chunking.py         # Tensor chunking utilities
│   └── class_naming.py            # Naming conventions
├── integration/                   # External integrations
│   └── rtl_conversion.py          # RTL Parser integration
└── examples/                      # Usage examples
    └── basic_usage.py             # Basic framework usage
```

### Design Principles

1. **Single Source of Truth**: [`DataflowModel`](core/dataflow_model.py:39) serves as the central repository for all interface metadata and computational relationships
2. **Mathematical Precision**: Three-tier dimension system (qDim/tDim/sDim) with clear tensor chunking relationships
3. **Flexible Constraints**: [`DataTypeConstraint`](core/dataflow_interface.py:71) system allows RTL creators to specify flexible datatype requirements
4. **Auto-Generation Friendly**: Base classes eliminate 80%+ of generated code through standardized implementations
5. **Validation-First**: Comprehensive validation at every level ensures correctness

## Key Concepts

### Interface Dimensions

The framework uses a three-tier dimension system that clarifies the distinction between original tensor shape and computed values:

- **qDim** (Query Dimensions): Original tensor shape (e.g., 768 for BERT hidden size)
- **tDim** (Tensor Processing Dimensions): Chunk size for processing (e.g., 96 elements per chunk)
- **sDim** (Stream Dimensions): Hardware parallelism (e.g., 8 elements per clock cycle)
- **num_tensors**: Computed as qDim ÷ tDim via `get_num_tensors()` method (e.g., 768 ÷ 96 = 8 chunks)

**Mathematical Relationships**:
```
qDim = original_tensor_shape[i]     # Original tensor dimension  
num_tensors[i] = qDim[i] ÷ tDim[i]   # Number of chunks to process
tDim[i] % sDim[i] = 0               # Valid streaming constraint
```

### Interface Types

- **INPUT**: AXI-Stream input for activation data
- **OUTPUT**: AXI-Stream output for result data  
- **WEIGHT**: AXI-Stream input for weight/parameter data
- **CONFIG**: AXI-Lite for runtime configuration
- **CONTROL**: Global control signals (clk, rst, etc.)

### Datatype System

Supports FINN-compatible datatypes with flexible constraints:

```python
# Create flexible datatype constraint
constraint = DataTypeConstraint(
    base_types=["INT", "UINT", "FIXED"],
    min_bitwidth=4,
    max_bitwidth=16,
    signed_allowed=True,
    unsigned_allowed=True
)

# Create specific datatype
dtype = DataflowDataType(
    base_type="INT",
    bitwidth=8,
    signed=True,
    finn_type="INT8"
)
```

## Installation

The framework is part of the Brainsmith ecosystem and requires:

```bash
# Core dependencies
pip install numpy

# Optional FINN integration
# Install FINN framework for full functionality
```

## Quick Start Guide

### Basic Interface Creation

```python
from brainsmith.dataflow import (
    DataflowInterface, DataflowInterfaceType, 
    DataflowDataType, DataflowModel
)

# Create a simple input interface
input_interface = DataflowInterface(
    name="input0",
    interface_type=DataflowInterfaceType.INPUT,
    qDim=[64],    # 64 elements total
    tDim=[16],    # 16 elements per calculation
    sDim=[4],     # 4 elements per clock cycle
    dtype=DataflowDataType("UINT", 8, False, "UINT8")
)

# Validate the interface
result = input_interface.validate_constraints()
if result.success:
    print("✓ Interface is valid")
```

### Unified Computational Model

```python
# Create dataflow model with multiple interfaces
model = DataflowModel([input_interface, weight_interface, output_interface], {})

# Calculate performance metrics with parallelism
iPar = {"input0": 4}
wPar = {"weights": 8}

intervals = model.calculate_initiation_intervals(iPar, wPar)
print(f"Latency: {intervals.L} cycles")
print(f"Bottleneck: {intervals.bottleneck_analysis['bottleneck_input']}")
```

### Tensor Chunking from ONNX

```python
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface

# Create interface from tensor chunking
interface = DataflowInterface.from_tensor_chunking(
    name="conv_input",
    interface_type=DataflowInterfaceType.INPUT,
    original_shape=[1, 64, 32, 32],  # NCHW
    tDim=[32, 32],                   # Process 32x32 at a time
    dtype=DataflowDataType("UINT", 8, False, "UINT8"),
    chunking_mode="broadcast"
)

print(f"Reconstructed shape: {interface.reconstruct_tensor_shape()}")
```

### RTL Integration

```python
from brainsmith.dataflow.integration.rtl_conversion import RTLInterfaceConverter

# Convert RTL Parser interfaces to DataflowInterfaces
converter = RTLInterfaceConverter(onnx_metadata)
dataflow_interfaces = converter.convert_interfaces(rtl_interfaces)

# Create unified model
model = DataflowModel(dataflow_interfaces, parameters)
```

## Integration with Brainsmith Ecosystem

### HW Kernel Generator Integration

The framework integrates seamlessly with the HW Kernel Generator:

1. **RTL Parser** discovers interfaces and pragmas
2. **RTL Conversion** transforms to DataflowInterface objects
3. **Code Generation** uses base classes for minimal template code
4. **Validation** ensures correctness throughout the pipeline

### FINN Optimization Integration

```python
# Get parallelism bounds for FINN optimization
bounds = model.get_parallelism_bounds()

for param_name, bound in bounds.items():
    print(f"{param_name}: [{bound.min_value}, {bound.max_value}]")
    print(f"Valid divisors: {bound.divisibility_constraints}")

# Optimize parallelism within constraints
optimal_config = model.optimize_parallelism(resource_constraints)
```

### Auto-Generated Base Classes

Instead of generating hundreds of lines of boilerplate code, templates can inherit from base classes:

```python
# Generated HWCustomOp (minimal template code)
class AutoThresholdingAxiHWCustomOp(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        dataflow_model = create_thresholding_model()  # Template-specific
        super().__init__(onnx_node, dataflow_model, **kwargs)
    
    def get_nodeattr_types(self):
        return self.get_enhanced_nodeattr_types()  # From base class
    
    # All other methods inherited from AutoHWCustomOp
```

## Advanced Features

### Constraint Validation

```python
# Validate mathematical constraints
validation_result = model.validate_mathematical_constraints()

# Validate tensor chunking
chunking_result = interface.validate_tensor_chunking([1, 64, 32, 32])

# Validate datatype against constraints
if interface.validate_datatype_string("UINT16"):
    print("Datatype allowed")
```

### Resource Estimation

```python
# Estimate resource requirements
parallelism_config = ParallelismConfiguration(iPar, wPar, derived_sDim)
resources = model.get_resource_requirements(parallelism_config)

print(f"Memory: {resources['memory_bits']} bits")
print(f"Bandwidth: {resources['transfer_bandwidth']} bits/cycle")
print(f"Compute cycles: {resources['computation_cycles']}")
```

### AXI Signal Generation

```python
# Generate AXI signal specifications
axi_signals = interface.get_axi_signals()
for signal_name, signal_info in axi_signals.items():
    print(f"{signal_name}: {signal_info['direction']} ({signal_info['width']} bits)")
```

## Error Handling and Validation

The framework provides comprehensive error handling:

```python
try:
    interface = DataflowInterface(name="test", ...)
except ValueError as e:
    print(f"Configuration error: {e}")

# Validation with detailed error reporting
result = interface.validate_constraints()
for error in result.errors:
    print(f"Error in {error.component}: {error.message}")
    print(f"Context: {error.context}")
```

## Performance Optimization

### Parallelism Optimization

```python
# Get valid parallelism bounds
bounds = model.get_parallelism_bounds()

# Find optimal configuration
optimal_config = model.optimize_parallelism({
    "max_dsp": 256,
    "max_bram": 128,
    "target_throughput": 1000
})
```

### Memory Footprint Analysis

```python
# Analyze memory requirements
for interface in model.interfaces.values():
    footprint = interface.get_memory_footprint()
    transfer_cycles = interface.get_transfer_cycles()
    
    print(f"{interface.name}: {footprint} bits, {transfer_cycles} cycles")
```

## Development and Testing

### Running Examples

```bash
# Run basic usage example
python -m brainsmith.dataflow.examples.basic_usage

# Run integration tests
python -m pytest tests/dataflow/
```

### Creating Custom Interfaces

```python
# Custom interface with specific constraints
custom_interface = DataflowInterface(
    name="custom_weights",
    interface_type=DataflowInterfaceType.WEIGHT,
    qDim=[256, 128],
    tDim=[32, 16], 
    sDim=[8, 4],
    dtype=DataflowDataType("INT", 8, True, "INT8"),
    allowed_datatypes=[
        DataTypeConstraint(
            base_types=["INT"],
            min_bitwidth=4,
            max_bitwidth=16,
            signed_allowed=True,
            unsigned_allowed=False
        )
    ]
)
```

## Migration Guide

### From Legacy FINN Integration

1. Replace manual dimension calculations with [`DataflowModel.calculate_initiation_intervals()`](core/dataflow_model.py:103)
2. Use [`DataTypeConstraint`](core/dataflow_interface.py:71) instead of hardcoded datatype lists
3. Inherit from [`AutoHWCustomOp`](core/auto_hw_custom_op.py:44) instead of writing boilerplate methods
4. Use RTL conversion pipeline instead of manual interface creation

### From Manual Code Generation

1. Replace template-heavy code generation with base class inheritance
2. Use [`DataflowInterface`](core/dataflow_interface.py:142) for standardized interface representation
3. Leverage validation framework for automatic constraint checking

## Contributing

1. Follow the mathematical precision principles for dimension relationships
2. Ensure all new components include comprehensive validation
3. Add examples for new features in the examples directory
4. Update API documentation for public interfaces

## License

Part of the Brainsmith ecosystem. See project root for license information.