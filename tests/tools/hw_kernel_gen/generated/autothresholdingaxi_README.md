# AutoThresholdingAxi - Auto-Generated HWCustomOp

## Overview

This document describes the auto-generated HWCustomOp implementation for `thresholding_axi`.

**Source RTL:** `/tmp/tmpquagqr66/thresholding_enhanced.sv`
**Generated Classes:**
- `AutoThresholdingAxi` - Main HWCustomOp implementation
- `AutoThresholdingAxiRTLBackend` - RTL backend for synthesis
- `TestAutoThresholdingAxi` - Comprehensive test suite

## Interface Specification

**Total Interfaces:** 4

### Input Interfaces

- **s_axis**
  - Type: DataflowInterfaceType.INPUT
  - Dimensions: qDim=[1, 32], tDim=[1, 32], stream_dims=[1, 1]
  - Data Type: UINT8

### Output Interfaces

- **m_axis**
  - Type: DataflowInterfaceType.OUTPUT
  - Dimensions: qDim=[1, 32], tDim=[1, 32], stream_dims=[1, 1]
  - Data Type: UINT1

### Configuration Interfaces

- **s_axilite**
  - Type: DataflowInterfaceType.CONFIG
  - Dimensions: qDim=[32], tDim=[32], stream_dims=[1]
  - Data Type: UINT32


## Usage Example

```python
from autothresholdingaxi import AutoThresholdingAxi
from finn.core.modelwrapper import ModelWrapper

# Create ONNX model with AutoThresholdingAxi node
# ... (model creation code)

# Get the node and create HWCustomOp instance
node = model.get_nodes_by_op_type("AutoThresholdingAxi")[0]
hw_op = AutoThresholdingAxi(node)

# Configure parallelism and datatypes
hw_op.set_nodeattr("ap_dtype", "UINT32")
hw_op.set_nodeattr("s_axis_dtype", "UINT8")
hw_op.set_nodeattr("m_axis_dtype", "UINT1")
hw_op.set_nodeattr("s_axilite_dtype", "UINT32")

# Verify node configuration
hw_op.verify_node()

# Get resource estimates
bram_usage = hw_op.bram_estimation()
lut_usage = hw_op.lut_estimation()
dsp_usage = hw_op.dsp_estimation("xcvu9p-flga2104-2-i")

print(f"Resource estimates - BRAM: {bram_usage}, LUT: {lut_usage}, DSP: {dsp_usage}")
```

## Generated Files

- `autothresholdingaxi.py` - Main HWCustomOp implementation
- `autothresholdingaxi_rtlbackend.py` - RTL backend implementation
- `test_autothresholdingaxi.py` - Comprehensive test suite
- `autothresholdingaxi_README.md` - This documentation file

## Resource Estimation

The generated classes include automatic resource estimation based on interface configuration:

- **BRAM Estimation:** Based on weight interface storage requirements and parallelism
- **LUT Estimation:** Based on interface complexity and control logic requirements
- **DSP Estimation:** Based on arithmetic operations and datatype bitwidths

Estimation modes:
- `automatic` - Balanced estimation (default)
- `conservative` - Higher resource estimates for safety margin
- `optimistic` - Lower resource estimates assuming optimal implementation

## Testing

Run the generated test suite:

```bash
pytest test_autothresholdingaxi.py -v
```

The test suite covers:
- Basic functionality and node creation
- Datatype constraint validation
- Parallelism configuration testing
- Resource estimation validation
- RTL backend integration
- End-to-end inference testing (when RTL simulation available)

## Interface-Wise Dataflow Modeling

This implementation uses the Interface-Wise Dataflow Modeling Framework which provides:

- **Unified Computational Model:** Consistent interface abstraction across input, output, weight, and config interfaces
- **Constraint Validation:** Automatic validation of datatype and parallelism constraints
- **Resource Estimation:** Interface-aware resource estimation algorithms
- **Template-Based Generation:** Production-quality code generation from RTL specifications

For more information about the framework, see the main documentation.
