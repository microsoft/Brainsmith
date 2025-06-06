# Current Architecture Analysis: Dataflow Modeling and HWCustomOp Generation

## Executive Summary

This document provides a comprehensive analysis of the current implementation of the Interface-Wise Dataflow Modeling framework and Hardware Kernel Generator (HKG) for automated HWCustomOp generation. The analysis covers the architecture, implementation status, key components, and integration points within the Brainsmith ecosystem.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components Analysis](#core-components-analysis)
3. [Implementation Status](#implementation-status)
4. [Integration Architecture](#integration-architecture)
5. [Code Generation Pipeline](#code-generation-pipeline)
6. [Testing and Validation Framework](#testing-and-validation-framework)
7. [Technology Stack](#technology-stack)
8. [Current Capabilities](#current-capabilities)

## Architecture Overview

### Conceptual Framework

The Interface-Wise Dataflow Modeling framework implements a **unified abstraction** for hardware kernel interfaces that bridges the gap between RTL implementation and ONNX/PyTorch operators. The architecture is built around three core principles:

1. **Interface Uniformity**: All kernel interfaces (INPUT, OUTPUT, WEIGHT, CONFIG, CONTROL) are represented using a consistent abstraction
2. **Computational Unity**: A single mathematical model describes performance characteristics across all interface types
3. **Template-Driven Generation**: Production-quality code generation using metadata-rich templates

### System Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RTL Source    │    │ Compiler Data   │    │   ONNX Model    │
│  (SystemVerilog)│    │   (Python)      │    │   Metadata      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Hardware Kernel Generator (HKG)                │
├─────────────────┬───────────────────────┬─────────────────────┤
│   RTL Parser    │  Dataflow Converter   │   Template Engine   │
│   Pipeline      │      Pipeline         │      Pipeline       │
└─────────┬───────┴───────────┬───────────┴─────────┬───────────┘
          │                   │                     │
          ▼                   ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   HWKernel      │  │ DataflowModel   │  │  Generated Code │
│   (RTL Data)    │  │ (Unified Model) │  │ (AutoHWCustomOp)│
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Data Flow Architecture

The framework implements a **three-tier data hierarchy** for interface modeling:

- **Query Level**: Complete hidden state or weight data streamed through interface
- **Tensor Level**: Data for a single calculation within the kernel  
- **Stream Level**: Data transferred per clock cycle

This hierarchy enables precise **parallelism optimization** through mathematical relationships:
- `qDim × tDim = original_tensor_shape` (with broadcasting)
- `tDim % sDim == 0` (streaming constraint)
- `sDim = f(iPar, wPar)` (parallelism application)

## Core Components Analysis

### 1. DataflowInterface (`brainsmith/dataflow/core/dataflow_interface.py`)

**Purpose**: Unified interface abstraction with constraint validation and datatype support.

**Key Features**:
- **Dimension Management**: qDim (query), tDim (tensor), sDim (stream) with validation
- **Datatype Constraints**: FINN-compatible datatype system with constraint checking
- **AXI Signal Generation**: Automatic AXI-Stream/AXI-Lite signal specification
- **Validation Framework**: Comprehensive constraint validation with detailed error reporting

**Implementation Quality**: ✅ **Complete and Production-Ready**

```python
# Example Usage
interface = DataflowInterface(
    name="input_stream",
    interface_type=DataflowInterfaceType.INPUT,
    qDim=[32],           # 32 tensors per query
    tDim=[16],           # 16 elements per tensor
    sDim=[4],            # 4 elements per cycle (4x parallelism)
    dtype=DataflowDataType(base_type="UINT", bitwidth=8, signed=False, finn_type="UINT8"),
    allowed_datatypes=[DataTypeConstraint(base_types=["UINT"], min_bitwidth=1, max_bitwidth=16)]
)
```

### 2. DataflowModel (`brainsmith/dataflow/core/dataflow_model.py`)

**Purpose**: Unified computational model implementing mathematical relationships between interfaces and parallelism parameters.

**Key Features**:
- **Initiation Interval Calculations**: cII, eII, L with bottleneck analysis
- **Parallelism Optimization**: iPar (input) and wPar (weight) parameter handling
- **Resource Estimation**: Memory, bandwidth, and computation requirements
- **Multi-Interface Support**: Handles arbitrary numbers of input, output, and weight interfaces

**Implementation Quality**: ✅ **Core Complete**, 🔄 **Optimization Algorithms In Progress**

**Mathematical Foundation**:
```python
# Core relationships implemented
cII_i = ∏(tDim_I,i / sDim_I,i)                    # Per-input calculation interval
eII_i = cII_i * max_j(∏(qDim_W,j / wPar_j))      # Per-input execution interval  
L = max_i(eII_i) * ∏(qDim_I,bottleneck)          # Overall inference latency
```

### 3. AutoHWCustomOp (`brainsmith/dataflow/core/auto_hw_custom_op.py`)

**Purpose**: Base class for auto-generated HWCustomOp implementations with standardized methods.

**Key Features**:
- **Standardized Implementations**: All common HWCustomOp methods implemented once
- **Dataflow Integration**: Direct integration with DataflowModel for performance calculations
- **Resource Estimation**: Automatic BRAM, LUT, DSP estimation based on interface configuration
- **FINN Compatibility**: Maintains full compatibility with existing FINN APIs

**Implementation Quality**: ✅ **Architecture Complete**, 🔄 **Some Methods Need Kernel-Specific Implementation**

**Standardized Methods**:
- `get_input_datatype()`, `get_output_datatype()` - Datatype management
- `get_normal_*_shape()`, `get_folded_*_shape()` - Shape inference  
- `get_instream_width()`, `get_outstream_width()` - Stream width calculations
- `get_exp_cycles()` - Performance estimation using unified model
- `generate_params()` - Parameter file generation

### 4. RTLInterfaceConverter (`brainsmith/dataflow/integration/rtl_conversion.py`)

**Purpose**: Converts RTL Parser Interface objects to DataflowInterface objects with pragma processing.

**Key Features**:
- **Interface Type Mapping**: AXI-Stream → INPUT/OUTPUT/WEIGHT, AXI-Lite → CONFIG
- **Dimension Extraction**: From ONNX metadata and TDIM pragmas
- **Datatype Constraint Conversion**: From DATATYPE pragmas to DataTypeConstraint objects
- **Metadata Integration**: Comprehensive pragma and AXI metadata preservation

**Implementation Quality**: ✅ **Complete and Robust**

**Conversion Pipeline**:
1. **Type Classification**: Determine DataflowInterfaceType from RTL interface characteristics
2. **Dimension Inference**: Extract qDim/tDim from ONNX metadata or pragmas
3. **Datatype Processing**: Convert pragma constraints to DataTypeConstraint objects
4. **Validation**: Ensure converted interfaces meet dataflow modeling requirements

### 5. HardwareKernelGenerator (`brainsmith/tools/hw_kernel_gen/hkg.py`)

**Purpose**: Main orchestrator for the complete RTL-to-HWCustomOp generation pipeline.

**Key Features**:
- **Multi-Phase Pipeline**: RTL parsing → dataflow conversion → model building → code generation
- **Enhanced Dataflow Support**: Full integration with dataflow modeling framework
- **Template-Based Generation**: Jinja2 templates with comprehensive context
- **Error Handling**: Detailed error reporting with phase-specific context

**Implementation Quality**: ✅ **Complete Pipeline**, 🔄 **Template Fine-Tuning Needed**

**Pipeline Phases**:
1. `parse_rtl` - RTL Parser execution
2. `parse_compiler_data` - Python module loading and AST analysis
3. `build_dataflow_model` - DataflowInterface conversion and model creation
4. `generate_hw_custom_op` - AutoHWCustomOp generation using templates
5. `generate_rtl_backend` - RTLBackend generation
6. `generate_documentation` - Comprehensive documentation generation

## Implementation Status

### ✅ Completed Components

| Component | Implementation Status | Test Coverage | Production Ready |
|-----------|----------------------|---------------|------------------|
| DataflowInterface | ✅ Complete | ✅ Comprehensive | ✅ Yes |
| DataflowModel | ✅ Core Complete | ✅ Unit Tests | ✅ Yes |
| RTLInterfaceConverter | ✅ Complete | ✅ Integration Tests | ✅ Yes |
| AutoHWCustomOp | ✅ Architecture Complete | ✅ Basic Tests | 🔄 Needs Kernel-Specific Methods |
| HKG Pipeline | ✅ Complete | ✅ End-to-End Tests | ✅ Yes |
| Template System | ✅ Basic Complete | ✅ Generation Tests | 🔄 Syntax Fixes Needed |

### 🔄 In Progress Components

| Component | Status | Next Steps |
|-----------|---------|------------|
| Validation Framework | Core complete | Enhanced constraint types |
| Resource Estimation | Basic implementation | Kernel-specific algorithms |
| Template Syntax | Working | Fix Jinja2 edge cases |
| Documentation Generation | Basic | Enhanced formatting |

### ❌ Missing Components

| Component | Priority | Description |
|-----------|----------|-------------|
| Utility Classes | High | Missing helper classes referenced in templates |
| Advanced Optimization | Medium | SetFolding algorithm integration |
| Performance Benchmarks | Low | Comprehensive performance analysis |

## Integration Architecture

### FINN Integration Points

The framework maintains **full compatibility** with FINN while adding enhancements:

```python
# Traditional FINN HWCustomOp
class TraditionalOp(HWCustomOp):
    def get_exp_cycles(self):
        # Manual implementation required
        return self.calc_manual_cycles()
    
    def get_instream_width(self):
        # Manual calculation required
        return self.manual_width_calc()

# Enhanced AutoHWCustomOp  
class EnhancedOp(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        # Dataflow model automatically provides all standard methods
        super().__init__(onnx_node, dataflow_model, **kwargs)
    
    # get_exp_cycles(), get_instream_width(), etc. automatically implemented
    # Only kernel-specific resource estimation needs implementation
```

### RTL Parser Integration

The framework leverages the **complete RTL Parser pipeline**:

```python
# RTL Parser Output → Dataflow Conversion
rtl_interfaces = rtl_parser.parse_file("kernel.sv")
converter = RTLInterfaceConverter(onnx_metadata)
dataflow_interfaces = converter.convert_interfaces(rtl_interfaces)
dataflow_model = DataflowModel(dataflow_interfaces, parameters)
```

## Code Generation Pipeline

### Template-Based Generation

The framework uses **Jinja2 templates** with enhanced context for production-quality code generation:

**Template Context Structure**:
```python
template_context = {
    # Kernel metadata
    "kernel_name": "thresholding_axi",
    "class_name": "AutoThresholdingAxi", 
    "source_file": "thresholding_axi.sv",
    
    # RTL Parser data
    "rtl_parameters": [Parameter(name="PE", default_value="1", ...)],
    "rtl_interfaces": {"s_axis": Interface(...), ...},
    
    # Dataflow framework data
    "dataflow_interfaces": [DataflowInterface(...), ...],
    "dataflow_model": DataflowModel(...),
    
    # Interface organization
    "input_interfaces": [...],
    "output_interfaces": [...], 
    "weight_interfaces": [...],
    
    # Computational model
    "has_unified_model": True,
    "parallelism_bounds": {...}
}
```

### Generated Code Structure

**AutoHWCustomOp Generation**:
```python
# Generated class inherits standardized implementations
class AutoThresholdingAxi(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, dataflow_model, **kwargs)
    
    # Kernel-specific resource estimation (only methods that need implementation)
    def bram_estimation(self): ...
    def lut_estimation(self): ...
    def dsp_estimation(self): ...
```

## Testing and Validation Framework

### End-to-End Integration Testing

The framework includes **comprehensive integration testing** using real-world examples:

**Test Coverage** (`tests/integration/test_end_to_end_thresholding.py`):
- ✅ RTL parsing of complex multi-interface modules
- ✅ Interface detection and classification
- ✅ Dataflow conversion with and without pragmas
- ✅ Dataflow model creation and validation
- ✅ Complete HKG pipeline execution
- ✅ Template-based code generation
- ✅ Performance and scalability testing

**Test Methodology**:
```python
def test_complete_hkg_pipeline(self):
    # Real thresholding_axi.sv with AXI-Stream, AXI-Lite, Global Control
    hkg = HardwareKernelGenerator(rtl_file, compiler_data, output_dir)
    generated_files = hkg.run(stop_after="generate_hw_custom_op")
    
    # Verify complete pipeline execution
    assert "rtl_template" in generated_files
    assert "hw_custom_op" in generated_files
    assert all(Path(f).exists() for f in generated_files.values())
```

### Validation Framework

**Constraint Validation**:
```python
# Automatic validation of interface constraints
validation_result = interface.validate_constraints()
if not validation_result.is_valid():
    for error in validation_result.errors:
        print(f"ERROR: {error.message}")
```

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary implementation language
- **Jinja2**: Template engine for code generation
- **tree-sitter**: RTL parsing with SystemVerilog grammar
- **NumPy**: Mathematical operations and array handling
- **pytest**: Comprehensive testing framework

### Integration Technologies  
- **FINN**: HWCustomOp and RTLBackend base classes
- **QONNX**: DataType integration and ONNX compatibility
- **pathlib**: Modern path handling
- **importlib**: Dynamic module loading

### Development Tools
- **dataclasses**: Type-safe data structures
- **typing**: Comprehensive type annotations
- **enum**: Type-safe enumerations
- **logging**: Structured logging throughout

## Current Capabilities

### ✅ Production-Ready Features

1. **Complete Dataflow Modeling**: Full interface abstraction with mathematical model
2. **RTL Integration**: Robust parsing and conversion pipeline
3. **Template Generation**: Working code generation for AutoHWCustomOp and RTLBackend
4. **FINN Compatibility**: Seamless integration with existing FINN workflows
5. **Validation Framework**: Comprehensive constraint checking and error reporting
6. **End-to-End Testing**: Real-world integration test using thresholding_axi.sv

### 🔄 Capabilities in Progress

1. **Template Syntax**: Minor Jinja2 syntax issues being resolved
2. **Resource Estimation**: Kernel-specific algorithms need implementation
3. **Documentation Generation**: Enhanced formatting and content
4. **Optimization Algorithms**: Advanced parallelism optimization

### 📋 Planned Capabilities

1. **SetFolding Integration**: FINN optimization algorithm integration
2. **Multi-Kernel Support**: Batch processing of multiple kernels
3. **Performance Benchmarking**: Comprehensive performance analysis tools
4. **Advanced Validation**: Cross-interface constraint validation

## Architecture Strengths

### 1. **Unified Abstraction**
- Single interface model works across all interface types
- Consistent mathematical relationships throughout

### 2. **Production Quality** 
- Comprehensive error handling and validation
- Full FINN compatibility maintained
- Extensive test coverage

### 3. **Extensibility**
- Template-based generation easily customizable
- Modular architecture enables easy enhancement
- Well-defined integration points

### 4. **Real-World Validation**
- Tested with complex real-world RTL (thresholding_axi.sv)
- End-to-end pipeline validation
- Performance and scalability testing

## Conclusion

The Interface-Wise Dataflow Modeling framework represents a **significant advancement** in hardware kernel development automation. The current implementation provides:

- **Complete architectural foundation** for dataflow modeling
- **Production-ready core components** with comprehensive testing
- **Seamless FINN integration** maintaining compatibility
- **Template-based code generation** for automated HWCustomOp creation
- **Robust validation framework** ensuring correctness

The architecture successfully bridges the gap between RTL implementation and high-level frameworks, enabling **automated generation of production-quality HWCustomOp classes** with standardized implementations and unified computational models.

**Ready for Production**: Core dataflow modeling and code generation capabilities are complete and tested.
**Enhancement Opportunities**: Template refinement, resource estimation algorithms, and advanced optimization features provide clear paths for continued development.