# FINN HWCustomOp System Architecture Analysis

## Executive Summary

This document provides a comprehensive analysis of the FINN HWCustomOp system architecture, building upon previous PE/SIMD parallelism analysis to understand the broader framework for custom hardware operation implementations. The analysis covers the three-tier architecture, backend specialization patterns, transformation passes, and the complete lifecycle from ONNX graphs to hardware implementations.

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [Three-Tier HWCustomOp Architecture](#three-tier-hwcustomop-architecture)
3. [Backend Specialization System](#backend-specialization-system)
4. [Registration and Discovery Mechanism](#registration-and-discovery-mechanism)
5. [Transformation Pass Integration](#transformation-pass-integration)
6. [Code Generation and Template System](#code-generation-and-template-system)
7. [Resource Estimation and Optimization](#resource-estimation-and-optimization)
8. [Interface Protocol Management](#interface-protocol-management)
9. [Simulation and Verification Framework](#simulation-and-verification-framework)
10. [Extension Points and Customization](#extension-points-and-customization)
11. [Questions for Expert Consultation](#questions-for-expert-consultation)

## Architectural Overview

FINN implements a sophisticated hardware custom operation system that transforms QONNX/ONNX computational graphs into optimized FPGA dataflow implementations. The system is built around a three-tier architecture that provides both flexibility and implementation efficiency.

### Core Design Principles

1. **Abstraction Separation**: Clear separation between generic operation logic, backend-specific implementation, and concrete hardware generation
2. **Backend Flexibility**: Support for both HLS (High-Level Synthesis) and RTL (Register Transfer Level) implementation approaches
3. **Template-Driven Generation**: Parameterized code generation using template substitution for both C++ and Verilog/SystemVerilog
4. **Streaming Dataflow**: AXI-Stream based interfaces for composable hardware modules
5. **Resource-Aware Optimization**: Built-in resource estimation and parallelism optimization

### Integration with PE/SIMD Parallelism

Building upon the previous PE/SIMD analysis, the HWCustomOp system provides the architectural framework within which PE/SIMD optimizations operate:

```
QONNX Graph → HWCustomOp Instances → Backend Specialization → Hardware Generation
                    ↓
            PE/SIMD Parallelism Configuration
                    ↓
            Resource Estimation & Optimization
                    ↓
            Template-Based Code Generation
```

## Three-Tier HWCustomOp Architecture

### Tier 1: Generic HWCustomOp Base Class

The foundation layer provides common functionality for all hardware-implementable operations:

```python
class HWCustomOp(CustomOp):
    """Base class for all FPGA dataflow custom operations"""
```

**Key Responsibilities:**
- Abstract method definitions for shape inference, datatype handling, and stream width calculation
- Resource estimation interfaces (BRAM, LUT, DSP, URAM)
- RTL simulation infrastructure (pyxsi integration)
- Common node attribute management
- AXI interface protocol handling
- Characteristic function derivation for FIFO sizing

**Critical Abstract Methods:**
```python
@abstractmethod
def get_number_output_values(self)
def get_input_datatype(self, ind=0)
def get_output_datatype(self, ind=0)
def get_normal_input_shape(self, ind=0)
def get_normal_output_shape(self, ind=0)
def get_folded_input_shape(self, ind=0)  # PE/SIMD folding
def get_folded_output_shape(self, ind=0)
def get_instream_width(self, ind=0)
def get_outstream_width(self, ind=0)
```

### Tier 2: Backend Interface Classes

Specialized interfaces for different implementation methodologies:

#### HLSBackend Interface
```python
class HLSBackend(ABC):
    """Interface for HLS-based implementations using finn-hlslib"""
```

**Capabilities:**
- C++ code generation with template substitution
- HLS pragma management for synthesis directives
- Vivado HLS project generation and compilation
- C++ simulation execution
- IP core packaging for integration

**Template Generation Pattern:**
```python
def code_generation_ipgen(self, model, fpgapart, clk):
    """Generates C++ code and TCL scripts for IP generation"""
    # Template substitution using self.code_gen_dict
    # Key-value replacement: "$VARIABLE$" -> actual_value
```

#### RTLBackend Interface
```python
class RTLBackend(ABC):
    """Interface for RTL-based implementations using finn-rtllib"""
```

**Capabilities:**
- Direct Verilog/SystemVerilog generation from templates
- Module instantiation and parameter configuration
- FPGA-specific optimization (DSP48, BRAM, URAM utilization)
- Vivado IPI integration for block design generation

### Tier 3: Concrete Implementations

Specific operation implementations inheriting from both generic operation class and backend interface:

```python
# HLS Implementation Example
class ChannelwiseOp_hls(ChannelwiseOp, HLSBackend):
    """HLS implementation of channelwise operations"""

# RTL Implementation Example  
class FMPadding_rtl(FMPadding, RTLBackend):
    """RTL implementation of feature map padding"""
```

## Backend Specialization System

### HLS Backend Specialization

The HLS backend transforms operations into C++ code that gets synthesized by Vivado HLS:

**Code Generation Components:**
1. **Global Includes**: C++ headers and finn-hlslib includes
2. **Defines**: Preprocessor definitions and type aliases
3. **Stream Declarations**: HLS stream variable declarations
4. **Parameters**: Weight/threshold tensor initialization
5. **Blackbox Function**: Top-level function signature
6. **Do Compute**: Core algorithmic implementation
7. **Pragmas**: HLS synthesis directives

**Template Substitution Example:**
```cpp
// Template: finn-hlslib/func_template.cpp
$GLOBALS$         // → #include "activations.hpp"
$DEFINES$         // → typedef ap_uint<32> input_t;
$STREAMDECLARATIONS$ // → hls::stream<input_t> in0_V;
$BLACKBOXFUNCTION$   // → void node_name(...)
$DOCOMPUTE$         // → Matrix_Vector_Activate<...>
$PRAGMAS$           // → #pragma HLS INTERFACE axis port=in0_V
```

### RTL Backend Specialization

The RTL backend directly generates optimized Verilog/SystemVerilog modules:

**Generation Approach:**
1. **Template Loading**: Load SystemVerilog templates from finn-rtllib
2. **Parameter Calculation**: Compute module parameters (widths, depths, etc.)
3. **Code Generation**: Replace template variables with computed values
4. **Module Instantiation**: Generate wrapper modules for AXI interfaces

**Template Substitution Example:**
```systemverilog
// Template: finn-rtllib/module_template.sv
module $MODULE_NAME$ #(
    parameter SIMD = $SIMD$,
    parameter PE = $PE$,
    parameter WEIGHT_WIDTH = $WEIGHT_WIDTH$
) (
    // AXI Stream interfaces
);
```

## Registration and Discovery Mechanism

### Custom Operation Registry

FINN uses a decorator-based registration system for automatic discovery:

```python
@register_custom_op
class ElementwiseAdd_hls(ElementwiseBinaryOperation_hls, 
                        elementwise_binary.ElementwiseAdd):
    pass
```

**Registration Process:**
1. **Import-Time Registration**: Classes are registered when modules are imported
2. **Type Validation**: Ensures proper inheritance from HWCustomOp and backend classes
3. **Dictionary Storage**: Operations stored in `custom_op` dictionaries by name
4. **Transformation Access**: Transformation passes can look up implementations by name

**Registry Structure:**
```python
# finn/custom_op/fpgadataflow/__init__.py
custom_op = dict()  # Generic operations

# finn/custom_op/fpgadataflow/hls/__init__.py  
custom_op = dict()  # HLS implementations

# finn/custom_op/fpgadataflow/rtl/__init__.py
custom_op = dict()  # RTL implementations
```

## Transformation Pass Integration

### Convert to HW Layers Transformation

The system integrates with FINN's transformation passes to convert standard ONNX operations to hardware implementations:

```python
class InferElementwiseBinaryOperation(Transformation):
    def apply(self, model: ModelWrapper):
        # Identify compatible operations
        # Replace with HWCustomOp instances
        # Set backend attribute to "fpgadataflow"
        # Configure PE/SIMD parameters
```

**Transformation Sequence:**
1. **Operation Identification**: Match ONNX op_type to available HWCustomOp implementations
2. **Node Replacement**: Replace ONNX node with HWCustomOp wrapper
3. **Attribute Configuration**: Set dataflow-specific attributes (PE, SIMD, etc.)
4. **Datatype Inference**: Propagate quantized datatypes through the graph
5. **Shape Inference**: Calculate folded shapes for streaming interfaces

### Folding and Optimization Passes

PE/SIMD optimization integrates directly with the HWCustomOp architecture:

```python
class SetFolding(Transformation):
    """Sets PE and SIMD attributes to meet throughput targets"""
    def apply(self, model):
        for node in model.graph.node:
            if node.op_type in hw_ops:
                inst = getCustomOp(node)
                # Optimize PE/SIMD based on constraints
                inst.set_nodeattr("PE", optimal_pe)
                inst.set_nodeattr("SIMD", optimal_simd)
```

## Code Generation and Template System

### Template-Based Generation Strategy

Both HLS and RTL backends use parameterized templates with variable substitution:

**HLS Template Structure:**
```cpp
// Top-level function template
$BLACKBOXFUNCTION$ {
    $STREAMDECLARATIONS$
    $PRAGMAS$
    $DOCOMPUTE$
}
```

**RTL Template Structure:**
```systemverilog
module $MODULE_NAME$ #(
    parameter $PARAMETER_LIST$
) (
    $PORT_DECLARATIONS$
);
    $MODULE_IMPLEMENTATION$
endmodule
```

### Code Generation Dictionary Pattern

All implementations use a consistent `code_gen_dict` pattern:

```python
self.code_gen_dict = {
    "$GLOBALS$": ["#include <headers>"],
    "$DEFINES$": ["#define PARAM value"],
    "$STREAMDECLARATIONS$": ["hls::stream<type> var;"],
    "$BLACKBOXFUNCTION$": ["void func_name(...)"],
    "$DOCOMPUTE$": ["algorithm_call(...)"],
    "$PRAGMAS$": ["#pragma HLS directive"]
}
```

## Resource Estimation and Optimization

### Multi-Resource Estimation Framework

The HWCustomOp system provides comprehensive resource estimation:

```python
def node_res_estimation(self, fpgapart):
    return {
        "BRAM_18K": self.bram_estimation(),
        "BRAM_efficiency": self.bram_efficiency_estimation(),
        "LUT": self.lut_estimation(), 
        "URAM": self.uram_estimation(),
        "URAM_efficiency": self.uram_efficiency_estimation(),
        "DSP": self.dsp_estimation(fpgapart)
    }
```

**Implementation-Specific Estimation:**
- **MVAU Operations**: DSP utilization based on precision and FPGA family
- **Memory Operations**: BRAM/URAM efficiency calculations
- **Logic Operations**: LUT estimation for control and datapath logic

### PE/SIMD Optimization Integration

Resource estimation directly feeds into PE/SIMD optimization algorithms:

```python
# From previous PE/SIMD analysis - now contextualized
def optimize_pe_simd(self, target_cycles, resource_budget):
    for simd in valid_simd_values:
        for pe in valid_pe_values:
            resources = self.estimate_resources(pe, simd)
            cycles = self.calculate_cycles(pe, simd)
            if cycles <= target_cycles and resources <= budget:
                return pe, simd
```

## Interface Protocol Management

### AXI Stream Interface Generation

The HWCustomOp system automatically generates AXI Stream interfaces:

```python
def get_verilog_top_module_intf_names(self):
    return {
        "clk": ["ap_clk"],
        "rst": ["ap_rst_n"], 
        "s_axis": [("in0_V", width), ("in1_V", width)],
        "m_axis": [("out0_V", width)],
        "aximm": [],  # Memory-mapped interfaces
        "axilite": []  # Control interfaces
    }
```

**Stream Width Calculation:**
- Input streams: `SIMD * input_datatype.bitwidth()` (padded to 8-bit boundaries)
- Output streams: `PE * output_datatype.bitwidth()` (padded to 8-bit boundaries)
- Memory interfaces: `PE * SIMD * weight_datatype.bitwidth()`

### Multi-Protocol Support

Advanced operations can combine multiple interface protocols:

1. **AXI Stream**: Primary data interfaces for streaming computation
2. **AXI Memory-Mapped**: For external memory access (weights, activations)
3. **AXI Lite**: For runtime configuration and control
4. **Protocol-less**: For simple control signals

## Simulation and Verification Framework

### RTL Simulation Infrastructure

The system provides comprehensive simulation capabilities through pyxsi integration:

```python
def prepare_rtlsim(self):
    """Creates XSI emulation library for RTL simulation"""
    verilog_files = self.get_all_verilog_filenames(abspath=True)
    sim = pyxsi_utils.compile_sim_obj(
        self.get_verilog_top_module_name(),
        verilog_files, 
        single_src_dir,
        debug=True
    )
    self.set_nodeattr("rtlsim_so", sim_library_path)
```

### Multi-Level Verification

1. **C++ Simulation**: Functional verification using HLS C++ simulation
2. **RTL Simulation**: Cycle-accurate verification using XSI
3. **Hardware Verification**: Actual FPGA execution for final validation

## Extension Points and Customization

### Adding New HWCustomOp Implementations

To extend the system with new operations:

1. **Define Generic Operation Class**:
```python
class NewOperation(HWCustomOp):
    """Generic operation definition"""
    def get_nodeattr_types(self):
        # Define operation-specific attributes
    
    # Implement abstract methods
```

2. **Create Backend Implementations**:
```python
@register_custom_op
class NewOperation_hls(NewOperation, HLSBackend):
    """HLS implementation"""
    
@register_custom_op  
class NewOperation_rtl(NewOperation, RTLBackend):
    """RTL implementation"""
```

3. **Add Transformation Support**:
```python
class InferNewOperation(Transformation):
    """Convert ONNX nodes to NewOperation HWCustomOp"""
```

### PE/SIMD Integration Points

New operations should integrate PE/SIMD parallelism by:

1. **Defining Parallelism Constraints**:
```python
def verify_node(self):
    assert self.get_nodeattr("Channels") % self.get_nodeattr("PE") == 0
    assert self.get_nodeattr("Width") % self.get_nodeattr("SIMD") == 0
```

2. **Implementing Resource Estimation**:
```python
def dsp_estimation(self, fpgapart):
    pe = self.get_nodeattr("PE")
    simd = self.get_nodeattr("SIMD") 
    # Calculate DSP usage based on parallelism
```

3. **Supporting Folding Optimization**:
```python
def get_exp_cycles(self):
    return (height / pe) * (width / simd) * batch_size
```

## Questions for Expert Consultation

Based on this analysis, here are key questions to clarify understanding:

### Expert Responses and Key Insights

Based on expert consultation, the following insights clarify critical aspects of FINN's architecture and identify key areas for future development:

### Architecture and Design Insights

1. **Backend Selection Strategy**: **RTL is always preferable when sufficient development time is available**. HLS is best only for quick prototyping. Although both backends are supported, **RTL is the preferred backend** for production implementations due to superior optimization capabilities.

2. **PE/SIMD Generalization Limitation**: The current PE/SIMD system's limitation to MVAU/VVAU operations is **a crucial gap and primary goal of the upcoming refactor**. Generalizing parallelism concepts for arbitrary operations with different parallelism dimensions is a key architectural challenge.

3. **Resource Estimation Validation**: The current system **relies on manual validation for each model**. Automated resource estimation analysis is an important future task but not within the current project scope.

4. **Template System Evolution**: While template system automation is a valuable consideration, it is **beyond the scope of the current project**.

### Implementation and Optimization Insights

5. **Mixed-Precision Support**: FINN supports mixed precision at multiple levels:
   - **Inter-Operation Granularity**: Each accelerator contains many HWCustomOps with different datatypes
   - **Intra-Operation Granularity**: Individual HWCustomOps with multiple inputs/weights can use mixed precision
   - **Static Operation Design**: Each HWCustomOp typically uses static datatypes internally

6. **Memory Subsystem Integration Gap**: **Memory bandwidth awareness is a crucial gap in current optimization algorithms** and is mostly handled manually. This represents a key consideration for the upcoming refactor design.

### Development Experience Limitations

7. **Debugging Infrastructure Gaps**:
   - **No uniform debugging flow** for simulation mismatches across C++/RTL/hardware phases
   - **No visualization tools** for PE/SIMD mapping and resource utilization (identified as wonderful future addition)
   - **Extremely manual development workflow** requiring in-depth FINN expertise

8. **Development Workflow Automation Need**: **Streamlining, canonizing, and highly automating the development process** for new operations is another primary goal of the refactor.

