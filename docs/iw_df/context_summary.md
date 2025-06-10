# Interface-Wise Dataflow Modeling: Context Summary Document

## Executive Summary

This document provides a comprehensive summary of the context, requirements, and technical foundations for developing an Interface-Wise Dataflow Modeling framework for the FINN/Brainsmith ecosystem. The framework aims to simplify HW Kernel design by abstracting hardware implementations to relationships between standardized interfaces, enabling automated generation of HWCustomOp classes and streamlining the development workflow for hardware acceleration.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Current FINN/Brainsmith Architecture](#current-finnbrainsmith-architecture)
3. [Existing Tools and Infrastructure](#existing-tools-and-infrastructure)
4. [Technical Requirements](#technical-requirements)
5. [Design Constraints and Decisions](#design-constraints-and-decisions)
6. [Key Reference Documents](#key-reference-documents)
7. [Expert Consultation Results](#expert-consultation-results)

## Project Overview

### Mission Statement
Develop an Interface-Wise Dataflow Modeling framework that provides a unified abstraction for HW Kernels, enabling automated generation of FINN HWCustomOp classes and significantly reducing the complexity of integrating custom RTL implementations into the FINN acceleration workflow.

### Primary Goals
1. **Automated HWCustomOp Generation**: Generate complete, functional HWCustomOp and RTLBackend classes automatically from RTL analysis
2. **Standardized Interface Abstraction**: Provide consistent interface-based abstraction across all kernel types
3. **Generalized Parallelism Model**: Extend beyond MVAU/VVAU-specific PE/SIMD to support arbitrary parallelism patterns
4. **Simplified Development Workflow**: Reduce manual implementation effort for new kernel integration
5. **Future FINN Refactoring Foundation**: Serve as architectural basis for modernizing FINN's HWCustomOp system

### Success Metrics
- Complete automation of HWCustomOp generation for standard interface patterns
- Successful integration and validation using thresholding_axi as reference implementation
- Clear migration path for existing FINN kernels to new framework
- Comprehensive test coverage ensuring reliability and robustness

## Current FINN/Brainsmith Architecture

### FINN HWCustomOp System Architecture

#### Three-Tier Architecture
The current FINN system implements a sophisticated three-tier architecture:

1. **Tier 1: Generic HWCustomOp Base Class**
   - Abstract method definitions for shape inference, datatype handling, stream width calculation
   - Resource estimation interfaces (BRAM, LUT, DSP, URAM)
   - RTL simulation infrastructure (pyxsi integration)
   - Common node attribute management
   - AXI interface protocol handling

2. **Tier 2: Backend Interface Classes**
   - **HLSBackend**: C++ code generation with template substitution, HLS pragma management
   - **RTLBackend**: Direct Verilog/SystemVerilog generation, FPGA-specific optimization

3. **Tier 3: Concrete Implementations**
   - Specific operation implementations inheriting from both generic operation and backend interface
   - Examples: `ChannelwiseOp_hls`, `FMPadding_rtl`

#### Key Design Principles
- **Abstraction Separation**: Clear separation between generic logic, backend-specific implementation, and hardware generation
- **Backend Flexibility**: Support for both HLS and RTL implementation approaches
- **Template-Driven Generation**: Parameterized code generation using template substitution
- **Streaming Dataflow**: AXI-Stream based interfaces for composable hardware modules
- **Resource-Aware Optimization**: Built-in resource estimation and parallelism optimization

### PE/SIMD Parallelism System

#### Mathematical Foundations
Current system implements sophisticated parallelism optimization:

- **PE (Processing Elements)**: Controls output parallelism (how many output elements computed simultaneously)
- **SIMD (Single Instruction Multiple Data)**: Controls input parallelism (how many input elements processed simultaneously)
- **Folding Factors**: Time multiplexing determining clock cycles needed
- **Constraints**: Mathematical divisibility requirements (MW % SIMD == 0, MH % PE == 0)

#### Cycle Calculations
```python
# For MVAU operations
exp_cycles = (MH / PE) * (MW / SIMD) * np.prod(num_input_vectors) / mmv

# Memory organization
omega = (MW * MH) / (PE * SIMD)  # Number of weight words stored
mem_width = SIMD * weight_bitwidth * PE  # Memory interface width
```

#### Optimization Strategies
- **SetFolding Transformation**: Intelligent optimization algorithm balancing performance and resources
- **Target-Driven Optimization**: User-specified cycles per frame with automatic parameter adjustment
- **Multi-Pass Relaxation**: Adaptive target adjustment based on achievable performance
- **Pipeline Balancing**: Bottleneck analysis and resource redistribution

#### Current Limitations
- **Limited Scope**: PE/SIMD system primarily designed for MVAU/VVAU operations
- **Manual Configuration**: Significant manual effort required for new operation types
- **Constraint Rigidity**: Mathematical constraints limit flexibility for diverse operation patterns

## Existing Tools and Infrastructure

### RTL Parser System

#### Architecture Overview
Sophisticated SystemVerilog analysis tool built using tree-sitter:

- **RTLParser**: Main orchestrator using tree-sitter for robust parsing
- **Grammar Handler**: SystemVerilog grammar loading via ctypes
- **Pragma Handler**: Brainsmith pragma extraction and validation
- **Interface Builder**: Coordinates interface identification
- **Interface Scanner**: Groups ports by naming conventions
- **Protocol Validator**: Validates interface protocol compliance

#### Data Structures
Comprehensive type system for representing parsed RTL:

```python
# Core Types
HWKernel: Top-level kernel representation
Parameter: Module parameters with template mapping  
Port: Module ports with direction and width
Interface: Validated interface groups (Global, AXI-Stream, AXI-Lite)
Pragma: Parsed Brainsmith directives

# Interface Types
GLOBAL_CONTROL: Clock/reset signals (clk, rst_n, optional clk2x)
AXI_STREAM: Data flow interfaces (TDATA, TVALID, TREADY, optional TLAST)
AXI_LITE: Configuration interfaces (supports partial implementations)

# Pragma Types  
TOP_MODULE: Specify target module when multiple exist
DATATYPE: Define supported datatypes for interfaces
DERIVED_PARAMETER: Add computed parameters from Python expressions
WEIGHT: Mark interfaces as weight inputs
```

#### Processing Pipeline
1. **Syntax Parsing**: Tree-sitter AST generation with error detection
2. **Module Selection**: Identify target module via pragma or heuristics
3. **Component Extraction**: Extract parameters, ports, pragmas
4. **Interface Analysis**: Scanner groups ports, validator checks compliance, builder creates validated objects
5. **Pragma Application**: Apply pragma effects to kernel data
6. **Validation**: Ensure all ports assigned to valid interfaces

### Hardware Kernel Generator (HKG)

#### Current Capabilities
Orchestrates integration of custom RTL designs into FINN/Brainsmith ecosystem:

- **RTL Analysis**: Full RTL Parser pipeline execution
- **Compiler Data Processing**: Dynamic Python module importing and validation
- **Code Generation**: RTL template generation using Jinja2
- **Integration Pipeline**: FINN-compatible class generation (planned)

#### Architecture Components
1. **HardwareKernelGenerator**: Main orchestrator class
2. **RTL Template Generator**: Jinja2-based wrapper generation  
3. **RTL Parser Integration**: Leverages full RTL Parser pipeline
4. **Compiler Data Handler**: Python module loading and AST analysis

#### Current Limitations
- **Limited Generation**: Only RTL templates currently implemented
- **Manual HWCustomOp Creation**: No automated HWCustomOp generation
- **Resource Estimation Gaps**: No automated resource modeling
- **FINN Integration Incomplete**: Missing HWCustomOp and RTLBackend generation

### Existing Brainsmith Extensions

#### Custom Operations
Current Brainsmith includes several custom operations:

- **HWSoftmax**: Softmax implementation with SIMD parallelism
- **LayerNorm**: Layer normalization with configurable precision
- **Shuffle**: Data reorganization operations
- **Crop**: Tensor cropping functionality

#### Design Patterns
Analysis of existing implementations reveals common patterns:

```python
# Typical HWCustomOp structure
class HWSoftmax(HWCustomOp):
    def get_nodeattr_types(self): # Define operation-specific attributes
    def get_normal_input_shape(self, ind=0): # ONNX tensor shapes
    def get_folded_input_shape(self, ind=0): # Hardware-folded shapes  
    def get_input_datatype(self, ind=0): # FINN DataType handling
    def get_instream_width(self, ind=0): # Stream width calculations
    def execute_node(self, context, graph): # Functional simulation
    def verify_node(self): # Constraint validation
```

## Technical Requirements

### Framework Requirements

#### Standalone Framework
- **Independence**: Framework must function independently of FINN for initial development
- **Future Integration**: Design for seamless future FINN integration without architectural changes
- **Modular Design**: Clean separation between framework core and integration layers

#### Interface Protocol Support
- **Standard Protocols Only**: AXI-Stream, AXI-Lite, Global Control interfaces
- **Protocol Compliance**: Full validation against standard protocol requirements
- **Naming Convention Support**: Flexible naming pattern matching for interface identification

#### Parallelism Parameter Mapping
- **Direct Mapping**: wPar = PE, iPar = SIMD for seamless future migration
- **Generalized Support**: Framework must work for any kernel type, not just MVAU/VVAU
- **Constraint Preservation**: Maintain mathematical divisibility and resource constraints

#### Method Standardization
- **Base Class Implementation**: Almost all HWCustomOp methods implemented in AutoHWCustomOp base class
- **Override Capability**: Allow user override of standardized methods when necessary
- **Resource Estimation Exception**: Resource estimation methods remain user-implemented

### Data Hierarchy Requirements

#### Interface Types
- **Input**: AXI-Stream input, streaming activation data into kernel
- **Output**: AXI-Stream output, streaming activation data out of kernel  
- **Weight**: AXI-Stream input, streaming weight data into kernel
- **Config**: AXI-Lite input/output/inout, streaming configuration data
- **Control**: Input control signals (clk, rst, optional clk2x)

#### Data Organization
- **Tensor**: Complete data of entire hidden state or weight streamed through interface (full tensor dimensions)
- **Block**: Data for single calculation in kernel (minimum required for computation)
- **Stream**: Data streamed each clock cycle
- **Element**: Single value with bitwidth defined by interface datatype

#### Constraint Requirements
- Each data level must tile into the next (stream → block → tensor)
- Tensor dimensions defined by ONNX pattern
- Block dimensions defined by calculation requirements
- Stream dimensions determined by parallelism parameters

### Computational Model Requirements

#### Mathematical Relationships
Core computational model must implement:

```python
# Parallelism parameters
iPar: Input parallelism (corresponds to SIMD)
wPar: Weight parallelism (corresponds to PE)

# Initiation intervals
cII: Calculation Initiation Interval (cycles per calculation)
eII: Execution Initiation Interval (cycles per execution)  
L: Inference Cycle Latency (cycles per inference)

# Simple case relationships
stream_dims_I = iPar
stream_dims_W = wPar * iPar * (block_dims_W / block_dims_I)
stream_dims_O = stream_dims_I * (block_dims_O / block_dims_I)
cII = ∏(block_dims_I / stream_dims_I)
eII = cII * ∏(tensor_dims_W / wPar)
L = eII * ∏(tensor_dims_I)
```

#### Multi-Interface Support
- Support arbitrary number of Input, Output, Weight interfaces
- Bottleneck calculation for mixed interface scenarios
- Constraint validation across all interface combinations

### Code Generation Requirements

#### AutoHWCustomOp Generation
- **Complete Class Generation**: Fully functional HWCustomOp subclasses
- **Template-Based**: Jinja2 templates for consistent generation
- **Metadata Integration**: RTL Parser metadata embedded in generated code
- **Validation Integration**: Generated code must pass FINN validation

#### Resource Estimation Integration
- **Placeholder Methods**: Clear error messages for unimplemented resource estimation
- **User Implementation Points**: Obvious extension points for user-provided estimation
- **Future Automation Ready**: Architecture ready for automated resource estimation

### Testing and Validation Requirements

#### Reference Implementation
- **Thresholding Validation**: thresholding_axi as primary test case
- **Complete Pipeline**: End-to-end validation from RTL to generated HWCustomOp
- **Functional Verification**: Generated code must produce correct results

#### Test Coverage
- **Comprehensive Testing**: All framework components thoroughly tested
- **Edge Case Coverage**: Boundary conditions and error cases
- **Integration Testing**: Full pipeline validation with real RTL examples
- **Performance Testing**: Validation of computational model accuracy

## Design Constraints and Decisions

### Architectural Constraints

#### FINN Compatibility
- **Future Integration**: Design must enable seamless FINN refactoring
- **Backward Compatibility**: Generated code compatible with existing FINN workflows
- **Migration Path**: Clear upgrade path for existing HWCustomOp implementations

#### Technology Constraints
- **Python 3.8+**: Minimum Python version requirement
- **Existing Dependencies**: Leverage existing RTL Parser and HKG infrastructure
- **Template System**: Jinja2 for code generation consistency

### Design Decisions

#### Framework Independence
**Decision**: Implement as standalone framework independent of FINN
**Rationale**: Enables rapid development and testing without FINN dependencies
**Implications**: Requires clear integration strategy for future FINN adoption

#### Standard Protocol Focus
**Decision**: Support only standard AXI protocols initially
**Rationale**: Reduces complexity while covering majority of use cases
**Implications**: Custom protocols require future extension

#### Parallelism Parameter Mapping
**Decision**: Direct mapping wPar=PE, iPar=SIMD
**Rationale**: Enables seamless migration of existing FINN optimizations
**Implications**: Framework inherits PE/SIMD mathematical constraints

#### Resource Estimation Strategy
**Decision**: User-implemented resource estimation with placeholders
**Rationale**: Automated estimation requires significant research effort
**Implications**: Generated code requires manual completion for production use

#### TDIM Pragma Extension
**Decision**: Add TDIM pragma for complex tensor dimension specification
**Rationale**: Provides flexibility for non-standard tensor chunking patterns
**Implications**: Requires RTL Parser extension and validation logic

### Quality and Performance Constraints

#### Code Quality
- **Production Ready**: Generated code must meet production quality standards
- **Documentation**: Comprehensive inline documentation and comments
- **Error Handling**: Robust error detection and reporting
- **Maintainability**: Clear code structure and logical organization

#### Performance Requirements
- **Generation Speed**: Fast code generation for rapid iteration
- **Runtime Performance**: Generated HWCustomOp must have minimal overhead
- **Memory Efficiency**: Reasonable memory usage for large kernels

## Key Reference Documents

### Primary Sources
1. **[FINN HWCustomOp Analysis](../hw_custom_op_analysis.md)**: Comprehensive analysis of current FINN architecture
2. **[PE/SIMD Parallelism Analysis](../pe_simd_parallelism_analysis.md)**: Mathematical foundations and optimization strategies
3. **[RTL Parser Documentation](../RTL_Parser.md)**: Current RTL Parser capabilities and data structures
4. **[HW Kernel Generator Documentation](../HW_Kernel_Gen.md)**: Current HKG architecture and limitations
5. **[Dataflow Modeling Specification](../dataflow_modeling.md)**: Proposed interface-wise modeling framework
6. **[Integration Requirements](../interface-wise_dataflow_modeling_prompt.md)**: Integration goals and workflow requirements

### Supporting Materials
1. **RTL Parser Prompts**: Original requirements and development guidelines
2. **HW Kernel Generator Prompts**: Integration objectives and scope definition
3. **Example Implementations**: thresholding_axi and existing Brainsmith operations
4. **FINN Developer Documentation**: Architectural context and development practices

### Code References
1. **RTL Parser Data Structures**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
2. **HKG Data Structures**: `brainsmith/tools/hw_kernel_gen/data.py`
3. **Example HWCustomOp**: `brainsmith/custom_op/fpgadataflow/hwsoftmax.py`
4. **Test Infrastructure**: `tests/tools/hw_kernel_gen/` directory structure

## Expert Consultation Results

### Scope and Implementation Strategy

#### Framework Independence Confirmation
**Expert Guidance**: Implement as completely standalone framework for independent development
**Impact**: Enables rapid prototyping and validation without FINN infrastructure dependencies
**Implementation**: Clean separation between framework core and future FINN integration layers

#### Protocol Support Scope
**Expert Guidance**: Focus on standard interface protocols only (AXI-Stream, AXI-Lite, Global Control)
**Impact**: Reduces initial complexity while covering majority of real-world use cases
**Implementation**: Comprehensive support for standard protocols with clear extension points

#### Migration Strategy
**Expert Guidance**: AutoHWCustomOp will eventually replace current HWCustomOp system
**Impact**: Framework must support complete method standardization with override capability
**Implementation**: Comprehensive base class with standardized method implementations

### Parallelism and Optimization

#### Parameter Mapping Strategy
**Expert Guidance**: Direct mapping wPar=PE, iPar=SIMD for seamless migration
**Impact**: Preserves existing optimization algorithms and mathematical constraints
**Implementation**: Framework inherits proven PE/SIMD optimization strategies

#### Generalization Requirements
**Expert Guidance**: Framework must work for any kernel type, not just MVAU/VVAU
**Impact**: Requires flexible parallelism model supporting diverse operation patterns
**Implementation**: Abstract parallelism concepts with kernel-specific constraint validation

#### TDIM Pragma Necessity
**Expert Guidance**: TDIM pragma required for complex tensor chunking cases
**Impact**: Provides necessary flexibility for non-standard operation patterns
**Implementation**: RTL Parser extension with pragma validation and application logic

### Resource Estimation and Quality

#### Resource Estimation Strategy
**Expert Guidance**: Resource estimation remains user-implemented with automation placeholders
**Impact**: Generated code requires manual completion but provides clear extension points
**Implementation**: Comprehensive placeholder methods with error reporting and documentation

#### Testing and Validation Requirements
**Expert Guidance**: Thorough test coverage using thresholding_axi as primary validation case
**Impact**: Ensures framework reliability and provides concrete validation reference
**Implementation**: Comprehensive test suite covering all framework components and integration scenarios

#### Quality Standards
**Expert Guidance**: Production-quality code generation with comprehensive documentation
**Impact**: Generated code must be maintainable and meet professional standards
**Implementation**: High-quality templates with extensive inline documentation and error handling

### Development Priorities

#### New Kernel Focus
**Expert Guidance**: Prioritize new kernel integration over legacy kernel migration
**Impact**: Framework optimized for streamlined development of new operations
**Implementation**: Development workflow designed for rapid new kernel integration

#### Future Automation Readiness
**Expert Guidance**: Architecture should support future resource estimation automation
**Impact**: Framework design must accommodate future automated analysis capabilities
**Implementation**: Extensible architecture with clear automation integration points

---

## Summary

This context summary provides the comprehensive foundation for developing the Interface-Wise Dataflow Modeling framework. The framework addresses critical gaps in the current FINN/Brainsmith architecture while maintaining compatibility with existing systems and providing a clear migration path for future development.

The technical requirements balance immediate needs for automated HWCustomOp generation with long-term goals for comprehensive FINN modernization. The design decisions reflect practical constraints while maintaining architectural flexibility for future enhancements.

The expert consultation results provide clear guidance on scope, priorities, and implementation strategies, ensuring the framework development aligns with broader project objectives and technical requirements.

This foundation enables confident progression to detailed architectural design and implementation planning, with clear understanding of requirements, constraints, and success criteria.
