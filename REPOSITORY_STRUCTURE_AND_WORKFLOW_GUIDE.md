# Brainsmith-2 Repository Structure and Workflow Guide

## Table of Contents
1. [Repository Overview](#repository-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Interaction Diagrams](#component-interaction-diagrams)
4. [Detailed Module Structure](#detailed-module-structure)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Workflow Sequences](#workflow-sequences)
7. [Development Workflows](#development-workflows)

## Repository Overview

Brainsmith-2 is an open-source platform for FPGA AI accelerators that automates the conversion of PyTorch models to RTL implementations for FPGA deployment using Interface-Wise Dataflow Modeling.

### Repository Structure at a Glance

```
brainsmith-2/
├── 📁 brainsmith/                    # Core framework
│   ├── 📁 dataflow/                  # Interface-wise modeling framework
│   ├── 📁 tools/hw_kernel_gen/       # Hardware kernel generator
│   ├── 📁 custom_op/                 # Custom FPGA operations
│   ├── 📁 hw_kernels/                # Hardware kernel implementations
│   └── 📁 transformation/            # Model transformation utilities
├── 📁 demos/                         # End-to-end demonstrations
├── 📁 examples/                      # Usage examples and tutorials
├── 📁 tests/                         # Comprehensive test suite
├── 📁 docs/                          # Documentation and guides
└── 📁 docker/                        # Containerized development environment
```

## High-Level Architecture

### System Overview Diagram

```mermaid
graph TB
    subgraph "Input Layer"
        A[PyTorch Model] --> B[Brevitas Quantization]
        C[SystemVerilog RTL] --> D[RTL Parser]
    end
    
    subgraph "Core Processing Layer"
        B --> E[ONNX Model]
        D --> F[Interface Metadata]
        E --> G[Dataflow Framework]
        F --> G
        G --> H[Hardware Kernel Generator]
    end
    
    subgraph "Output Layer"
        H --> I[FINN Integration Files]
        H --> J[Verilog Wrappers]
        H --> K[Test Suites]
        I --> L[FPGA Deployment]
        J --> L
    end
    
    subgraph "Support Systems"
        M[Validation Framework] --> G
        N[Template System] --> H
        O[Error Handling] --> H
    end
    
    style A fill:#e1f5fe
    style C fill:#e1f5fe
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style L fill:#e8f5e8
```

### Technology Stack

```mermaid
graph LR
    subgraph "Languages & Frameworks"
        A[Python 3.12] --> B[PyTorch 2.7]
        A --> C[ONNX]
        A --> D[Jinja2]
        E[SystemVerilog] --> F[Tree-sitter]
        G[VHDL/Verilog] --> H[Vivado HLS]
    end
    
    subgraph "FPGA Ecosystem"
        I[FINN Framework] --> J[Brevitas]
        I --> K[QONNX]
        L[Xilinx Vivado] --> M[AMD/Xilinx FPGAs]
    end
    
    subgraph "Development Tools"
        N[Docker] --> O[pytest]
        N --> P[Makefile]
        Q[Git] --> R[GitHub Actions]
    end
```

## Component Interaction Diagrams

### Core System Interactions

```mermaid
graph TB
    subgraph "External Inputs"
        A["User RTL File<br/>(.sv/.v)"]
        B["Compiler Data<br/>(JSON)"]
        C["ONNX Model<br/>(Optional)"]
    end
    
    subgraph "RTL Processing Pipeline"
        A --> D[RTL Parser]
        D --> E[Interface Scanner]
        E --> F[Interface Builder]
        F --> G[Protocol Validator]
        G --> H[Pragma Processor]
    end
    
    subgraph "Dataflow Framework"
        H --> I[RTL Conversion]
        B --> I
        C --> I
        I --> J[DataflowInterface]
        J --> K[DataflowModel]
        K --> L[Validation Engine]
    end
    
    subgraph "Code Generation"
        L --> M[Template Manager]
        M --> N[HWCustomOp Generator]
        M --> O[RTLBackend Generator]
        M --> P[Wrapper Generator]
        M --> Q[Test Generator]
    end
    
    subgraph "Output Artifacts"
        N --> R[Python Classes]
        O --> S[FINN Integration]
        P --> T[Verilog Wrappers]
        Q --> U[Test Suites]
    end
    
    style D fill:#ffeb3b
    style I fill:#4caf50
    style M fill:#2196f3
```

### Dataflow Framework Internal Architecture

```mermaid
graph TB
    subgraph "Interface Layer"
        A[DataflowInterface<br/>Core Abstraction] --> B[DataflowDataType<br/>Type System]
        A --> C[Constraint System<br/>Validation Rules]
        A --> D[AXI Signal Generation<br/>Protocol Mapping]
    end
    
    subgraph "Model Layer"
        E[DataflowModel<br/>Computational Model] --> F[InitiationInterval<br/>Performance Calc]
        E --> G[ParallelismConfig<br/>Optimization]
        E --> H[ResourceRequirements<br/>Estimation]
    end
    
    subgraph "Auto-Generation Layer"
        I[AutoHWCustomOp<br/>Base Class] --> J[Runtime Shape<br/>Extraction]
        I --> K[FINN Integration<br/>Methods]
        I --> L[Resource Estimation<br/>Algorithms]
        M[AutoRTLBackend<br/>Base Class] --> N[Synthesis Flow<br/>Integration]
        M --> O[Build System<br/>Automation]
    end
    
    subgraph "Utility Layer"
        P[TensorChunking<br/>Shape Management] --> Q[ChunkingStrategy<br/>Algorithms]
        P --> R[ONNX Integration<br/>Shape Inference]
        S[ClassNaming<br/>Conventions] --> T[CamelCase<br/>Generation]
        U[Validation<br/>Framework] --> V[Error Reporting<br/>Context Management]
    end
    
    A --> E
    E --> I
    E --> M
    I --> P
    M --> P
    P --> U
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style I fill:#fff3e0
    style M fill:#fff3e0
```

## Detailed Module Structure

### Brainsmith Core Structure

```
brainsmith/
├── 📁 dataflow/                           # Interface-Wise Dataflow Framework
│   ├── 📄 API_REFERENCE.md               # API documentation
│   ├── 📄 README.md                      # Framework overview
│   ├── 📁 core/                          # Core framework components
│   │   ├── 📄 dataflow_interface.py      # ⭐ Core interface abstraction
│   │   ├── 📄 dataflow_model.py          # ⭐ Computational modeling
│   │   ├── 📄 auto_hw_custom_op.py       # ⭐ Generated HWCustomOp base
│   │   ├── 📄 auto_rtl_backend.py        # ⭐ Generated RTLBackend base
│   │   ├── 📄 validation.py              # Constraint validation
│   │   ├── 📄 tensor_chunking.py         # Tensor shape utilities
│   │   ├── 📄 interface_metadata.py      # Metadata containers
│   │   └── 📄 class_naming.py            # Naming conventions
│   ├── 📁 integration/                   # External system integration
│   │   └── 📄 rtl_conversion.py          # RTL → Dataflow conversion
│   └── 📁 examples/                      # Usage examples
│       └── 📄 basic_usage.py             # Framework demonstration
│
├── 📁 tools/hw_kernel_gen/                # Hardware Kernel Generator
│   ├── 📄 README.md                      # HWKG documentation
│   ├── 📄 hkg.py                         # ⭐ Main CLI entry point
│   ├── 📄 errors.py                      # Error handling framework
│   ├── 📁 rtl_parser/                    # SystemVerilog parsing engine
│   │   ├── 📄 README.md                  # Parser documentation
│   │   ├── 📄 parser.py                  # ⭐ Main parsing logic
│   │   ├── 📄 interface_scanner.py       # Interface discovery
│   │   ├── 📄 interface_builder.py       # Interface construction
│   │   ├── 📄 protocol_validator.py      # AXI protocol validation
│   │   ├── 📄 pragma.py                  # Pragma extraction
│   │   ├── 📄 grammar.py                 # Tree-sitter grammar
│   │   └── 📄 sv.so                      # Compiled grammar binary
│   ├── 📁 generators/                    # Code generation engines
│   │   ├── 📄 enhanced_hw_custom_op_generator.py  # ⭐ HWCustomOp generation
│   │   ├── 📄 enhanced_rtl_backend_generator.py   # ⭐ RTLBackend generation
│   │   ├── 📄 hw_custom_op_generator.py           # Legacy generator
│   │   └── 📄 rtl_template_generator.py           # Verilog wrapper gen
│   ├── 📁 templates/                     # Jinja2 template system
│   │   ├── 📄 hw_custom_op_slim.py.j2    # ⭐ Minimal HWCustomOp template
│   │   ├── 📄 rtl_backend.py.j2          # ⭐ RTLBackend template
│   │   ├── 📄 rtl_wrapper.v.j2           # Verilog wrapper template
│   │   ├── 📄 test_suite.py.j2           # Test generation template
│   │   └── 📄 documentation.md.j2        # Documentation template
│   ├── 📁 orchestration/                 # Pipeline coordination
│   │   ├── 📄 pipeline_orchestrator.py   # ⭐ Main pipeline controller
│   │   ├── 📄 generation_workflow.py     # Workflow definitions
│   │   ├── 📄 generator_factory.py       # Generator instantiation
│   │   └── 📄 integration_orchestrator.py # Integration management
│   ├── 📁 analysis/                      # Advanced analysis tools
│   │   ├── 📄 enhanced_interface_analyzer.py  # Interface analysis
│   │   └── 📄 enhanced_pragma_processor.py    # Pragma processing
│   └── 📁 compatibility/                 # Backward compatibility
│       ├── 📄 backward_compatibility.py  # Legacy support
│       └── 📄 legacy_adapter.py          # Legacy system adapters
```

⭐ = Critical components for code review

### Custom Operations Structure

```
custom_op/
├── 📁 fpgadataflow/                      # FPGA-specific dataflow operations
│   ├── 📄 brainsmith_hlsbackend.py      # HLS backend integration
│   ├── 📄 brainsmith_templates.py       # Template utilities
│   ├── 📁 hls/                          # HLS kernel implementations
│   │   ├── 📄 crop_hls.py               # Crop operation HLS
│   │   ├── 📄 hwsoftmax_hls.py          # Softmax operation HLS
│   │   ├── 📄 layernorm_hls.py          # LayerNorm operation HLS
│   │   └── 📄 shuffle_hls.py            # Shuffle operation HLS
│   ├── 📄 crop.py                       # Crop operation
│   ├── 📄 hwsoftmax.py                  # Hardware softmax
│   ├── 📄 layernorm.py                  # Layer normalization
│   └── 📄 shuffle.py                    # Channel shuffle
└── 📁 general/                          # General-purpose operations
    └── 📄 norms.py                      # Normalization operations
```

### Hardware Kernels Structure

```
hw_kernels/
├── 📁 hls/                              # High-Level Synthesis headers
│   ├── 📄 bs_utils.hpp                  # Brainsmith utilities
│   ├── 📄 input_gen.hpp                 # Input generation utilities
│   ├── 📄 layernorm.hpp                 # LayerNorm implementation
│   └── 📄 softmax.hpp                   # Softmax implementation
└── 📁 rtl/                              # RTL module implementations
    └── 📄 README.md                     # RTL documentation
```

## Data Flow Diagrams

### End-to-End Pipeline Data Flow

```mermaid
flowchart TD
    subgraph "Input Stage"
        A[SystemVerilog RTL<br/>thresholding.sv] --> A1[RTL Text Content]
        B[Compiler Data<br/>config.json] --> B1[Configuration Parameters]
        C[ONNX Model<br/>model.onnx] --> C1[Model Metadata<br/>Optional]
    end
    
    subgraph "Parsing Stage"
        A1 --> D[Tree-sitter Parser]
        D --> D1[Abstract Syntax Tree]
        D1 --> E[Interface Scanner]
        E --> E1[Raw Interface Data]
        E1 --> F[Interface Builder]
        F --> F1[RTLInterface Objects]
    end
    
    subgraph "Analysis Stage"
        F1 --> G[Protocol Validator]
        B1 --> G
        G --> G1[Validated Interfaces]
        G1 --> H[Pragma Processor]
        H --> H1[Enhanced Interface Metadata]
    end
    
    subgraph "Conversion Stage"
        H1 --> I[RTL Conversion Engine]
        C1 --> I
        I --> I1[DataflowInterface Objects]
        I1 --> J[DataflowModel Builder]
        J --> J1[Complete DataflowModel]
    end
    
    subgraph "Validation Stage"
        J1 --> K[Constraint Validator]
        K --> K1[Validation Results]
        K1 --> L{Valid?}
        L -->|No| M[Error Reporter]
        L -->|Yes| N[Template Context Builder]
    end
    
    subgraph "Generation Stage"
        N --> N1[Template Context]
        N1 --> O[Template Manager]
        O --> P[HWCustomOp Generator]
        O --> Q[RTLBackend Generator]
        O --> R[Wrapper Generator]
        O --> S[Test Generator]
    end
    
    subgraph "Output Stage"
        P --> T[Python HWCustomOp Class]
        Q --> U[Python RTLBackend Class]
        R --> V[Verilog Wrapper Module]
        S --> W[Test Suite Files]
        T --> X[FINN Integration Package]
        U --> X
        V --> X
        W --> X
    end
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style I fill:#f3e5f5
    style J fill:#f3e5f5
    style O fill:#fff3e0
    style X fill:#e8f5e8
```

### Interface Processing Detail Flow

```mermaid
flowchart LR
    subgraph "RTL Interface Processing"
        A[Raw RTL Signals] --> B[Signal Classification]
        B --> C{Signal Type?}
        C -->|Clock/Reset| D[Control Interface]
        C -->|AXI-Stream| E[Data Interface]
        C -->|AXI-Lite| F[Config Interface]
        C -->|Other| G[Skip/Warning]
    end
    
    subgraph "Interface Metadata Extraction"
        D --> H[Extract Control Metadata]
        E --> I[Extract Stream Metadata]
        F --> J[Extract Config Metadata]
        H --> K[Control InterfaceMetadata]
        I --> L[Stream InterfaceMetadata]
        J --> M[Config InterfaceMetadata]
    end
    
    subgraph "Dataflow Conversion"
        K --> N[Create Control DataflowInterface]
        L --> O[Create Stream DataflowInterface]
        M --> P[Create Config DataflowInterface]
        N --> Q[Validate Control Interface]
        O --> R[Validate Stream Interface]
        P --> S[Validate Config Interface]
    end
    
    subgraph "Model Integration"
        Q --> T[Add to DataflowModel]
        R --> T
        S --> T
        T --> U[Complete Model Validation]
        U --> V[Ready for Code Generation]
    end
    
    style A fill:#ffebee
    style B fill:#fff3e0
    style T fill:#e8f5e8
```

## Workflow Sequences

### HWKG Generation Sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI as HKG CLI
    participant Parser as RTL Parser
    participant Converter as RTL Converter
    participant Dataflow as Dataflow Framework
    participant Generator as Code Generator
    participant FileSystem as File System
    
    User->>CLI: hkg input.sv config.json -o output/
    CLI->>Parser: parse_rtl_file(input.sv)
    Parser->>Parser: tokenize_and_parse()
    Parser->>Parser: extract_interfaces()
    Parser->>Parser: validate_protocols()
    Parser-->>CLI: RTLInterface[]
    
    CLI->>Converter: convert_to_dataflow(RTLInterface[])
    Converter->>Dataflow: create_dataflow_interfaces()
    Dataflow->>Dataflow: validate_constraints()
    Dataflow-->>Converter: DataflowInterface[]
    Converter->>Dataflow: create_dataflow_model()
    Dataflow-->>CLI: DataflowModel
    
    CLI->>Generator: generate_code(DataflowModel)
    Generator->>Generator: build_template_context()
    Generator->>Generator: render_templates()
    Generator->>FileSystem: write_hwcustomop.py
    Generator->>FileSystem: write_rtlbackend.py
    Generator->>FileSystem: write_wrapper.v
    Generator->>FileSystem: write_tests.py
    Generator-->>CLI: GenerationResult
    
    CLI-->>User: Success + file paths
```

### Dataflow Model Building Sequence

```mermaid
sequenceDiagram
    participant Client
    participant AutoHWCustomOp as AutoHWCustomOp
    participant ModelWrapper as FINN ModelWrapper
    participant TensorChunking as TensorChunking
    participant DataflowModel as DataflowModel
    participant Validation as Validation
    
    Client->>AutoHWCustomOp: __init__(onnx_node, metadata)
    AutoHWCustomOp->>AutoHWCustomOp: _interface_metadata = metadata
    AutoHWCustomOp->>AutoHWCustomOp: _dataflow_model = None (lazy)
    
    Client->>AutoHWCustomOp: get_exp_cycles()
    AutoHWCustomOp->>AutoHWCustomOp: _ensure_dataflow_model_built()
    AutoHWCustomOp->>AutoHWCustomOp: _build_dataflow_model()
    
    AutoHWCustomOp->>TensorChunking: extract_tensor_shape()
    TensorChunking->>ModelWrapper: get_tensor_shape(interface_name)
    ModelWrapper-->>TensorChunking: runtime_shape
    TensorChunking-->>AutoHWCustomOp: actual_shape
    
    AutoHWCustomOp->>AutoHWCustomOp: compute_chunking(metadata, shape)
    AutoHWCustomOp->>DataflowModel: create_dataflow_interfaces()
    DataflowModel->>Validation: validate_constraints()
    Validation-->>DataflowModel: validation_result
    DataflowModel-->>AutoHWCustomOp: built_model
    
    AutoHWCustomOp->>DataflowModel: calculate_initiation_intervals()
    DataflowModel-->>AutoHWCustomOp: performance_metrics
    AutoHWCustomOp-->>Client: expected_cycles
```

### Error Handling Flow

```mermaid
flowchart TD
    A[Operation Start] --> B{Try Operation}
    B --> C[RTL Parsing]
    C --> D{Parse Success?}
    D -->|No| E[RTLParsingError]
    D -->|Yes| F[Interface Validation]
    F --> G{Validation Success?}
    G -->|No| H[ValidationError]
    G -->|Yes| I[Code Generation]
    I --> J{Generation Success?}
    J -->|No| K[GenerationError]
    J -->|Yes| L[Success]
    
    E --> M[Error Context Collection]
    H --> M
    K --> M
    M --> N[Error Message Formatting]
    N --> O[User-Friendly Error Report]
    O --> P[Cleanup & Exit]
    
    subgraph "Error Context"
        M1[File Location]
        M2[Line Numbers]
        M3[Interface Names]
        M4[Constraint Details]
        M5[Stack Trace]
    end
    
    M --> M1
    M --> M2
    M --> M3
    M --> M4
    M --> M5
    
    style E fill:#ffcdd2
    style H fill:#ffcdd2
    style K fill:#ffcdd2
    style O fill:#fff3e0
    style L fill:#e8f5e8
```

## Development Workflows

### Developer Workflow for Adding New RTL Kernels

```mermaid
flowchart TD
    subgraph "Development Phase"
        A[Create SystemVerilog Module] --> B[Add Interface Pragmas]
        B --> C[Test RTL Simulation]
        C --> D[Create Compiler Data JSON]
        D --> E[Run HWKG Generator]
    end
    
    subgraph "Validation Phase"
        E --> F[Verify Generated Code]
        F --> G[Run Unit Tests]
        G --> H[Integration Testing]
        H --> I{All Tests Pass?}
        I -->|No| J[Debug & Fix Issues]
        J --> F
        I -->|Yes| K[Code Review]
    end
    
    subgraph "Integration Phase"
        K --> L[Merge to Main Branch]
        L --> M[Update Documentation]
        M --> N[Release Notes]
        N --> O[Deploy to Production]
    end
    
    subgraph "Testing Tools"
        P[pytest Unit Tests]
        Q[Golden Reference Tests]
        R[BERT Demo End-to-End]
        S[Performance Benchmarks]
    end
    
    G --> P
    H --> Q
    H --> R
    H --> S
    
    style A fill:#e3f2fd
    style E fill:#f3e5f5
    style K fill:#fff3e0
    style O fill:#e8f5e8
```

### FINN Integration Workflow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant HWKG as HWKG Tool
    participant FINN as FINN Compiler
    participant Vivado as Vivado Synthesis
    participant FPGA as Target FPGA
    
    Dev->>HWKG: Generate RTL integration
    HWKG->>HWKG: Create HWCustomOp class
    HWKG->>HWKG: Create RTLBackend class
    HWKG->>HWKG: Generate Verilog wrapper
    HWKG-->>Dev: Integration package
    
    Dev->>FINN: Register custom operation
    FINN->>FINN: Load HWCustomOp class
    FINN->>FINN: Instantiate for model node
    FINN->>FINN: Extract runtime shapes
    FINN->>FINN: Configure parallelism
    
    FINN->>Vivado: Synthesize with RTLBackend
    Vivado->>Vivado: Place & Route
    Vivado->>Vivado: Generate bitstream
    Vivado-->>FINN: Synthesis results
    
    FINN->>FPGA: Deploy bitstream
    FPGA->>FPGA: Execute neural network
    FPGA-->>FINN: Execution results
    FINN-->>Dev: Performance metrics
```

### Testing Pipeline Workflow

```mermaid
flowchart LR
    subgraph "Continuous Integration"
        A[Git Push] --> B[GitHub Actions Trigger]
        B --> C[Docker Environment Setup]
        C --> D[Dependency Installation]
        D --> E[Code Quality Checks]
    end
    
    subgraph "Test Execution"
        E --> F[Unit Tests<br/>pytest tests/]
        F --> G[Integration Tests<br/>End-to-end scenarios]
        G --> H[Golden Reference Tests<br/>Generated code comparison]
        H --> I[Performance Tests<br/>BERT demo validation]
    end
    
    subgraph "Quality Gates"
        I --> J{All Tests Pass?}
        J -->|Yes| K[Code Coverage Check<br/>>90% required]
        K --> L[Static Analysis<br/>Code quality metrics]
        L --> M[Documentation Check<br/>API docs updated]
        M --> N[Merge Approval]
    end
    
    subgraph "Failure Handling"
        J -->|No| O[Generate Test Report]
        K -->|Fail| O
        L -->|Fail| O
        M -->|Fail| O
        O --> P[Notify Developer]
        P --> Q[Block Merge]
    end
    
    style N fill:#e8f5e8
    style Q fill:#ffcdd2
```

## Summary

This repository implements a sophisticated FPGA acceleration framework with the following key characteristics:

### 🏗️ **Architectural Highlights**
- **Modular Design**: Clear separation between parsing, modeling, and generation
- **Interface-Driven**: Everything flows through standardized interface abstractions
- **Runtime Adaptation**: Dynamic shape extraction eliminates static configuration
- **Base Class Inheritance**: 90% reduction in generated code complexity

### 🔄 **Data Flow Characteristics**
- **Linear Pipeline**: RTL → Parse → Model → Generate → Integrate
- **Validation Gates**: Multiple validation points prevent invalid configurations
- **Error Recovery**: Comprehensive error handling with actionable messages
- **Lazy Initialization**: On-demand model building for performance

### 🛠️ **Development Features**
- **Docker-First**: Consistent development environment
- **Test-Driven**: Comprehensive test coverage with multiple test types
- **Documentation-Rich**: Extensive guides and API documentation
- **CI/CD Ready**: Automated testing and quality gates

The system successfully bridges the gap between low-level RTL hardware and high-level neural network frameworks, providing an automated pathway from SystemVerilog modules to FPGA-deployed neural networks.