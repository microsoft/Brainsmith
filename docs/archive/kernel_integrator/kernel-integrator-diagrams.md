# Kernel Integrator Architecture Diagrams and Analysis

This document provides a comprehensive visual analysis of the kernel_integrator tool, including various diagrams and architectural descriptions.

## Table of Contents
1. [Overview](#overview)
2. [High-Level Architecture](#1-high-level-architecture-diagram)
3. [Data Flow](#2-data-flow-diagram)
4. [Class Hierarchy](#3-class-hierarchy-diagram)
5. [Pragma Processing](#4-pragma-processing-flow)
6. [Template Generation](#5-template-generation-flow)
7. [Interface Detection](#6-interface-detection-ascii-diagram)
8. [Parameter Linking](#7-parameter-linking-process)
9. [Processing Pipeline](#8-complete-processing-pipeline)
10. [Artifact Dependencies](#9-artifact-dependencies)
11. [Component Interaction](#10-component-interaction-matrix)

## Overview

The kernel_integrator is a tool that parses SystemVerilog RTL files and generates FINN-compatible HWCustomOp implementations. It follows a clean pipeline architecture:

```
RTL File → Parser → Metadata → Generator → Output Files
```

### Key Components

- **Entry Point & CLI**: Command-line interface for operations
- **RTL Parser System**: Tree-sitter based parser with multiple sub-components
- **Data Structures**: Core types for ports, parameters, and metadata
- **Pragma System**: Special comments that provide additional metadata
- **Code Generator**: Template-based code generation using Jinja2

## 1. High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "Input"
        RTL["SystemVerilog RTL File<br/>(.sv, .v)"]
    end
    
    subgraph "Kernel Integrator Pipeline"
        CLI["CLI<br/>(cli.py)"]
        
        subgraph "RTL Parser"
            Parser["RTLParser<br/>(parser.py)"]
            AST["ASTParser<br/>(tree-sitter)"]
            ModEx["ModuleExtractor"]
            KB["KernelBuilder"]
            PL["ParameterLinker"]
            
            Parser --> AST
            AST --> ModEx
            ModEx --> KB
            KB --> PL
        end
        
        Metadata["KernelMetadata<br/>(metadata.py)"]
        
        Generator["KernelGenerator<br/>(generator.py)"]
        
        CLI --> Parser
        PL --> Metadata
        Metadata --> Generator
    end
    
    subgraph "Outputs"
        HWOp["HWCustomOp<br/>(.py)"]
        Backend["RTL Backend<br/>(.py)"]
        Wrapper["RTL Wrapper<br/>(.v)"]
        Infer["Inference Transform<br/>(.py)"]
        
        Generator --> HWOp
        Generator --> Backend
        Generator --> Wrapper
        Generator --> Infer
    end
    
    RTL --> CLI
    
    style RTL fill:#4CAF50
    style HWOp fill:#F44336
    style Backend fill:#F44336
    style Wrapper fill:#F44336
    style Infer fill:#F44336
```

## 2. Data Flow Diagram

```mermaid
graph LR
    subgraph "Raw Data"
        RTL["RTL Source"]
        Pragmas["@brainsmith pragmas"]
    end
    
    subgraph "Parsed Data"
        Ports["Port[]"]
        Params["Parameter[]"]
        PragmaList["Pragma[]"]
    end
    
    subgraph "Structured Data"
        PM["ParsedModule"]
        Interfaces["Interface Metadata"]
        DF["Dataflow Metadata"]
    end
    
    subgraph "Final Data"
        KM["KernelMetadata"]
    end
    
    RTL --> Ports
    RTL --> Params
    Pragmas --> PragmaList
    
    Ports --> PM
    Params --> PM
    PragmaList --> PM
    
    PM --> Interfaces
    PM --> DF
    
    Interfaces --> KM
    DF --> KM
```

## 3. Class Hierarchy Diagram

```mermaid
classDiagram
    class InterfaceMetadata {
        <<abstract>>
        +name: str
        +signals: List[str]
        +compiler_name: str
        +get_compiler_signals()
    }
    
    class AXIStreamMetadata {
        +direction: str
        +datatype: DatatypeParameters
        +is_weight: bool
        +sdim: List[int]
        +bdim: List[int]
    }
    
    class AXILiteMetadata {
        +is_weight: bool
        +config_params: List[str]
    }
    
    class ControlMetadata {
        +clock: str
        +reset: str
    }
    
    class KernelMetadata {
        +name: str
        +module_name: str
        +file_path: str
        +interfaces: List[InterfaceMetadata]
        +dataflow: DataflowMetadata
        +extra_params: List[Parameter]
    }
    
    InterfaceMetadata <|-- AXIStreamMetadata
    InterfaceMetadata <|-- AXILiteMetadata
    InterfaceMetadata <|-- ControlMetadata
    
    KernelMetadata o-- InterfaceMetadata
    AXIStreamMetadata o-- DatatypeParameters
```

## 4. Pragma Processing Flow

```
┌─────────────────────┐
│   RTL Comments      │
│ // @brainsmith ...  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Pragma Extraction  │
│  (find_pragmas)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Pragma Parsing    │
│  (parse_pragma)     │
└──────────┬──────────┘
           │
           ▼
    ┌──────┴──────┐
    │ Pragma Type │
    └──────┬──────┘
           │
  ┌────────┼────────┬───────┬────────┐
  ▼        ▼        ▼       ▼        ▼
┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌─────┐
│TOP │  │BDIM│  │TYPE│  │AXIL│  │ ... │
└────┘  └────┘  └────┘  └────┘  └─────┘
  │        │        │       │        │
  ▼        ▼        ▼       ▼        ▼
┌─────────────────────────────────────┐
│     Apply to Metadata Objects       │
└─────────────────────────────────────┘
```

## 5. Template Generation Flow

```mermaid
sequenceDiagram
    participant CLI
    participant Generator
    participant Templates
    participant FileSystem
    
    CLI->>Generator: generate(metadata, artifact_type)
    
    Generator->>Generator: Load template for artifact_type
    
    Generator->>Templates: Render with metadata context
    Templates-->>Generator: Generated code
    
    Generator->>Generator: Create output path
    
    Generator->>FileSystem: Write file
    
    Note over Generator: Repeat for each artifact type:<br/>- autohwcustomop<br/>- rtlbackend<br/>- wrapper<br/>- infer
```

## 6. Interface Detection ASCII Diagram

```
Module Ports Analysis:
┌────────────────────────────────────────────┐
│              RTL Module                     │
├────────────────────────────────────────────┤
│ Ports:                                     │
│  ┌─────────────┐                          │
│  │ ap_clk      │ ──> Control Interface    │
│  │ ap_rst_n    │                          │
│  └─────────────┘                          │
│  ┌─────────────┐                          │
│  │ s_axis_*    │ ──> AXI-Stream Input     │
│  │ m_axis_*    │ ──> AXI-Stream Output    │
│  └─────────────┘                          │
│  ┌─────────────┐                          │
│  │ s_axilite_* │ ──> AXI-Lite Config      │
│  └─────────────┘                          │
└────────────────────────────────────────────┘
```

## 7. Parameter Linking Process

```mermaid
graph TD
    subgraph "Module Level"
        MP["Module Parameters<br/>(WIDTH, HEIGHT, etc.)"]
    end
    
    subgraph "Pragma Processing"
        DP["@brainsmith:datatype<br/>WIDTH=width"]
        AP["@brainsmith:axilite_param<br/>THRESHOLD"]
    end
    
    subgraph "Interface Level"
        IDP["Interface Datatype Params<br/>(width, signed, etc.)"]
        ICP["Interface Config Params<br/>(threshold, etc.)"]
    end
    
    MP --> DP
    MP --> AP
    
    DP --> IDP
    AP --> ICP
    
    style DP fill:#FF9800
    style AP fill:#FF9800
```

## 8. Complete Processing Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  RTL File    │────▶│    Parse     │────▶│   Extract    │
│  (.sv/.v)    │     │ (tree-sitter)│     │   Modules    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Generate   │◀────│    Build     │◀────│    Apply     │
│   Outputs    │     │   Metadata   │     │   Pragmas    │
└──────────────┘     └──────────────┘     └──────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                   Output Files                        │
├───────────────┬───────────────┬────────────┬────────┤
│ HWCustomOp.py │ RTLBackend.py │ Wrapper.v  │Infer.py│
└───────────────┴───────────────┴────────────┴────────┘
```

## 9. Artifact Dependencies

```mermaid
graph TD
    subgraph "Generated Artifacts"
        HW["HWCustomOp<br/>(Python Class)"]
        RTL["RTL Backend<br/>(Implementation)"]
        WRP["RTL Wrapper<br/>(SystemVerilog)"]
        INF["Inference Transform<br/>(Optimizer Pass)"]
    end
    
    subgraph "FINN Integration"
        FINN["FINN Framework"]
        OPT["ONNX Graph"]
    end
    
    HW --> FINN
    RTL --> HW
    WRP --> RTL
    INF --> OPT
    OPT --> HW
    
    style HW fill:#2196F3
    style RTL fill:#2196F3
    style WRP fill:#2196F3
    style INF fill:#2196F3
```

## 10. Component Interaction Matrix

```
┌─────────────┬────────┬────────┬──────────┬───────────┬──────────┐
│ Component   │  CLI   │ Parser │ Metadata │ Generator │ Templates│
├─────────────┼────────┼────────┼──────────┼───────────┼──────────┤
│ CLI         │   ●    │   ▶    │    ▷     │     ▶     │    ○     │
│ Parser      │   ◀    │   ●    │    ▶     │     ○     │    ○     │
│ Metadata    │   ◁    │   ◀    │    ●     │     ▶     │    ○     │
│ Generator   │   ◀    │   ○    │    ◀     │     ●     │    ▶     │
│ Templates   │   ○    │   ○    │    ○     │     ◀     │    ●     │
└─────────────┴────────┴────────┴──────────┴───────────┴──────────┘

Legend:  ● Self   ▶ Direct Use   ▷ Indirect Use   ○ No Interaction
```

## Key Design Patterns

1. **Pragma-Driven Configuration**: RTL annotations drive behavior
2. **Interface Abstraction**: Different interface types with common base
3. **Parameter Migration**: Parameters move from module to interfaces based on pragmas
4. **Compiler Name Mapping**: RTL names → standardized compiler names
5. **Template Substitution**: Parameters use `$PARAM_NAME$` placeholders

## Processing Stages Summary

### Stage 1: Parse RTL File
- Extract pragmas from comments
- Find and select target module
- Extract ports, parameters
- Build ParsedModule

### Stage 2: Build Metadata
- Create interface objects (AXI-Stream, AXI-Lite, Control)
- Apply pragmas to modify metadata
- Auto-link parameters to interfaces
- Format for compiler export (assign compiler names)

### Stage 3: Generate Outputs
- Load Jinja2 templates
- Render with KernelMetadata context
- Write output files

## Supported Pragma Types

- `TOP_MODULE`: Specify target module
- `DATATYPE_CONSTRAINT`: Constrain interface datatypes
- `WEIGHT`: Mark interfaces as weights
- `BDIM`/`SDIM`: Define block/stream dimensions
- `DATATYPE`: Map RTL params to datatype properties
- `ALIAS`: Expose params with different names
- `DERIVED_PARAMETER`: Define computed parameters
- `AXILITE_PARAM`: Mark AXI-Lite config params
- `RELATIONSHIP`: Define interface relationships

This architecture provides a clean separation of concerns with distinct parsing, metadata representation, and generation phases, making it maintainable and extensible.