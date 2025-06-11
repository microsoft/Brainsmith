# Brainsmith System Architecture Analysis

## Executive Summary

Brainsmith implements a sophisticated **Interface-Wise Dataflow Modeling** framework that converts PyTorch models to RTL implementations for FPGA deployment. The system consists of two main components: the **Dataflow Modeling System** and the **Hardware Kernel Generator (HKG)**, connected by integration layers that enable seamless RTL-to-FINN conversion.

## System Overview

```mermaid
graph TB
    subgraph "Input Sources"
        RTL[SystemVerilog RTL Files]
        ONNX[ONNX Model Metadata]
        Pragma[Brainsmith Pragmas]
    end
    
    subgraph "Hardware Kernel Generator"
        Parser[RTL Parser]
        Scanner[Interface Scanner]
        Validator[Protocol Validator]
        Builder[Interface Builder]
    end
    
    subgraph "Integration Layer"
        Converter[RTL Interface Converter]
        Mapper[Interface Mapper]
        PragmaProc[Pragma Processor]
    end
    
    subgraph "Dataflow Modeling System"
        DFInterface[DataflowInterface]
        DFModel[DataflowModel]
        AutoOp[AutoHWCustomOp]
        AutoBackend[AutoRTLBackend]
    end
    
    subgraph "FINN Integration"
        HWCustomOp[FINN HWCustomOp]
        RTLBackend[FINN RTLBackend]
        Templates[Generated Code]
    end
    
    RTL --> Parser
    ONNX --> Converter
    Pragma --> PragmaProc
    
    Parser --> Scanner
    Scanner --> Validator
    Validator --> Builder
    Builder --> Converter
    
    Converter --> DFInterface
    Mapper --> DFInterface
    PragmaProc --> DFInterface
    
    DFInterface --> DFModel
    DFModel --> AutoOp
    DFModel --> AutoBackend
    
    AutoOp --> HWCustomOp
    AutoBackend --> RTLBackend
    HWCustomOp --> Templates
    RTLBackend --> Templates
```

## Mathematical Foundation: Three-Tier Dimension Hierarchy

The core mathematical foundation follows the relationship:

**tensor_dims → block_dims → stream_dims → element**

```mermaid
graph TB
    subgraph "Mathematical Foundation"
        TD[tensor_dims: Original Shape<br/>[1, 128, 768]]
        BD[block_dims: Processing Chunks<br/>[1, 8, 96]]
        SD[stream_dims: Hardware Parallelism<br/>[1, 1, 8]]
        NB[num_blocks: Computed<br/>[1, 16, 8]]
    end
    
    subgraph "Constraints"
        C1[tensor_dims[i] % block_dims[i] == 0]
        C2[block_dims[i] % stream_dims[i] == 0]
        C3[num_blocks[i] = tensor_dims[i] ÷ block_dims[i]]
    end
    
    TD --> BD
    BD --> SD
    BD --> NB
    TD --> C1
    BD --> C2
    TD --> C3
```

## RTL Parsing Pipeline

```mermaid
graph TB
    subgraph "Stage 1: Initial Parse"
        RTLFile[SystemVerilog File]
        TreeSitter[Tree-sitter Parser]
        AST[Abstract Syntax Tree]
        ModSelect[Module Selection]
        PragmaExt[Pragma Extraction]
    end
    
    subgraph "Stage 2: Component Extraction"
        ModName[Module Name]
        Params[Parameters]
        Ports[Port Declarations]
    end
    
    subgraph "Stage 3: Interface Analysis"
        PortGroups[Port Grouping]
        ProtocolVal[Protocol Validation]
        InterfaceClass[Interface Classification]
    end
    
    subgraph "Stage 4: Pragma Application"
        DataType[DATATYPE Pragmas]
        Weight[WEIGHT Pragmas]
        BDIM[BDIM Pragmas]
        Derived[DERIVED_PARAMETER]
    end
    
    RTLFile --> TreeSitter
    TreeSitter --> AST
    AST --> ModSelect
    AST --> PragmaExt
    
    ModSelect --> ModName
    AST --> Params
    AST --> Ports
    
    Ports --> PortGroups
    PortGroups --> ProtocolVal
    ProtocolVal --> InterfaceClass
    
    PragmaExt --> DataType
    PragmaExt --> Weight
    PragmaExt --> BDIM
    PragmaExt --> Derived
```

## Interface Detection and Classification

```mermaid
graph TB
    subgraph "Port Analysis"
        Ports[SystemVerilog Ports]
        Patterns[Naming Patterns]
        Groups[Port Groups]
    end
    
    subgraph "Protocol Validation"
        AXIStream[AXI-Stream Validation<br/>TDATA, TVALID, TREADY]
        AXILite[AXI-Lite Validation<br/>AW, W, B, AR, R channels]
        GlobalCtrl[Global Control<br/>ap_clk, ap_rst_n]
    end
    
    subgraph "Interface Classification"
        INPUT[INPUT<br/>AXI-Stream Input]
        OUTPUT[OUTPUT<br/>AXI-Stream Output]
        WEIGHT[WEIGHT<br/>AXI-Stream Parameters]
        CONFIG[CONFIG<br/>AXI-Lite Configuration]
        CONTROL[CONTROL<br/>Global Control Signals]
    end
    
    Ports --> Patterns
    Patterns --> Groups
    
    Groups --> AXIStream
    Groups --> AXILite
    Groups --> GlobalCtrl
    
    AXIStream --> INPUT
    AXIStream --> OUTPUT
    AXIStream --> WEIGHT
    AXILite --> CONFIG
    GlobalCtrl --> CONTROL
```

## Pragma Processing System

```mermaid
graph TB
    subgraph "Pragma Types"
        BDIM[BDIM: Block Dimensions<br/>@brainsmith BDIM in0 -1 [16]]
        DATATYPE[DATATYPE: Type Constraints<br/>@brainsmith DATATYPE in0 INT,UINT 1 16]
        WEIGHT[WEIGHT: Weight Marking<br/>@brainsmith WEIGHT weights_V]
        TOP[TOP_MODULE: Module Selection<br/>@brainsmith TOP_MODULE my_module]
    end
    
    subgraph "Processing Pipeline"
        Extract[Pragma Extraction]
        Parse[Pragma Parsing]
        Validate[Pragma Validation]
        Apply[Pragma Application]
    end
    
    subgraph "Output Effects"
        ChunkStrat[Chunking Strategies]
        TypeConst[Datatype Constraints]
        InterfaceMeta[Interface Metadata]
        ModSelect[Module Selection]
    end
    
    BDIM --> Extract
    DATATYPE --> Extract
    WEIGHT --> Extract
    TOP --> Extract
    
    Extract --> Parse
    Parse --> Validate
    Validate --> Apply
    
    Apply --> ChunkStrat
    Apply --> TypeConst
    Apply --> InterfaceMeta
    Apply --> ModSelect
```

## RTL to DataflowInterface Conversion

```mermaid
graph TB
    subgraph "Input Data"
        RTLInt[RTL Interface]
        ONNXMeta[ONNX Metadata]
        PragmaData[Pragma Data]
    end
    
    subgraph "Conversion Steps"
        TypeMap[1. Interface Type Mapping]
        DimExt[2. Dimension Extraction]
        DataExt[3. Datatype Extraction]
        ConsExt[4. Constraint Extraction]
        Create[5. DataflowInterface Creation]
    end
    
    subgraph "Dimension Sources (Priority Order)"
        BDIMPragma[BDIM Pragma Override]
        ONNXShape[ONNX Shape Metadata]
        RTLWidth[RTL Port Width Inference]
        Defaults[Sensible Defaults]
    end
    
    subgraph "Output"
        DFInterface[DataflowInterface<br/>tensor_dims, block_dims, stream_dims<br/>dtype, constraints, metadata]
    end
    
    RTLInt --> TypeMap
    ONNXMeta --> DimExt
    PragmaData --> ConsExt
    
    TypeMap --> DimExt
    DimExt --> DataExt
    DataExt --> ConsExt
    ConsExt --> Create
    
    BDIMPragma --> DimExt
    ONNXShape --> DimExt
    RTLWidth --> DimExt
    Defaults --> DimExt
    
    Create --> DFInterface
```

## DataflowModel and Performance Calculations

```mermaid
graph TB
    subgraph "Input Interfaces"
        InputIF[INPUT Interfaces]
        OutputIF[OUTPUT Interfaces]
        WeightIF[WEIGHT Interfaces]
    end
    
    subgraph "DataflowModel"
        IntCollection[Interface Collection]
        Relationships[Interface Relationships]
        Constraints[Mathematical Constraints]
        PerfCalc[Performance Calculations]
    end
    
    subgraph "Performance Metrics"
        cII[cII: Calculation Interval<br/>∏(block_dims[i] / stream_dims[i])]
        eII[eII: Execution Interval<br/>cII * max_weight_cycles]
        Latency[L: Total Latency<br/>eII_bottleneck * num_blocks]
        Bottleneck[Bottleneck Analysis]
    end
    
    subgraph "Parallelism Configuration"
        iPar[iPar: Input Parallelism]
        wPar[wPar: Weight Parallelism]
        StreamUpdate[stream_dims Updates]
    end
    
    InputIF --> IntCollection
    OutputIF --> IntCollection
    WeightIF --> IntCollection
    
    IntCollection --> Relationships
    Relationships --> Constraints
    Constraints --> PerfCalc
    
    PerfCalc --> cII
    PerfCalc --> eII
    PerfCalc --> Latency
    PerfCalc --> Bottleneck
    
    iPar --> StreamUpdate
    wPar --> StreamUpdate
    StreamUpdate --> PerfCalc
```

## Template Generation and FINN Integration

```mermaid
graph TB
    subgraph "Template Generation Paths"
        Path1[Path 1: DataflowModel → Templates<br/>Full Pipeline]
        Path2[Path 2: EnhancedRTLParsingResult → Templates<br/>Direct Generation]
    end
    
    subgraph "Generated Components"
        HWCustomOp[HWCustomOp Class<br/>FINN Integration]
        RTLBackend[RTLBackend Class<br/>RTL Generation]
        VerilogWrapper[Verilog Wrapper<br/>Signal Mapping]
        TestSuite[Test Suite<br/>Validation]
        Documentation[Documentation<br/>README]
    end
    
    subgraph "FINN Integration Features"
        DataTypes[get_input_datatype()]
        FoldedShape[get_folded_shape()]
        ExpCycles[get_exp_cycles()]
        RTLSignals[RTL Signal Mapping]
        ParamFiles[Parameter Files]
    end
    
    Path1 --> HWCustomOp
    Path1 --> RTLBackend
    Path2 --> VerilogWrapper
    Path2 --> TestSuite
    Path2 --> Documentation
    
    HWCustomOp --> DataTypes
    HWCustomOp --> FoldedShape
    HWCustomOp --> ExpCycles
    RTLBackend --> RTLSignals
    RTLBackend --> ParamFiles
```

## Complete End-to-End Workflow

```mermaid
graph TB
    subgraph "Phase 1: RTL Analysis"
        A1[SystemVerilog RTL File]
        A2[RTL Parser]
        A3[Interface Detection]
        A4[Pragma Extraction]
        A5[RTLParsingResult]
    end
    
    subgraph "Phase 2: Integration"
        B1[Interface Mapping]
        B2[Pragma Processing]
        B3[Dimension Calculation]
        B4[DataflowInterface Creation]
    end
    
    subgraph "Phase 3: Modeling"
        C1[DataflowModel Creation]
        C2[Performance Analysis]
        C3[Parallelism Configuration]
        C4[Validation]
    end
    
    subgraph "Phase 4: Generation"
        D1[Template Context]
        D2[Code Generation]
        D3[FINN Integration]
        D4[Output Files]
    end
    
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5
    
    A5 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    
    B4 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    
    C4 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
```

## Key Architectural Strengths

### 1. **Mathematical Rigor**
- Enforced divisibility relationships ensure valid hardware implementations
- Unified performance calculations with bottleneck analysis
- Three-tier dimension hierarchy provides systematic tensor decomposition

### 2. **Separation of Concerns**
- **RTL Analysis**: SystemVerilog parsing and interface detection
- **Dataflow Modeling**: Mathematical modeling and performance analysis
- **Template Generation**: Code generation and FINN integration
- **Integration Layers**: Clean conversion between systems

### 3. **Pragma-Driven Configuration**
- `@brainsmith` pragmas enable fine-grained control
- Systematic conversion from comments to typed objects
- Backward compatibility with legacy formats

### 4. **Performance Optimization**
- Lightweight RTLParsingResult reduces overhead by ~800 lines
- Direct template generation path bypasses DataflowModel when needed
- Template context caching for repeated operations

### 5. **FINN Integration**
- Seamless integration with AMD's FINN framework
- Auto-generated HWCustomOp and RTLBackend classes
- Compatible with existing FINN workflows and optimizations

## Error Handling and Validation

```mermaid
graph TB
    subgraph "Validation Layers"
        V1[RTL Parser: Syntax & Pragma Validation]
        V2[Interface Scanner: Pattern Matching]
        V3[Protocol Validator: Signal Completeness]
        V4[Conversion: Interface Compatibility]
        V5[DataflowModel: Dimensional Consistency]
        V6[FINN Integration: Template Validation]
    end
    
    subgraph "Error Recovery"
        E1[Graceful Degradation]
        E2[Detailed Error Reports]
        E3[Debug Information]
        E4[Legacy Fallback]
    end
    
    V1 --> V2
    V2 --> V3
    V3 --> V4
    V4 --> V5
    V5 --> V6
    
    V1 --> E1
    V2 --> E2
    V3 --> E3
    V4 --> E4
```

This comprehensive architecture enables Brainsmith to systematically convert SystemVerilog RTL modules with semantic annotations into FINN-compatible hardware accelerators while maintaining mathematical rigor, performance optimization, and extensive validation capabilities.