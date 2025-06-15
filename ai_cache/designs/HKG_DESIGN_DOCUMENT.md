# Hardware Kernel Generator (HKG) - Design Document

## Overview

The Hardware Kernel Generator (HKG) transforms SystemVerilog RTL modules into FINN-compatible HWCustomOp classes with automatic interface detection, datatype constraints, and template-based code generation.

## Complete HKG Workflow

```mermaid
graph TB
    subgraph "📄 Input"
        A[SystemVerilog RTL Module]
        A1["• Matrix multiply, convolution, etc.<br/>• Standard AXI interfaces<br/>• Optional @brainsmith pragmas"]
        A --> A1
    end
    
    subgraph "🔍 RTL Analysis"
        B[Tree-sitter Parser]
        C[Interface Detection]
        D[Protocol Validation]
        E[Pragma Processing]
        
        B1["• Parameters (PE, SIMD, widths)<br/>• Ports with directions<br/>• Comments with pragmas"]
        C1["• AXI-Stream → INPUT/OUTPUT/WEIGHT<br/>• AXI-Lite → CONFIG<br/>• Clock/Reset → CONTROL"]
        D1["• Required signals present<br/>• Direction consistency<br/>• Protocol compliance"]
        E1["• BDIM: Block chunking<br/>• DATATYPE: Constraints<br/>• WEIGHT: Force type override"]
        
        B --> B1
        C --> C1
        D --> D1
        E --> E1
    end
    
    subgraph "🏗️ Code Generation"
        F[Template Engine]
        G[Context Building]
        H[File Writing]
        
        F1["• Jinja2 templates<br/>• Interface-aware generation<br/>• Parameter mapping"]
        G1["• Interface categorization<br/>• Legacy FINN compatibility<br/>• Metadata organization"]
        H1["• HWCustomOp Python class<br/>• RTL wrapper<br/>• Test suite"]
        
        F --> F1
        G --> G1
        H --> H1
    end
    
    subgraph "🎯 FINN Integration"
        I[Generated HWCustomOp]
        I1["• get_nodeattr_types()<br/>• get_interface_metadata()<br/>• AutoHWCustomOp inheritance"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> I1
    
    style A fill:#e1f5fe
    style I fill:#e8f5e8
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#fff3e0
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#f3e5f5
```

## Interface Detection & Type Assignment

```mermaid
graph TD
    subgraph "🔌 Port Classification"
        A[RTL Ports] --> B{Signal Pattern?}
        B -->|*_tdata, *_tvalid, *_tready| C[AXI-Stream Candidate]
        B -->|*_awaddr, *_wdata, *_rdata| D[AXI-Lite Candidate]
        B -->|*clk, *rst_n| E[Control Candidate]
        B -->|Other patterns| F[Unassigned]
    end
    
    subgraph "✅ Protocol Validation"
        C --> G{Valid AXI-Stream?}
        D --> H{Valid AXI-Lite?}
        E --> I{Valid Control?}
        
        G -->|Required signals present| J[✓ AXI-Stream Interface]
        G -->|Missing signals| K[✗ Invalid]
        H -->|Complete channels| L[✓ AXI-Lite Interface]  
        H -->|Incomplete| K
        I -->|Has clk & rst_n| M[✓ Control Interface]
        I -->|Missing signals| K
    end
    
    subgraph "🏷️ Type Assignment"
        J --> N{Port Direction & Name}
        N -->|INPUT ports + weight patterns| O[WEIGHT Interface]
        N -->|INPUT ports + other names| P[INPUT Interface]
        N -->|OUTPUT ports| Q[OUTPUT Interface]
        
        L --> R[CONFIG Interface]
        M --> S[CONTROL Interface]
        
        subgraph "🎯 Pragma Override"
            T["@brainsmith WEIGHT interface_name"]
            T --> U[Force WEIGHT Type]
        end
        
        O --> U
        P --> U
    end
    
    subgraph "📊 Final Interface Types"
        V[INPUT: Activations]
        W[WEIGHT: Parameters] 
        X[OUTPUT: Results]
        Y[CONFIG: AXI-Lite]
        Z[CONTROL: Clock/Reset]
    end
    
    P --> V
    U --> W
    Q --> X
    R --> Y
    S --> Z
    
    style C fill:#e3f2fd
    style J fill:#e8f5e8
    style O fill:#fff3e0
    style W fill:#fff3e0
    style V fill:#e1f5fe
    style X fill:#fce4ec
```

## Pragma System & Template Generation

```mermaid
graph TB
    subgraph "📝 RTL Pragmas"
        A["@brainsmith BDIM in0 -1 [SIMD]"]
        B["@brainsmith DATATYPE weights FIXED 8 8"] 
        C["@brainsmith WEIGHT weights_V"]
        D["@brainsmith TOP my_module"]
        
        A1["Block Dimension<br/>Chunking Strategy"]
        B1["Datatype Constraints<br/>Bit Width Limits"]
        C1["Force Interface Type<br/>Override Direction-Based"]
        D1["Module Selection<br/>When Multiple Present"]
        
        A --> A1
        B --> B1
        C --> C1
        D --> D1
    end
    
    subgraph "🏗️ Template Context Building"
        E[Interface Categorization]
        F[Parameter Processing]
        G[Legacy FINN Mapping]
        H[Code Generation Helpers]
        
        E1["inputs: [in0, in1]<br/>weights: [weights_V]<br/>outputs: [out0]<br/>config: [axilite]"]
        F1["PE = 4 (with fallback)<br/>SIMD = 8 (with fallback)<br/>CUSTOM (required)"]
        G1["SIMD → iPar<br/>PE → wPar<br/>widths → datatypes"]
        H1["Class naming<br/>Import generation<br/>Method templates"]
        
        E --> E1
        F --> F1
        G --> G1
        H --> H1
    end
    
    subgraph "📦 Generated Output"
        I[HWCustomOp Class]
        J[RTL Wrapper]
        K[Test Suite]
        L[Documentation]
        
        I1["class MatMulHWCustomOp(AutoHWCustomOp):<br/>  def get_nodeattr_types():<br/>  def get_interface_metadata():"]
        J1["module mat_mul_wrapper;<br/>  // Parameter mappings<br/>  // Interface connections"]
        K1["def test_interface_creation():<br/>def test_parameter_validation():<br/>def test_dataflow_integration():"]
        L1["# MatMul HWCustomOp<br/>Auto-generated from RTL<br/>Interface specifications"]
        
        I --> I1
        J --> J1
        K --> K1
        L --> L1
    end
    
    A1 --> E
    B1 --> E
    C1 --> E
    D1 --> F
    E --> I
    F --> I
    G --> I
    H --> I
    E --> J
    F --> J
    I --> K
    I --> L
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style D fill:#e3f2fd
    style I fill:#e8f5e8
    style J fill:#fff3e0
    style K fill:#f3e5f5
    style L fill:#fce4ec
```

## FINN Integration Architecture

```mermaid
graph TB
    subgraph "🎯 Generated HWCustomOp"
        A[MatMulHWCustomOp Class]
        B[get_nodeattr_types Method]
        C[get_interface_metadata Method]
        D[AutoHWCustomOp Inheritance]
        
        B1["PE: Required integer<br/>SIMD: Required integer<br/>ACTIVATION_WIDTH: Optional<br/>ram_style: Optional enum"]
        C1["INPUT: in0 (INT8 constraints)<br/>WEIGHT: weights (INT4 constraints)<br/>OUTPUT: out0 (INT32 constraints)<br/>CONTROL: clk, rst_n"]
        D1["get_legacy_attr(): SIMD/PE mapping<br/>verify_node(): Validation<br/>Resource estimation methods"]
        
        B --> B1
        C --> C1
        D --> D1
    end
    
    subgraph "🔗 FINN Integration Points"
        E[ONNX Graph Node]
        F[Node Attributes]
        G[Dataflow Model]
        H[Compilation Pipeline]
        
        E1["op_type: MatMul<br/>inputs: [activations, weights]<br/>outputs: [result]"]
        F1["PE=4, SIMD=8<br/>inputDataType=INT8<br/>weightDataType=INT4"]
        G1["DataflowInterface objects<br/>Block chunking strategies<br/>Tensor shape inference"]
        H1["RTL generation<br/>IP integration<br/>Hardware synthesis"]
        
        E --> E1
        F --> F1
        G --> G1
        H --> H1
    end
    
    subgraph "📊 Interface Metadata Flow"
        I[RTL Interface Detection]
        J[Pragma Enhancement]
        K[InterfaceMetadata Creation]
        L[DataflowInterface Binding]
        
        I1["s_axis_input → INPUT<br/>s_axis_weights → WEIGHT<br/>m_axis_output → OUTPUT"]
        J1["BDIM: Chunking rules<br/>DATATYPE: Bit constraints<br/>WEIGHT: Type override"]
        K1["Unified interface types<br/>Constraint groups<br/>Chunking strategies"]
        L1["AutoHWCustomOp integration<br/>Legacy compatibility<br/>Resource estimation"]
        
        I --> I1
        J --> J1
        K --> K1
        L --> L1
    end
    
    A --> E
    B --> F
    C --> G
    D --> L
    I --> K
    J --> K
    K --> C
    L --> D
    
    E --> H
    F --> H
    G --> H
    
    style A fill:#e8f5e8
    style E fill:#e1f5fe
    style I fill:#fff3e0
    style H fill:#fce4ec
```

## Key Benefits for FINN Users

```mermaid
graph LR
    subgraph "🚀 Developer Experience"
        A[Zero Manual Coding]
        B[Automatic Interface Detection]
        C[Legacy FINN Compatibility]
        D[Template Customization]
    end
    
    subgraph "🎯 Integration Quality" 
        E[Validated Protocols]
        F[Consistent Naming]
        G[Complete Test Suites]
        H[Documentation Generation]
    end
    
    subgraph "🔧 Extensibility"
        I[Pragma System]
        J[Template Engine]
        K[Custom Constraints]
        L[Multi-Interface Support]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    E --> I
    F --> J
    G --> K
    H --> L
    
    style A fill:#e8f5e8
    style E fill:#e1f5fe
    style I fill:#f3e5f5
```

## Usage Summary

1. **Input**: SystemVerilog RTL module with standard AXI interfaces
2. **Command**: `hkg generate my_module.sv output_dir/`
3. **Output**: Complete FINN-compatible HWCustomOp package
4. **Integration**: Drop into FINN compilation pipeline

**Key Features:**
- 🔍 **Automatic Detection**: AXI-Stream, AXI-Lite, and Control interfaces
- 🏷️ **Smart Typing**: Direction-based INPUT/OUTPUT, name-based WEIGHT detection
- 📝 **Pragma Enhancement**: BDIM chunking, DATATYPE constraints, WEIGHT overrides
- 🎯 **FINN Ready**: AutoHWCustomOp inheritance with legacy compatibility
- 📦 **Complete Package**: Python class, RTL wrapper, tests, documentation