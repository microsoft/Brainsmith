# Hardware Kernel Generator (HKG) - Design Document

## Overview

The Hardware Kernel Generator (HKG) Phase 4 is a modular system that transforms SystemVerilog RTL modules into FINN-compatible HWCustomOp classes with automatic interface detection, datatype constraints, and extensible template-based code generation.

**Latest Updates (Phase 4):**
- ✅ Modular generator architecture with auto-discovery
- ✅ Extensible generator system (hw_custom_op, rtl_wrapper, test_suite)
- ✅ KernelIntegrator orchestration replacing UnifiedGenerator
- ✅ Complete dead code cleanup and deprecation warnings
- ✅ Comprehensive @brainsmith pragma support across all hw_kernels

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
    
    subgraph "🏗️ Phase 4 Modular Generation"
        F[KernelIntegrator]
        G[GeneratorManager]
        H[Auto-Discovery System]
        I[Modular Generators]
        
        F1["• Orchestrates generation workflow<br/>• Template context building<br/>• Result aggregation"]
        G1["• Auto-discovers generators<br/>• Manages template rendering<br/>• Extensible architecture"]
        H1["• Package-based introspection<br/>• Dynamic generator loading<br/>• Custom context processing"]
        I1["• hw_custom_op_generator<br/>• rtl_wrapper_generator<br/>• test_suite_generator"]
        
        F --> F1
        G --> G1
        H --> H1
        I --> I1
    end
    
    subgraph "🎯 FINN Integration"
        J[Generated HWCustomOp]
        J1["• get_nodeattr_types()<br/>• get_interface_metadata()<br/>• AutoHWCustomOp inheritance"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> J1
    
    style A fill:#e1f5fe
    style J fill:#e8f5e8
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#fff3e0
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style I fill:#f3e5f5
```

## Phase 4 Modular Architecture

```mermaid
graph TB
    subgraph "🚀 Phase 4 Components"
        A[KernelIntegrator]
        B[GeneratorManager] 
        C[GeneratorBase]
        D[Individual Generators]
        
        A1["• Orchestrates workflow<br/>• Template context generation<br/>• Result aggregation<br/>• File writing coordination"]
        B1["• Auto-discovers generators<br/>• Package introspection<br/>• Template rendering<br/>• Error handling"]
        C1["• Base class for generators<br/>• context_to_dict() helper<br/>• Extensible interface<br/>• Custom context processing"]
        D1["• hw_custom_op_generator<br/>• rtl_wrapper_generator<br/>• test_suite_generator<br/>• Easily extensible"]
        
        A --> A1
        B --> B1
        C --> C1
        D --> D1
    end
    
    subgraph "🔍 Auto-Discovery System"
        E[Package Introspection]
        F[Generator Registration]
        G[Template Association]
        
        E1["• imports in __init__.py<br/>• Class introspection<br/>• No file globbing<br/>• Elegant Python patterns"]
        F1["• Automatic registration<br/>• Name-based discovery<br/>• No manual configuration<br/>• Zero boilerplate"]
        G1["• Template selection<br/>• Fallback handling<br/>• Version compatibility<br/>• Jinja2 integration"]
        
        E --> E1
        F --> F1
        G --> G1
    end
    
    subgraph "⚡ Benefits"
        H[Extensibility]
        I[Maintainability]
        J[Performance]
        
        H1["• Add new generators easily<br/>• Custom context processing<br/>• Template customization<br/>• Zero core changes needed"]
        I1["• Clean separation of concerns<br/>• Eliminated dead code<br/>• Deprecation warnings<br/>• Single generation path"]
        J1["• 60-65ms generation time<br/>• Efficient context passing<br/>• Minimal overhead<br/>• Full functionality retained"]
        
        H --> H1
        I --> I1
        J --> J1
    end
    
    A --> E
    B --> F
    C --> G
    E --> H
    F --> I
    G --> J
    
    style A fill:#e8f5e8
    style B fill:#e1f5fe
    style C fill:#fff3e0
    style D fill:#f3e5f5
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
        A["@brainsmith BDIM s_axis_weights [MH,MW] [PE,SIMD]"]
        B["@brainsmith DATATYPE weights FIXED WEIGHT_WIDTH WEIGHT_WIDTH"] 
        C["@brainsmith WEIGHT s_axis_weights"]
        D["@brainsmith TOP my_module"]
        
        A1["Block Dimensions: [MH,MW]<br/>Stream Dimensions: [PE,SIMD]<br/>Tensor vs Parallelization"]
        B1["Parameterized Datatypes<br/>Fixed-Point Specification<br/>RTL Parameter References"]
        C1["Force Interface Type<br/>Mark Weight Streams<br/>Override Direction-Based"]
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

1. **Input**: SystemVerilog RTL module with @brainsmith pragmas
2. **Command**: `./smithy exec "python -m brainsmith.tools.hw_kernel_gen my_module.sv -o output_dir/ --debug"`
3. **Output**: Complete FINN-compatible HWCustomOp package
4. **Integration**: Drop into FINN compilation pipeline

**Phase 4 Examples:**

```bash
# FMPadding with complete pragmas
./smithy exec "python -m brainsmith.tools.hw_kernel_gen brainsmith/hw_kernels/fmpadding/fmpadding_axi.sv -o output/"

# MVU/VVU matrix operations
./smithy exec "python -m brainsmith.tools.hw_kernel_gen brainsmith/hw_kernels/mvu/mvu_vvu_axi.sv -o output/"

# Thresholding/activation functions  
./smithy exec "python -m brainsmith.tools.hw_kernel_gen brainsmith/hw_kernels/thresholding/thresholding_axi.sv -o output/"
```

**Key Phase 4 Features:**
- 🚀 **Modular Architecture**: Extensible generator system with auto-discovery
- 🔍 **Enhanced Pragmas**: Block vs stream dimensions, parameterized datatypes
- 🏷️ **Smart Interface Detection**: AXI-Stream typing with weight stream support
- 📝 **Comprehensive Pragmas**: All hw_kernels updated with proper @brainsmith annotations
- 🎯 **FINN Ready**: AutoHWCustomOp inheritance with validated interfaces
- ⚡ **Performance**: 60-65ms generation time with complete functionality
- 🧹 **Clean Codebase**: Dead code eliminated, deprecation warnings for legacy components
- 📦 **Complete Package**: Python class, RTL wrapper, comprehensive test suites

**Supported RTL Kernels:**
- ✅ Matrix Vector Units (MVU/VVU) with weight streams
- ✅ Feature Map Padding with channel processing
- ✅ Thresholding/Activation functions with PE parallelization
- ✅ Memory streaming with parameterized datatypes
- ✅ FIFO/Queue modules with width parameters