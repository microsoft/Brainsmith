# HWKG Template Generation Flow - Detailed Architecture

This document provides clear, organized Mermaid diagrams showing the template generation workflow in the Hardware Kernel Generator (HKG) Phase 4, starting from parsed KernelMetadata through final file generation.

## High-Level Template Generation Flow

```mermaid
graph TD
    subgraph "Data Flow Spine"
        A[KernelMetadata<br/>Parsed RTL] 
        B[TemplateContext<br/>Processed Data]
        C[Generated Content<br/>Template Output]
        D[Written Files<br/>Disk Output]
    end
    
    subgraph "Processing Steps"
        P1[Context Generator<br/>Transform Metadata]
        P2[GeneratorManager<br/>Coordinate Rendering]
        P3[File Writer<br/>Organize Output]
    end
    
    A -->|transforms| P1
    P1 -->|produces| B
    B -->|distributes to| P2
    P2 -->|renders| C
    C -->|writes| P3
    P3 -->|creates| D
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style P1 fill:#e3f2fd
    style P2 fill:#e3f2fd
    style P3 fill:#e3f2fd
```

## Context Generation Process

```mermaid
graph TD
    subgraph "Input Components"
        KM[KernelMetadata]
        P[Parameters<br/>PE, SIMD, MW, MH]
        I[Interfaces<br/>weights, input, output]
        PR[Pragmas<br/>BDIM, DATATYPE]
    end
    
    subgraph "Processing Operations"
        PC[Parameter<br/>Processor]
        IC[Interface<br/>Classifier]
        PRC[Pragma<br/>Extractor]
    end
    
    subgraph "Context Assembly"
        TCG[Template Context<br/>Generator]
        TC[TemplateContext<br/>Complete]
    end
    
    KM --> P
    KM --> I  
    KM --> PR
    
    P -->|processes| PC
    I -->|categorizes| IC
    PR -->|extracts| PRC
    
    PC -->|parameter data| TCG
    IC -->|interface data| TCG
    PRC -->|constraint data| TCG
    
    TCG -->|assembles| TC
    
    style KM fill:#e1f5fe
    style TC fill:#e8f5e8
    style PC fill:#fff3e0
    style IC fill:#fff3e0
    style PRC fill:#fff3e0
    style TCG fill:#f3e5f5
```

## Parallel Generator Execution

```mermaid
graph TD
    subgraph "Input"
        TC[TemplateContext]
    end
    
    subgraph "Coordination"
        GM[GeneratorManager]
        AD[Auto-Discovery<br/>System]
    end
    
    subgraph "Parallel Generation"
        G1[HWCustomOp<br/>Generator]
        G2[RTL Wrapper<br/>Generator] 
        G3[Test Suite<br/>Generator]
    end
    
    subgraph "Template Rendering"
        T1[hw_custom_op_phase2.py.j2]
        T2[rtl_wrapper_v2.v.j2]
        T3[test_suite_v2.py.j2]
    end
    
    subgraph "Generated Content"
        F1[.py file content]
        F2[.v file content]
        F3[.py test content]
    end
    
    TC -->|provides data| GM
    GM -->|coordinates| AD
    AD -->|discovers| G1
    AD -->|discovers| G2
    AD -->|discovers| G3
    
    G1 -->|loads| T1
    G2 -->|loads| T2
    G3 -->|loads| T3
    
    T1 -->|renders with TC| F1
    T2 -->|renders with TC| F2
    T3 -->|renders with TC| F3
    
    style TC fill:#e1f5fe
    style GM fill:#e8f5e8
    style AD fill:#fff3e0
    style G1 fill:#f3e5f5
    style G2 fill:#f3e5f5
    style G3 fill:#f3e5f5
    style T1 fill:#fce4ec
    style T2 fill:#fce4ec
    style T3 fill:#fce4ec
    style F1 fill:#e8f5e8
    style F2 fill:#e8f5e8
    style F3 fill:#e8f5e8
```

## Template Context Data Structure

```mermaid
graph LR
    TC[TemplateContext]
    TC --> Core[Core Info<br/>• module_name<br/>• class_name<br/>• source_file]
    TC --> Intf[Interfaces<br/>• inputs<br/>• weights<br/>• outputs<br/>• control]
    TC --> Param[Parameters<br/>• definitions<br/>• defaults<br/>• required]
    TC --> Help[Helpers<br/>• imports<br/>• attributes<br/>• flags]
    
    style TC fill:#e1f5fe
    style Core fill:#fff3e0
    style Intf fill:#fff3e0
    style Param fill:#fff3e0
    style Help fill:#fff3e0
```

## Template Rendering Detail

```mermaid
graph TD
    subgraph "Template Processing"
        TC[TemplateContext]
        CD[context_to_dict helper]
        DICT[Context Dictionary]
    end
    
    subgraph "Jinja2 Engine"
        TPL[Template File<br/>hw_custom_op_phase2.py.j2]
        J2[Jinja2 Renderer]
    end
    
    subgraph "Generated Sections"
        CLS[Class Definition<br/>MvuVvuAxi extends AutoHWCustomOp]
        NAT[get_nodeattr_types<br/>Method]
        IFM[get_interface_metadata<br/>Method]
    end
    
    subgraph "Final Output"
        OUT[Complete Python File<br/>mvu_vvu_axi_hw_custom_op.py]
    end
    
    TC -->|converts| CD
    CD -->|produces| DICT
    DICT -->|feeds| J2
    TPL -->|loaded by| J2
    
    J2 -->|renders| CLS
    J2 -->|renders| NAT
    J2 -->|renders| IFM
    
    CLS -->|combines into| OUT
    NAT -->|combines into| OUT
    IFM -->|combines into| OUT
    
    style TC fill:#e1f5fe
    style DICT fill:#fff3e0
    style J2 fill:#f3e5f5
    style OUT fill:#e8f5e8
```

## File Writing and Result Aggregation

```mermaid
graph LR
    subgraph "Generation Results"
        GR[Generated Content<br/>• hw_custom_op.py<br/>• wrapper.v<br/>• test_suite.py]
    end
    
    subgraph "KernelIntegrator Processing"
        KI[File Writing<br/>1. Determine output directory<br/>2. Write generated files<br/>3. Create metadata files<br/>4. Track performance metrics]
    end
    
    subgraph "Output Structure"
        OUT[output_dir/kernel_name/<br/>├── kernel_name_hw_custom_op.py<br/>├── kernel_name_wrapper.v<br/>├── test_kernel_name.py<br/>├── generation_metadata.json<br/>└── generation_summary.txt]
    end
    
    GR --> KI
    KI --> OUT
    
    style GR fill:#e1f5fe
    style KI fill:#fff3e0
    style OUT fill:#e8f5e8
```

## Performance Metrics

```mermaid
graph LR
    A[Parsing Time] --> E[Total: 60-65ms]
    B[Context Generation] --> E
    C[Template Rendering] --> E
    D[File Writing] --> E
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#fce4ec
    style E fill:#e8f5e8
```

## Key Phase 4 Improvements Summary

### 🚀 **Modular Architecture Benefits**
- **Extensibility**: Add new generators without core changes
- **Maintainability**: Clean separation of concerns
- **Auto-Discovery**: Zero-configuration generator registration
- **Performance**: 60-65ms generation time maintained

### 🎯 **Template System Enhancements**
- **Unified Context**: Single TemplateContext for all generators
- **Standardized Processing**: context_to_dict() helper method
- **Fallback Support**: Template version compatibility
- **Error Handling**: Graceful degradation and reporting

### 📊 **Data Flow Optimization**
- **Single Parse**: KernelMetadata parsed once
- **Efficient Context**: Full context passed to all generators
- **Minimal Overhead**: Package introspection over file globbing
- **Result Aggregation**: Comprehensive GenerationResult tracking

This architecture provides a robust, extensible foundation for HWKG template generation while maintaining the simplicity and performance of the original system.