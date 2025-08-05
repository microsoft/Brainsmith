# Kernel Integrator Data Flow Analysis

This document visualizes the data transformation pipeline in the Kernel Integrator, showing how SystemVerilog RTL is progressively abstracted into Python code for FINN integration.

## Data Flow Diagram

```mermaid
flowchart TB
    %% Input Stage
    RTL["SystemVerilog RTL File<br/>with Pragmas"]
    
    %% Parsing Stage
    subgraph Parsing ["Parsing Stage (rtl_parser/)"]
        AST["ASTParser<br/>Tree-sitter"]
        ME[ModuleExtractor]
        PH[PragmaHandler]
        
        AST --> ParsedData["ParsedData<br/>- module_name<br/>- parameters: List[Parameter]<br/>- ports: List[Port]<br/>- pragmas: List[Pragma]"]
        ME --> ParsedData
        PH --> ParsedData
    end
    
    %% Types at Parsing Stage
    subgraph RTLTypes ["RTL Types (types/rtl.py)"]
        Port["Port<br/>- name<br/>- direction<br/>- width"]
        Param["Parameter<br/>- name<br/>- type<br/>- default_value"]
        PortGroup["PortGroup<br/>- interface_type<br/>- ports: Dict"]
    end
    
    %% Interface Building Stage
    subgraph InterfaceBuilding ["Interface Building Stage"]
        IB[InterfaceBuilder]
        IS["InterfaceScanner<br/>Protocol Detection"]
        PV[ProtocolValidator]
        
        IB --> IM[InterfaceMetadata]
        IS --> IM
        PV --> IM
    end
    
    %% Metadata Stage
    subgraph MetadataStage ["Metadata Construction"]
        PL["ParameterLinker<br/>Auto-linking"]
        PA["Pragma Application"]
        
        PL --> KM[KernelMetadata]
        PA --> KM
    end
    
    %% Metadata Types
    subgraph MetadataTypes ["Metadata Types (types/metadata.py)"]
        InterfaceMeta["InterfaceMetadata<br/>- name<br/>- interface_type<br/>- datatype_constraints<br/>- bdim_shape<br/>- sdim_shape"]
        KernelMeta["KernelMetadata<br/>- name<br/>- interfaces: List[InterfaceMetadata]<br/>- parameters: List[Parameter]<br/>- exposed_parameters<br/>- relationships"]
        DatatypeMeta["DatatypeMetadata<br/>- name<br/>- width<br/>- signed<br/>- format"]
    end
    
    %% Template Context Stage
    subgraph TemplateStage ["Template Context Generation"]
        TCG[TemplateContextGenerator]
        CBB["CodegenBinding Builder"]
        
        TCG --> TC[TemplateContext]
        CBB --> CB[CodegenBinding]
    end
    
    %% Template Types
    subgraph TemplateTypes ["Template Types"]
        TemplateCtx["TemplateContext<br/>- module_name<br/>- class_name<br/>- interface_metadata<br/>- parameter_definitions<br/>- codegen_binding"]
        Binding["CodegenBinding<br/>- io_specs: List[IOSpec]<br/>- attributes: List[AttributeBinding]<br/>- interface_bindings"]
    end
    
    %% Code Generation Stage
    subgraph CodeGen ["Code Generation Stage"]
        GM[GeneratorManager]
        HWG[HWCustomOpGenerator]
        RBG[RTLBackendGenerator]
        RWG[RTLWrapperGenerator]
        
        GM --> GR[GenerationResult]
        HWG --> GR
        RBG --> GR
        RWG --> GR
    end
    
    %% Output Types
    subgraph OutputTypes ["Output Types (types/generation.py)"]
        GenResult["GenerationResult<br/>- generated_files<br/>- validation_result<br/>- performance_metrics"]
        GenFile["GeneratedFile<br/>- path<br/>- content<br/>- description"]
    end
    
    %% Flow connections
    RTL --> Parsing
    ParsedData --> InterfaceBuilding
    ParsedData --> Port
    ParsedData --> Param
    
    IM --> PortGroup
    IM --> InterfaceMeta
    
    InterfaceMeta --> KernelMeta
    KernelMeta --> MetadataStage
    
    KM --> TemplateStage
    TC --> TemplateCtx
    CB --> Binding
    
    TemplateCtx --> CodeGen
    Binding --> CodeGen
    
    GR --> GenResult
    GenResult --> GenFile
    
    %% Final outputs
    GenFile --> PythonCode["Python HWCustomOp<br/>Class"]
    GenFile --> SVWrapper["SystemVerilog<br/>Wrapper"]
    GenFile --> Backend["RTL Backend<br/>Python Module"]
    
    %% Styling
    classDef inputStyle fill:#e1f5e1,stroke:#4caf50,stroke-width:2px
    classDef typeStyle fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef stageStyle fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef outputStyle fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    
    class RTL inputStyle
    class Port,Param,PortGroup,InterfaceMeta,KernelMeta,DatatypeMeta,TemplateCtx,Binding,GenResult,GenFile typeStyle
    class Parsing,InterfaceBuilding,MetadataStage,TemplateStage,CodeGen stageStyle
    class PythonCode,SVWrapper,Backend outputStyle
```

## Abstraction Levels

The diagram shows five distinct abstraction levels:

### 1. **RTL Level** (Green)
- **Input**: Raw SystemVerilog source with pragma annotations
- **Types**: `Port`, `Parameter`, `Pragma`
- **Focus**: Hardware-specific details like port widths and directions

### 2. **Protocol Level** (Orange - Interface Building)
- **Process**: Group ports into logical interfaces
- **Types**: `PortGroup`, early `InterfaceMetadata`
- **Focus**: Protocol detection (AXI-Stream, AXI-Lite)

### 3. **Metadata Level** (Orange - Metadata Construction)
- **Process**: Apply pragmas, link parameters
- **Types**: `KernelMetadata`, complete `InterfaceMetadata`
- **Focus**: High-level kernel semantics and relationships

### 4. **Template Level** (Orange - Template Context)
- **Process**: Prepare for code generation
- **Types**: `TemplateContext`, `CodegenBinding`
- **Focus**: Python/C++ bindings and attribute mappings

### 5. **Output Level** (Pink)
- **Process**: Generate final artifacts
- **Types**: `GeneratedFile`, `GenerationResult`
- **Output**: Python HWCustomOp classes, SystemVerilog wrappers

## Key Transformations

1. **Parsing**: SystemVerilog → Structured RTL data
2. **Interface Building**: Individual ports → Grouped interfaces
3. **Metadata Construction**: RTL constructs → Semantic kernel model
4. **Context Generation**: Kernel model → Template-ready bindings
5. **Code Generation**: Templates + Context → Executable code

## Data Type Progression

```
Raw SV → Port/Parameter → PortGroup → InterfaceMetadata → KernelMetadata → TemplateContext → Generated Files
```

Each stage adds semantic information while abstracting away lower-level details, ultimately producing FINN-compatible Python operators from SystemVerilog RTL modules.