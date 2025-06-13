# Hardware Kernel Generator (HWKG) Architecture Analysis

## Overview

This document provides a comprehensive analysis of the Hardware Kernel Generator (HWKG) system based on analysis of test files and core components in `brainsmith/tools/hw_kernel_gen/`. The HWKG transforms SystemVerilog RTL modules with embedded pragmas into FINN-compatible Python components through a sophisticated 3-phase pipeline.

## High-Level Architecture

```mermaid
graph TB
    A[SystemVerilog RTL File] --> B[Phase 1: RTL Parsing]
    B --> C[KernelMetadata]
    C --> D[Phase 2: Template Context Generation]
    D --> E[TemplateContext]
    E --> F[Phase 3: Code Generation]
    F --> G[Generated Files]
    G --> H[Phase 4: Result Handling]
    H --> I[File System Output]
    
    subgraph "Phase 1: RTL Parsing"
        B1[RTLParser]
        B2[PragmaHandler]
        B3[InterfaceBuilder]
        B4[ProtocolValidator]
    end
    
    subgraph "Phase 2: Context Generation"
        D1[TemplateContextGenerator]
        D2[ParameterClassification]
        D3[InterfaceCategorization]
        D4[AlgorithmInference]
    end
    
    subgraph "Phase 3: Code Generation"
        F1[UnifiedGenerator]
        F2[Jinja2 Templates]
        F3[Template Validation]
        F4[Code Assembly]
    end
    
    subgraph "Phase 4: Result Handling"
        H1[ResultHandler]
        H2[FileOrganization]
        H3[MetadataGeneration]
        H4[ErrorHandling]
    end
```

## Phase 1: RTL Parsing & Metadata Extraction

### Component Architecture

```mermaid
classDiagram
    class RTLParser {
        -tree: Node
        -pragmas: List[Pragma]
        -module_node: Node
        -name: str
        -parameters: List[Parameter]
        -ports: List[Port]
        -interface_metadata_list: List[InterfaceMetadata]
        +parse_file(file_path: str) KernelMetadata
        +parse(code: str, source_name: str) KernelMetadata
        -_initial_parse(file_path: str) void
        -_extract_kernel_components() void
        -_analyze_and_validate_interfaces() void
    }
    
    class InterfaceBuilder {
        -scanner: InterfaceScanner
        -validator: ProtocolValidator
        +build_interface_metadata(ports: List[Port], pragmas: List[Pragma]) Tuple[List[InterfaceMetadata], List[Port]]
        -_create_base_metadata(group: PortGroup) InterfaceMetadata
    }
    
    class PragmaHandler {
        -pragma_constructors: Dict[PragmaType, Callable]
        +extract_pragmas(root_node: Node) List[Pragma]
        -_validate_pragma(node: Node, line_number: int) Optional[Pragma]
    }
    
    class KernelMetadata {
        +name: str
        +source_file: Path
        +interfaces: List[InterfaceMetadata]
        +parameters: List[Parameter]
        +pragmas: List[Pragma]
        +parsing_warnings: List[str]
    }
    
    class InterfaceMetadata {
        +name: str
        +interface_type: InterfaceType
        +datatype_constraints: List[DatatypeConstraintGroup]
        +chunking_strategy: ChunkingStrategy
        +description: Optional[str]
    }
    
    RTLParser --> InterfaceBuilder: uses
    RTLParser --> PragmaHandler: uses
    RTLParser --> KernelMetadata: produces
    InterfaceBuilder --> InterfaceMetadata: creates
    KernelMetadata --> InterfaceMetadata: contains
```

### Pragma Processing Flow

```mermaid
sequenceDiagram
    participant P as RTLParser
    participant PH as PragmaHandler
    participant IB as InterfaceBuilder
    participant Pragma as Pragma Classes
    
    P->>PH: extract_pragmas(root_node)
    PH->>PH: find comment nodes
    PH->>Pragma: instantiate pragma subclasses
    Pragma->>Pragma: _parse_inputs() & validate
    PH->>P: return List[Pragma]
    
    P->>IB: build_interface_metadata(ports, pragmas)
    IB->>IB: create base metadata from ports
    
    loop For each pragma
        IB->>Pragma: applies_to_interface_metadata(metadata)
        alt Pragma applies
            IB->>Pragma: apply_to_metadata(metadata)
            Pragma->>IB: return modified metadata
        end
    end
    
    IB->>P: return (metadata_list, unassigned_ports)
```

### Key Data Transformations

```mermaid
graph LR
    A[SystemVerilog Ports] --> B[PortGroup]
    B --> C[Base InterfaceMetadata]
    C --> D[Pragma Application]
    D --> E[Final InterfaceMetadata]
    
    F[BDIM Pragma] --> G[BlockChunkingStrategy]
    H[DATATYPE Pragma] --> I[DatatypeConstraintGroup]
    J[WEIGHT Pragma] --> K[InterfaceType.WEIGHT]
    
    G --> E
    I --> E
    K --> E
```

## Phase 2: Template Context Generation

### Core Components

```mermaid
classDiagram
    class TemplateContextGenerator {
        +generate_template_context(kernel_metadata: KernelMetadata) TemplateContext
        +generate_context(kernel_metadata: KernelMetadata) Dict[str, Any]
        -_analyze_parallelism_parameters(kernel_metadata: KernelMetadata) Dict[str, Any]
        -_infer_algorithm_parameters(kernel_metadata: KernelMetadata) Dict[str, Any]
        -_get_interfaces_by_type(kernel_metadata: KernelMetadata, interface_type: InterfaceType) List[InterfaceMetadata]
        -_template_context_to_dict(template_context: TemplateContext) Dict[str, Any]
    }
    
    class TemplateContext {
        +module_name: str
        +class_name: str
        +source_file: Path
        +interface_metadata: List[InterfaceMetadata]
        +parameter_definitions: List[ParameterDefinition]
        +whitelisted_defaults: Dict[str, int]
        +required_attributes: List[str]
        +input_interfaces: List[InterfaceMetadata]
        +output_interfaces: List[InterfaceMetadata]
        +weight_interfaces: List[InterfaceMetadata]
        +config_interfaces: List[InterfaceMetadata]
        +control_interfaces: List[InterfaceMetadata]
        +get_node_attribute_definitions() Dict[str, Tuple[str, bool, Any]]
        +get_runtime_parameter_extraction_code() List[str]
        +get_interface_metadata_code() List[str]
        +validate() List[str]
    }
    
    class ParameterDefinition {
        +name: str
        +param_type: Optional[str]
        +default_value: Optional[int]
        +description: Optional[str]
        +line_number: int
        +template_param_name: Optional[str]
        +is_whitelisted: bool
        +is_required: bool
    }
    
    TemplateContextGenerator --> TemplateContext: creates
    TemplateContext --> ParameterDefinition: contains
```

### Parameter Classification System

```mermaid
graph TB
    A[RTL Parameters] --> B{Is Whitelisted?}
    B -->|Yes| C{Has RTL Default?}
    B -->|No| D[Required Attribute]
    
    C -->|Yes| E[Use RTL Default]
    C -->|No| F[Use System Default]
    
    E --> G[Add to whitelisted_defaults]
    F --> G
    D --> H[Add to required_attributes]
    
    G --> I[Optional FINN Attribute]
    H --> J[Required FINN Attribute]
    
    subgraph "Whitelisted Parameters"
        K["PE, SIMD, DEPTH<br/>CHANNELS, FILTERS<br/>KERNEL_SIZE, etc."]
    end
    
    subgraph "Non-Whitelisted Parameters"
        L["Custom parameters<br/>Algorithm-specific<br/>Domain-specific"]
    end
```

### Template Context Generation Flow

```mermaid
sequenceDiagram
    participant KM as KernelMetadata
    participant TCG as TemplateContextGenerator
    participant PC as ParameterClassifier
    participant IC as InterfaceCategorizer
    participant TC as TemplateContext
    
    KM->>TCG: generate_template_context()
    
    TCG->>PC: analyze parameters
    loop For each parameter
        PC->>PC: check whitelist status
        PC->>PC: determine default value
        PC->>PC: classify as required/optional
    end
    PC->>TCG: return parameter_definitions, whitelisted_defaults, required_attributes
    
    TCG->>IC: categorize interfaces
    IC->>IC: group by InterfaceType
    IC->>TCG: return categorized interfaces
    
    TCG->>TC: create TemplateContext
    TC->>TC: validate consistency
    TCG->>KM: return TemplateContext
```

## Phase 3: Code Generation

### UnifiedGenerator Architecture

```mermaid
classDiagram
    class UnifiedGenerator {
        -template_dir: Path
        -template_context_generator: TemplateContextGenerator
        -jinja_env: Environment
        +generate_hw_custom_op(kernel_metadata: KernelMetadata) str
        +generate_rtl_wrapper(kernel_metadata: KernelMetadata) str
        +generate_test_suite(kernel_metadata: KernelMetadata) str
        +generate_all(kernel_metadata: KernelMetadata) Dict[str, str]
        +get_available_templates() List[str]
        +validate_templates() Dict[str, bool]
        -_template_context_to_dict(template_context: TemplateContext) Dict[str, Any]
    }
    
    class TemplateSystem {
        <<interface>>
        +hw_custom_op_phase2.py.j2
        +rtl_wrapper_v2.v.j2
        +test_suite_v2.py.j2
        +rtl_wrapper.v.j2 [fallback]
        +test_suite.py.j2 [fallback]
    }
    
    UnifiedGenerator --> TemplateSystem: uses
    UnifiedGenerator --> TemplateContextGenerator: uses
```

### Template Generation Flow

```mermaid
sequenceDiagram
    participant UG as UnifiedGenerator
    participant TCG as TemplateContextGenerator
    participant J2 as Jinja2Environment
    participant T as Templates
    
    UG->>TCG: generate_template_context(kernel_metadata)
    TCG->>UG: return TemplateContext
    
    UG->>UG: _template_context_to_dict(template_context)
    
    loop For each template type
        UG->>J2: get_template(template_name)
        J2->>T: load template
        T->>J2: return template object
        J2->>UG: return template
        
        UG->>J2: template.render(**context_dict)
        J2->>UG: return rendered code
    end
    
    UG->>UG: assemble generated_files dict
```

### Generated Files Structure

```mermaid
graph TB
    A[UnifiedGenerator.generate_all] --> B[Generated Files Dict]
    
    B --> C["module_name_hw_custom_op.py"]
    B --> D["module_name_wrapper.v"]
    B --> E["test_module_name.py"]
    
    C --> C1["AutoHWCustomOp Subclass"]
    C --> C2["Runtime Parameter Extraction"]
    C --> C3["Interface Metadata Definition"]
    C --> C4["Node Attribute Definitions"]
    
    D --> D1["SystemVerilog Wrapper"]
    D --> D2["Parameter Validation"]
    D --> D3["Interface Width Calculations"]
    D --> D4["Error Handling"]
    
    E --> E1["pytest Test Class"]
    E --> E2["Parameter Validation Tests"]
    E --> E3["Instantiation Tests"]
    E --> E4["Interface Configuration Tests"]
```

## Phase 4: Result Handling

### ResultHandler Architecture

```mermaid
classDiagram
    class ResultHandler {
        -output_dir: Path
        +write_result(result: GenerationResult) Path
        +cleanup_failed_generation(kernel_name: str) void
        +get_existing_results() List[str]
        +load_result_metadata(kernel_name: str) Optional[Dict]
        -_write_metadata(kernel_dir: Path, result: GenerationResult) void
        -_write_summary_log(kernel_dir: Path, result: GenerationResult, file_paths: List[str]) void
    }
    
    class GenerationResult {
        +kernel_name: str
        +source_file: Path
        +generated_files: Dict[str, str]
        +template_context: Optional[TemplateContext]
        +kernel_metadata: Optional[KernelMetadata]
        +validation_passed: bool
        +errors: List[str]
        +warnings: List[str]
        +generation_time_ms: Optional[float]
        +add_error(message: str) void
        +add_warning(message: str) void
        +is_success() bool
        +get_summary() Dict[str, Any]
    }
    
    ResultHandler --> GenerationResult: processes
```

### File Organization Structure

```mermaid
graph TB
    A[output_dir/] --> B[kernel_name/]
    
    B --> C[Generated Source Files]
    B --> D[Metadata Files]
    B --> E[Documentation]
    
    C --> C1["{kernel_name}_hw_custom_op.py"]
    C --> C2["{kernel_name}_wrapper.v"]
    C --> C3["test_{kernel_name}.py"]
    
    D --> D1["generation_metadata.json"]
    D --> D2["generation_summary.txt"]
    
    E --> E1["README.md [optional]"]
    E --> E2["documentation.md [optional]"]
```

## Complete End-to-End Flow

### Comprehensive System Flow

```mermaid
sequenceDiagram
    participant RTL as SystemVerilog RTL
    participant P as RTLParser
    participant TCG as TemplateContextGenerator
    participant UG as UnifiedGenerator
    participant RH as ResultHandler
    participant FS as File System
    
    Note over RTL,FS: Phase 1: RTL Parsing
    RTL->>P: parse_file(rtl_path)
    P->>P: _initial_parse() - AST generation
    P->>P: _extract_kernel_components() - extract parameters/ports
    P->>P: _analyze_and_validate_interfaces() - create InterfaceMetadata
    P->>TCG: return KernelMetadata
    
    Note over TCG,UG: Phase 2: Template Context Generation
    TCG->>TCG: generate_template_context(kernel_metadata)
    TCG->>TCG: classify parameters (whitelisted vs required)
    TCG->>TCG: categorize interfaces by type
    TCG->>TCG: infer algorithm parameters
    TCG->>UG: return TemplateContext
    
    Note over UG,RH: Phase 3: Code Generation
    UG->>UG: generate_all(kernel_metadata)
    UG->>UG: generate_hw_custom_op() - AutoHWCustomOp subclass
    UG->>UG: generate_rtl_wrapper() - SystemVerilog wrapper
    UG->>UG: generate_test_suite() - pytest tests
    UG->>RH: return GenerationResult with generated_files
    
    Note over RH,FS: Phase 4: Result Handling
    RH->>RH: write_result(generation_result)
    RH->>FS: create kernel directory
    RH->>FS: write generated source files
    RH->>FS: write generation_metadata.json
    RH->>FS: write generation_summary.txt
    RH->>P: return kernel_output_directory
```

## Key Features & Capabilities

### 1. **Runtime Parameter Extraction**

Generated `AutoHWCustomOp` subclasses extract parameters from ONNX nodes:

```python
# Generated in __init__ method:
self.runtime_parameters = {}
self.runtime_parameters["PE"] = self.get_nodeattr("PE")
self.runtime_parameters["SIMD"] = self.get_nodeattr("SIMD")
# Required parameters fail if not provided by FINN
```

### 2. **Symbolic BDIM Validation**

BDIM pragmas reference RTL parameters and are validated during parsing:

```systemverilog
// @brainsmith bdim input0 [PE]           // Valid: PE is RTL parameter
// @brainsmith bdim weights [SIMD,PE]     // Valid: both are RTL parameters  
// @brainsmith bdim output0 [16]          // ERROR: magic numbers forbidden
```

### 3. **Parameter Whitelist System**

Parameters are classified for FINN integration:

- **Whitelisted** (PE, SIMD, DEPTH, etc.): Can have defaults, become optional ONNX attributes
- **Non-Whitelisted**: Must be provided by FINN, become required ONNX attributes

### 4. **Interface Type System**

Unified interface types with protocol binding:

- `INPUT/OUTPUT/WEIGHT` → Always AXI-Stream
- `CONFIG` → Always AXI-Lite  
- `CONTROL` → Always global signals (clk, rst)

### 5. **Enhanced Error Handling**

Comprehensive validation at each phase:

- **Phase 1**: Syntax validation, pragma validation, parameter reference checking
- **Phase 2**: Parameter classification validation, interface categorization
- **Phase 3**: Template validation, code generation validation
- **Phase 4**: File system validation, metadata consistency

## Performance Characteristics

Based on test analysis:

- **Parsing**: < 3s for complex modules (50+ parameters, 20+ interfaces)
- **Context Generation**: < 1s for parameter classification and interface categorization
- **Code Generation**: < 5s for multi-template generation (3 templates)
- **Total Pipeline**: < 10s end-to-end for complex kernels
- **Memory Usage**: Reasonable growth with module complexity
- **Scalability**: Handles production-scale FPGA kernels efficiently

## Testing Architecture

### Test Organization

```mermaid
graph TB
    A[HWKG Tests] --> B[Unit Tests]
    A --> C[Integration Tests]
    A --> D[End-to-End Tests]
    
    B --> B1[Template Context Unit Tests]
    B --> B2[Result Handler Unit Tests]
    B --> B3[Unified Generator Unit Tests]
    
    C --> C1[RTL → Metadata Integration]
    C --> C2[Template Generation Integration]
    C --> C3[Parameter Extraction Integration]
    
    D --> D1[Phase 3 End-to-End]
    D --> D2[Complex RTL Integration]
    D --> D3[Constraint Groups Integration]
```

### Test Coverage Areas

1. **RTL Parser Integration**: Complete RTL → `KernelMetadata` pipeline
2. **Template Context Generation**: Parameter classification, interface categorization
3. **Code Generation**: Template rendering, validation, file generation  
4. **Result Handling**: File organization, metadata generation, error handling
5. **End-to-End**: Complete RTL → generated files pipeline with real examples

## Architecture Evolution

### Legacy vs. Current

```mermaid
graph LR
    subgraph "Legacy Architecture"
        A1[Multiple Generator Classes]
        A2[DataflowModel Intermediary]
        A3[Dual Interface Types]
        A4[Manual Template Context]
        A5[Complex Template Logic]
    end
    
    subgraph "Current Architecture (Phase 2/3)"
        B1[UnifiedGenerator]
        B2[Direct RTL → Template]
        B3[Unified Interface Types]
        B4[Automated Context Generation]
        B5[Minimal Template Logic]
    end
    
    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4
    A5 --> B5
```

### Architectural Improvements

1. **Unified Generation**: Single `UnifiedGenerator` class replaces multiple generator classes
2. **Performance**: Direct RTL → Template pipeline bypasses `DataflowModel` overhead
3. **Validation**: Comprehensive parameter and pragma validation at parse time
4. **Extensibility**: Template-based approach enables easy extension for new kernel types
5. **Integration**: Seamless FINN integration with proper node attribute handling

## Conclusion

The HWKG represents a sophisticated, well-architected system for transforming custom RTL modules into FINN-compatible components. The 4-phase pipeline provides clear separation of concerns while maintaining high performance and extensibility. The comprehensive test coverage and validation at each phase ensure reliability for production FPGA AI accelerator development.

The architecture successfully bridges the gap between custom hardware implementations and the FINN compilation framework, enabling hardware engineers to leverage their existing RTL while integrating seamlessly with the broader FPGA AI ecosystem.