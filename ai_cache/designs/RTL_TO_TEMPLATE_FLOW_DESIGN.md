# RTL to AutoHWCustomOp Template Generation Flow

**Date:** January 6, 2025  
**Status:** Complete Design with Implementation  
**Version:** 2.0 (Post BDIM Pragma Redesign)

## Overview

This document describes the complete end-to-end flow from SystemVerilog RTL modules to generated AutoHWCustomOp templates in the Brainsmith Hardware Kernel Generator. The flow includes RTL parsing, pragma processing, interface metadata generation, block chunking strategy application, parameter resolution, and template generation.

## Architecture Overview

```mermaid
graph TB
    subgraph "Input Layer"
        RTL[SystemVerilog RTL Module]
        PRAGMAS[Brainsmith Pragmas]
    end
    
    subgraph "Parsing Layer"
        PARSER[Tree-sitter RTL Parser]
        PRAGMA_PROC[Pragma Processor]
        IFACE_EXTRACT[Interface Extractor]
    end
    
    subgraph "Metadata Layer"
        IFACE_META[Interface Metadata]
        CHUNK_STRAT[Block Chunking Strategy]
        DTYPE_CONST[DataType Constraints]
        PARAM_DEF[Parameter Definitions]
    end
    
    subgraph "Integration Layer"
        PARAM_RES[Parameter Resolution Bridge]
        DATAFLOW[DataflowModel Creation]
        AUTO_OP[AutoHWCustomOp Instance]
    end
    
    subgraph "Template Layer"
        TEMPLATE_CTX[Template Context Generation]
        CODE_GEN[Code Generation]
        OUTPUTS[Generated Files]
    end
    
    RTL --> PARSER
    PRAGMAS --> PRAGMA_PROC
    PARSER --> IFACE_EXTRACT
    PRAGMA_PROC --> IFACE_META
    IFACE_EXTRACT --> IFACE_META
    
    IFACE_META --> CHUNK_STRAT
    IFACE_META --> DTYPE_CONST
    IFACE_META --> PARAM_DEF
    
    CHUNK_STRAT --> PARAM_RES
    PARAM_DEF --> PARAM_RES
    PARAM_RES --> DATAFLOW
    DATAFLOW --> AUTO_OP
    
    AUTO_OP --> TEMPLATE_CTX
    TEMPLATE_CTX --> CODE_GEN
    CODE_GEN --> OUTPUTS
    
    style RTL fill:#e1f5fe
    style OUTPUTS fill:#e8f5e8
    style PARAM_RES fill:#fff3e0
    style AUTO_OP fill:#f3e5f5
```

## Detailed Flow Stages

### Stage 1: RTL Parsing and Interface Extraction

```mermaid
sequenceDiagram
    participant User
    participant CLI as HKG CLI
    participant Parser as RTL Parser
    participant TreeSitter as Tree-sitter
    participant Extractor as Interface Extractor
    
    User->>CLI: hkg parse module.sv
    CLI->>Parser: parse_rtl_file(module.sv)
    Parser->>TreeSitter: parse SystemVerilog
    TreeSitter-->>Parser: AST
    Parser->>Extractor: extract_interfaces(AST)
    Extractor-->>Parser: Raw Interface Data
    Parser-->>CLI: Parsed RTL Result
    CLI-->>User: Interface Summary
```

**Key Data Structures:**

```python
# Raw interface data from RTL parsing
@dataclass
class RawInterfaceData:
    name: str                    # "in0_V_data_V"
    direction: str              # "input" | "output" 
    width: int                  # Signal width in bits
    interface_type: str         # "axi_stream" | "axi_lite"
    protocol_signals: Dict[str, Any]  # TDATA, TVALID, TREADY, etc.
```

### Stage 2: Pragma Processing and Application

```mermaid
flowchart TD
    subgraph "Pragma Types"
        BDIM["@brainsmith bdim interface [shape] RINDEX=n"]
        DTYPE["@brainsmith datatype interface TYPE bits"]
        WEIGHT["@brainsmith weight interface"]
        PARAM["@brainsmith parameter NAME default_value"]
    end
    
    subgraph "Pragma Processing"
        PARSE_PRAGMA[Parse Pragma Syntax]
        VALIDATE[Validate Pragma Parameters]
        APPLY[Apply to Interface Metadata]
    end
    
    subgraph "BDIM Processing Detail"
        EXTRACT_SHAPE[Extract Block Shape]
        PARSE_PARAMS[Parse Parameter Names]
        CREATE_STRATEGY[Create BlockChunkingStrategy]
    end
    
    BDIM --> PARSE_PRAGMA
    DTYPE --> PARSE_PRAGMA
    WEIGHT --> PARSE_PRAGMA
    PARAM --> PARSE_PRAGMA
    
    PARSE_PRAGMA --> VALIDATE
    VALIDATE --> APPLY
    
    BDIM --> EXTRACT_SHAPE
    EXTRACT_SHAPE --> PARSE_PARAMS
    PARSE_PARAMS --> CREATE_STRATEGY
    
    style BDIM fill:#ffe0b2
    style EXTRACT_SHAPE fill:#fff3e0
```

**BDIM Pragma Processing:**

```python
# New BDIM pragma format (parameter names only)
"@brainsmith bdim in0_V_data_V [PE] RINDEX=0"
"@brainsmith bdim weights_V [SIMD,PE] RINDEX=1" 
"@brainsmith bdim out0_V [CHANNELS,:] RINDEX=0"

# Parsed BDIM pragma data
@dataclass
class BDimPragmaData:
    interface_name: str         # "in0_V_data_V"
    block_shape: List[str]      # ["PE"] or ["SIMD", "PE"] or ["CHANNELS", ":"]
    rindex: int                 # 0, 1, 2, etc.
    
# Resulting block chunking strategy
class BlockChunkingStrategy:
    def compute_chunking(self, tensor_shape: List[int], interface_name: str) -> Tuple[List[int], List[Union[int, str]]]:
        # Returns: (tensor_dims, block_dims)
        # tensor_dims = [1, 128, 128, 256]  # Original tensor shape
        # block_dims = [1, 128, "SIMD", "PE"]  # Symbolic parameters preserved
```

### Stage 3: Interface Metadata Generation

```mermaid
graph LR
    subgraph "Input Sources"
        RTL_DATA[Raw RTL Interface Data]
        PRAGMA_DATA[Applied Pragma Data]
        DEFAULTS[System Defaults]
    end
    
    subgraph "Metadata Construction"
        COMBINE[Combine Data Sources]
        VALIDATE_META[Validate Metadata]
        CREATE_META[Create InterfaceMetadata]
    end
    
    subgraph "Metadata Components"
        IFACE_TYPE[Interface Type]
        CHUNK_STRATEGY[Chunking Strategy]
        DTYPE_CONSTRAINTS[DataType Constraints]
        AXI_CONFIG[AXI Protocol Config]
    end
    
    RTL_DATA --> COMBINE
    PRAGMA_DATA --> COMBINE
    DEFAULTS --> COMBINE
    
    COMBINE --> VALIDATE_META
    VALIDATE_META --> CREATE_META
    
    CREATE_META --> IFACE_TYPE
    CREATE_META --> CHUNK_STRATEGY
    CREATE_META --> DTYPE_CONSTRAINTS
    CREATE_META --> AXI_CONFIG
    
    style CREATE_META fill:#e1f5fe
```

**Interface Metadata Structure:**

```python
@dataclass
class InterfaceMetadata:
    name: str                           # "in0_V_data_V"
    interface_type: InterfaceType       # INPUT, OUTPUT, WEIGHT, CONFIG
    allowed_datatypes: List[DataTypeConstraint]  # UINT8, INT16, etc.
    chunking_strategy: BlockChunkingStrategy     # BDIM pragma strategy
    default_layout: Optional[str]       # AXI layout information
    description: Optional[str]          # Human-readable description
    
    def get_default_datatype(self) -> Optional[DataTypeConstraint]:
        return self.allowed_datatypes[0] if self.allowed_datatypes else None
```

### Stage 4: Parameter Resolution and DataflowModel Creation

```mermaid
sequenceDiagram
    participant Template as Template Generator
    participant AutoOp as AutoHWCustomOp
    participant Resolver as Parameter Resolver
    participant Strategy as BlockChunkingStrategy
    participant Dataflow as DataflowModel
    
    Template->>AutoOp: __init__(metadata, runtime_parameters)
    Note over Template,AutoOp: runtime_parameters = {"PE": 8, "SIMD": 4}
    
    AutoOp->>AutoOp: _build_dataflow_model_with_defaults()
    
    loop For each interface metadata
        AutoOp->>AutoOp: _extract_tensor_shape_from_onnx()
        Note over AutoOp: tensor_shape = [1, 128, 128, 256]
        
        AutoOp->>Strategy: compute_chunking(tensor_shape, interface_name)
        Strategy-->>AutoOp: (tensor_dims, symbolic_block_dims)
        Note over Strategy,AutoOp: symbolic_block_dims = [1, 128, "SIMD", "PE"]
        
        AutoOp->>Resolver: _resolve_block_dimensions(symbolic_block_dims)
        
        alt Parameter found
            Resolver-->>AutoOp: [1, 128, 4, 8]
        else Parameter missing with runtime_params
            Resolver-->>AutoOp: ValueError("Parameter 'PE' not found")
        else No runtime_params
            Resolver-->>AutoOp: [1, 128, 1, 1] (defaults)
        end
        
        AutoOp->>Dataflow: Create DataflowInterface
        Note over Dataflow: block_dims = [1, 128, 4, 8] (concrete integers only)
    end
    
    AutoOp->>Dataflow: DataflowModel(interfaces, {})
    Dataflow-->>AutoOp: Concrete DataflowModel
```

**Parameter Resolution Algorithm:**

```python
def _resolve_block_dimensions(self, block_dims: List[Union[int, str]], 
                              interface_name: str, tensor_shape: List[int]) -> List[int]:
    """Convert symbolic parameters to concrete integers."""
    resolved = []
    for i, dim in enumerate(block_dims):
        if isinstance(dim, int):
            resolved.append(dim)  # Already concrete
        elif isinstance(dim, str):
            if dim == ":":
                # Full dimension - use corresponding tensor dimension
                resolved.append(tensor_shape[i] if i < len(tensor_shape) else 1)
            elif dim in self._runtime_parameters:
                # Parameter name - resolve to concrete value
                resolved.append(self._runtime_parameters[dim])
            else:
                # Missing parameter handling
                if not self._runtime_parameters:
                    resolved.append(1)  # Default when no runtime params
                else:
                    raise ValueError(f"Parameter '{dim}' not found in runtime_parameters")
    return resolved
```

### Stage 5: Template Context Generation

```mermaid
graph TB
    subgraph "Context Inputs"
        AUTO_OP[AutoHWCustomOp Instance]
        KERNEL_META[Kernel Metadata]
        PARAM_OVERRIDES[Parameter Overrides]
    end
    
    subgraph "Context Generation"
        EXTRACT_DATAFLOW[Extract Dataflow Model]
        EXTRACT_INTERFACES[Extract Interface Configs]
        EXTRACT_PARAMS[Extract Parameters]
        BUILD_CONTEXT[Build Template Context]
    end
    
    subgraph "Context Components"
        DATAFLOW_CTX[Dataflow Model Context]
        INTERFACE_CTX[Interface Configurations]
        PARAM_CTX[Parameter Context]
        SYMBOLIC_CTX[Symbolic Parameter Context]
        CLASS_META[Class Metadata]
    end
    
    AUTO_OP --> EXTRACT_DATAFLOW
    KERNEL_META --> EXTRACT_PARAMS
    PARAM_OVERRIDES --> EXTRACT_PARAMS
    
    EXTRACT_DATAFLOW --> BUILD_CONTEXT
    EXTRACT_INTERFACES --> BUILD_CONTEXT
    EXTRACT_PARAMS --> BUILD_CONTEXT
    
    BUILD_CONTEXT --> DATAFLOW_CTX
    BUILD_CONTEXT --> INTERFACE_CTX
    BUILD_CONTEXT --> PARAM_CTX
    BUILD_CONTEXT --> SYMBOLIC_CTX
    BUILD_CONTEXT --> CLASS_META
    
    style BUILD_CONTEXT fill:#f3e5f5
    style SYMBOLIC_CTX fill:#fff3e0
```

**Template Context Structure:**

```python
@dataclass
class TemplateContext:
    # Core components
    auto_hw_custom_op: AutoHWCustomOp           # Concrete dataflow model
    dataflow_model: DataflowModel               # Always contains integers
    
    # Parameter contexts
    runtime_parameters: Dict[str, int]          # Concrete values: {"PE": 8, "SIMD": 4}
    symbolic_parameters: Dict[str, str]         # Template placeholders: {"PE": "${PE}$"}
    
    # Interface configurations  
    interface_configs: Dict[str, Dict]          # Per-interface configuration
    stream_widths: Dict[str, int]              # Calculated stream widths
    parallelism_bounds: Dict[str, ParallelismBounds]  # Valid parallelism ranges
    
    # Class generation metadata
    class_name: str                            # "MyAcceleratorHWCustomOp"
    module_name: str                           # "my_accelerator"
    base_imports: List[str]                    # Required imports
    
    # Resource estimation
    resource_estimates: Dict[str, Any]         # BRAM, LUT, DSP estimates
```

### Stage 6: Code Generation and Template Processing

```mermaid
flowchart TD
    subgraph "Template Selection"
        HW_TEMPLATE[HWCustomOp Template]
        RTL_TEMPLATE[RTLBackend Template]
        WRAPPER_TEMPLATE[Verilog Wrapper Template]
        TEST_TEMPLATE[Test Suite Template]
    end
    
    subgraph "Template Processing"
        JINJA_ENGINE[Jinja2 Template Engine]
        CONTEXT_INJECT[Inject Template Context]
        RENDER[Render Templates]
    end
    
    subgraph "Generated Outputs"
        HW_OP_PY[my_accelerator_hw_custom_op.py]
        RTL_BACKEND_PY[my_accelerator_rtl_backend.py]
        WRAPPER_SV[my_accelerator_wrapper.sv]
        TEST_PY[test_my_accelerator.py]
        DOCS_MD[my_accelerator_docs.md]
    end
    
    HW_TEMPLATE --> JINJA_ENGINE
    RTL_TEMPLATE --> JINJA_ENGINE
    WRAPPER_TEMPLATE --> JINJA_ENGINE
    TEST_TEMPLATE --> JINJA_ENGINE
    
    JINJA_ENGINE --> CONTEXT_INJECT
    CONTEXT_INJECT --> RENDER
    
    RENDER --> HW_OP_PY
    RENDER --> RTL_BACKEND_PY
    RENDER --> WRAPPER_SV
    RENDER --> TEST_PY
    RENDER --> DOCS_MD
    
    style JINJA_ENGINE fill:#e8f5e8
    style CONTEXT_INJECT fill:#fff3e0
```

## Key Parameter Flow Examples

### Example 1: Simple PE Parameter

```mermaid
graph TD
    subgraph "RTL Input"
        RTL1["module simple_pe(...);<br/>// @brainsmith bdim in0_V [PE]<br/>input [31:0] in0_V_data_V;"]
    end
    
    subgraph "Pragma Processing"
        BDIM1["BDimPragma:<br/>interface: 'in0_V'<br/>block_shape: ['PE']<br/>rindex: 0"]
    end
    
    subgraph "Chunking Strategy"
        CHUNK1["BlockChunkingStrategy:<br/>block_shape: ['PE']<br/>compute_chunking() →<br/>([1,128,128,256], [1,128,128,'PE'])"]
    end
    
    subgraph "Parameter Resolution"
        RESOLVE1["runtime_parameters: {'PE': 8}<br/>_resolve_block_dimensions() →<br/>[1,128,128,8]"]
    end
    
    subgraph "Template Context"
        CONTEXT1["symbolic_parameters: {'PE': '${PE}$'}<br/>runtime_parameters: {'PE': 8}<br/>dataflow_model.block_dims: [1,128,128,8]"]
    end
    
    RTL1 --> BDIM1
    BDIM1 --> CHUNK1
    CHUNK1 --> RESOLVE1
    RESOLVE1 --> CONTEXT1
    
    style RTL1 fill:#e1f5fe
    style CONTEXT1 fill:#e8f5e8
```

### Example 2: Multi-Parameter with Colon

```mermaid
graph TD
    subgraph "RTL Input"
        RTL2["module multi_param(...);<br/>// @brainsmith bdim weights_V [SIMD,:,PE] RINDEX=1<br/>input [127:0] weights_V_data_V;"]
    end
    
    subgraph "Pragma Processing"
        BDIM2["BDimPragma:<br/>interface: 'weights_V'<br/>block_shape: ['SIMD',':','PE']<br/>rindex: 1"]
    end
    
    subgraph "Chunking Strategy"
        CHUNK2["tensor_shape: [1,128,128,256]<br/>block_shape at positions [0,1,2]:<br/>block_dims: ['SIMD',128,'PE',256]"]
    end
    
    subgraph "Parameter Resolution"
        RESOLVE2["runtime_parameters: {'SIMD': 4, 'PE': 8}<br/>':' → tensor_shape[1] = 128<br/>final: [4,128,8,256]"]
    end
    
    subgraph "Template Context"
        CONTEXT2["symbolic: {'SIMD': '${SIMD}$', 'PE': '${PE}$'}<br/>concrete: [4,128,8,256]<br/>templates use symbolic for generation"]
    end
    
    RTL2 --> BDIM2
    BDIM2 --> CHUNK2
    CHUNK2 --> RESOLVE2
    RESOLVE2 --> CONTEXT2
    
    style RTL2 fill:#e1f5fe
    style CONTEXT2 fill:#e8f5e8
```

## Data Flow Summary

### Complete Parameter Journey

```mermaid
flowchart LR
    subgraph "RTL Layer"
        PRAGMA_TEXT["@brainsmith bdim in0_V [PE]"]
    end
    
    subgraph "Parsing Layer"
        PARSED_PRAGMA["BDimPragma.block_shape = ['PE']"]
    end
    
    subgraph "Strategy Layer"
        SYMBOLIC_DIMS["block_dims = [1,128,128,'PE']"]
    end
    
    subgraph "Resolution Layer"
        CONCRETE_DIMS["block_dims = [1,128,128,8]"]
        RUNTIME_PARAMS["runtime_parameters = {'PE': 8}"]
    end
    
    subgraph "Template Layer"
        SYMBOLIC_TEMPLATE["Template uses '${PE}$'"]
        CONCRETE_MODEL["DataflowModel uses [1,128,128,8]"]
    end
    
    PRAGMA_TEXT --> PARSED_PRAGMA
    PARSED_PRAGMA --> SYMBOLIC_DIMS
    SYMBOLIC_DIMS --> CONCRETE_DIMS
    RUNTIME_PARAMS --> CONCRETE_DIMS
    CONCRETE_DIMS --> CONCRETE_MODEL
    SYMBOLIC_DIMS --> SYMBOLIC_TEMPLATE
    
    style PRAGMA_TEXT fill:#ffe0b2
    style CONCRETE_DIMS fill:#fff3e0
    style CONCRETE_MODEL fill:#e8f5e8
    style SYMBOLIC_TEMPLATE fill:#f3e5f5
```

## Architecture Benefits

### 1. Separation of Concerns

- **RTL Parsing**: Pure SystemVerilog analysis without parameter resolution
- **Pragma Processing**: Symbolic parameter extraction and validation
- **Parameter Resolution**: Bridge between symbolic and concrete representations
- **Template Generation**: Uses both symbolic (for code gen) and concrete (for modeling)

### 2. Flexibility and Extensibility

- **Parameter System**: Easy to add new parameter types and validation rules
- **Template System**: Jinja2 templates can access both symbolic and concrete contexts
- **Backend Support**: Works with any FINN-compatible backend

### 3. Error Handling and Validation

```mermaid
graph TB
    subgraph "Validation Layers"
        RTL_VALID[RTL Syntax Validation]
        PRAGMA_VALID[Pragma Syntax Validation]
        PARAM_VALID[Parameter Resolution Validation]
        DATAFLOW_VALID[DataflowModel Validation]
        TEMPLATE_VALID[Template Generation Validation]
    end
    
    subgraph "Error Types"
        PARSE_ERR[Parse Errors]
        PRAGMA_ERR[Pragma Errors]
        PARAM_ERR[Missing Parameter Errors]
        TYPE_ERR[Type Mismatch Errors]
        TEMPLATE_ERR[Template Errors]
    end
    
    RTL_VALID --> PARSE_ERR
    PRAGMA_VALID --> PRAGMA_ERR
    PARAM_VALID --> PARAM_ERR
    DATAFLOW_VALID --> TYPE_ERR
    TEMPLATE_VALID --> TEMPLATE_ERR
    
    style PARAM_VALID fill:#fff3e0
    style PARAM_ERR fill:#ffebee
```

## Implementation Status

### ✅ Completed Components

1. **RTL Parser**: Tree-sitter based SystemVerilog parsing
2. **BDIM Pragma System**: New parameter-only format with validation
3. **Block Chunking Strategy**: Symbolic parameter preservation
4. **Parameter Resolution Bridge**: AutoHWCustomOp integration
5. **Template Context Generation**: Dual symbolic/concrete context
6. **Test Coverage**: Comprehensive test suite (44 tests passing)

### 🔄 Integration Points

1. **Template Generator Integration**: Runtime parameter extraction from module parameters
2. **FINN Backend Integration**: AutoHWCustomOp instantiation with resolved parameters
3. **CLI Integration**: Parameter override support in command-line interface

## Usage Examples

### CLI Usage

```bash
# Parse RTL and show interface analysis
python -m brainsmith.tools.hw_kernel_gen.cli parse my_accelerator.sv

# Generate templates with parameter overrides
python -m brainsmith.tools.hw_kernel_gen.cli generate my_accelerator.sv compiler_data.json \
    -o output_dir \
    --param PE=8 --param SIMD=4 --param CHANNELS=32
```

### Programmatic Usage

```python
# Create AutoHWCustomOp with parameter resolution
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp

auto_op = AutoHWCustomOp(
    onnx_node=onnx_node,
    interface_metadata=parsed_metadata,
    runtime_parameters={"PE": 8, "SIMD": 4, "CHANNELS": 32}
)

# DataflowModel contains only concrete integers
assert auto_op.dataflow_model.input_interfaces[0].block_dims == [1, 128, 4, 8]

# Generate template context for code generation
context = TemplateContextGenerator.generate_context(
    kernel_metadata=kernel_meta,
    parameter_overrides={"PE": 8, "SIMD": 4}
)

# Templates access both symbolic and concrete representations
template_code = template.render(context)
```

This design provides a robust, extensible framework for RTL-to-template generation with proper parameter handling, clear separation of concerns, and comprehensive error handling throughout the pipeline.