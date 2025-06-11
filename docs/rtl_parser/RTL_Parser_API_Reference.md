# RTL Parser API Reference

## Overview

The RTL Parser is a sophisticated SystemVerilog analysis engine that extracts module interfaces, parameters, and Brainsmith-specific pragmas for hardware kernel integration. It provides two parsing modes: standard RTL parsing for DataflowModel conversion and enhanced parsing for direct template generation.

## API Entry Points

### Core Parsing Functions

#### `parse_rtl_file(rtl_file, advanced_pragmas=False) -> RTLParsingResult`

Parse RTL file and return lightweight parsing result optimized for DataflowModel conversion.

**Parameters:**
- `rtl_file` (str|Path): Path to SystemVerilog RTL file
- `advanced_pragmas` (bool): Enable enhanced BDIM pragma processing (default: False)

**Returns:**
- `RTLParsingResult`: Lightweight parsing result containing only essential data

**Raises:**
- `RTLParsingError`: If RTL parsing fails

**Usage:**
```python
from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file

result = parse_rtl_file("thresholding_axi.sv", advanced_pragmas=True)
print(f"Module: {result.name}")
print(f"Interfaces: {list(result.interfaces.keys())}")
```

#### `parse_rtl_file_enhanced(rtl_file, advanced_pragmas=False) -> EnhancedRTLParsingResult`

Parse RTL file and return enhanced parsing result for direct template generation, eliminating DataflowModel conversion overhead.

**Parameters:**
- `rtl_file` (str|Path): Path to SystemVerilog RTL file  
- `advanced_pragmas` (bool): Enable enhanced BDIM pragma processing (default: False)

**Returns:**
- `EnhancedRTLParsingResult`: Enhanced parsing result with template context generation

**Raises:**
- `RTLParsingError`: If RTL parsing fails

**Usage:**
```python
from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file_enhanced

enhanced_result = parse_rtl_file_enhanced("thresholding_axi.sv")
template_context = enhanced_result.get_template_context()
print(f"Template variables: {len(template_context)}")
```

### Core Classes

#### `RTLParser`

Main orchestrator using tree-sitter for robust SystemVerilog parsing.

**Methods:**
- `parse_file(file_path: str) -> HWKernel`: Parse RTL file and return full HWKernel object
- `parse_string(rtl_content: str) -> HWKernel`: Parse RTL from string content

#### `ProtocolValidator`

Validates interface protocol compliance against AXI standards.

**Methods:**
- `validate_interface(port_group: PortGroup) -> ValidationResult`: Validate protocol compliance

## Data Structures

### Core Result Types

#### `RTLParsingResult`

Lightweight result containing only data needed for DataflowModel conversion (6 essential properties).

**Attributes:**
- `name` (str): Module name
- `interfaces` (Dict[str, Interface]): RTL Interface objects
- `pragmas` (List[Pragma]): Parsed pragma objects
- `parameters` (List[Parameter]): Module parameters
- `source_file` (Optional[Path]): Source RTL file path
- `pragma_sophistication_level` (str): Pragma complexity level ("simple" | "advanced")
- `parsing_warnings` (List[str]): Parser warnings

#### `EnhancedRTLParsingResult`

Enhanced result with template-ready metadata generation, eliminating DataflowModel overhead.

**Attributes:**
- All `RTLParsingResult` attributes plus:
- `_template_context` (Dict[str, Any]): Cached template context

**Methods:**
- `get_template_context() -> Dict[str, Any]`: Get complete template context with 22+ variables

### Interface Types

#### `Interface`

Represents a fully validated and identified interface.

**Attributes:**
- `name` (str): Interface name (e.g., "global", "in0", "config")
- `type` (InterfaceType): Interface type (GLOBAL_CONTROL, AXI_STREAM, AXI_LITE)
- `ports` (Dict[str, Port]): Maps signal suffix/name to Port object
- `validation_result` (ValidationResult): Protocol validation result
- `metadata` (Dict[str, Any]): Interface metadata (data width, address width, etc.)
- `wrapper_name` (Optional[str]): Template wrapper name

#### `InterfaceType` (Enum)

Supported interface types:
- `GLOBAL_CONTROL`: Clock/reset signals (ap_clk, ap_rst_n, optional ap_clk2x)
- `AXI_STREAM`: Data flow interfaces (TDATA, TVALID, TREADY, optional TLAST)
- `AXI_LITE`: Configuration interfaces (read-only, write-only, or full)
- `UNKNOWN`: Ports not part of recognized interface

### Port and Parameter Types

#### `Port`

SystemVerilog port representation.

**Attributes:**
- `name` (str): Port identifier
- `direction` (Direction): Port direction (INPUT, OUTPUT, INOUT)
- `width` (Optional[str]): Bit width expression (preserved as string)
- `data_type` (Optional[str]): Port data type (logic, wire, etc.)
- `description` (Optional[str]): Documentation from RTL comments

#### `Parameter`

SystemVerilog parameter representation.

**Attributes:**
- `name` (str): Parameter identifier
- `param_type` (Optional[str]): Parameter datatype
- `default_value` (Optional[str]): Default value if specified
- `description` (Optional[str]): Documentation from RTL comments
- `template_param_name` (str): Template parameter name (computed, e.g., "$NAME$")

### Pragma System

#### `Pragma` (Base Class)

Brainsmith pragma representation following format: `// @brainsmith <type> <inputs...>`

**Attributes:**
- `type` (PragmaType): Pragma type identifier
- `inputs` (List[str]): Space-separated inputs
- `line_number` (int): Source line number for error reporting
- `parsed_data` (Dict): Processed data from pragma handler

**Methods:**
- `_parse_inputs() -> Dict`: Abstract method to parse pragma inputs
- `apply(**kwargs) -> Any`: Abstract method to apply pragma effects

#### `PragmaType` (Enum)

Valid pragma types:
- `TOP_MODULE`: Specify target module if multiple exist
- `DATATYPE`: Define supported datatypes for interfaces  
- `DERIVED_PARAMETER`: Link module param to python function
- `WEIGHT`: Mark interfaces as weight inputs
- `BDIM`: Override block dimensions for interface (preferred)
- `TDIM`: Override tensor dimensions for interface (deprecated)

## Interface Protocol Specifications

### Global Control Interface

**Required Signals:**
- `ap_clk` (input): Core clock
- `ap_rst_n` (input): Active-low reset

**Optional Signals:**
- `ap_clk2x` (input): Double-rate clock

### AXI-Stream Interface

**Required Signals:**
- `TDATA` (input/output): Data (width % 8 == 0)
- `TREADY` (output/input): Ready
- `TVALID` (input/output): Valid

**Optional Signals:**
- `TLAST` (input/output): Last

**Naming Patterns:**
- Input interfaces: `in{i}_V_*` or `s_axis_*`
- Output interfaces: `out{j}_V_*` or `m_axis_*`

### AXI-Lite Interface

**Write Channel (Optional):**
- `*_AWADDR`, `*_AWPROT`, `*_AWVALID`, `*_AWREADY`
- `*_WDATA`, `*_WSTRB`, `*_WVALID`, `*_WREADY`
- `*_BRESP`, `*_BVALID`, `*_BREADY`

**Read Channel (Optional):**
- `*_ARADDR`, `*_ARPROT`, `*_ARVALID`, `*_ARREADY`
- `*_RDATA`, `*_RRESP`, `*_RVALID`, `*_RREADY`

**Naming Patterns:**
- Configuration interfaces: `config_*`, `s_axilite_*`

## Template Context Generation

The `EnhancedRTLParsingResult.get_template_context()` method provides 22+ template variables:

### Core Variables
- `kernel_name`: Module name
- `class_name`: Generated Python class name
- `rtl_parameters`: Module parameters list
- `dataflow_interfaces`: AXI-Stream interfaces
- `dimensional_metadata`: Tensor/block/stream dimensions
- `interface_metadata`: Interface-specific metadata

### Interface Categorization
- `input_interfaces`: Input AXI-Stream interfaces
- `output_interfaces`: Output AXI-Stream interfaces
- `weight_interfaces`: Weight interfaces
- `input_interfaces_count`: Count of input interfaces
- `output_interfaces_count`: Count of output interfaces
- `weight_interfaces_count`: Count of weight interfaces

### Boolean Flags
- `has_inputs`: True if input interfaces exist
- `has_outputs`: True if output interfaces exist
- `has_weights`: True if weight interfaces exist

### Additional Context
- `complexity_level`: Kernel complexity ("low", "medium", "high")
- `compiler_data`: Compiler configuration data
- `interface_types`: Interface type mapping
- `datatype_constraints`: Data type constraints
- `InterfaceType`: Interface type enum
- `DataType`: Data type enum

## Error Handling

### Exception Types
- `ParserError`: General RTL parsing errors
- `RTLParsingError`: Specific RTL parsing failures
- `PragmaError`: Pragma parsing/validation errors

### Validation
- Protocol validation ensures interface compliance
- Pragma validation checks syntax and semantics
- Warning collection for non-critical issues

## Performance Characteristics

### RTLParsingResult
- **Code Reduction**: ~800 lines eliminated vs HWKernel
- **Performance**: 25% improvement over full HWKernel
- **Utilization**: 100% property utilization (vs 22% for HWKernel)

### EnhancedRTLParsingResult  
- **Template Generation**: 40% faster than DataflowModel approach
- **Memory**: Reduced overhead (no DataflowModel objects)
- **Direct Pipeline**: RTL → Enhanced Result → Templates

## Usage Examples

### Basic RTL Parsing
```python
from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file

# Parse with simple pragma processing
result = parse_rtl_file("my_kernel.sv")
print(f"Found {len(result.interfaces)} interfaces")

# Parse with advanced BDIM pragma processing
result = parse_rtl_file("my_kernel.sv", advanced_pragmas=True)
print(f"Sophistication level: {result.pragma_sophistication_level}")
```

### Enhanced Template Generation
```python
from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file_enhanced

# Parse for direct template generation
enhanced_result = parse_rtl_file_enhanced("my_kernel.sv")

# Get complete template context
context = enhanced_result.get_template_context()

# Use with Jinja2 template
from jinja2 import Template
template = Template("{{ kernel_name }} has {{ dataflow_interfaces|length }} interfaces")
output = template.render(**context)
```

### Interface Analysis
```python
result = parse_rtl_file("my_kernel.sv")

for name, interface in result.interfaces.items():
    print(f"Interface {name}:")
    print(f"  Type: {interface.type.value}")
    print(f"  Ports: {list(interface.ports.keys())}")
    print(f"  Valid: {interface.validation_result.valid}")
```

### Pragma Processing
```python
result = parse_rtl_file("my_kernel.sv", advanced_pragmas=True)

for pragma in result.pragmas:
    print(f"Pragma {pragma.type.value} at line {pragma.line_number}")
    if hasattr(pragma, 'parsed_data'):
        print(f"  Data: {pragma.parsed_data}")
```