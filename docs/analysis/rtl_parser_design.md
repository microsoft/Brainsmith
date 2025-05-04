# RTL Parser Design Document

## Overview
The RTL Parser is a key component of the Hardware Kernel Generator (HKG) that extracts interface information from SystemVerilog modules. It parses the module's top-level interface, parameters, and custom pragmas to enable integration with the FINN compiler infrastructure.

## Architecture

### Component Structure
```
rtl_parser/
├── data.py       - Core data structures
├── pragma.py     - Pragma processing
├── parser.py     - Main parser implementation  
├── interface.py  - Interface analysis
└── sv.so         - SystemVerilog grammar
```

### Core Components

1. **Data Structures** (`data.py`)
   - `Direction`: Enum for port directions (INPUT/OUTPUT/INOUT)
   - `Parameter`: Represents module parameters
     - name: Parameter identifier
     - param_type: Parameter datatype
     - default_value: Optional default value
     - description: Optional documentation
   - `Port`: Represents module ports
     - name: Port identifier
     - direction: Port direction
     - width: Bit width expression (preserved as string)
     - description: Optional documentation
   - `Pragma`: Represents @brainsmith pragmas
     - type: Pragma type identifier
     - inputs: List of space-separated inputs
     - line_number: Source code line number
     - processed_data: Handler-specific data
   - `HWKernel`: Top-level container
     - name: Module name
     - parameters: List of Parameter objects
     - ports: List of Port objects
     - pragmas: List of Pragma objects
     - metadata: Additional extraction data

2. **Pragma Processing** (`pragma.py`)
   - PragmaType Enum: 
     - INTERFACE: Specify interface protocol
     - PARAMETER: Parameter configuration
     - RESOURCE: Resource utilization hints
     - TIMING: Timing constraints
     - FEATURE: Optional features
   - Format: `// @brainsmith <type> <inputs...>`
   - Handler functions for each pragma type
   - Validation and error reporting

3. **Main Parser** (`parser.py`)
   - Uses tree-sitter for parsing
   - Loads SystemVerilog grammar
   - Main entry point: `parse_file()`
   - Error handling for:
     - Missing files
     - Invalid syntax
     - Missing module definitions

4. **Interface Analysis** (`interface.py`)
   - Specialized parsing functions:
     - `parse_port_declaration()`: Extract port information
     - `parse_parameter_declaration()`: Extract parameter information
     - `extract_module_header()`: Process module interface
   - Helper functions for AST traversal
   - Handles complex SystemVerilog syntax variations

## Data Flow

1. **Input Processing**
   ```
   RTL File -> tree-sitter Parser -> AST
   ```

2. **Information Extraction**
   ```
   AST -> Module Header -> Parameters
                       -> Ports
                       -> Pragmas
   ```

3. **Data Assembly**
   ```
   Components -> HWKernel Object -> Validation -> Output
   ```

## Key Features

### 1. Parameter Handling
- Extracts parameter name, type, default value
- Preserves type information for compiler
- Validates parameter uniqueness

### 2. Port Analysis
- Supports complex port declarations
- Preserves width expressions without simplification
- Handles various SystemVerilog port styles

### 3. Pragma System
- Extensible pragma framework
- Type-specific validation
- Structured data extraction
- Line number tracking for errors

### 4. Error Handling
- Syntax validation
- Detailed error messages
- Debug logging support
- Input validation

### 5. AST Processing
- Efficient tree traversal
- Robust node type checking
- Flexible child node search
- UTF-8 text handling

## Implementation Notes

### Parameter Processing
- Skips local parameters
- Defaults to "logic" type if unspecified
- Preserves expression-based default values

### Port Processing
- Handles packed dimensions
- Supports multiple declaration styles
- Preserves complex width expressions

### Pragma Processing
- Custom handler per type
- Structured data extraction
- Optional debug output
- Error recovery

## Usage Example

```python
# Initialize parser
parser = RTLParser()

# Parse RTL file
kernel = parser.parse_file("module.sv")

# Access extracted data
module_name = kernel.name
parameters = kernel.parameters
ports = kernel.ports
pragmas = kernel.pragmas

# Example pragma processing
for pragma in pragmas:
    if pragma.type == "interface":
        protocol = pragma.processed_data["protocol"]
        options = pragma.processed_data["options"]
```

## Practical Example: Thresholding AXI Module

The following example demonstrates how the RTL Parser processes a real SystemVerilog module from the FINN project:

### Source Module Interface
```systemverilog
module thresholding_axi #(
    int unsigned  N,              // output precision
    int unsigned  WI,             // input precision
    int unsigned  WT,             // threshold precision
    int unsigned  C = 1,          // Channels
    int unsigned  PE = 1,         // Processing Parallelism
    bit  SIGNED = 1,             // signed inputs
    bit  FPARG  = 0,             // floating-point inputs
    int  BIAS   = 0,             // output bias
    parameter  THRESHOLDS_PATH = "",
    bit  USE_AXILITE            // AXI-Lite interface control
)(
    // Global Control
    input  logic  ap_clk,
    input  logic  ap_rst_n,
    
    // AXI Lite Interface
    input  logic                  s_axilite_AWVALID,
    output logic                  s_axilite_AWREADY,
    input  logic [ADDR_BITS-1:0] s_axilite_AWADDR
    // ... additional ports omitted for brevity
);
```

### Parser Output Structure

1. **Extracted Parameters**
```python
hw_kernel.parameters = [
    Parameter(
        name="N",
        param_type="int unsigned",
        description="output precision"
    ),
    Parameter(
        name="C",
        param_type="int unsigned",
        default_value="1",
        description="Channels"
    ),
    Parameter(
        name="SIGNED",
        param_type="bit",
        default_value="1",
        description="signed inputs"
    ),
    Parameter(
        name="THRESHOLDS_PATH",
        param_type="parameter",
        default_value='""',
        description=None
    )
]
```

2. **Extracted Ports**
```python
hw_kernel.ports = [
    Port(
        name="ap_clk",
        direction=Direction.INPUT,
        width="1",
        description="Global clock"
    ),
    Port(
        name="s_axilite_AWVALID",
        direction=Direction.INPUT,
        width="1",
        description="AXI-Lite write address valid"
    ),
    Port(
        name="s_axilite_AWADDR",
        direction=Direction.INPUT,
        width="ADDR_BITS-1:0",
        description="AXI-Lite write address"
    )
]
```

### Key Processing Features Demonstrated

1. **Complex Parameter Types**
   - Mixed parameter types (int unsigned, bit, parameter)
   - Default value handling
   - Documentation comment extraction

2. **Port Width Expressions**
   - Parameterized widths (e.g., "ADDR_BITS-1:0")
   - Default single-bit widths
   - Width expression preservation

3. **Interface Organization**
   - Logical grouping by interface type (Global Control, AXI Lite)
   - Direction inference for custom types (logic)
   - Protocol-specific signal patterns

4. **Documentation Processing**
   - Inline comment extraction
   - Port group comments
   - Parameter descriptions

This real-world example showcases how the RTL Parser handles:
- Complex SystemVerilog syntax
- Mixed parameter types and configurations
- Standardized interface protocols (AXI)
- Documentation and metadata extraction
- Expression preservation in port widths

## Future Extensions

1. **Interface Grouping**
   - Group related ports into interfaces
   - Support standard protocols (AXI, Avalon)
   - Custom interface definitions

2. **Enhanced Pragma System**
   - New pragma types
   - Complex pragma relationships
   - Multi-line pragma support

3. **Documentation Generation**
   - Extract documentation from comments
   - Generate interface diagrams
   - Protocol compliance checking

## Error Cases

1. **Syntax Errors**
   - Invalid SystemVerilog syntax
   - Malformed pragmas
   - Invalid identifiers

2. **Semantic Errors**
   - Duplicate parameters
   - Duplicate ports
   - Invalid pragma inputs

3. **Resource Errors**
   - Missing files
   - Grammar load failures
   - Encoding issues

## Dependencies

- tree-sitter: AST parsing
- SystemVerilog grammar (sv.so)
- Python standard library
- Logging infrastructure

## Testing Strategy

1. **Unit Tests**
   - Individual component testing
   - Error case validation
   - Edge case handling

2. **Integration Tests**
   - End-to-end parsing
   - Complex module testing
   - Error recovery

3. **Example Files**
   - Standard test cases
   - Complex test cases
   - Error test cases