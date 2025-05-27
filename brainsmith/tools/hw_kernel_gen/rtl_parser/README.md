# RTL Parser

A SystemVerilog parser component for the Brainsmith Hardware Kernel Generator (HKG) that extracts, validates, and processes hardware interface information from RTL source code.

## Overview

The RTL Parser analyzes SystemVerilog files to identify and validate hardware interfaces, module parameters, and special compiler directives (pragmas) needed by the Hardware Kernel Generator. It serves as the critical bridge between custom RTL implementations and the FINN compiler toolchain, enabling hardware engineers to integrate their designs into the Brainsmith ecosystem.

The RTL Parser operates as the first stage in the Hardware Kernel Generator pipeline, taking SystemVerilog RTL files with embedded pragmas as input, processing and validating hardware interface information, and producing a structured `HWKernel` object containing all relevant data for subsequent wrapper template generation and compiler integration.

### Key Capabilities

- **Interface Recognition**: Automatically identifies and validates AXI-Stream, AXI-Lite, and Global Control interfaces using case-insensitive suffix detection (uppercase preferred)
- **Parameter Extraction**: Extracts module parameters while preserving bit-width expressions
- **Pragma Processing**: Parses `@brainsmith` compiler directives for additional metadata
- **Protocol Validation**: Ensures interfaces conform to expected signal naming and direction requirements
- **Extensible Design**: Modular architecture supports future interface types and pragma extensions

### Integration with Hardware Kernel Generator

The extracted information enables the HKG to:
- Generate parameterized wrapper templates
- Create FINN compiler integration files (HWCustomOp instances)
- Perform design space exploration
- Validate interface compatibility

## Architecture

The RTL Parser follows a multi-stage pipeline architecture:

```
SystemVerilog File → Tree-sitter AST → Interface Scanning → Protocol Validation → HWKernel Object
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **`parser.py`** | Main orchestrator and tree-sitter integration |
| **`data.py`** | Core data structures and type definitions |
| **`grammar.py`** | SystemVerilog grammar loading via tree-sitter |
| **`interface_scanner.py`** | Port grouping based on naming conventions |
| **`protocol_validator.py`** | Interface protocol compliance validation |
| **`interface_builder.py`** | Coordination between scanning and validation |
| **`pragma.py`** | Pragma extraction and processing |

### Processing Pipeline

1. **Initial Parse**: Load and parse SystemVerilog using tree-sitter, extract pragmas, select target module
2. **Component Extraction**: Extract module parameters and ports from the AST
3. **Interface Analysis**: Group ports into potential interfaces and validate against protocol specifications
4. **Pragma Application**: Apply compiler directives to modify interface and parameter metadata

## Quick Start

### Basic Usage

```python
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser

# Initialize parser
parser = RTLParser(debug=False)

# Parse SystemVerilog file
hw_kernel = parser.parse_file("path/to/module.sv")

# Access extracted information
print(f"Module: {hw_kernel.name}")
print(f"Parameters: {[p.name for p in hw_kernel.parameters]}")
print(f"Interfaces: {list(hw_kernel.interfaces.keys())}")
```

### Debug Mode

```python
# Enable detailed logging for troubleshooting
parser = RTLParser(debug=True)
hw_kernel = parser.parse_file("module.sv")
```

## Supported Interfaces

The RTL Parser recognizes three categories of hardware interfaces based on signal naming conventions:

### 1. Global Control Signals

Required timing and control signals for all modules:

| Signal | Direction | Required | Description |
|--------|-----------|----------|-------------|
| `*_clk` | Input | Yes | Primary clock |
| `*_rst_n` | Input | Yes | Active-low reset |
| `*_clk2x` | Input | No | Double-rate clock |

**Example:**
```systemverilog
input wire ap_clk,
input wire ap_rst_n,
input wire ap_clk2x  // Optional
```

### 2. AXI-Stream Interfaces

Primary data flow interfaces supporting both input and output directions:

| Signal | Direction (Slave) | Required | Description |
|--------|-------------------|----------|-------------|
| `*_TDATA` | Input | Yes | Data payload |
| `*_TVALID` | Input | Yes | Valid signal |
| `*_TREADY` | Output | Yes | Ready signal |
| `*_TLAST` | Input | No | Last transfer |

**Example:**
```systemverilog
// Input stream (slave interface)
input wire [31:0] in0_V_TDATA,
input wire in0_V_TVALID,
output wire in0_V_TREADY,
input wire in0_V_TLAST,

// Output stream (master interface)
output wire [31:0] out0_V_TDATA,
output wire out0_V_TVALID,
input wire out0_V_TREADY
```

### 3. AXI-Lite Interfaces

Configuration and control interfaces (read and/or write channels):

| Signal | Direction | Required | Description |
|--------|-----------|----------|-------------|
| `*_AWADDR` | Input | Yes* | Write address |
| `*_AWVALID` | Input | Yes* | Write address valid |
| `*_AWREADY` | Output | Yes* | Write address ready |
| `*_WDATA` | Input | Yes* | Write data |
| `*_WSTRB` | Input | Yes* | Write strobe |
| `*_WVALID` | Input | Yes* | Write data valid |
| `*_WREADY` | Output | Yes* | Write data ready |
| `*_BRESP` | Output | Yes* | Write response |
| `*_BVALID` | Output | Yes* | Write response valid |
| `*_BREADY` | Input | Yes* | Write response ready |
| `*_ARADDR` | Input | Yes** | Read address |
| `*_ARVALID` | Input | Yes** | Read address valid |
| `*_ARREADY` | Output | Yes** | Read address ready |
| `*_RDATA` | Output | Yes** | Read data |
| `*_RRESP` | Output | Yes** | Read response |
| `*_RVALID` | Output | Yes** | Read data valid |
| `*_RREADY` | Input | Yes** | Read data ready |

*Required if write channel is present  
**Required if read channel is present

**Example:**
```systemverilog
// AXI-Lite interface (both read and write)
input wire [4:0] s_axi_control_AWADDR,
input wire s_axi_control_AWVALID,
output wire s_axi_control_AWREADY,
input wire [31:0] s_axi_control_WDATA,
input wire [3:0] s_axi_control_WSTRB,
input wire s_axi_control_WVALID,
output wire s_axi_control_WREADY,
output wire [1:0] s_axi_control_BRESP,
output wire s_axi_control_BVALID,
input wire s_axi_control_BREADY,
input wire [4:0] s_axi_control_ARADDR,
input wire s_axi_control_ARVALID,
output wire s_axi_control_ARREADY,
output wire [31:0] s_axi_control_RDATA,
output wire [1:0] s_axi_control_RRESP,
output wire s_axi_control_RVALID,
input wire s_axi_control_RREADY
```

## Pragma System

Pragmas are special comments that provide additional metadata to the Hardware Kernel Generator. They follow the format:

```
// @brainsmith <pragma_type> <arguments...>
```

### Supported Pragmas

#### 1. Top Module Selection
```systemverilog
// @brainsmith top_module my_target_module
```
Specifies which module to use when multiple modules exist in the file.

#### 2. Interface Datatype Constraints
```systemverilog
// @brainsmith datatype in0 8
// @brainsmith datatype config 1 32
```
Restricts supported datatypes for interfaces. First form specifies fixed size, second form specifies range. *Note: This pragma handler is currently a placeholder that needs to be defined based on future HWCustomOp improvements and expansions.*

#### 3. Derived Parameters
```systemverilog
// @brainsmith derived_parameter my_function param1 param2
```
Links module parameters to Python functions for complex parameter derivation. *Note: This pragma handler is currently a placeholder that needs to be defined based on future HWCustomOp improvements and expansions.*

#### 4. Weight Interfaces
```systemverilog
// @brainsmith weight in1
```
Marks an interface as carrying weight data to inform HWCustomOp generation.

### Pragma Extensibility

The pragma system is designed for extensibility. New pragma types can be added by:

1. Adding the pragma type to `PragmaType` enum in `data.py`
2. Creating a new pragma subclass inheriting from `Pragma`
3. Implementing `_parse_inputs()` and `apply()` methods
4. Registering the pragma constructor in `PragmaHandler`

## Data Structures

### Core Types

- **`HWKernel`**: Top-level representation of a parsed hardware module
- **`Interface`**: Validated interface with associated ports and metadata
- **`Port`**: Individual SystemVerilog port with direction and width information
- **`Parameter`**: Module parameter with type and default value
- **`Pragma`**: Compiler directive with parsed arguments

### Interface Types

```python
class InterfaceType(Enum):
    GLOBAL_CONTROL = "global"
    AXI_STREAM = "axistream" 
    AXI_LITE = "axilite"
    UNKNOWN = "unknown"
```

### Direction Types

```python
class Direction(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
```

## API Reference

### RTLParser Class

The main parser interface:

```python
class RTLParser:
    def __init__(self, grammar_path: Optional[str] = None, debug: bool = False)
    def parse_file(self, file_path: str) -> HWKernel
```

**Parameters:**
- `grammar_path`: Path to tree-sitter grammar library (uses default if None)
- `debug`: Enable detailed logging
- `file_path`: Path to SystemVerilog file to parse

**Returns:** `HWKernel` object containing all extracted information

### HWKernel Object

```python
@dataclass
class HWKernel:
    name: str                                    # Module name
    parameters: List[Parameter]                  # Module parameters
    interfaces: Dict[str, Interface]             # Validated interfaces
    pragmas: List[Pragma]                        # Found pragmas
    metadata: Dict[str, Any]                     # Additional metadata
```

### Interface Object

```python
@dataclass
class Interface:
    name: str                                    # Interface name (e.g., "in0", "config")
    type: InterfaceType                          # Interface type
    ports: Dict[str, Port]                       # Signal name to Port mapping
    validation_result: ValidationResult          # Validation status
    metadata: Dict[str, Any]                     # Protocol-specific metadata
```

## Dependencies

### Runtime Dependencies

- **Python 3.7+**
- **tree-sitter**: Python bindings for tree-sitter parser
- **SystemVerilog Grammar**: Pre-compiled grammar library (`sv.so`)

### Grammar Library

The parser currently uses a pre-compiled SystemVerilog grammar (`sv.so`) for tree-sitter. This is a temporary solution that will be replaced with a more robust system to build the grammar from the open-source tree-sitter-verilog repository during Docker generation.

## Error Handling

The parser provides comprehensive error reporting with specific guidance for common issues:

### Syntax Errors
- Invalid SystemVerilog syntax
- Malformed module definitions

### Interface Validation Errors
- Missing required signals
- Incorrect signal directions
- Invalid interface configurations

### Pragma Errors
- Invalid pragma syntax
- Missing required arguments
- Conflicting pragma specifications

All errors include line numbers and specific guidance for resolution.

## Development Guide

### Extending Interface Support

To add support for new interface types:

1. **Define Protocol Specification**
   ```python
   NEW_INTERFACE_SUFFIXES = {
       "SIGNAL1": {"direction": Direction.INPUT, "required": True},
       "SIGNAL2": {"direction": Direction.OUTPUT, "required": False},
   }
   ```

2. **Add Interface Type**
   ```python
   class InterfaceType(Enum):
       # ... existing types ...
       NEW_INTERFACE = "new_interface"
   ```

3. **Implement Validation Logic**
   ```python
   def validate_new_interface(self, group: PortGroup) -> ValidationResult:
       # Validation implementation
   ```

4. **Update Scanner Configuration**
   ```python
   self.suffixes[InterfaceType.NEW_INTERFACE] = NEW_INTERFACE_SUFFIXES
   ```

### Adding Custom Pragmas

1. **Define Pragma Type**
   ```python
   class PragmaType(Enum):
       # ... existing types ...
       CUSTOM_PRAGMA = "custom_pragma"
   ```

2. **Create Pragma Subclass**
   ```python
   @dataclass
   class CustomPragma(Pragma):
       def _parse_inputs(self) -> Dict:
           # Input parsing logic
           
       def apply(self, **kwargs) -> Any:
           # Application logic
   ```

3. **Register in Handler**
   ```python
   self.pragma_constructors[PragmaType.CUSTOM_PRAGMA] = CustomPragma
   ```

### Testing Guidelines

When developing extensions, ensure comprehensive validation and error checking:

- Add appropriate validation for new signal patterns
- Include comprehensive error messages with line numbers
- Test with both valid and invalid input cases
- Verify proper metadata extraction

## Naming Conventions

### Signal Naming Requirements

For proper interface recognition, signals must follow these conventions. The parser performs case-insensitive suffix detection, but uppercase is the preferred style:

- **Global Control**: `<prefix>_clk`, `<prefix>_rst_n`, `<prefix>_clk2x`
- **AXI-Stream**: `<prefix>_TDATA`, `<prefix>_TVALID`, `<prefix>_TREADY`, `<prefix>_TLAST`
- **AXI-Lite**: `<prefix>_AWADDR`, `<prefix>_WDATA`, etc. (see full list above)

### Interface Naming

The parser automatically assigns interface names:
- Global Control: Uses signal names directly
- AXI-Stream: `in0`, `in1`, ... for inputs; `out0`, `out1`, ... for outputs  
- AXI-Lite: `config` for configuration interfaces

## Limitations and Future Work

### Current Limitations

- **Grammar Dependency**: Relies on pre-compiled SystemVerilog grammar
- **Interface Coverage**: Limited to Global Control, AXI-Stream, and AXI-Lite
- **Parameter Expressions**: Preserves but doesn't evaluate complex expressions

### Planned Enhancements

- **Dynamic Grammar Building**: Replace static grammar with build-time compilation from the open-source tree-sitter-verilog repository

## License

Copyright (c) Microsoft Corporation. Licensed under the MIT License.

---

*This documentation corresponds to the RTL Parser implementation as part of the Brainsmith Hardware Kernel Generator project.*
