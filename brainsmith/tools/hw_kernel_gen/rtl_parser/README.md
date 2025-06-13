# RTL Parser

A SystemVerilog parser component for the Brainsmith Hardware Kernel Generator (HKG) that extracts, validates, and processes hardware interface information from RTL source code.

## Overview

The RTL Parser analyzes SystemVerilog files to identify and validate hardware interfaces, module parameters, and special compiler directives (pragmas) needed by the Hardware Kernel Generator. It serves as the critical bridge between custom RTL implementations and the FINN compiler toolchain, enabling hardware engineers to integrate their designs into the Brainsmith ecosystem.

The RTL Parser operates as the first stage in the Hardware Kernel Generator pipeline, taking SystemVerilog RTL files with embedded pragmas as input, processing and validating hardware interface information, and producing a structured `KernelMetadata` object containing `InterfaceMetadata` objects for subsequent wrapper template generation and compiler integration.

### Key Capabilities

- **Interface Recognition**: Automatically identifies and validates AXI-Stream, AXI-Lite, and Global Control interfaces using case-insensitive suffix detection (uppercase preferred)
- **Parameter Extraction**: Extracts module parameters while preserving bit-width expressions and validating pragma parameter references
- **Pragma Processing**: Parses `@brainsmith` compiler directives with robust error isolation and chain-of-responsibility pattern
- **Extensible Design**: Modular architecture supports future interface types and pragma extensions

### Integration with Hardware Kernel Generator

The extracted information is packaged in a `KernelMetadata` object that directly informs the generation of the FINN compiler integration files for the input Kernel. It also enforces restrictions on Interfaces and parameters (types, amounts, formatting, etc.), validating compatibility across the compilation pipeline.

## Architecture

The RTL Parser follows a multi-stage pipeline architecture with direct metadata generation:

```
SystemVerilog File → Tree-sitter AST → Interface Scanning → Protocol Validation → Pragma Application → KernelMetadata Object
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **`parser.py`** | Main orchestrator, tree-sitter integration, and 3-stage parsing pipeline |
| **`data.py`** | Core data structures: `Parameter`, `Port`, `Pragma` subclasses with `InterfaceNameMatcher` |
| **`grammar.py`** | SystemVerilog grammar loading via tree-sitter |
| **`interface_scanner.py`** | Port grouping based on naming conventions into `PortGroup` objects |
| **`protocol_validator.py`** | Interface protocol compliance validation for grouped ports |
| **`interface_builder.py`** | Direct `InterfaceMetadata` creation from validated `PortGroup` objects |
| **`pragma.py`** | `PragmaHandler` for extraction and pragma subclass instantiation |

### Processing Pipeline

The RTL Parser follows a rigorous 3-stage pipeline:

1. **Initial Parse** (`_initial_parse`): 
   - Load and parse SystemVerilog using tree-sitter
   - Extract `@brainsmith` pragmas from comment nodes
   - Select target module (handles multiple modules via `TOP_MODULE` pragma)
   
2. **Component Extraction** (`_extract_kernel_components`):
   - Extract module name, parameters (excluding localparams), and ports from AST
   - Set module parameters for BDIM pragma validation
   - Support ANSI-style port declarations only
   
3. **Interface Analysis & Validation** (`_analyze_and_validate_interfaces`):
   - Use `InterfaceBuilder.build_interface_metadata()` for direct metadata creation
   - Apply pragmas with error isolation using chain-of-responsibility pattern
   - Validate required Global Control and AXI-Stream interfaces

## Quick Start

### Basic Usage

```python
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser

# Initialize parser
parser = RTLParser(debug=False)

# Parse SystemVerilog file
kernel_metadata = parser.parse_file("path/to/module.sv")

# Access extracted information
print(f"Module: {kernel_metadata.name}")
print(f"Parameters: {[p.name for p in kernel_metadata.parameters]}")
print(f"Interfaces: {[iface.name for iface in kernel_metadata.interfaces]}")
print(f"Interface Types: {[iface.interface_type.value for iface in kernel_metadata.interfaces]}")

# Access interface metadata details
for iface in kernel_metadata.interfaces:
    print(f"Interface {iface.name}:")
    print(f"  Type: {iface.interface_type.value}")
    print(f"  Datatype Constraints: {len(iface.datatype_constraints)}")
    print(f"  Chunking Strategy: {type(iface.chunking_strategy).__name__}")
```

**Note**: The RTL Parser currently supports only ANSI-style port declarations (ports declared in the module header). Non-ANSI style declarations are not supported.

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

The pragma system uses a chain-of-responsibility pattern with robust error isolation, meaning individual pragma failures don't break the entire parsing process.

### Supported Pragmas

#### 1. Top Module Selection
```systemverilog
// @brainsmith top_module my_target_module
```
Specifies which module to use when multiple modules exist in the file.

#### 2. Interface Datatype Constraints
```systemverilog
// @brainsmith datatype in0 UINT 8 16
// @brainsmith datatype weights FIXED 8 8
// @brainsmith datatype config INT 4 12
```
Defines datatype constraint groups for interfaces using the new QONNX integration format:
- `interface_name`: Target interface (supports flexible name matching)
- `base_type`: UINT, INT, or FIXED
- `min_width`: Minimum bit width (must be positive)
- `max_width`: Maximum bit width (must be >= min_width)

Creates `DatatypeConstraintGroup` objects that are added to `InterfaceMetadata.datatype_constraints`.

#### 3. Block Dimension Chunking (BDIM)
```systemverilog
// @brainsmith bdim in0 [PE]
// @brainsmith bdim in0 [SIMD,PE]
// @brainsmith bdim weights [:,:,PE] RINDEX=1
// @brainsmith bdim out0 [TILE_SIZE,:]
```
**NEW SIMPLIFIED FORMAT** - Defines block chunking strategies with parameter names only:
- `interface_name`: Target interface (supports flexible name matching)  
- `[shape]`: Block shape using parameter names and `:` (full dimension)
- `RINDEX=n`: Optional starting index for chunking (default: 0)

**IMPORTANT**: Magic numbers are explicitly forbidden - only parameter names and `:` allowed. This ensures parameterizability. Parameter names are validated against module parameters.

Creates `BlockChunkingStrategy` objects replacing default chunking strategies.

#### 4. Weight Interfaces
```systemverilog
// @brainsmith weight weights
// @brainsmith weight weights bias params
```
Marks interfaces as carrying weight data by changing their `InterfaceType` to `WEIGHT`. Supports multiple interface names in a single pragma.

#### 5. Derived Parameters  
```systemverilog
// @brainsmith derived_parameter my_function param1 param2
```
Links module parameters to Python functions for complex parameter derivation. Adds derived parameters to the kernel metadata.

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
# Uses unified interface types from brainsmith.dataflow.core.interface_types
class InterfaceType(Enum):
    INPUT = "input"          # AXI-Stream input interface
    OUTPUT = "output"        # AXI-Stream output interface  
    WEIGHT = "weight"        # Weight/parameter interface (AXI-Stream)
    CONFIG = "config"        # AXI-Lite configuration interface
    CONTROL = "control"      # Global control signals (clk, rst)
```

**Note**: The RTL Parser has been updated to use the unified interface type system. Interface roles are inherently tied to protocols (e.g., INPUT/OUTPUT/WEIGHT are always AXI-Stream, CONFIG is always AXI-Lite, CONTROL is always global signals).

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
    def parse_file(self, file_path: str) -> KernelMetadata
    def parse(self, systemverilog_code: str, source_name: str = "<string>", module_name: Optional[str] = None) -> KernelMetadata
```

**Parameters:**
- `grammar_path`: Path to tree-sitter grammar library (uses default if None)
- `debug`: Enable detailed logging
- `file_path`: Path to SystemVerilog file to parse
- `systemverilog_code`: SystemVerilog source code string
- `source_name`: Name for logging/error messages
- `module_name`: Optional target module name

**Returns:** `KernelMetadata` object containing all extracted information

### KernelMetadata Object

```python
@dataclass
class KernelMetadata:
    name: str                                    # Module name
    source_file: Path                            # Source file path
    interfaces: List[InterfaceMetadata]          # Interface metadata objects
    parameters: List[Parameter]                  # Module parameters
    pragmas: List[Pragma]                        # Found pragmas
    parsing_warnings: List[str]                  # Warnings during parsing
```

### InterfaceMetadata Object

```python
@dataclass  
class InterfaceMetadata:
    name: str                                    # Interface name (e.g., "in0", "config")
    interface_type: InterfaceType                # Interface type (INPUT/OUTPUT/WEIGHT/CONFIG/CONTROL)
    datatype_constraints: List[DatatypeConstraintGroup]  # QONNX datatype constraints
    chunking_strategy: ChunkingStrategy          # Block chunking strategy
    description: Optional[str]                   # Optional description
```

### Parameter Object

```python
@dataclass
class Parameter:
    name: str                                    # Parameter identifier
    param_type: Optional[str]                    # Parameter datatype ("int", "type", "derived", etc.)
    default_value: Optional[str]                 # Default value if specified
    description: Optional[str]                   # Optional documentation
    template_param_name: str                     # Template parameter name (computed: $NAME$)
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

## RTL Restrictions

The RTL Parser intentionally enforces strict restrictions on input RTL code to ensure compatibility with the Brainsmith Hardware Kernel Generator pipeline. These are **design requirements**, not limitations:

### Interface Type Restrictions

- **Approved Interface Types Only**: Only AXI-Stream, AXI-Lite, and Global Control interfaces are supported
- **Strict Protocol Compliance**: All interfaces must conform exactly to the specified signal naming and direction requirements
- **No Custom Interfaces**: Any interface that doesn't match the approved patterns is treated as a **hard error**
- **Unassigned Ports Forbidden**: All module ports must be assigned to a recognized interface type

### Port Naming Restrictions

Signals must follow exact suffix patterns for interface recognition:

- **Global Control**: Must use `*_clk`, `*_rst_n`, `*_clk2x` suffixes
- **AXI-Stream**: Must use `*_TDATA`, `*_TVALID`, `*_TREADY`, `*_TLAST` suffixes  
- **AXI-Lite**: Must use complete AXI-Lite suffix set (`*_AWADDR`, `*_WDATA`, etc.)
- **Case Sensitivity**: Detection is case-insensitive, but uppercase is strongly preferred
- **No Deviations**: Signal names that don't match these patterns will not be recognized

### Module Structure Restrictions

- **ANSI-Style Only**: Only ANSI-style port declarations (ports in module header) are supported
- **Parameter Exposure**: Regular parameters are exposed as FINN parameters; localparams are allowed but not exposed
- **Single Module Target**: Multi-module files require explicit `TOP_MODULE` pragma for disambiguation

### Pragma Restrictions

- **BDIM Parameter Validation**: All parameter names in BDIM pragmas must exist in the module
- **Magic Number Prohibition**: BDIM pragmas explicitly forbid magic numbers to enforce parameterizability
- **Datatype Constraint Format**: DATATYPE pragmas must use exact QONNX-compatible format
- **Error Isolation**: Individual pragma errors don't halt parsing, but invalid pragmas are ignored

These restrictions ensure that all RTL modules can be reliably processed by the Hardware Kernel Generator pipeline and integrated with FINN.

## Limitations and Future Work

### Current Technical Limitations

- **Grammar Dependency**: Relies on pre-compiled SystemVerilog grammar
- **Parameter Expressions**: Preserves but doesn't evaluate complex expressions  
- **BDIM Validation**: Parameter validation occurs during pragma application, not during initial parsing

### Recent Improvements (Based on Test Analysis)

- **Pragma Error Isolation**: Individual pragma failures don't break the entire parsing process
- **Direct Metadata Creation**: No longer creates temporary `Interface` objects, works directly with `InterfaceMetadata`
- **Robust Name Matching**: `InterfaceNameMatcher` supports multiple naming patterns (exact, prefix, AXI patterns)
- **Parameter Validation**: BDIM pragmas validate parameter names against actual module parameters
- **QONNX Integration**: Datatype constraints use `DatatypeConstraintGroup` for QONNX compatibility
- **Magic Number Prevention**: BDIM pragmas explicitly reject magic numbers to ensure parameterizability

### Planned Enhancements

- **Dynamic Grammar Building**: Replace static grammar with build-time compilation from the open-source tree-sitter-verilog repository
- **Non-ANSI Port Support**: Add support for non-ANSI port declaration styles

## License

Copyright (c) Microsoft Corporation. Licensed under the MIT License.

---

*This documentation corresponds to the RTL Parser implementation as part of the Brainsmith Hardware Kernel Generator project.*
