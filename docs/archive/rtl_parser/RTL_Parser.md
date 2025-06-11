# RTL Parser – Current Design (Updated)

## Overview
The RTL Parser is a sophisticated SystemVerilog analysis tool that extracts module interfaces, parameters, and Brainsmith-specific pragmas for hardware kernel integration. Built using tree-sitter for robust parsing, it provides comprehensive interface identification and protocol validation.

## Architecture

### Core Components
1. **RTLParser** (`parser.py`) - Main orchestrator using tree-sitter
2. **Grammar Handler** (`grammar.py`) - SystemVerilog grammar loading via ctypes
3. **Pragma Handler** (`pragma.py`) - Brainsmith pragma extraction and validation
4. **Interface Builder** (`interface_builder.py`) - Coordinates interface identification
5. **Interface Scanner** (`interface_scanner.py`) - Groups ports by naming conventions
6. **Protocol Validator** (`protocol_validator.py`) - Validates interface protocol compliance
7. **Data Structures** (`data.py`) - Comprehensive type definitions

### Data Structures

#### Core Types
- **HWKernel**: Top-level kernel representation
- **Parameter**: Module parameters with template mapping
- **Port**: Module ports with direction and width
- **Interface**: Validated interface groups (Global, AXI-Stream, AXI-Lite)
- **Pragma**: Parsed Brainsmith directives

#### Interface Types
- **GLOBAL_CONTROL**: Clock/reset signals (required: clk, rst_n; optional: clk2x)
- **AXI_STREAM**: Data flow interfaces (required: TDATA, TVALID, TREADY; optional: TLAST)
- **AXI_LITE**: Configuration interfaces (supports read-only, write-only, or full)

#### Pragma Types
- **TOP_MODULE**: Specify target module when multiple exist
- **DATATYPE**: Define supported datatypes for interfaces
- **DERIVED_PARAMETER**: Add computed parameters from Python expressions
- **WEIGHT**: Mark interfaces as weight inputs
- **CALCULATIONS_OVERRIDE**: Override calculation counts per execution
- **CALCULATIONS_MODIFIER**: Apply multipliers to default calculations  
- **PARALLELISM_CONSTRAINT**: Set PE/SIMD constraints (4 constraint types supported)

### Processing Pipeline

1. **Syntax Parsing**: Tree-sitter AST generation with error detection
2. **Module Selection**: Identify target module (via pragma or heuristics)
3. **Component Extraction**: Extract parameters, ports, and pragmas
4. **Interface Analysis**: 
   - Scanner groups ports by naming patterns
   - Validator checks protocol compliance
   - Builder creates validated Interface objects
5. **Pragma Application**: Apply pragma effects to kernel data
6. **Validation**: Ensure all ports assigned to valid interfaces

### Key Features

#### Robust Interface Detection
- Regex-based pattern matching for standard protocols
- Flexible naming conventions (prefixes/suffixes)
- Comprehensive validation against protocol requirements
- Support for partial AXI-Lite implementations

#### Advanced Pragma System
- Type-safe pragma parsing with validation
- Extensible pragma handler architecture
- Line-number tracking for error reporting
- Support for interface-wise modeling constraints

#### Error Handling
- Comprehensive error types (ParserError, SyntaxError, PragmaError)
- Detailed error messages with source locations
- Graceful handling of unrecognized constructs
- Debug mode with detailed AST inspection

## Technology Stack
- **Parser**: py-tree-sitter with SystemVerilog grammar
- **Grammar**: Pre-compiled sv.so library loaded via ctypes
- **Language**: Python 3.8+ with dataclasses and type hints
- **Validation**: Protocol-specific validation rules

## Output Format
Returns a fully populated `HWKernel` object containing:
- Module metadata (name, source file)
- Typed parameters with template mappings
- Validated interfaces with protocol compliance
- Applied pragma effects
- Comprehensive error reporting for invalid constructs
