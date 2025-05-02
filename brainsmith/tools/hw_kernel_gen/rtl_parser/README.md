# RTL Parser for Hardware Kernel Generator

This package implements the RTL parsing functionality for the Hardware Kernel Generator, extracting information from SystemVerilog modules to enable FINN integration.

## Overview

The RTL Parser examines the top-level interface of SystemVerilog modules to extract:
- Module parameters
- Port definitions 
- @brainsmith pragmas

This information is used by the Hardware Kernel Generator to create parameterized wrappers and integration files for FINN.

## Usage

```python
from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser

# Create parser instance
parser = RTLParser()

# Parse RTL file
kernel = parser.parse_file("path/to/module.sv")

# Access extracted information
print(f"Module name: {kernel.name}")
print(f"Parameters: {kernel.parameters}")
print(f"Ports: {kernel.ports}")
print(f"Pragmas: {kernel.pragmas}")
```

## Components

### Data Structures (data.py)
- `HWKernel`: Top-level representation of parsed module
- `Parameter`: Module parameter information
- `Port`: Port definition with direction and width
- `Pragma`: @brainsmith pragma data

### Parser (parser.py)
Main parser implementation using tree-sitter to parse SystemVerilog and extract relevant information.

### Interface Analysis (interface.py)
Specialized parsing functions for module interface components:
- Parameter declarations
- Port declarations
- Module headers

### Pragma Processing (pragma.py)
Handling of @brainsmith pragmas for:
- Interface specifications
- Parameter configurations
- Resource utilization hints
- Timing constraints
- Feature flags

## Pragma Syntax

Pragmas use the following format:
```systemverilog
// @brainsmith <type> <inputs...>
```

Supported pragma types:
- `interface`: Specify interface protocol
  ```systemverilog
  // @brainsmith interface AXI_STREAM
  ```
- `parameter`: Parameter configuration
  ```systemverilog
  // @brainsmith parameter STATIC WIDTH
  ```
- `resource`: Resource utilization hints
  ```systemverilog
  // @brainsmith resource DSP 4
  ```
- `timing`: Timing constraints
  ```systemverilog
  // @brainsmith timing LATENCY 2
  ```
- `feature`: Optional features
  ```systemverilog
  // @brainsmith feature PIPELINE enabled
  ```

## Testing

Run tests with pytest:
```bash
pytest brainsmith/tools/hw_kernel_gen/rtl_parser/tests/
```

Test coverage includes:
- Basic module parsing
- Parameter extraction
- Port analysis
- Pragma processing
- Error handling

## Dependencies

- py-tree-sitter: For SystemVerilog parsing
- SystemVerilog grammar (sv.so)

## Assumptions

- Only parses top-level module interface
- Assumes valid SystemVerilog syntax
- Ignores module implementation details