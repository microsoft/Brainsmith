# Brainsmith SystemVerilog RTL Parser

## Overview

This directory contains the SystemVerilog RTL parser used within the Brainsmith framework. Its primary purpose is to analyze SystemVerilog hardware kernel files (`.sv`) and extract crucial information about the module's interface, parameters, and specific Brainsmith pragmas. This extracted information, structured into an `HWKernel` object, is essential for downstream tools, particularly the hardware kernel generator, to understand how to interact with and integrate the kernel.

The parser focuses on identifying standard interface protocols like AXI-Stream and AXI-Lite, along with global control signals, based on common naming conventions.

## How it Works

The parsing process involves several steps:

1.  **Grammar Loading:** It loads a pre-compiled `tree-sitter` SystemVerilog grammar library (`sv.so`) using the `grammar.py` module.
2.  **AST Generation:** The `RTLParser` class in `parser.py` uses the loaded grammar to parse the input SystemVerilog file into an Abstract Syntax Tree (AST).
3.  **Syntax Check:** It performs a basic check for syntax errors reported by `tree-sitter`.
4.  **Pragma Extraction:** It scans the source code comments for `//@brainsmith` pragmas using `pragma.py`.
5.  **Module Selection:** It identifies all `module` declarations in the AST. If multiple modules exist, it uses the `//@brainsmith TOP_MODULE <name>` pragma to select the target module. If no pragma is present and multiple modules exist, or if no modules are found, it raises an error.
6.  **Header Extraction:** For the target module, it extracts the module name, parameters (name, type, default value), and ports (name, direction, width) by traversing the relevant AST nodes. This logic resides primarily within `parser.py`.
7.  **Interface Building:**
    *   The `InterfaceBuilder` (`interface_builder.py`) is invoked with the list of extracted ports.
    *   It uses the `InterfaceScanner` (`interface_scanner.py`) to group ports based on naming conventions (e.g., `in0_TDATA`, `config_AWADDR`, `ap_clk`) into potential `PortGroup` objects representing Global, AXI-Stream, or AXI-Lite interfaces.
    *   Each potential `PortGroup` is then validated by the `ProtocolValidator` (`protocol_validator.py`) to ensure it meets the minimum requirements for its identified type (e.g., required signals, correct directions).
    *   Only valid `PortGroup` objects are kept.
8.  **Result Aggregation:** The parser aggregates the module name, parameters, valid interfaces, and pragmas into an `HWKernel` data object (`data.py`).
9.  **Post-Validation:** The parser performs final checks, ensuring at least one AXI-Stream interface was found and raising an error if any ports remain unassigned to a valid interface.
10. **Return Value:** The `HWKernel` object is returned.

## Key Components

*   `parser.py`: Main class `RTLParser` orchestrating the parsing flow.
*   `grammar.py`: Handles loading the `tree-sitter` grammar (`.so` file) and defines node type constants.
*   `data.py`: Defines core data structures (`HWKernel`, `Port`, `Parameter`, `ModuleSummary`, `Direction`).
*   `interface_types.py`: Defines interface-related structures (`PortGroup`, `InterfaceType`, `ValidationResult`).
*   `pragma.py`: Logic for extracting `//@brainsmith` pragmas from comments.
*   `interface_scanner.py`: Identifies potential interface groups based on port naming conventions. Defines expected signal patterns.
*   `protocol_validator.py`: Validates that identified `PortGroup` objects adhere to protocol rules (required signals, directions).
*   `interface_builder.py`: Coordinates the `InterfaceScanner` and `ProtocolValidator` to produce a dictionary of validated interfaces.
*   `sv.so` (Assumed): The pre-compiled `tree-sitter` SystemVerilog grammar library.

## Usage

```python
import logging
from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser, ParserError, SyntaxError

# Configure logging if desired (optional, application should configure)
# logging.basicConfig(level=logging.INFO)
# logging.getLogger('brainsmith.tools.hw_kernel_gen.rtl_parser').setLevel(logging.DEBUG)

sv_file_path = "path/to/your/kernel.sv"
grammar_so_path = None # Optional: Path to sv.so if not in default location

try:
    # Instantiate the parser
    # Set debug=True for verbose logging during parsing
    parser = RTLParser(grammar_path=grammar_so_path, debug=False)

    # Parse the file
    hw_kernel_info = parser.parse_file(sv_file_path)

    # Access the extracted information
    print(f"Successfully parsed kernel: {hw_kernel_info.name}")

    print("\nParameters:")
    for param in hw_kernel_info.parameters:
        print(f"- {param.name} (Type: {param.param_type}, Default: {param.default_value})")

    print("\nInterfaces:")
    for iface_name, interface in hw_kernel_info.interfaces.items():
        print(f"- {iface_name} (Type: {interface.type.name})")
        # Access ports within the interface: interface.ports (dict mapping base name to Port object)
        # Example: print(interface.ports.keys())

    print("\nPragmas:")
    for pragma in hw_kernel_info.pragmas:
        print(f"- Type: {pragma.type.name}, Args: {pragma.args}")

except SyntaxError as e:
    print(f"Syntax Error: {e}")
except ParserError as e:
    print(f"Parsing Error: {e}")
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    # Catch other potential errors during parsing
    print(f"An unexpected error occurred: {e}")
    logging.exception("Unexpected parsing error") # Log stack trace
