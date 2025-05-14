# RTL Parser Test Suite

This directory contains the test suite for the RTL Parser component of the Brainsmith Hardware Kernel Generator.

## Overview

The test suite covers all aspects of the RTL Parser, which is responsible for parsing SystemVerilog files and extracting module interfaces, parameters, and pragmas. The parser uses tree-sitter for SystemVerilog syntax parsing and implements specialized logic for identifying standard hardware interfaces.

## Test Structure

The test suite follows a modular design with each test file focusing on different aspects of the parser:

1. **test_rtl_parser.py**: Main test suite covering core parser functionality:
   - `TestParserCore`: Basic file handling, module selection, syntax validation
   - `TestParameterParsing`: Extraction of module parameters
   - `TestPortParsing`: Port extraction with various syntaxes
   - `TestPragmaHandling`: Handling of custom @brainsmith pragmas
   - `TestInterfaceAnalysis`: Interface identification and validation

2. **test_interface_builder.py**: Tests for the interface building component:
   - Construction of interfaces from port groups
   - Validation of interface protocols
   - Handling of unassigned ports

3. **test_interface_scanner.py**: Tests for the interface scanning component:
   - Port grouping based on naming conventions
   - Recognition of interface types
   - Pattern matching algorithms

4. **test_protocol_validator.py**: Tests for protocol validation:
   - Validation of Global Control interface
   - Validation of AXI-Stream interfaces
   - Validation of AXI-Lite interfaces

5. **test_width_parsing.py**: Tests specific to width expression extraction:
   - Simple widths (e.g., [7:0])
   - Parametric widths (e.g., [WIDTH-1:0])
   - Complex expressions (e.g., [$clog2(DEPTH)-1:0])

## Test Fixtures

Common test fixtures are defined in `test_fixtures.py` and imported through `conftest.py`:

- `parser`: A configured RTLParser instance (module scope)
- `parser_debug`: A RTLParser with debug enabled (function scope)
- `temp_sv_file`: Creates temporary SystemVerilog files
- Port fixtures: Pre-defined port collections for common interfaces
- Content fixtures: Pre-defined SystemVerilog content for testing

## Testing Approaches

The suite uses three main testing approaches:

1. **Unit Testing**: Testing specific methods directly (e.g., `_extract_kernel_components`)
2. **Integration Testing**: Testing full parsing flow with `parse_file`
3. **Error Handling Testing**: Verifying correct errors are raised for invalid inputs

## Coverage Analysis

The test suite provides coverage for all major components of the RTL Parser:

1. **Parser Core**: 100% coverage of initialization, file handling, and module selection
2. **Parameter Parsing**: 100% coverage of parameter types, defaults, and complex expressions
3. **Port Parsing**: 100% coverage of port directions, widths, and declaration styles
4. **Pragma Handling**: 100% coverage of supported pragma types and error handling
5. **Interface Analysis**: 100% coverage of interface identification and validation rules

## Best Practices

When extending or modifying the test suite:

1. **Use Fixtures**: Reuse fixtures from `test_fixtures.py` for common objects
2. **Parameterize Tests**: Use pytest's parametrize feature for similar test cases
3. **Test Error Cases**: Include tests for expected error conditions
4. **Document Tests**: Add clear docstrings explaining test purpose and assertions
5. **Separate Parsing Stages**: Tests should be clear about which parsing stages they target
   - Stage 1: AST parsing and pragma extraction
   - Stage 2: Module component extraction (parameters, ports)
   - Stage 3: Interface analysis and validation

## Adding New Tests

1. Determine the appropriate test file based on component being tested
2. Reuse fixtures where possible
3. Follow existing patterns for similar tests
4. Add detailed docstrings
5. Use clear assertion messages

## Running Tests

```bash
# Run all RTL parser tests
pytest tests/tools/hw_kernel_gen/rtl_parser

# Run specific test file
pytest tests/tools/hw_kernel_gen/rtl_parser/test_rtl_parser.py

# Run specific test class
pytest tests/tools/hw_kernel_gen/rtl_parser/test_rtl_parser.py::TestParameterParsing

# Run specific test
pytest tests/tools/hw_kernel_gen/rtl_parser/test_rtl_parser.py::TestParameterParsing::test_parameter_real_types
```
