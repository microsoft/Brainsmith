# AST Comparison Test Fixtures

This directory contains test fixtures for AST serialization and comparison testing.

## Purpose

These tests verify the initial phase of the RTL parser by:
1. Parsing SystemVerilog RTL files into AST (Abstract Syntax Tree)
2. Serializing the AST to human-readable text formats
3. Comparing parsed AST with expected output
4. Providing debugging utilities for parser development

## Files

### Input RTL Files
- `simple_module.sv` - Basic module with minimal structure
- `parameterized_module.sv` - Module with parameters and complex logic
- `module_with_pragmas.sv` - Module with brainsmith pragma annotations
- `malformed_module.sv` - Module with syntax errors for error testing

### Expected Output Files
- `*_expected.tree` - Expected AST output in tree format
- `*_expected.json` - Expected AST output in JSON format
- `*_expected.compact` - Expected AST output in compact format

## Usage

### Running Tests
```bash
# Run all AST serialization tests
./smithy "python -m pytest tests/tools/kernel_integrator/rtl_parser/test_ast_serialization.py -v"

# Run specific test
./smithy "python -m pytest tests/tools/kernel_integrator/rtl_parser/test_ast_serialization.py::TestASTSerialization::test_simple_module_serialization -v"

# Generate reference files
./smithy "python -m pytest tests/tools/kernel_integrator/rtl_parser/test_ast_serialization.py::TestASTSerialization::test_generate_reference_files -v"
```

### Debugging Parser Issues
The AST serializer provides several formats for debugging:

1. **Tree Format** - Visual tree structure with connectors
2. **JSON Format** - Structured data for programmatic analysis  
3. **Compact Format** - Single-line representation for diffs

### Creating New Test Cases
1. Add new RTL file to this directory
2. Run the test to generate AST output
3. Verify the AST is correct
4. Save as expected output file
5. Add test case to `test_ast_serialization.py`

## AST Serializer Features

- **Configurable depth** - Limit tree traversal depth
- **Text truncation** - Control node text display length
- **Position tracking** - Include/exclude line:col information
- **Node filtering** - Exclude specific node types
- **Diff support** - Compare two AST trees

## Example Output

Tree format shows the hierarchical structure:
```
└── source_file [0:0-15:9]
    └── module_declaration [6:0-15:9]
        ├── module_ansi_header [6:0-10:2]
        │   ├── module_keyword "module" [6:0-6:6]
        │   ├── simple_identifier "simple_module" [6:7-6:20]
        │   └── list_of_port_declarations [6:21-10:1]
        └── endmodule [15:0-15:9]
```