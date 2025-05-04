# RTL Parser and AST Structure Documentation

## Overview
The RTL parser uses tree-sitter to parse SystemVerilog source code into an Abstract Syntax Tree (AST). This document explains the AST structure and how the parser navigates it.

## AST Structure for Module Parameters

### Basic Structure
```
source_file
└── module_declaration
    └── module_ansi_header
        ├── module_keyword ("module")
        ├── simple_identifier (module name)
        ├── parameter_port_list
        │   ├── "#" token
        │   ├── "(" token
        │   ├── parameter nodes and comments
        │   └── ")" token
        └── list_of_port_declarations
```

### Parameter Port List Structure
The `parameter_port_list` contains parameters and their associated comments in sequential order. For example:

```
parameter_port_list
├── "#"
├── "("
├── comment ("// Processing elements")
├── comment ("// requires C = k*PE")
├── parameter_port_declaration
│   ├── data_type ("int unsigned")
│   └── list_of_param_assignments
│       └── param_assignment
│           ├── simple_identifier ("PE")
│           ├── "="
│           └── constant_param_expression ("1")
├── ","
├── comment ("/* Floating point ... */")
├── parameter_port_declaration
│   ├── data_type ("bit")
│   └── list_of_param_assignments
└── ")"
```

### Comment Types and Locations
Comments appear in the AST in three ways:

1. **Standalone Comments**: Direct child nodes of type "comment" in the parameter_port_list
   ```
   parameter_port_list
   ├── comment ("// Processing elements")
   ├── comment ("// requires C = k*PE")
   ├── parameter_port_declaration
   ```

2. **Inline Comments**: Part of the parameter_port_declaration text
   ```verilog
   int unsigned PE = 1, // This is inline
   ```

3. **Multi-line Comments**: C-style comments that appear as comment nodes
   ```verilog
   /* Floating point format:
    * [sign] | exponent | mantissa
    */
   ```

### Parameter Node Structure
Each parameter is represented by a `parameter_port_declaration` node with:

```
parameter_port_declaration
├── data_type 
│   └── type information nodes
├── list_of_param_assignments
│   └── param_assignment
│       ├── simple_identifier (parameter name)
│       ├── "="
│       └── constant_param_expression (default value)
```

## Comment Association Rules

The parser needs to associate comments with parameters according to these rules:

1. Comments appearing before a parameter (up until the previous parameter or list start) belong to that parameter
2. Comments after a comma and before the next parameter belong to the next parameter
3. Inline comments in a parameter's own text belong to that parameter

## Parser Navigation

The parser navigates the AST using these key methods:

1. `_find_module_definition`: Locates the module_declaration node
2. `_extract_parameters`: 
   - Finds the parameter_port_list
   - Iterates through children to find parameters and comments
3. `_get_parameter_comments`:
   - Takes current parameter node index
   - Looks backward for associated comments
   - Checks for inline comments

## Common AST Node Types

- **module_declaration**: Top-level module node
- **module_ansi_header**: Module interface definition
- **parameter_port_list**: Container for parameters
- **parameter_port_declaration**: Individual parameter definition
- **comment**: Comment nodes (both // and /* style)
- **simple_identifier**: Names (module, parameter, etc)
- **data_type**: Parameter type information
- **constant_param_expression**: Parameter default values

## Test Case Example
The example in test_parameter_comments shows this structure:

```verilog
module test #(
    // Processing elements    <- comment node
    // requires C = k*PE     <- comment node
    int unsigned PE = 1,     <- parameter_port_declaration
    
    /* Floating point...     <- comment node
     * [sign] | exponent...
     */
    bit FPARG = 0           <- parameter_port_declaration
)(
    input logic clk
);
