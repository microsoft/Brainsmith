# RTL Parser Pragma Reference

This document provides a comprehensive reference for all pragmas supported by the Brainsmith RTL Parser. Pragmas are special comments that provide semantic information to enhance the parser's understanding of SystemVerilog modules.

## Pragma Syntax

All pragmas follow the format:
```
// @brainsmith <pragma_type> <arguments>
```

Pragmas must:
- Start with `// @brainsmith` (single-line comment)
- Be placed before the module declaration
- Use exact parameter names from the module

## Module Selection Pragmas

### TOP_MODULE

Specifies which module to parse in files containing multiple modules.

**Syntax:**
```systemverilog
// @brainsmith top_module <module_name>
```

**Example:**
```systemverilog
// @brainsmith top_module my_accelerator
module helper_module (...);
endmodule

module my_accelerator (...);  // This will be parsed
endmodule
```

**Validation:**
- Module name must exist in the file
- Only one TOP_MODULE pragma allowed per file

## Interface Configuration Pragmas

### DATATYPE

Constrains the allowed datatypes for an interface.

**Syntax:**
```systemverilog
// @brainsmith datatype <interface_name> <base_type> <width1> [<width2> ...]
```

**Base Types:**
- `UINT`: Unsigned integer
- `INT`: Signed integer
- `UFIXED`: Unsigned fixed-point
- `FIXED`: Signed fixed-point
- `BFLOAT`: Brain floating-point
- `FLOAT`: Standard floating-point

**Examples:**
```systemverilog
// Fixed 8-bit unsigned
// @brainsmith datatype input0 UINT 8

// Multiple allowed widths
// @brainsmith datatype weights UINT 8 16 32

// Signed integer
// @brainsmith datatype output0 INT 16 32

// Fixed-point types
// @brainsmith datatype pixels UFIXED 16 8
```

**Behavior:**
- Creates constraints for code generation
- Widths are validated against actual port widths
- Last width value for fixed-point types represents fractional bits

### WEIGHT

Marks an interface as containing weight/parameter data.

**Syntax:**
```systemverilog
// @brainsmith weight <interface_name>
```

**Example:**
```systemverilog
// @brainsmith weight kernel_weights
module conv2d #(
    parameter WEIGHT_WIDTH = 8
) (
    input s_axis_kernel_weights_tvalid,
    output s_axis_kernel_weights_tready,
    input [WEIGHT_WIDTH-1:0] s_axis_kernel_weights_tdata,
    ...
);
```

**Behavior:**
- Sets `is_weight=True` on the interface
- Affects code generation and optimization
- Only applies to input interfaces

### DATATYPE_PARAM

Links RTL parameters to datatype properties of an interface.

**Syntax:**
```systemverilog
// @brainsmith datatype_param <interface_name> <property>=<parameter_name>
```

**Properties:**
- `width`: Bit width
- `signed`: Signed flag
- `bias`: Bias value
- `format`: Format parameter
- `fractional_width`: Fixed-point fractional bits
- `exponent_width`: Floating-point exponent bits
- `mantissa_width`: Floating-point mantissa bits

**Examples:**
```systemverilog
// @brainsmith datatype_param input0 width=DATA_WIDTH
// @brainsmith datatype_param input0 signed=IS_SIGNED
// @brainsmith datatype_param output0 fractional_width=FRAC_BITS
module processor #(
    parameter DATA_WIDTH = 16,
    parameter IS_SIGNED = 0,
    parameter FRAC_BITS = 8
) (...);
```

## Dimension Pragmas

### BDIM

Specifies block dimensions for tiled processing.

**Syntax:**
```systemverilog
// Basic form
// @brainsmith bdim <interface_name> [<param1>, <param2>, ...]

// With shape mapping
// @brainsmith bdim <interface_name> [<param1>, <param2>] shape=[<shape1>, <shape2>]
```

**Examples:**
```systemverilog
// Simple 2D tiling
// @brainsmith bdim input0 [TILE_H, TILE_W]

// With shape mapping
// @brainsmith bdim pixels [BH, BW] shape=[H, W]

// 3D tiling
// @brainsmith bdim volume [BLOCK_D, BLOCK_H, BLOCK_W]

// Singleton dimension
// @brainsmith bdim input0 [TILE_SIZE, 1]
```

**Behavior:**
- Parameters are moved from kernel to interface
- Shape mapping enables dimension transformations
- Singleton dimensions ("1") are allowed but require real parameters

### SDIM

Specifies stream dimensions for data streaming.

**Syntax:**
```systemverilog
// Single dimension
// @brainsmith sdim <interface_name> <parameter_name>

// Multiple dimensions
// @brainsmith sdim <interface_name> [<param1>, <param2>]

// With shape mapping
// @brainsmith sdim <interface_name> [<param1>] shape=[<shape1>]
```

**Examples:**
```systemverilog
// Simple stream dimension
// @brainsmith sdim input0 NUM_BLOCKS

// Multi-dimensional streaming
// @brainsmith sdim pixels [ROWS, COLS]

// With shape
// @brainsmith sdim data STREAM_LEN shape=[TOTAL_SIZE]
```

## Parameter Management Pragmas

### ALIAS

Creates parameter aliases for compatibility or clarity.

**Syntax:**
```systemverilog
// @brainsmith alias <new_name>=<existing_parameter>
```

**Examples:**
```systemverilog
// @brainsmith alias WIDTH=DATA_WIDTH
// @brainsmith alias HEIGHT=IMAGE_HEIGHT
module processor #(
    parameter DATA_WIDTH = 32,
    parameter IMAGE_HEIGHT = 1080
) (...);
```

**Behavior:**
- Creates a new parameter with the aliased name
- Original parameter value is used
- Useful for matching expected parameter names

### DERIVED_PARAMETER

Defines computed parameters based on expressions.

**Syntax:**
```systemverilog
// @brainsmith derived_parameter <name> <expression>
```

**Examples:**
```systemverilog
// @brainsmith derived_parameter TOTAL_SIZE TILE_H * TILE_W
// @brainsmith derived_parameter BUFFER_SIZE 2 * MAX_WIDTH + 1
// @brainsmith derived_parameter LOG2_WIDTH $clog2(WIDTH)
```

**Supported Operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `**`
- Functions: `$clog2()`, `max()`, `min()`
- Parentheses for grouping

**Behavior:**
- Parameter is evaluated during parsing
- Can reference other parameters
- Becomes a regular parameter in metadata

### AXILITE_PARAM

Links parameters to AXI-Lite register addresses.

**Syntax:**
```systemverilog
// @brainsmith axilite_param <interface_name> <parameter_name> <address>
```

**Examples:**
```systemverilog
// @brainsmith axilite_param config THRESHOLD 0x10
// @brainsmith axilite_param config MODE 0x14
// @brainsmith axilite_param control ENABLE 0x00
module configurable #(
    parameter THRESHOLD = 128,
    parameter MODE = 0,
    parameter ENABLE = 1
) (
    input s_axilite_arvalid,
    ...
);
```

**Behavior:**
- Parameter is moved to the AXI-Lite interface
- Address can be decimal or hexadecimal
- Used for runtime configuration

## Relationship Pragmas

### RELATIONSHIP

Defines constraints between parameters.

**Syntax:**
```systemverilog
// @brainsmith relationship <expression>
```

**Examples:**
```systemverilog
// @brainsmith relationship TILE_H <= MAX_HEIGHT
// @brainsmith relationship TILE_W * TILE_H <= 4096
// @brainsmith relationship INPUT_WIDTH == OUTPUT_WIDTH
```

**Supported Operators:**
- Comparison: `<`, `<=`, `>`, `>=`, `==`, `!=`
- Arithmetic: `+`, `-`, `*`, `/`, `%`

**Behavior:**
- Used for validation and constraint solving
- Not enforced by parser but passed to generators
- Can reference any kernel parameters

## Pragma Application Order

Pragmas are applied in a specific order to ensure correct behavior:

1. **Module Selection** (TOP_MODULE)
2. **Interface Configuration** (DATATYPE, WEIGHT)
3. **Dimension Assignment** (BDIM, SDIM)
4. **Parameter Enhancement** (ALIAS, DERIVED_PARAMETER)
5. **Parameter Linking** (DATATYPE_PARAM, AXILITE_PARAM)
6. **Relationships** (RELATIONSHIP)

## Error Handling

### Warnings vs Errors

- **Warnings**: Invalid pragmas produce warnings but parsing continues
- **Errors**: SystemVerilog syntax errors stop parsing

### Common Warning Examples

```
Warning: Invalid pragma at line 5: Unknown parameter 'MISSING_PARAM'
Warning: Failed to apply pragma bdim at line 8: Interface 'unknown' not found
Warning: Invalid pragma syntax at line 12: Expected format: @brainsmith <type> <args>
```

### Debugging Tips

1. **Check Parameter Names**: Ensure parameters match module declaration exactly
2. **Verify Interface Names**: Use actual interface names from detected ports
3. **Validate Syntax**: Follow exact syntax including brackets for lists
4. **Check Line Numbers**: Warnings include line numbers for easy debugging

## Best Practices

### 1. Group Related Pragmas

```systemverilog
// Input configuration
// @brainsmith datatype input0 UINT 8 16
// @brainsmith bdim input0 [TILE_H, TILE_W]
// @brainsmith sdim input0 NUM_TILES

// Output configuration  
// @brainsmith datatype output0 UINT 16 32
// @brainsmith bdim output0 [TILE_H, TILE_W]
```

### 2. Document Complex Pragmas

```systemverilog
// Shape mapping: BH,BW are block dimensions, H,W are full image
// @brainsmith bdim pixels [BH, BW] shape=[H, W]
```

### 3. Use Meaningful Names

```systemverilog
// Clear parameter relationships
// @brainsmith derived_parameter PIXELS_PER_BLOCK BLOCK_HEIGHT * BLOCK_WIDTH
// @brainsmith relationship PIXELS_PER_BLOCK <= MAX_LOCAL_MEMORY
```

### 4. Validate Early

Test pragma effects immediately:
```python
kernel = parse_rtl_file("module.sv")
# Verify pragma was applied
assert kernel.inputs[0].bdim == ["TILE_H", "TILE_W"]
```

## Extension Guide

To add new pragma types:

1. Create new class in `pragmas/` inheriting from `Pragma` or `InterfacePragma`
2. Implement `_parse_inputs()` for parsing pragma arguments
3. Implement `apply_to_kernel()` for applying effects
4. Register in `PragmaType` enum
5. Add to pragma factory in parser

Example skeleton:
```python
class MyPragma(InterfacePragma):
    def _parse_inputs(self, inputs: List[str]) -> None:
        # Parse pragma arguments
        self.interface_name = inputs[0]
        self.value = inputs[1]
    
    def apply_to_kernel(self, kernel: KernelMetadata) -> None:
        # Apply pragma effects
        interface = self._get_interface(kernel, self.interface_name)
        interface.my_property = self.value
```