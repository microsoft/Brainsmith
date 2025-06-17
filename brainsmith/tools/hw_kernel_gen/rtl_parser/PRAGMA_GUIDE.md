# Brainsmith RTL Parser Pragma Guide

A quick reference for using `@brainsmith` pragmas in SystemVerilog RTL modules.

## Pragma Syntax

All pragmas follow this format:
```systemverilog
// @brainsmith <type> <arguments...>
```

## Supported Pragmas

### TOP_MODULE
Specify the top module when multiple modules exist in the same file.
```systemverilog
// @brainsmith TOP_MODULE my_accelerator
module my_accelerator #(...) (...);
```

### DATATYPE
Constrain datatype for an interface.
```systemverilog
// @brainsmith DATATYPE input0 UINT 8 16
// @brainsmith DATATYPE weights FIXED 8 8
```
**Format**: `DATATYPE <interface_name> <base_type> <min_bits> <max_bits>`

### DATATYPE_PARAM
Map interface datatype properties to specific RTL parameters.
```systemverilog
// @brainsmith DATATYPE_PARAM s_axis_input0 width INPUT0_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_input0 signed SIGNED_INPUT0
// @brainsmith DATATYPE_PARAM s_axis_query width QUERY_WIDTH
```
**Format**: `DATATYPE_PARAM <interface_name> <property_type> <parameter_name>`

**Property types**: `width`, `signed`, `format`, `bias`, `fractional_width`

### WEIGHT
Mark input interfaces as weight interfaces.
```systemverilog
// @brainsmith WEIGHT weights_V
// @brainsmith WEIGHT bias_stream
```

### BDIM
Link interface to block dimension parameter with optional shape specification.
```systemverilog
// @brainsmith BDIM s_axis_input0 INPUT0_BDIM
// @brainsmith BDIM weights_V WEIGHTS_BLOCK_SIZE SHAPE=[C,PE] RINDEX=0
// @brainsmith BDIM out0 OUTPUT0_BDIM SHAPE=[TILE_SIZE,:] RINDEX=1
```
**Format**: `BDIM <interface_name> <param_name> [SHAPE=<shape>] [RINDEX=<n>]`

**Notes**: Parameter name is mandatory. SHAPE and RINDEX are optional. Use parameter names (not magic numbers) in shapes.

### SDIM
Link interface to stream dimension parameter.
```systemverilog
// @brainsmith SDIM s_axis_input0 INPUT0_SDIM
// @brainsmith SDIM weights_V WEIGHTS_STREAM_SIZE
```
**Format**: `SDIM <interface_name> <param_name>`

**Notes**: Stream shape is inferred from BDIM configuration.

## Multi-Interface Example

For modules with multiple interfaces of the same type:

```systemverilog
// Multi-input elementwise add
// @brainsmith DATATYPE_PARAM s_axis_input0 width INPUT0_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_input0 signed SIGNED_INPUT0
// @brainsmith DATATYPE_PARAM s_axis_input1 width INPUT1_WIDTH  
// @brainsmith DATATYPE_PARAM s_axis_input1 signed SIGNED_INPUT1

module elementwise_add #(
    parameter INPUT0_WIDTH = 8,
    parameter SIGNED_INPUT0 = 0,
    parameter INPUT1_WIDTH = 8,
    parameter SIGNED_INPUT1 = 0
) (
    input clk,
    input rst_n,
    
    input [INPUT0_WIDTH-1:0] s_axis_input0_tdata,
    input s_axis_input0_tvalid,
    output s_axis_input0_tready,
    
    input [INPUT1_WIDTH-1:0] s_axis_input1_tdata,
    input s_axis_input1_tvalid,
    output s_axis_input1_tready,
    
    output [7:0] m_axis_output0_tdata,
    output m_axis_output0_tvalid,
    input m_axis_output0_tready
);
```

## Default Parameter Naming

Without pragmas, interfaces get automatic parameter names:
- `s_axis_input0` → `INPUT0_WIDTH`, `SIGNED_INPUT0`, `INPUT0_BDIM`, `INPUT0_SDIM`
- `s_axis_input1` → `INPUT1_WIDTH`, `SIGNED_INPUT1`, `INPUT1_BDIM`, `INPUT1_SDIM`
- `m_axis_output0` → `OUTPUT0_WIDTH`, `SIGNED_OUTPUT0`, `OUTPUT0_BDIM`, `OUTPUT0_SDIM`
- `weights_V` → `WEIGHTS_WIDTH`, `SIGNED_WEIGHTS`, `WEIGHTS_BDIM`, `WEIGHTS_SDIM`

## Interface Name Matching

Pragmas support flexible interface name matching:
- Exact match: `input0` matches `input0`
- Prefix match: `input0` matches `input0_V_data_V`
- Base name match: `query` matches `s_axis_query`

## Best Practices

1. **Use parameter names**: Always reference RTL parameters, not magic numbers
2. **Be specific**: Use full interface names to avoid ambiguity
3. **Document purpose**: Add comments explaining pragma usage
4. **Test coverage**: Verify pragmas work with your specific RTL patterns

## Generated HWCustomOp

Pragmas control parameter names in generated HWCustomOp classes:

```python
def get_nodeattr_types(self):
    my_attrs = {}
    # Generated from DATATYPE_PARAM pragmas
    my_attrs["INPUT0_WIDTH"] = ("i", True, 8)
    my_attrs["SIGNED_INPUT0"] = ("i", False, 0, {0, 1})
    my_attrs["INPUT1_WIDTH"] = ("i", True, 8) 
    my_attrs["SIGNED_INPUT1"] = ("i", False, 0, {0, 1})
    # ...
```