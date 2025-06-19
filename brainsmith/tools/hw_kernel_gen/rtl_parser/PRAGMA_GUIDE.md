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

### ALIAS
Expose RTL parameters with user-friendly names in the Python API.
```systemverilog
// @brainsmith ALIAS PE parallelism_factor
// @brainsmith ALIAS C num_channels
// @brainsmith ALIAS FOLD folding_factor
```
**Format**: `ALIAS <rtl_parameter> <nodeattr_name>`

**Notes**: 
- RTL parameter must exist and be exposed
- Node attribute name must not conflict with other parameters
- Allows hardware-specific names in RTL with clean Python API

### DERIVED_PARAMETER
Compute parameter values from Python expressions instead of exposing them.
```systemverilog
// @brainsmith DERIVED_PARAMETER SIMD self.get_input_datatype().bitwidth()
// @brainsmith DERIVED_PARAMETER MEM_DEPTH self.calc_wmem()
// @brainsmith DERIVED_PARAMETER LATENCY self.get_nodeattr("parallelism_factor") * 2 + 3
```
**Format**: `DERIVED_PARAMETER <parameter_name> <python_expression>`

**Notes**:
- Parameter is computed in HWCustomOp/RTLBackend context
- Can reference other parameters via `self.get_nodeattr()`
- Not exposed as node attributes

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

## Parameter Naming Conventions

### Automatic Parameter Detection

The parser automatically detects parameters following consistent naming patterns:

**Required Pattern**: `{interface}_{property}`

- `s_axis_input0_WIDTH` ✓
- `s_axis_input0_SIGNED` ✓
- `SIGNED_s_axis_input0` ✗ (generates warning)

### Default Parameter Names

For common interface names:
- `s_axis_input0` → `s_axis_input0_WIDTH`, `s_axis_input0_SIGNED`, `s_axis_input0_BDIM`, `s_axis_input0_SDIM`
- `m_axis_output0` → `m_axis_output0_WIDTH`, `m_axis_output0_SIGNED`, `m_axis_output0_BDIM`, `m_axis_output0_SDIM`
- `weights_V` → `weights_V_WIDTH`, `weights_V_SIGNED`, `weights_V_BDIM`, `weights_V_SDIM`

### Parameter Handling Hierarchy

1. **Parameter Pragmas** (highest priority)
   - ALIAS: Custom node attribute names
   - DERIVED_PARAMETER: Computed values

2. **Interface Pragmas** (medium priority)
   - DATATYPE_PARAM: Explicit linkage
   - BDIM/SDIM: Dimension linkage

3. **Auto-detection** (lowest priority)
   - Parameters following naming conventions

Only parameters not handled by higher-priority mechanisms remain exposed as node attributes.

## Interface Name Matching

Pragmas support flexible interface name matching:
- Exact match: `input0` matches `input0`
- Prefix match: `input0` matches `input0_V_data_V`
- Base name match: `query` matches `s_axis_query`

## Best Practices

1. **Use parameter names**: Always reference RTL parameters, not magic numbers
2. **Follow naming conventions**: Use `{interface}_{property}` pattern for auto-detection
3. **Be specific**: Use full interface names to avoid ambiguity
4. **Document purpose**: Add comments explaining pragma usage
5. **Use ALIAS for clean APIs**: Hide hardware-specific names behind user-friendly aliases
6. **Test coverage**: Verify pragmas work with your specific RTL patterns

## Common Patterns

### Clean API with ALIAS
```systemverilog
// Hardware uses PE, software sees parallelism_factor
// @brainsmith ALIAS PE parallelism_factor
parameter PE = 16,
```

### Multi-Interface Modules
```systemverilog
// Each interface gets its own parameters
// @brainsmith DATATYPE_PARAM input0 width input0_WIDTH
// @brainsmith DATATYPE_PARAM input1 width input1_WIDTH
```

### Computed Parameters
```systemverilog
// SIMD computed from input datatype
// @brainsmith DERIVED_PARAMETER SIMD self.get_input_datatype().bitwidth()
parameter SIMD = 8,  // Will be overridden
```

## Generated Code Examples

### HWCustomOp with ALIAS
```python
def get_nodeattr_types(self):
    my_attrs = {}
    # ALIAS pragmas create user-friendly names
    my_attrs["parallelism_factor"] = ("i", True, 16)
    my_attrs["num_channels"] = ("i", True, 64)
    my_attrs["folding_factor"] = ("i", True, 4)
    # Exposed parameters without aliases
    my_attrs["CONFIG_WIDTH"] = ("i", True, 32)
    return my_attrs
```

### RTLBackend with All Pragma Types
```python
def prepare_codegen_rtl_values(self):
    code_gen_dict = {}
    
    # ALIAS parameters (from node attributes)
    code_gen_dict["$PE$"] = [str(self.get_nodeattr("parallelism_factor"))]
    code_gen_dict["$C$"] = [str(self.get_nodeattr("num_channels"))]
    
    # DERIVED parameters (computed)
    code_gen_dict["$SIMD$"] = [str(self.get_input_datatype().bitwidth())]
    code_gen_dict["$MEM_DEPTH$"] = [str(self.calc_wmem())]
    
    # Auto-detected interface parameters
    code_gen_dict["$S_AXIS_WIDTH$"] = [str(self.get_input_datatype().bitwidth())]
    code_gen_dict["$S_AXIS_SIGNED$"] = [str(int(self.get_input_datatype().signed()))]
    
    return code_gen_dict
```

## Troubleshooting

### Warning: Parameter not found
```
WARNING: Interface 'weights_V' expects signed parameter 'weights_V_SIGNED' but it was not found
```
**Solution**: Either add the missing parameter or use DATATYPE_PARAM pragma

### Warning: ALIAS parameter not exposed
```
WARNING: ALIAS pragma at line 5: Parameter 'PE' is not in exposed parameters list
```
**Solution**: Check if parameter is already linked via interface pragmas or auto-detection

### Warning: Parameter linked by multiple interfaces
```
WARNING: Parameter 'SHARED_WIDTH' is linked by multiple interfaces: 'input0' and 'input1'
```
**Solution**: Use interface-specific parameters or DATATYPE_PARAM pragmas