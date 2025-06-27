# Brainsmith RTL Parser Pragma Guide

A quick reference for using `@brainsmith` pragmas in SystemVerilog RTL modules.

## Pragma Syntax

All pragmas follow this format:
```systemverilog
// @brainsmith <type> <arguments...>
```

## Interface Type Compatibility

The following table shows which pragmas and auto-linking features apply to each interface type:

| Feature | INPUT | OUTPUT | WEIGHT | CONFIG | CONTROL |
|---------|-------|---------|---------|---------|----------|
| DATATYPE | ✓ | ✓ | ✓ | ✓ | ✗ |
| DATATYPE_PARAM | ✓ | ✓ | ✓ | ✓ | ✗ |
| WEIGHT pragma | ✓ | ✗ | N/A | ✗ | ✗ |
| BDIM | ✓ | ✓ | ✓ | ✗ | ✗ |
| SDIM | ✓ | ✗ | ✓ | ✗ | ✗ |
| Auto-link datatypes | ✓ | ✓ | ✓ | ✓ | ✗ |
| Auto-link BDIM | ✓ | ✓ | ✓ | ✗ | ✗ |
| Auto-link SDIM | ✓ | ✗ | ✓ | ✗ | ✗ |

**Key Points:**
- BDIM (Block Dimensions) apply to all dataflow interfaces: INPUT, OUTPUT, and WEIGHT
- SDIM (Stream Dimensions) only apply to INPUT and WEIGHT interfaces
- Datatype features (DATATYPE pragma, DATATYPE_PARAM pragma, auto-linking) apply to all data-carrying interfaces: INPUT, OUTPUT, WEIGHT, and CONFIG
- CONTROL interfaces (clock/reset signals) do not support any parameterization
- The WEIGHT pragma marks an input interface as containing weights

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
Specify block dimension parameters for an interface.
```systemverilog
// Single parameter syntax
// @brainsmith BDIM s_axis_input0 INPUT0_BDIM
// @brainsmith BDIM weights_V WEIGHTS_BLOCK_SIZE

// Multi-dimensional syntax
// @brainsmith BDIM s_axis_input0 [TILE_H, TILE_W]
// @brainsmith BDIM weights_V [1, KERNEL_SIZE]      # Singleton first dimension
// @brainsmith BDIM output0 [OUT_H, OUT_W, OUT_C]
```
**Formats**: 
- `BDIM <interface_name> <param_name>` - Single dimension
- `BDIM <interface_name> [<param1>, <param2>, ...]` - Multi-dimensional

**Interface Types**: INPUT, OUTPUT, WEIGHT only (not CONFIG)

**Special Values**: 
- `1` - Singleton dimension (only in lists with at least one parameter)
- Parameter names - Actual block dimensions

**Notes**: Lists containing "1" must have at least one real parameter. No magic numbers except "1".

### SDIM
Specify stream dimension parameters for an interface.
```systemverilog
// Single parameter syntax
// @brainsmith SDIM s_axis_input0 INPUT0_SDIM
// @brainsmith SDIM weights_V WEIGHTS_STREAM_SIZE

// Multi-dimensional syntax
// @brainsmith SDIM input0 [SDIM_H, SDIM_W, SDIM_C]
// @brainsmith SDIM weights [1, STREAM_DIM]          # Singleton first dimension
```
**Formats**: 
- `SDIM <interface_name> <param_name>` - Single dimension
- `SDIM <interface_name> [<param1>, <param2>, ...]` - Multi-dimensional

**Interface Types**: INPUT, WEIGHT only (not OUTPUT or CONFIG)

**Special Values**: 
- `1` - Singleton dimension (only in lists with at least one parameter)
- Parameter names - Actual stream dimensions

**Notes**: Stream dimensions only make sense for inputs and weights. Lists containing "1" must have at least one real parameter.

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

## Auto-Linking for Dimension Parameters

The RTL parser supports automatic linking of BDIM/SDIM parameters without pragmas using naming conventions:

### Single Dimension Parameters
```systemverilog
parameter s_axis_BDIM = 64;   // Auto-linked to s_axis interface
parameter s_axis_SDIM = 128;  // Auto-linked to s_axis interface
```

### Indexed Multi-Dimensional Parameters
```systemverilog
// Contiguous indexed BDIM (3D tensor: H x W x C)
parameter input_BDIM0 = 16;   // Height
parameter input_BDIM1 = 16;   // Width  
parameter input_BDIM2 = 3;    // Channels
// Auto-linked as: bdim_params = ["input_BDIM0", "input_BDIM1", "input_BDIM2"]

// Non-contiguous indexed (missing BDIM1 treated as singleton)
parameter weights_BDIM0 = 32;  // Batch
parameter weights_BDIM2 = 64;  // Features
// Auto-linked as: bdim_params = ["weights_BDIM0", "1", "weights_BDIM2"]

// Mixed case supported
parameter output_sdim0 = 256;
parameter output_sdim1 = 128;
// Auto-linked as: sdim_params = ["output_sdim0", "output_sdim1"]
```

### Precedence Rules
1. **Pragma wins**: If BDIM/SDIM pragma exists, it overrides auto-linking
2. **Single before indexed**: `interface_BDIM` takes precedence over `interface_BDIM0/1/2`
3. **Default fallback**: If no parameters found, defaults to `interface_BDIM` and `interface_SDIM`

### Examples

#### Complete Example with Mixed Styles
```systemverilog
module example #(
    // Indexed parameters for input (supports both BDIM and SDIM)
    parameter s_axis_input_BDIM0 = 16,
    parameter s_axis_input_BDIM1 = 16,
    parameter s_axis_input_BDIM2 = 3,
    parameter s_axis_input_SDIM0 = 1024,
    parameter s_axis_input_SDIM1 = 512,
    
    // Single parameter for weights (supports both BDIM and SDIM)
    parameter weights_BDIM = 64,
    parameter weights_SDIM = 512,
    
    // Output parameters (BDIM only - SDIM ignored on outputs)
    parameter m_axis_output_BDIM = 128,
    parameter m_axis_output_SDIM = 256,  // This will be ignored!
    
    // Config parameters (neither BDIM nor SDIM apply)
    parameter s_axilite_config_BDIM = 32,  // This will be ignored!
    parameter s_axilite_config_SDIM = 64,  // This will be ignored!
    
    // Pragma override example
    parameter OUT_H = 8,
    parameter OUT_W = 8
) (
    // ... ports ...
);

// @brainsmith BDIM m_axis_result [OUT_H, OUT_W]  // This pragma wins over any auto-linking
```

#### Interface Type Examples
```systemverilog
// ✓ VALID: Input interface with both BDIM and SDIM
parameter s_axis_data_BDIM = 64;
parameter s_axis_data_SDIM = 1024;

// ✓ VALID: Weight interface with both BDIM and SDIM  
parameter weights_V_BDIM0 = 32;
parameter weights_V_BDIM1 = 64;
parameter weights_V_SDIM = 512;

// ⚠️ PARTIAL: Output interface - only BDIM applies
parameter m_axis_result_BDIM = 16;
parameter m_axis_result_SDIM = 128;  // Ignored by auto-linking

// ✗ INVALID: Config interface - neither BDIM nor SDIM apply
parameter s_axilite_ctrl_BDIM = 8;   // Ignored by auto-linking
parameter s_axilite_ctrl_SDIM = 256; // Ignored by auto-linking
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