# DATATYPE_PARAM Pragma Deep Dive

## Overview

The DATATYPE_PARAM pragma is a powerful mechanism that creates explicit linkages between interface datatype properties and RTL parameters. It's part of the interface pragma family and provides fine-grained control over how hardware parameters map to interface configurations.

## Syntax & Format

```systemverilog
// @brainsmith DATATYPE_PARAM <interface_name> <property_type> <parameter_name>
```

### Components:
- **interface_name**: Name of the interface to configure (e.g., `s_axis_input0`, `weights_V`)
- **property_type**: The datatype property being mapped (lowercase)
- **parameter_name**: The RTL parameter that controls this property

### Valid Property Types:
1. **`width`** - Bit width of the interface data
2. **`signed`** - Whether data is signed (0/1)
3. **`format`** - Data format (e.g., FIXED, FLOAT)
4. **`bias`** - Bias/offset value for the data
5. **`fractional_width`** - Number of fractional bits in fixed-point

## Implementation Details

### Class Structure
```python
@dataclass 
class DatatypeParamPragma(InterfacePragma):
    """Maps specific RTL parameters to interface datatype properties."""
```

The pragma inherits from `InterfacePragma`, giving it:
- Interface name matching capabilities
- Integration with the pragma application pipeline
- Validation framework

### Key Methods

1. **`_parse_inputs()`**
   - Validates exactly 3 arguments
   - Checks property_type against valid list
   - Returns parsed dictionary with interface_name, property_type, parameter_name

2. **`apply_to_metadata()`**
   - Updates InterfaceMetadata's `datatype_params` dictionary
   - Preserves existing mappings while adding new ones
   - Returns updated metadata following immutability pattern

## Usage Examples

### Basic Width/Signed Mapping
```systemverilog
// Map interface width to specific parameter
// @brainsmith DATATYPE_PARAM s_axis_input0 width INPUT0_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_input0 signed SIGNED_INPUT0

module my_accelerator #(
    parameter INPUT0_WIDTH = 8,
    parameter SIGNED_INPUT0 = 0
) (
    input [INPUT0_WIDTH-1:0] s_axis_input0_tdata,
    ...
);
```

### Multi-Interface Configuration
```systemverilog
// Different parameters for each interface
// @brainsmith DATATYPE_PARAM s_axis_query width QUERY_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_key width KEY_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_value width VALUE_WIDTH
```

### Format and Bias Mapping (thresholding example)
```systemverilog
// @brainsmith DATATYPE_PARAM in format FPARG
// @brainsmith DATATYPE_PARAM out bias BIAS

module thresholding_axi #(
    bit  FPARG  = 0,    // floating-point inputs
    int  BIAS   = 0,    // output offset
    ...
);
```

## How It Works

### 1. Pragma Processing Flow
```
RTL File → AST Parsing → Comment Extraction → Pragma Validation → Metadata Application
```

### 2. Parameter Linkage
When applied, the pragma:
- Checks if it matches the interface name
- Updates the interface's `datatype_params` dictionary
- Links the property to the specified RTL parameter

### 3. Template Generation Impact
The linked parameters are used in:
- **HWCustomOp**: For datatype constraint validation
- **RTLBackend**: For parameter value assignment in `prepare_codegen_rtl_values()`
- **Wrapper generation**: Proper parameter propagation

## Interaction with Other Systems

### 1. Auto-detection Fallback
If no DATATYPE_PARAM pragma is specified, the system falls back to auto-detection:
```python
# Default naming convention
width_param = f"{interface_name}_WIDTH"   # e.g., s_axis_input0_WIDTH
signed_param = f"{interface_name}_SIGNED" # e.g., s_axis_input0_SIGNED
```

### 2. Priority Hierarchy
```
DATATYPE_PARAM pragma (highest) → Auto-detection (lowest)
```

### 3. Parameter Exposure
Parameters linked via DATATYPE_PARAM are automatically removed from the exposed parameters list, as they're now controlled by the interface configuration.

## Benefits

1. **Flexibility**: Decouple RTL parameter names from interface names
2. **Legacy Support**: Work with existing RTL that doesn't follow naming conventions
3. **Multi-Interface**: Handle modules with many similar interfaces
4. **Clean APIs**: Hide implementation details from users

## Common Patterns

### 1. Legacy RTL Integration
```systemverilog
// Old RTL with non-standard naming
// @brainsmith DATATYPE_PARAM input_stream width DATA_W
// @brainsmith DATATYPE_PARAM input_stream signed IS_SIGNED
parameter DATA_W = 16;
parameter IS_SIGNED = 1;
```

### 2. Shared Parameters
```systemverilog
// Multiple interfaces share width but have separate signed flags
// @brainsmith DATATYPE_PARAM in0 width SHARED_WIDTH
// @brainsmith DATATYPE_PARAM in1 width SHARED_WIDTH
// @brainsmith DATATYPE_PARAM in0 signed IN0_SIGNED
// @brainsmith DATATYPE_PARAM in1 signed IN1_SIGNED
```

### 3. Format Control
```systemverilog
// Control data interpretation
// @brainsmith DATATYPE_PARAM weights format WEIGHT_FORMAT
// @brainsmith DATATYPE_PARAM weights fractional_width WEIGHT_FRAC
```

## Validation & Error Handling

### 1. Property Type Validation
```python
valid_properties = ['width', 'signed', 'format', 'bias', 'fractional_width']
if property_type not in valid_properties:
    raise PragmaError(f"Invalid property_type '{property_type}'")
```

### 2. Interface Matching
- Uses exact string matching for precision
- Logs detailed matching attempts for debugging
- Skips if interface name doesn't match

### 3. Parameter Existence
- Warning generated if linked parameter doesn't exist
- Validation happens during template generation phase

## Best Practices

1. **Be Explicit**: Always specify DATATYPE_PARAM for non-standard naming
2. **Document Intent**: Add comments explaining why specific mappings exist
3. **Consistency**: Use consistent property mappings across similar interfaces
4. **Validation**: Ensure linked parameters actually exist in the module

## Troubleshooting

### Warning: Parameter not found
```
WARNING: DATATYPE_PARAM references parameter 'FOO_WIDTH' which doesn't exist
```
**Solution**: Check parameter name spelling and existence

### No Effect Observed
**Check**:
1. Interface name matches exactly
2. Property type is valid (lowercase)
3. Pragma appears before module declaration
4. No syntax errors in pragma

### Multiple Pragmas Same Property
Last pragma wins - be careful with ordering

## Integration Example

Complete example showing DATATYPE_PARAM in context:

```systemverilog
// @brainsmith DATATYPE in0 FIXED 8 32
// @brainsmith DATATYPE_PARAM in0 width IN_WIDTH
// @brainsmith DATATYPE_PARAM in0 signed IN_SIGNED
// @brainsmith BDIM in0 IN_BLOCK_DIM
// @brainsmith ALIAS IN_WIDTH input_precision

module processor #(
    parameter IN_WIDTH = 16,
    parameter IN_SIGNED = 1,
    parameter IN_BLOCK_DIM = 64
) (
    input [IN_WIDTH-1:0] in0_tdata,
    ...
);
```

This creates:
- Datatype constraints via DATATYPE
- Parameter linkage via DATATYPE_PARAM
- Block dimensions via BDIM
- User-friendly naming via ALIAS