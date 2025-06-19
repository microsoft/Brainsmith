# RTL Parser Pragma State Report

## Executive Summary

The RTL parser pragma system is fully implemented with support for 8 pragma types that enable fine-grained control over hardware kernel generation. The system uses a clean architecture with pragma handlers, interface metadata integration, and template context generation.

## Current Pragma Types

### Interface Pragmas
These pragmas modify interface metadata and properties:

1. **DATATYPE** - Constrain datatype for interfaces
   - Format: `@brainsmith DATATYPE <interface_name> <base_type> <min_bits> <max_bits>`
   - Example: `@brainsmith DATATYPE in FIXED 1 32`

2. **DATATYPE_PARAM** - Map interface datatype properties to RTL parameters
   - Format: `@brainsmith DATATYPE_PARAM <interface_name> <property> <parameter_name>`
   - Properties: width, signed, format, bias, fractional_width
   - Example: `@brainsmith DATATYPE_PARAM in format FPARG`

3. **WEIGHT** - Mark input interfaces as weight interfaces
   - Format: `@brainsmith WEIGHT <interface_name>`
   - Changes interface type from INPUT to WEIGHT

4. **BDIM** - Block dimension configuration with optional shape
   - Format: `@brainsmith BDIM <interface_name> <param_name> [SHAPE=<shape>] [RINDEX=<n>]`
   - Example: `@brainsmith BDIM weights_V WEIGHTS_BLOCK_SIZE SHAPE=[C,PE] RINDEX=0`

5. **SDIM** - Stream dimension configuration
   - Format: `@brainsmith SDIM <interface_name> <param_name>`
   - Stream shape inferred from BDIM configuration

### Parameter Pragmas
These pragmas control parameter exposure and naming:

6. **ALIAS** - Expose RTL parameters with user-friendly names
   - Format: `@brainsmith ALIAS <rtl_param> <nodeattr_name>`
   - Example: `@brainsmith ALIAS PE parallelism_factor`

7. **DERIVED_PARAMETER** - Compute parameters from Python expressions
   - Format: `@brainsmith DERIVED_PARAMETER <param_name> <python_expression>`
   - Example: `@brainsmith DERIVED_PARAMETER SIMD self.get_input_datatype().bitwidth()`

### Module Pragmas

8. **TOP_MODULE** - Specify top module when multiple exist
   - Format: `@brainsmith TOP_MODULE <module_name>`

## Architecture

### Core Classes

1. **PragmaHandler** (`pragma.py`)
   - Extracts pragmas from AST comments
   - Validates pragma syntax
   - Applies pragmas to interface metadata and parameters

2. **Pragma Base Class** (`data.py`)
   - Abstract base with parsed_data dictionary
   - Methods: `_parse_inputs()`, `apply()`, `applies_to_interface_metadata()`, `apply_to_metadata()`

3. **InterfacePragma** (`data.py`)
   - Base class for interface-targeting pragmas
   - Implements name matching logic
   - Subclassed by DATATYPE, WEIGHT, BDIM, SDIM, DATATYPE_PARAM

4. **ParameterPragma** (`data.py`)
   - Base class for parameter pragmas
   - Subclassed by ALIAS, DERIVED_PARAMETER

### Processing Flow

1. **Extraction**: PragmaHandler walks AST finding `// @brainsmith` comments
2. **Parsing**: Each pragma type has custom `_parse_inputs()` implementation
3. **Validation**: Pragmas validate against module parameters/interfaces
4. **Application**: 
   - Interface pragmas modify InterfaceMetadata via `apply_to_metadata()`
   - Parameter pragmas update exposed parameters list

## Key Features

### Parameter Naming Conventions
- Auto-detection pattern: `{interface}_{property}` (e.g., `s_axis_input0_WIDTH`)
- Fallback to defaults if not found
- Warning generation for missing expected parameters

### Interface Name Matching
- Exact match: `input0` matches `input0`
- Supports full AXI names: `s_axis_input0` 

### Processing Hierarchy
1. Parameter pragmas (ALIAS, DERIVED_PARAMETER) - highest priority
2. Interface pragmas (DATATYPE_PARAM, BDIM, SDIM) - medium priority  
3. Auto-detection from naming conventions - lowest priority

## Example Usage (thresholding_axi_bw.sv)

```systemverilog
// @brainsmith DATATYPE in FIXED 1 32
// @brainsmith DATATYPE out FIXED 1 32
// @brainsmith DATATYPE_PARAM in format FPARG
// @brainsmith DATATYPE_PARAM out bias BIAS

module thresholding_axi #(
    int unsigned  in_WIDTH,    
    int unsigned  out_WIDTH,   
    bit  FPARG  = 0,           // linked to 'in' interface
    int  BIAS   = 0,           // linked to 'out' interface
    ...
```

This demonstrates:
- DATATYPE pragmas constraining input/output to FIXED types
- DATATYPE_PARAM linking format property of 'in' to FPARG parameter
- DATATYPE_PARAM linking bias property of 'out' to BIAS parameter

## Template Integration

The pragma data flows to template generation:
- Interface pragmas affect datatype constraints and chunking strategies
- ALIAS pragmas create user-friendly nodeattr names in HWCustomOp
- DERIVED_PARAMETER pragmas generate computed values in RTLBackend
- DATATYPE_PARAM links are used for parameter value assignment

## Current State Assessment

**Strengths:**
- Clean, extensible architecture
- Comprehensive pragma type coverage
- Good error handling and validation
- Well-integrated with dataflow concepts

**Areas Working Well:**
- Pragma extraction from SystemVerilog comments
- Interface metadata modification
- Parameter aliasing and derivation
- Template context generation

**Recent Updates:**
- Unified interface type system (removed dual RTL/Dataflow types)
- Enhanced BDIM/SDIM pragma support with shapes
- Improved parameter validation using actual module parameters
- Better warning messages for missing parameters