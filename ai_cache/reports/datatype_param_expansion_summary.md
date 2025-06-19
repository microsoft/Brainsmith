# DATATYPE_PARAM Pragma Expansion Summary (Simplified)

## Overview

Successfully expanded the DATATYPE_PARAM pragma to support both interface datatype parameters (existing) and internal kernel mechanism datatype parameters (new). This enables explicit parameter mappings for code generation to connect RTL to FINN appropriately.

## Implementation Changes

### 1. Created DatatypeMetadata Class
- **Location**: `brainsmith/dataflow/core/datatype_metadata.py`
- **Purpose**: Explicit binding between RTL parameters and datatype properties
- **Structure**:
  - `name`: Identifier for the datatype
  - `width`: RTL parameter name for bit width (required)
  - `signed`: Optional RTL parameter name for signedness
  - `format`: Optional RTL parameter name for format selection
  - `bias`: Optional RTL parameter name for bias/offset
  - `fractional_width`: Optional RTL parameter name for fractional bits
  - `exponent_width`: Optional RTL parameter name for exponent bits
  - `mantissa_width`: Optional RTL parameter name for mantissa bits

### 2. Updated InterfaceMetadata
- Replaced `datatype_params: Dict[str, str]` with `datatype_metadata: DatatypeMetadata`
- Maintains backward compatibility through accessor methods
- Cleaner object model with dedicated datatype binding

### 3. Extended KernelMetadata
- Added `internal_datatypes: List[DatatypeMetadata]` field
- Stores datatype bindings for internal mechanisms
- Enables future datatype inference and validation

### 4. Enhanced DATATYPE_PARAM Pragma
- Now creates DatatypeMetadata objects
- If target matches interface: updates interface's datatype metadata
- If no interface match: creates standalone internal datatype

### 5. Updated Processing Pipeline
- PragmaHandler collects internal datatype pragmas
- RTL Parser adds them to KernelMetadata
- Templates access via new structure

## Usage Example

```systemverilog
// Interface datatype parameters (existing)
// @brainsmith DATATYPE_PARAM in width in_WIDTH
// @brainsmith DATATYPE_PARAM in format in_FORMAT

// Internal mechanism datatype parameters (new)
// @brainsmith DATATYPE_PARAM threshold width thresh_WIDTH
// @brainsmith DATATYPE_PARAM threshold signed THRESH_SIGNED
// @brainsmith DATATYPE_PARAM accumulator width ACC_WIDTH
// @brainsmith DATATYPE_PARAM accumulator format ACC_FORMAT

module advanced_processor #(
    int unsigned thresh_WIDTH = 16,
    bit THRESH_SIGNED = 1,
    int unsigned ACC_WIDTH = 32,
    int ACC_FORMAT = 0  // 0=INT, 1=UINT, 2=FIXED
);
```

## Benefits

1. **Explicit Mappings**: Clear, structured parameter bindings
2. **Unified Model**: Single mechanism for all datatype parameters
3. **Documentation**: Clear specification of internal precision requirements
4. **Code Generation**: Direct mapping for RTL-to-FINN connection
5. **Clean Architecture**: Separation of concerns with dedicated metadata

## Demonstration Files

- `demo_internal_datatype_pragmas.sv`: Example SystemVerilog with pragmas
- `demo_internal_datatype_pragmas.py`: Python script showing parsing results

## Key Design Decisions

1. **Explicit Attributes**: Each datatype property is an explicit optional field rather than a generic dictionary
2. **Width Required**: Only `width` is mandatory since all datatypes need bit width
3. **No Construction Logic**: DatatypeMetadata is purely for parameter mapping, not runtime datatype construction
4. **Direct Access**: Code generation can directly access needed parameters without dictionary lookups