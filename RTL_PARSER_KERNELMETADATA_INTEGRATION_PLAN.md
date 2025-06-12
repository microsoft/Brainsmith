# RTL Parser Integration Plan for KernelMetadata

## Overview

This plan outlines the minimal, careful changes needed to make the RTL Parser output `KernelMetadata` instead of `ParsedKernelData`. The RTL Parser is complex and works well, so we'll preserve its core parsing logic while only changing the output format.

## Current State Analysis

### What RTL Parser Currently Does Well
1. **Parses SystemVerilog** using tree-sitter (robust, tested)
2. **Extracts interfaces** with type classification (AXI-Stream, AXI-Lite, etc.)
3. **Parses parameters** with names, types, and default values
4. **Processes pragmas** including BDIM and datatype pragmas
5. **Validates protocols** for AXI interfaces
6. **Generates warnings** for parsing ambiguities

### Current Output: ParsedKernelData
```python
@dataclass
class ParsedKernelData:
    name: str
    source_file: Path
    parameters: List[Parameter]
    interfaces: Dict[str, Interface]  # Key difference: Dict of Interface objects
    pragmas: List[Pragma]
    parsing_warnings: List[str]
```

### Target Output: KernelMetadata
```python
@dataclass
class KernelMetadata:
    name: str
    source_file: Path
    interfaces: List[InterfaceMetadata]  # Key difference: List of InterfaceMetadata
    parameters: List[Parameter]
    pragmas: List[Pragma]
    parsing_warnings: List[str]
```

## Integration Strategy: Minimal Changes

### Key Insight
The RTL Parser already extracts ALL the information needed for `InterfaceMetadata`:
- Interface names and types
- Port information for datatype constraints
- Pragma associations for chunking strategies

We just need to construct `InterfaceMetadata` objects instead of `Interface` objects.

## Proposed Changes

### 1. Add InterfaceMetadata Construction (ONE New Method)

Add a single method to convert parsed interface data to InterfaceMetadata:

```python
# In rtl_parser/parser.py

def _create_interface_metadata(self, interface: Interface, pragmas: List[Pragma]) -> InterfaceMetadata:
    """
    Convert parsed Interface to InterfaceMetadata.
    
    This is the ONLY new method needed - it packages existing data
    into the format needed by AutoHWCustomOp.
    """
    # Extract datatype constraints from interface
    allowed_datatypes = []
    
    # Use existing bit width info from interface
    if hasattr(interface, 'data_width'):
        # Create constraints based on width
        bit_width = interface.data_width
        allowed_datatypes = [
            DataTypeConstraint(finn_type=f"UINT{bit_width}", bit_width=bit_width, signed=False),
            DataTypeConstraint(finn_type=f"INT{bit_width}", bit_width=bit_width, signed=True),
        ]
    else:
        # Default constraints
        allowed_datatypes = [
            DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False),
            DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True),
        ]
    
    # Find relevant pragmas for this interface
    interface_pragmas = [p for p in pragmas if self._pragma_applies_to_interface(p, interface)]
    
    # Determine chunking strategy from pragmas
    chunking_strategy = DefaultChunkingStrategy()  # Default
    for pragma in interface_pragmas:
        if pragma.type == PragmaType.BDIM:
            # Parser already extracts BDIM info - use it
            chunking_strategy = self._pragma_to_chunking_strategy(pragma)
            break
    
    return InterfaceMetadata(
        name=interface.name,
        interface_type=interface.type,  # Already classified!
        allowed_datatypes=allowed_datatypes,
        chunking_strategy=chunking_strategy
    )
```

### 2. Modify parse() Return Type (ONE Line Change)

Change the main parse method to return KernelMetadata:

```python
def parse(self, sv_file: Union[str, Path]) -> KernelMetadata:  # Changed return type
    """Parse SystemVerilog file and extract complete kernel metadata."""
    
    # ... existing parsing logic remains UNCHANGED ...
    
    # At the very end, instead of creating ParsedKernelData:
    
    # Convert interfaces Dict[str, Interface] to List[InterfaceMetadata]
    interface_metadata_list = []
    for interface_name, interface in interfaces.items():
        metadata = self._create_interface_metadata(interface, pragmas)
        interface_metadata_list.append(metadata)
    
    # Return KernelMetadata instead of ParsedKernelData
    return KernelMetadata(
        name=module_name,
        source_file=Path(sv_file),
        interfaces=interface_metadata_list,  # List instead of Dict
        parameters=parameters,
        pragmas=pragmas,
        parsing_warnings=self.warnings
    )
```

### 3. Import Updates (Minimal)

Add imports at the top of parser.py:
```python
from brainsmith.dataflow.core.kernel_metadata import KernelMetadata
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy
```

### 4. Remove ParsedKernelData (Clean Removal)

Since this is a pure refactor with no backwards compatibility:
- Remove `ParsedKernelData` class from `rtl_parser/data.py`
- It's no longer needed anywhere

## What We're NOT Changing

### Preserve ALL Core Parsing Logic
- ✅ Tree-sitter parsing remains untouched
- ✅ Interface extraction logic unchanged
- ✅ Parameter parsing unchanged
- ✅ Pragma parsing unchanged
- ✅ Protocol validation unchanged
- ✅ Warning generation unchanged

### Keep Existing Helper Methods
- ✅ `_extract_axi_interfaces()` - works perfectly
- ✅ `_classify_interface_type()` - already does the classification
- ✅ `_validate_axi_stream_signals()` - important validation
- ✅ `_parse_pragmas()` - sophisticated pragma handling

## Testing Strategy

### 1. Update Existing Tests
```python
# Old test
def test_parse_module():
    result = parser.parse("test.sv")
    assert isinstance(result, ParsedKernelData)
    assert "s_axis_input" in result.interfaces

# New test  
def test_parse_module():
    result = parser.parse("test.sv")
    assert isinstance(result, KernelMetadata)
    assert any(iface.name == "s_axis_input" for iface in result.interfaces)
```

### 2. Verify Information Preservation
Ensure all information is preserved in the new format:
- Interface names, types, and properties
- Parameter values and types
- Pragma associations
- Warning messages

## Benefits of This Approach

1. **Minimal Risk**: Core parsing logic untouched
2. **Clean Architecture**: Direct path from RTL to AutoHWCustomOp metadata
3. **No Information Loss**: Everything parsed is preserved
4. **Simplified Pipeline**: No intermediate transformations needed
5. **Type Safety**: Strong typing with dataclasses throughout

## Implementation Order

1. **Add imports** to parser.py
2. **Add `_create_interface_metadata()` method**
3. **Modify `parse()` return statement** 
4. **Update test files**
5. **Remove `ParsedKernelData` class**
6. **Update any remaining references**

## Potential Gotchas to Check

1. **Pragma Association**: Ensure `_pragma_applies_to_interface()` logic works correctly
2. **Interface Ordering**: List order might matter for some templates
3. **Data Width Extraction**: Verify interface.data_width is available
4. **Type Compatibility**: Ensure Interface.type maps cleanly to InterfaceType enum

## Summary

This plan makes minimal, surgical changes to the RTL Parser:
- **One new method** to create InterfaceMetadata
- **One changed return type** in parse()
- **Remove one obsolete class** (ParsedKernelData)

Everything else remains unchanged, preserving the complex parsing logic that already works well.