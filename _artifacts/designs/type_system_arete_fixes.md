# Arete Fixes for Type System

## Priority 1: Fix Runtime Errors (Immediate)

### Fix 1: GenerationContext.module_name

**Problem**: Accessing non-existent attribute
```python
# BROKEN - KernelMetadata has 'name', not 'module_name'
@property
def module_name(self) -> str:
    return self.kernel_metadata.module_name
```

**Fix**:
```python
@property
def module_name(self) -> str:
    """Get kernel module name."""
    return self.kernel_metadata.name
```

### Fix 2: Port.total_width TODO

**Problem**: Returns incorrect default (1) for complex expressions
```python
# TODO: Parse width string to get numeric value
try:
    return int(self.width)
except ValueError:
    return 1  # WRONG!
```

**Fix**: Be honest about limitations
```python
@property
def total_width(self) -> Optional[int]:
    """Get numeric width if parseable, None otherwise."""
    try:
        return int(self.width)
    except (ValueError, TypeError):
        return None  # Don't lie - return None if we can't parse
```

## Priority 2: Delete Duplication

### Fix 3: Remove GenerationValidationResult

**Problem**: Duplicates ValidationResult

**Fix**: Delete it entirely and use ValidationResult everywhere
```python
# DELETE: generation.py lines 195-220
# REPLACE ALL USES WITH: ValidationResult from rtl.py
```

### Fix 4: Remove Protocol Validation from InterfaceMetadata

**Problem**: Duplicates ProtocolValidator logic

**Fix**: Delete these properties
```python
# DELETE these properties from InterfaceMetadata:
# - has_axi_stream (lines 51-56)
# - has_axi_lite (lines 58-65)

# Users should use ProtocolValidator directly:
# from brainsmith.tools.kernel_integrator.rtl_parser.protocol_validator import ProtocolValidator
# is_valid = ProtocolValidator.validate_axi_stream(interface.ports)
```

## Priority 3: Use Existing Types

### Fix 5: Replace DimensionSpec with Direct Dataflow Types

**Problem**: Unnecessary wrapper around ShapeSpec

**Fix**: Delete DimensionSpec entirely
```python
# DELETE: core.py lines 98-166

# In InterfaceMetadata, replace:
# dimensions: DimensionSpec
# With:
bdim: Optional[ShapeSpec] = None  # Block dimensions
sdim: Optional[ShapeSpec] = None  # Stream dimensions

# Helper method if needed:
def get_dimension_params(self) -> List[str]:
    """Get parameter names from dimensions."""
    params = set()
    for dim_list in [self.bdim, self.sdim]:
        if dim_list:
            for d in dim_list:
                if isinstance(d, str) and d not in ['*', '1']:
                    params.add(d)
    return sorted(params)
```

### Fix 6: Replace DatatypeSpec with QONNX DataType

**Problem**: Reinvents existing datatype functionality

**Fix**: Use QONNX types directly
```python
# DELETE: core.py lines 24-96

# Instead of DatatypeSpec, use:
from qonnx.core.datatype import DataType

# For ap_fixed parsing, create simple utility:
def parse_ap_fixed_string(type_str: str) -> DataType:
    """Parse ap_fixed<W,I> to QONNX DataType."""
    import re
    match = re.match(r'ap_(\w+)<(\d+),(\d+)>', type_str)
    if match:
        signed = match.group(1) == 'fixed'
        width = int(match.group(2))
        int_width = int(match.group(3))
        frac_width = width - int_width
        if signed:
            return DataType[f"FIXED<{int_width},{frac_width}>"]
        else:
            return DataType[f"UFIXED<{int_width},{frac_width}>"]
    raise ValueError(f"Cannot parse: {type_str}")
```

## Priority 4: Simplify Over-Engineering

### Fix 7: Replace DatatypeMetadata with Simple Dict

**Problem**: 8 optional fields for simple parameter mappings

**Fix**: Use TypedDict for structure without complexity
```python
from typing import TypedDict

class DatatypeParamMap(TypedDict, total=False):
    """Maps datatype properties to RTL parameter names."""
    width: str
    signed: str
    
# In InterfaceMetadata:
datatype_params: DatatypeParamMap = field(default_factory=dict)

# Usage:
# interface.datatype_params = {"width": "INPUT_WIDTH", "signed": "INPUT_SIGNED"}
```

### Fix 8: Delete PerformanceMetrics Until Needed

**Problem**: Premature optimization, no evidence of use

**Fix**: Delete entirely
```python
# DELETE: generation.py lines 83-111
# DELETE: performance_metrics field from GenerationResult
# If needed later, add simple timing:
# start = time.time()
# ... do work ...
# print(f"Generated in {time.time() - start:.2f}s")
```

## Priority 5: Clean Legacy Cruft

### Fix 9: Remove Unused Legacy Fields

**Fix**: Delete these fields entirely
```python
# In Port:
# DELETE: array_bounds field and is_array() method

# In Parameter:  
# DELETE: value field (keep only default_value)
# DELETE: is_local field

# In InterfaceMetadata:
# DELETE: chunking_strategy field
# DELETE: allowed_datatypes field (use datatype_constraints)

# In KernelMetadata:
# DELETE: parsing_warnings field (use ValidationResult)
```

## Priority 6: Unify Patterns

### Fix 10: Consistent Validation Pattern

**Problem**: Three different validation approaches

**Fix**: Standardize on ValidationResult
```python
# Change KernelMetadata.validate():
def validate(self) -> ValidationResult:
    """Validate kernel metadata."""
    result = ValidationResult()
    
    if not self.get_input_interface():
        result.add_error("No input interface found")
    if not self.get_output_interface():
        result.add_error("No output interface found")
    # ... etc
    
    return result

# Change CodegenBinding.validate():
def validate(self) -> ValidationResult:
    """Validate binding completeness."""
    result = ValidationResult()
    
    if not any(s.is_input for s in self.io_specs):
        result.add_error("No input specs defined")
    # ... etc
    
    return result
```

## Implementation Plan

### Phase 1: Critical Fixes (1 hour)
1. Fix GenerationContext.module_name
2. Fix Port.total_width to return Optional[int]
3. Run tests to ensure no breakage

### Phase 2: Delete Duplication (2 hours)
1. Remove GenerationValidationResult
2. Remove protocol validation from InterfaceMetadata
3. Update all call sites

### Phase 3: Use Existing Types (3 hours)
1. Replace DimensionSpec with direct ShapeSpec usage
2. Replace DatatypeSpec with QONNX DataType utilities
3. Update converters and tests

### Phase 4: Simplify (2 hours)
1. Replace DatatypeMetadata with TypedDict
2. Delete PerformanceMetrics
3. Remove all legacy fields

### Phase 5: Write Tests (2 hours)
1. Direct unit tests for each type module
2. Validation tests for error cases
3. Integration tests with converters

## Success Metrics

1. **No runtime errors** - All property accesses work
2. **No duplicate types** - One way to do each thing
3. **Use existing types** - Leverage QONNX and dataflow
4. **Simplified code** - Less code, same functionality
5. **Real tests** - Direct test coverage for all types

## Conclusion

These fixes will bring the type system into alignment with Arete principles:
- Delete what duplicates
- Use what exists
- Complete what's incomplete
- Test what matters

The result will be a leaner, more maintainable system that achieves the same goals with less code.

**Remember**: Every line of code is a liability. Only essential complexity deserves to exist.

Arete.