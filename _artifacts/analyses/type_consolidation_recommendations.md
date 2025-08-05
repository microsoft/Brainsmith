# Type Consolidation Recommendations

## Current Dependencies

### kernel_integrator → dataflow
```python
# In kernel_integrator/metadata.py
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup, validate_datatype_against_constraints
from brainsmith.core.dataflow.relationships import DimensionRelationship, RelationType

# In kernel_integrator/data.py
from brainsmith.core.dataflow.constraint_types import (
    DatatypeConstraintGroup,
    validate_datatype_against_constraints
)
```

### No Reverse Dependencies ✅
- dataflow does not import from kernel_integrator
- This is the correct dependency direction

## Specific Type Analysis

### 1. InterfaceType Enum

**Current Location**: `kernel_integrator/data.py`

```python
class InterfaceType(Enum):
    INPUT = "input"      # AXI-Stream input
    OUTPUT = "output"    # AXI-Stream output  
    WEIGHT = "weight"    # AXI-Stream weight
    CONFIG = "config"    # AXI-Lite config
    CONTROL = "control"  # Global control
    UNKNOWN = "unknown"  # Unknown
```

**Recommendation**: Move to `dataflow/types.py`
- This is a fundamental categorization of interfaces
- Used by both kernel modeling and RTL integration
- Has properties like `is_dataflow`, `protocol`, `direction`

### 2. Shape Types

**dataflow Types**:
```python
Shape = Tuple[int, ...]
RaggedShape = Union[Shape, List[Shape]]
```

**kernel_integrator Usage**:
```python
# In InterfaceMetadata
bdim_params: Optional[List[str]] = None
sdim_params: Optional[List[str]] = None
bdim_shape: Optional[List[Union[str, int]]] = None
sdim_shape: Optional[List[Union[str, int]]] = None
```

**Recommendation**: Create unified shape expression types
```python
# In dataflow/types.py
ShapeExpr = Union[int, str]  # Single dimension expression
ShapeSpec = List[ShapeExpr]  # Complete shape specification

# Then in kernel_integrator:
bdim_shape: Optional[ShapeSpec] = None
sdim_shape: Optional[ShapeSpec] = None
```

### 3. Datatype System

**Already Unified**: ✅
- Both use QONNX datatypes
- Both use `DatatypeConstraintGroup`
- Validation functions are shared

**No Action Needed**

### 4. Parameter Types

**dataflow**:
```python
@dataclass
class ParameterBinding:
    parameters: Dict[str, Union[int, float, str]]
    constants: Dict[str, Union[int, float]] = None
```

**kernel_integrator**:
```python
@dataclass
class Parameter:  # SystemVerilog parameter
    name: str
    param_type: Optional[str] = None
    default_value: Optional[str] = None
    description: Optional[str] = None
```

**Recommendation**: Keep separate
- Different purposes (runtime vs RTL)
- No overlap in functionality

### 5. Metadata Types

**kernel_integrator**:
```python
@dataclass
class InterfaceMetadata:
    name: str
    interface_type: InterfaceType
    datatype_constraints: List[DatatypeConstraintGroup]
    # RTL-specific fields...
```

**dataflow**:
```python
@dataclass
class InputDefinition(BaseDefinition):
    name: str
    datatype_constraints: List[DatatypeConstraintGroup]
    block_tiling: Optional[List[Union[int, str]]]
    # Kernel modeling fields...
```

**Recommendation**: Keep separate but share common fields
```python
# In dataflow/interface_base.py
@dataclass
class InterfaceConstraints:
    """Common interface constraints."""
    name: str
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
```

## Implementation Plan

### Phase 1: Move InterfaceType (Low Risk)

1. Move `InterfaceType` from `kernel_integrator/data.py` to `dataflow/types.py`
2. Update imports in kernel_integrator
3. Add to dataflow exports

### Phase 2: Create Shape Expression Types (Medium Risk)

1. Add to `dataflow/types.py`:
```python
# Shape expression types
ShapeExpr = Union[int, str]  # 1, ":", "SIMD", etc.
ShapeSpec = List[ShapeExpr]  # [1, "CH_TILES", ":", ":"]
```

2. Update kernel_integrator to use these types

### Phase 3: Interface Base Class (Medium Risk)

1. Create `dataflow/interface_base.py`:
```python
@dataclass
class InterfaceConstraints:
    """Base constraints for any interface."""
    name: str
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    
    def validates_datatype(self, datatype: BaseDataType) -> bool:
        """Check if datatype satisfies constraints."""
        return validate_datatype_against_constraints(
            datatype, self.datatype_constraints
        )
```

2. Have both systems inherit/use this base

## Benefits

1. **Type Safety**: Proper types instead of raw lists
2. **Consistency**: Single source of truth for shared concepts
3. **Maintainability**: Clear dependency hierarchy
4. **Documentation**: Types serve as documentation

## Risks & Mitigation

### Risk 1: Breaking Changes
- **Mitigation**: Phase implementation, thorough testing

### Risk 2: Circular Dependencies
- **Mitigation**: Only move truly shared types to dataflow

### Risk 3: Over-abstraction
- **Mitigation**: Keep RTL-specific types in kernel_integrator

## Testing Strategy

1. **Unit Tests**: Ensure moved types work correctly
2. **Integration Tests**: Verify kernel_integrator still works
3. **Type Checking**: Run mypy to catch type errors
4. **Example Validation**: Test with existing RTL examples

## Conclusion

The consolidation should focus on:
1. Moving `InterfaceType` enum to dataflow
2. Creating proper shape expression types
3. Keeping RTL-specific types in kernel_integrator

This maintains clean architecture while eliminating redundancy.