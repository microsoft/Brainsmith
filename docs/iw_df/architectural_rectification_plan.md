# AutoHWCustomOp Architectural Rectification Plan

## Overview
This plan addresses the fundamental architectural violation where pragma concepts polluted the pure Dataflow Modeling layer. We will clean the separation of concerns and implement a proper layered architecture.

## Current Problems

### 1. **Pragma Pollution in Dataflow Layer**
```python
# ❌ WRONG: Pragmas in dataflow layer
InterfaceMetadata(
    name="in0_V_data_V",
    pragma_metadata={"enhanced_tdim": {"chunk_index": -1}}  # This shouldn't exist
)
```

### 2. **Complex Pragma Logic in Core**
- `tensor_chunking.py` parses pragma strings 
- Dataflow layer interprets HWKG-specific logic
- AutoHWCustomOp handles pragma processing

### 3. **Confused Layer Responsibilities**
```
Current (Wrong):
RTL → HWKG → Dataflow Model (pragma-polluted) → Complex pragma-aware chunking

Should Be:
RTL → HWKG (pragma parser) → Simple chunking override → Clean Dataflow Model
```

## Rectification Strategy

### Phase 1: Clean the Dataflow Layer

#### 1.1 **Purify InterfaceMetadata**
```python
# ✅ CORRECT: Pure computational properties only
@dataclass
class InterfaceMetadata:
    name: str
    interface_type: DataflowInterfaceType
    allowed_datatypes: List[DataTypeConstraint]
    # Remove: pragma_metadata completely
```

#### 1.2 **Simplify Tensor Chunking System**
Replace complex pragma-aware chunking with simple override pattern:

```python
@dataclass
class ChunkingOverride:
    """Simple chunking override - no pragma knowledge."""
    start_index: int           # Starting dimension index
    shape: List[Union[str, int]]  # [tdim1, tdim2] or [:] 
    
class TensorChunker:
    def __init__(self):
        self._override: Optional[ChunkingOverride] = None
        
    def set_chunking_override(self, override: ChunkingOverride):
        """Set simple override - called by HWKG layer."""
        self._override = override
        
    def compute_chunking(self, tensor_shape: List[int]) -> Tuple[List[int], List[int]]:
        """Compute qDim, tDim using override or default complex logic."""
        if self._override:
            return self._apply_simple_override(tensor_shape)
        else:
            return self._apply_complex_default_chunking(tensor_shape)
```

#### 1.3 **Broadcasting Rules**
```python
def _apply_simple_override(self, tensor_shape: List[int]) -> Tuple[List[int], List[int]]:
    """Apply simple override with broadcasting."""
    start_idx = self._override.start_index
    shape = self._override.shape
    
    # Handle shape formats
    if shape == [":"] or shape == ":":
        # Full input shape
        actual_shape = tensor_shape
    else:
        # Module parameters: [tdim1, tdim2, ...]
        actual_shape = [self._resolve_param(s) for s in shape]
    
    # Broadcasting rules
    # 2D shape → broadcasts against start_idx and start_idx+1
    # 3D shape → broadcasts against start_idx, start_idx+1, start_idx+2
    qDim, tDim = self._broadcast_shape(tensor_shape, actual_shape, start_idx)
    return qDim, tDim
```

### Phase 2: Move Pragma Logic to HWKG

#### 2.1 **RTL Pragma Parser in HWKG**
```python
# In HWKG template generation
class RTLPragmaParser:
    def parse_tdim_pragmas(self, rtl_content: str) -> Dict[str, ChunkingOverride]:
        """Parse @brainsmith TDIM pragmas from RTL."""
        pragmas = {}
        for line in rtl_content.split('\n'):
            if '@brainsmith TDIM' in line:
                # Parse: @brainsmith TDIM intf_name start_idx [shape]
                interface_name, override = self._parse_tdim_pragma(line)
                pragmas[interface_name] = override
        return pragmas
```

#### 2.2 **HWKG Template Integration**
```jinja2
{# In rtl_backend.py.j2 template #}
{% if pragma_overrides %}
# Apply chunking overrides from RTL pragmas
{% for intf_name, override in pragma_overrides.items() %}
self.get_chunker_for_interface("{{ intf_name }}").set_chunking_override(
    ChunkingOverride(
        start_index={{ override.start_index }},
        shape={{ override.shape }}
    )
)
{% endfor %}
{% endif %}
```

#### 2.3 **Clean Workflow**
```
1. HWKG reads RTL file
2. HWKG parses @brainsmith TDIM pragmas  
3. HWKG creates ChunkingOverride objects
4. HWKG generates Python class with override configuration
5. Dataflow layer applies simple override (no pragma knowledge)
```

### Phase 3: Architecture Validation

#### 3.1 **Layer Purity Checks**
- [ ] Dataflow layer has zero pragma imports
- [ ] No pragma parsing in `tensor_chunking.py`
- [ ] `InterfaceMetadata` contains only computational properties
- [ ] HWKG layer handles all pragma interpretation

#### 3.2 **Interface Contracts**
```python
# Dataflow Layer Interface (pragma-free)
class AutoHWCustomOp:
    def set_chunking_override(self, interface_name: str, override: ChunkingOverride):
        """Set simple override - no pragma knowledge required."""
        
# HWKG Layer Interface (pragma-aware)  
class HWKernelGenerator:
    def parse_rtl_pragmas(self, rtl_path: str) -> Dict[str, ChunkingOverride]:
        """Parse pragmas and return simple overrides."""
```

## Implementation Steps

### Step 1: Clean InterfaceMetadata (High Priority)
- Remove `pragma_metadata` field completely
- Update all tests to remove pragma references
- Ensure only computational properties remain

### Step 2: Simplify TensorChunker (High Priority)
- Replace complex pragma parsing with simple override pattern
- Implement `ChunkingOverride` dataclass
- Add broadcasting logic for shape expansion

### Step 3: Create RTL Pragma Parser (Medium Priority)
- Build pragma parser in HWKG layer
- Parse `@brainsmith TDIM` statements
- Generate `ChunkingOverride` objects

### Step 4: Update HWKG Templates (Medium Priority)
- Modify `rtl_backend.py.j2` to use pragma parser
- Generate code that sets chunking overrides
- Remove pragma handling from generated dataflow code

### Step 5: Integration Testing (Low Priority)
- Test end-to-end pragma → override → chunking flow
- Validate layer separation is maintained
- Ensure performance is maintained or improved

## Benefits of Clean Architecture

1. **Pure Dataflow Layer**: Computational model with no pragma pollution
2. **Simple Override Pattern**: Easy to understand and test `(start_index, shape)` overrides
3. **Clear Responsibilities**: HWKG handles pragmas, Dataflow handles computation
4. **Better Testability**: Each layer can be tested independently
5. **Future Extensibility**: New pragma types won't pollute dataflow layer

## Migration Strategy

### Backward Compatibility
- Maintain public interfaces during transition
- Deprecate pragma-related APIs with clear migration path
- Provide compatibility shims where necessary

### Validation Points
- All existing tests must pass after refactoring
- New tests validate layer separation
- Performance benchmarks confirm no regression

This rectification will establish a clean, maintainable architecture that properly separates pragma interpretation (HWKG) from computational modeling (Dataflow).