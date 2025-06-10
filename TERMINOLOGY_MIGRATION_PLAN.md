# Brainsmith Terminology Migration Plan

## Overview

This document outlines the plan to migrate from the old terminology to the new Interface-Wise Dataflow Modeling terminology defined in `INTERFACE_WISE_DATAFLOW_AXIOMS.md`.

## Terminology Changes

### Core Mapping
| Old Term | New Term | Description |
|----------|----------|-------------|
| `query` | `tensor` | Complete data (entire hidden state/weight) - DEPRECATED TERM |
| `tensor` | `block` | Minimum data for one calculation |
| `qDim` | `tDim` | Full tensor shape (no batch dimension) |
| `tDim` | `bDim` | Shape of each block |
| `num_tensors` | `num_blocks` | Number of blocks available |
| `stream_dims` | `stream_dims` | Data streamed per clock cycle (unchanged) |

## Critical Gotchas Identified

### 1. **"tensor" Overloading Risk - HIGH PRIORITY**
```python
# CONFLICT: PyTorch/ONNX tensors vs dataflow tensors
import torch
model_tensor = torch.tensor([1, 2, 3])  # Keep unchanged
onnx_tensor_shape = model.get_tensor_shape("input")  # Keep unchanged

# OLD dataflow usage (to be changed):
tensor_dims = interface.tDim  # This becomes block_dims
tensor_count = interface.num_tensors  # This becomes num_blocks
```

**Solution**: Use context-specific naming:
- Keep `tensor` for PyTorch/ONNX contexts
- Use `block` for dataflow calculation units
- Use full names like `tensor_dims`/`block_dims` for clarity

### 2. **Mathematical Formula Consistency**
```python
# OLD formulas (throughout codebase):
num_tensors[i] = qDim[i] // tDim[i]
cII = ∏(tDim_i / stream_dims_i)

# NEW formulas:
num_blocks[i] = tDim[i] // bDim[i]  
cII = ∏(bDim_i / stream_dims_i)
```

### 3. **Template Generation Impact**
- 15+ generated files use current terminology
- Jinja2 templates need coordinated updates
- Generated code examples in tests

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
**Goal**: Update core dataflow classes with backward compatibility

**Files to Update**:
1. `brainsmith/dataflow/core/dataflow_interface.py` (PRIMARY)
2. `brainsmith/dataflow/core/tensor_chunking.py` → `block_chunking.py`
3. `brainsmith/dataflow/core/dataflow_model.py`
4. `brainsmith/dataflow/core/auto_hw_custom_op.py`

**Strategy**: Direct replacement with no backward compatibility aliases
```python
class DataflowInterface:
    # New properties (direct replacement)
    @property
    def tensor_dims(self) -> List[int]:
        """Full tensor shape (no batch dimension) - replaces qDim"""
        return self._tensor_dims
    
    @property  
    def block_dims(self) -> List[int]:
        """Shape of each block - replaces tDim"""
        return self._block_dims
    
    # NO deprecated aliases - clean break for clarity
    # All references must be updated simultaneously
```

### Phase 2: Template System (Week 2)
**Goal**: Update template generation and examples

**Files to Update**:
1. `brainsmith/tools/hw_kernel_gen/templates/*.j2` (ALL)
2. `brainsmith/tools/hw_kernel_gen/enhanced_template_context.py`
3. `tests/tools/hw_kernel_gen/generated/*` (regenerate all)

**Strategy**: Update templates to use new terminology
```jinja2
{# OLD template variables #}
qDim={{qDim}}, tDim={{tDim}}, num_tensors={{num_tensors}}

{# NEW template variables #}
tDim={{tDim}}, bDim={{bDim}}, num_blocks={{num_blocks}}
```

### Phase 3: Documentation (Week 3) 
**Goal**: Update all documentation and comments

**Files to Update**:
1. `INTERFACE_WISE_DATAFLOW_AXIOMS.md` (already updated)
2. `docs/iw_df/*.md` (28 files)
3. All docstrings in core classes
4. README files throughout codebase

### Phase 4: Tests and Examples (Week 4)
**Goal**: Update test suite and example code

**Files to Update**:
1. `tests/dataflow/core/*` (15+ files)
2. `tests/tools/hw_kernel_gen/*` (20+ files)
3. `examples/` directory
4. Demo files in `test_builds/`

### Phase 5: Final Validation (Week 5)
**Goal**: Comprehensive testing and validation

**Tasks**:
1. Full integration testing
2. Performance validation  
3. Documentation review
4. End-to-end demo validation

## File-by-File Migration Plan

### Core Files (Phase 1)

#### `brainsmith/dataflow/core/dataflow_interface.py`
**Lines to Update**: 159, 194-197, 221-247
**Changes**:
- `qDim` → `tensor_dims` (property)
- `tDim` → `block_dims` (property) 
- `num_tensors` → `num_blocks` (method)
- Update docstrings and comments

#### `brainsmith/dataflow/core/tensor_chunking.py`
**Rename to**: `brainsmith/dataflow/core/block_chunking.py`
**Changes**:
- All function names: `tensor_*` → `block_*`
- Internal variables and comments
- Mathematical formulas in comments

#### `brainsmith/dataflow/core/auto_hw_custom_op.py`
**Lines to Update**: 8 occurrences of `num_tensors`
**Changes**:
- Method calls: `get_num_tensors()` → `get_num_blocks()`
- Variable names throughout

### Template Files (Phase 2)

#### `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_slim.py.j2`
**Changes**:
- Template variables: `{{qDim}}` → `{{tDim}}`
- Template variables: `{{tDim}}` → `{{bDim}}`
- Template variables: `{{num_tensors}}` → `{{num_blocks}}`
- Method names in generated code

#### Generated Test Files
**Location**: `tests/tools/hw_kernel_gen/generated/*`
**Strategy**: Regenerate all files with new templates

### Documentation Files (Phase 3)

#### `docs/iw_df/dataflow_modeling.md`
**Already Updated**: Core document updated in previous conversation

#### Other Documentation
**Files**: All README.md files, API documentation
**Changes**: Systematic find/replace with context validation

## Validation Strategy

### 1. Automated Testing
```bash
# Phase 1 validation
python -m pytest tests/dataflow/core/ -v

# Phase 2 validation  
python -m pytest tests/tools/hw_kernel_gen/ -v

# Full integration testing
python -m pytest tests/ -v
```

### 2. Generated Code Validation
```bash
# Test code generation
python -m brainsmith.tools.hw_kernel_gen.hkg examples/thresholding/thresholding_axi.sv

# Validate generated content
python test_builds/test_data_layout_chunking_validation.py
```

### 3. Documentation Validation
- Spell check all documentation
- Validate mathematical formulas
- Test code examples in documentation

## Risk Mitigation

### Clean Transition Strategy
- Atomic updates across all related files in each phase
- No deprecated aliases to avoid confusion
- Comprehensive test coverage during transition
- All changes must be completed together per phase

### Rollback Plan
- Git branches for each phase
- Ability to revert individual phases
- Automated testing at each step

### Communication Plan
- Update CHANGELOG.md with breaking changes
- Migration guide for users
- Clear timeline and migration path

## Success Criteria

1. **All tests pass** with new terminology
2. **Generated code functions** identically to old code
3. **Documentation consistency** across all files
4. **No performance regression** in core algorithms
5. **Clean terminology** - no old terminology anywhere in codebase

## Timeline

- **Week 1**: Phase 1 (Core Infrastructure)
- **Week 2**: Phase 2 (Templates)  
- **Week 3**: Phase 3 (Documentation)
- **Week 4**: Phase 4 (Tests and Examples)
- **Week 5**: Phase 5 (Cleanup and Validation)

**Total Duration**: 5 weeks with daily validation and testing.

## Post-Migration

### Maintenance
- Monitor for any missed terminology
- Update any new contributions to use new terminology
- Consider tooling to prevent old terminology introduction

### Future Considerations
- Establish terminology guidelines for new features
- Regular terminology audits
- Documentation style guide updates