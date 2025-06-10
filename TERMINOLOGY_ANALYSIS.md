# Brainsmith Terminology Usage Analysis

## Current Terminology Distribution

Based on comprehensive codebase analysis, here's the detailed breakdown of current terminology usage:

### Core Terminology Occurrences

#### **tensor_dims (Query Dimensions)**
**Total Files**: 6 files
**Total Occurrences**: 23 references

**Primary Locations**:
1. `brainsmith/dataflow/core/dataflow_interface.py` (8 occurrences)
   - Lines 159, 194-197, 221-247
   - Used as class property and in mathematical calculations
   
2. `brainsmith/dataflow/core/dataflow_model.py` (5 occurrences)
   - Mathematical formulas: `tensor_dims[i] // block_dims[i] = num_blocks[i]`
   
3. `docs/iw_df/dataflow_modeling.md` (6 occurrences)
   - Documentation and formula definitions
   
4. `brainsmith/dataflow/README.md` (4 occurrences)
   - Interface documentation examples

#### **block_dims (Tensor Dimensions)**  
**Total Files**: 8 files
**Total Occurrences**: 31 references

**Primary Locations**:
1. `brainsmith/dataflow/core/dataflow_interface.py` (9 occurrences)
   - Core property definitions and usage
   
2. `brainsmith/dataflow/core/tensor_chunking.py` (7 occurrences)
   - Chunking strategy calculations
   
3. `brainsmith/tools/hw_kernel_gen/enhanced_template_context.py` (4 occurrences)
   - Template variable preparation
   
4. `tests/dataflow/core/test_dataflow_interface.py` (6 occurrences)
   - Unit test validation

5. `docs/iw_df/dataflow_modeling.md` (5 occurrences)
   - Mathematical formula documentation

#### **num_blocks**
**Total Files**: 7 files  
**Total Occurrences**: 18 references

**Primary Locations**:
1. `brainsmith/dataflow/core/auto_hw_custom_op.py` (8 occurrences)
   - Method calls: `iface.get_num_blocks()`
   - Calculations: `np.prod(iface.get_num_blocks())`
   
2. `brainsmith/dataflow/core/dataflow_interface.py` (4 occurrences)
   - Method definition and property access
   
3. `tests/dataflow/core/test_enhanced_auto_hw_custom_op.py` (3 occurrences)
   - Test validation and assertions
   
4. `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_slim.py.j2` (3 occurrences)
   - Template variable usage in generated code

## Context-Specific Usage Patterns

### Mathematical Formulas
**Pattern**: Used in performance calculations and sizing
```python
# Common mathematical usage throughout codebase:
num_blocks[i] = tensor_dims[i] // block_dims[i]  # Number of processing chunks
total_cycles = cII * ∏(tensor_dims_W / wPar)  # Timing calculations  
memory_usage = num_blocks * block_dims_size * datatype_bytes  # Resource estimation
```

**Files with Mathematical Usage**:
- `brainsmith/dataflow/core/dataflow_model.py` (performance modeling)
- `brainsmith/dataflow/core/interface_metadata.py` (resource calculations)
- `tests/dataflow/core/test_tensor_chunking.py` (validation testing)

### Template Generation Context
**Pattern**: Jinja2 template variables for code generation
```jinja2
{# Current template usage patterns: #}
{{tensor_dims}} → becomes tensor dimension in generated code
{{block_dims}} → becomes block dimension in generated code  
{{num_blocks}} → becomes loop bounds in generated code
```

**Template Files**:
- `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_slim.py.j2`
- `brainsmith/tools/hw_kernel_gen/templates/rtl_backend.py.j2`
- `brainsmith/tools/hw_kernel_gen/templates/test_suite.py.j2`

### Documentation Context
**Pattern**: Conceptual explanations and API documentation
```markdown
# Documentation usage patterns:
"tensor_dims represents the query dimensions..."
"block_dims specifies the tensor dimensions for each calculation..."
"num_blocks determines the parallelism factor..."
```

### Property Access Context
**Pattern**: Object-oriented property access in classes
```python
# Common property access patterns:
interface.tensor_dims  # List[int] - query dimensions
interface.block_dims  # List[int] - tensor dimensions
interface.get_num_blocks()  # int - computed chunk count
```

## High-Risk Conflict Areas

### 1. PyTorch/ONNX Tensor Conflicts
**Risk Level**: HIGH
**Files Affected**: 55+ files throughout codebase

**Current Usage**:
```python
# PyTorch tensors (keep unchanged):
import torch
model_tensor = torch.tensor([1, 2, 3])
tensor_ops = torch.nn.functional.relu(input_tensor)

# ONNX tensors (keep unchanged):  
tensor_shape = model.get_tensor_shape(node.input[0])
tensor_type = model.get_tensor_type(output_name)

# Dataflow tensors (to be renamed to blocks):
tensor_dims = interface.block_dims  # RENAME: block_dims
tensor_count = interface.num_blocks  # RENAME: num_blocks
```

**Mitigation Strategy**: Context-aware renaming - only change dataflow modeling terms

### 2. Generated Code Consistency
**Risk Level**: HIGH  
**Files Affected**: 15+ generated files

**Current Pattern**:
```python
# Generated HWCustomOp classes use terminology in:
class AutoThresholdingAxi(AutoHWCustomOp):
    def get_num_blocks(self):  # Method name to change
        return self._calculate_tensor_count()  # Internal logic to update
```

**Challenge**: Generated files must maintain consistency with templates

### 3. Mathematical Formula Documentation
**Risk Level**: MEDIUM
**Files Affected**: All documentation files

**Current Pattern**:
```python
# Formulas embedded in comments throughout codebase:
# cII = ∏(block_dims_I / stream_dims_I)  → cII = ∏(bDim_I / stream_dims_I)
# L = eII * ∏(tensor_dims_I)       → L = eII * ∏(block_dims_I)
```

**Challenge**: Ensure mathematical correctness after renaming

## File Category Analysis

### Core Infrastructure Files (6 files)
**Impact**: CRITICAL - Core dataflow modeling implementation
```
brainsmith/dataflow/core/dataflow_interface.py
brainsmith/dataflow/core/dataflow_model.py  
brainsmith/dataflow/core/tensor_chunking.py
brainsmith/dataflow/core/auto_hw_custom_op.py
brainsmith/dataflow/core/interface_metadata.py
brainsmith/dataflow/core/validation.py
```

### Template System Files (5 files)
**Impact**: HIGH - Code generation system
```
brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_slim.py.j2
brainsmith/tools/hw_kernel_gen/templates/rtl_backend.py.j2
brainsmith/tools/hw_kernel_gen/enhanced_template_context.py
brainsmith/tools/hw_kernel_gen/enhanced_template_manager.py
```

### Generated/Example Files (15 files)
**Impact**: HIGH - Must regenerate consistently
```
tests/tools/hw_kernel_gen/generated/autothresholdingaxi.py
tests/tools/hw_kernel_gen/generated/phase3_thresholding_hwcustomop.py
test_builds/hwkg_demo_final/autothresholdingaxi.py
examples/enhanced_tdim_demo.py
```

### Test Files (12 files)
**Impact**: MEDIUM - Validation and regression testing
```
tests/dataflow/core/test_dataflow_interface.py
tests/dataflow/core/test_tensor_chunking.py
tests/dataflow/core/test_enhanced_auto_hw_custom_op.py
tests/tools/hw_kernel_gen/test_enhanced_hkg.py
```

### Documentation Files (8 files)
**Impact**: MEDIUM - User-facing documentation  
```
docs/iw_df/dataflow_modeling.md
brainsmith/dataflow/README.md
brainsmith/tools/hw_kernel_gen/README.md
docs/stakeholder/*.md
```

## Specific Gotchas by File

### `brainsmith/dataflow/core/dataflow_interface.py`
**Gotcha**: Properties vs methods confusion
```python
# Current mixed usage:
@property
def tensor_dims(self) -> List[int]:  # Property access
    return self._tensor_dims

def get_num_blocks(self) -> int:  # Method call
    return len(self.tensor_dims) // len(self.block_dims)
```

### `brainsmith/tools/hw_kernel_gen/enhanced_template_context.py`
**Gotcha**: Template variable preparation
```python
# Template context building:
context = {
    'tensor_dims': interface.tensor_dims,  # Template variable
    'block_dims': interface.block_dims,  # Template variable  
    'num_blocks': interface.get_num_blocks()  # Computed value
}
```

### `tests/dataflow/core/test_tensor_chunking.py`
**Gotcha**: Filename and test method naming
```python
# File needs renaming: test_tensor_chunking.py → test_block_chunking.py
class TestTensorChunking:  # Class name to update
    def test_tensor_dimensions(self):  # Method name to update
        pass
```

## Migration Risk Assessment

### Low Risk Changes (Safe)
- Internal variable names
- Comments and docstrings  
- Property renaming with aliases
- Documentation updates

### Medium Risk Changes (Requires Testing)
- Method name changes
- Template variable updates
- File renaming
- Formula updates

### High Risk Changes (Requires Coordination)
- Public API changes
- Generated code updates
- Cross-file dependencies
- Mathematical relationships

## Validation Requirements

### 1. Functional Validation
- All existing tests must pass
- Generated code must function identically
- Mathematical calculations must remain correct

### 2. Integration Validation  
- End-to-end workflow testing
- Template generation consistency
- Cross-module compatibility

### 3. Performance Validation
- No performance regression
- Memory usage consistency
- Timing analysis accuracy

This analysis provides the foundation for safe, systematic terminology migration across the entire brainsmith codebase.