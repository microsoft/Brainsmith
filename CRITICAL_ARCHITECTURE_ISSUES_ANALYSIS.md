# Critical Architecture Issues Analysis and Fix Plan

## Executive Summary

Two critical architectural issues have been identified in the Hardware Kernel Generator (HKG) and dataflow modeling system that fundamentally misrepresent the system's purpose and violate the separation between compile-time generation and runtime configuration.

## Issue 1: qDim Conceptual Error

### Problem Description
The current documentation and implementation incorrectly describes `qDim` as the "original tensor dimension" or "full tensor size". This is **fundamentally wrong**.

### Correct Conceptual Model
- **`qDim` = `num_tensors`**: The number of tensor chunks that fit into the input shape when divided by `tDim`
- **`tDim`**: The size/shape of each individual tensor chunk processed per operation
- **`stream_dims`**: The streaming parallelism (elements processed per clock cycle)

### Mathematical Relationship
```
Input_Shape = qDim × tDim = num_tensors × tensor_chunk_size
qDim = Input_Shape ÷ tDim
```

**Example**:
- Input shape: [1, 768] (BERT hidden vector)
- tDim: [1, 96] (process 96 elements at a time)
- qDim: [1, 8] (8 chunks of 96 elements each: 8 × 96 = 768)

### Current Incorrect Implementation
In multiple files, qDim is incorrectly set to static large values (768, 512, etc.) representing full tensor sizes instead of chunk counts.

## Issue 2: Static Dimension Assignment vs Runtime Configuration

### Problem Description
The HKG currently generates code with **hardcoded static dimensions**, violating the fundamental principle that:
- **HKG generates reusable components** for future use
- **FINN compiler sets dimensions at runtime** when instantiating components with actual model data

### Correct Architecture
1. **Generation Time**: HKG creates flexible components with dimension parameters
2. **Runtime**: FINN compiler extracts actual tensor shapes from model and configures dimensions
3. **Lazy Evaluation**: DataflowModel builds dimensions only when ModelWrapper provides actual data

### Current Incorrect Implementation
Examples from generated code:
```python
# WRONG: Static hardcoded dimensions
qDim=768, tDim=96, stream_dims=8

# CORRECT: Runtime-configurable dimensions
qDim=self._extract_num_tensors_from_model(),
tDim=self._extract_tensor_chunk_size(),
stream_dims=self._extract_stream_parallelism()
```

## Root Cause Analysis

### Issue Origins
1. **Documentation Propagation**: Initial conceptual errors in documentation spread throughout codebase
2. **Tutorial Implementation**: HKG tutorial reinforced incorrect patterns with static examples
3. **Missing Runtime Extraction**: Lazy DataflowModel building exists but is not properly leveraged
4. **Template System**: Templates generate static code instead of runtime-configurable code

### Affected Components
1. **Documentation**: All stakeholder docs incorrectly describe qDim/tDim/stream_dims relationships
2. **HKG Tutorial**: Vector dot product example uses static dimensions
3. **Generated Code**: Templates produce hardcoded dimension values
4. **Test Suite**: Tests validate incorrect conceptual model
5. **Auto Classes**: `AutoHWCustomOp` base class has flawed dimension handling

## Comprehensive Fix Plan

### Phase 1: Conceptual Corrections (High Priority)

#### 1.1 Rename qDim → num_tensors Throughout Codebase
```bash
# Files requiring qDim → num_tensors rename:
- brainsmith/dataflow/core/dataflow_interface.py
- brainsmith/dataflow/core/auto_hw_custom_op.py  
- brainsmith/dataflow/core/tensor_chunking.py
- brainsmith/dataflow/core/chunking_strategy.py
- All test files
- All documentation files
- HKG tutorial example
```

#### 1.2 Fix Mathematical Relationships
**Current (Wrong)**:
```python
# Constraints: stream_dims ≤ tDim ≤ qDim
# Relationship: qDim × tDim = original_tensor_shape
```

**Correct**:
```python
# Constraints: stream_dims ≤ tDim, num_tensors = input_shape ÷ tDim
# Relationship: input_shape = num_tensors × tDim
# Stream constraint: tDim must be divisible by stream_dims
```

#### 1.3 Update Documentation
- Fix all stakeholder documentation (6 files)
- Correct HKG tutorial mathematical explanations
- Update API reference documentation
- Fix inline code comments

### Phase 2: Runtime Configuration Implementation (High Priority)

#### 2.1 Enhance AutoHWCustomOp for Runtime Dimensions
```python
class AutoHWCustomOp(HWCustomOp):
    def _build_dataflow_model(self):
        """Build model with runtime-extracted dimensions."""
        for metadata in self._interface_metadata_collection.interfaces:
            # Extract actual tensor shape from ModelWrapper
            if self._model_wrapper:
                tensor_shape = self._extract_tensor_shape_from_model(metadata.name)
                num_tensors, tDim = self._compute_runtime_chunking(tensor_shape, metadata)
            else:
                # Fallback to defaults during generation
                num_tensors, tDim = metadata.get_default_chunking()
```

#### 2.2 Fix Template Generation
**Current Templates (Wrong)**:
```jinja2
DataflowInterface(
    qDim=768,  # Hardcoded static value
    tDim=96,   # Hardcoded static value
    stream_dims=8     # Hardcoded static value
)
```

**Correct Templates**:
```jinja2
DataflowInterface(
    num_tensors=self._compute_num_tensors({{interface.name}}),
    tDim=self._compute_tensor_chunk_size({{interface.name}}),
    stream_dims=self._compute_stream_parallelism({{interface.name}})
)
```

#### 2.3 Implement Runtime Extraction Methods
```python
def _compute_num_tensors(self, interface_name: str) -> List[int]:
    """Extract number of tensors from model at runtime."""
    if self._model_wrapper:
        tensor_shape = self._model_wrapper.get_tensor_shape(interface_name)
        return self._chunking_strategy.compute_num_tensors(tensor_shape)
    return [1]  # Default during generation

def _compute_tensor_chunk_size(self, interface_name: str) -> List[int]:
    """Extract tensor chunk size from configuration."""
    # Implementation depends on operation requirements
    pass

def _compute_stream_parallelism(self, interface_name: str) -> List[int]:
    """Extract streaming parallelism from configuration."""
    # Implementation depends on hardware constraints
    pass
```

### Phase 3: Template System Overhaul (Medium Priority)

#### 3.1 Create Runtime-Aware Templates
- Modify all HKG templates to generate runtime-configurable code
- Remove hardcoded dimension values
- Add model extraction methods to generated classes

#### 3.2 Update Metadata System
- Modify metadata files to specify dimension computation strategies
- Remove static dimension specifications
- Add runtime extraction hints

### Phase 4: Test System Updates (Medium Priority)

#### 4.1 Fix Test Conceptual Model
- Update all tests to use correct num_tensors concept
- Add runtime configuration testing
- Remove tests validating incorrect static behavior

#### 4.2 Add Runtime Testing
- Test dimension extraction from ModelWrapper
- Validate lazy DataflowModel building
- Test FINN integration with runtime dimensions

### Phase 5: Documentation Comprehensive Update (Medium Priority)

#### 5.1 Stakeholder Documentation
- Rewrite architecture sections with correct conceptual model
- Update all examples and tutorials
- Fix mathematical relationships throughout

#### 5.2 API Documentation
- Update method signatures and descriptions
- Correct parameter explanations
- Add runtime configuration examples

## Implementation Priority

### Immediate (Week 1)
1. Rename qDim → num_tensors in core files
2. Fix mathematical relationships in DataflowInterface
3. Update AutoHWCustomOp dimension handling
4. Correct stakeholder documentation

### Short-term (Week 2-3)
1. Implement runtime dimension extraction
2. Update template generation system
3. Fix HKG tutorial with correct concepts
4. Update test suite

### Medium-term (Week 4-6)
1. Complete template system overhaul
2. Add comprehensive runtime testing
3. Update all documentation
4. Validate FINN integration

## Risk Assessment

### High Risk
- **Breaking Changes**: Renaming qDim affects entire codebase
- **FINN Integration**: Changes may affect existing FINN workflows
- **Template Compatibility**: Generated code structure changes

### Mitigation Strategies
1. **Gradual Migration**: Implement changes incrementally with compatibility layers
2. **Comprehensive Testing**: Extensive testing at each phase
3. **Documentation Sync**: Keep documentation current with implementation
4. **Stakeholder Communication**: Clear communication about architectural changes

## Success Metrics

### Technical Metrics
- [ ] All qDim references renamed to num_tensors
- [ ] Runtime dimension extraction functional
- [ ] Generated code uses runtime configuration
- [ ] All tests pass with new conceptual model
- [ ] FINN integration maintained

### Documentation Metrics
- [ ] Stakeholder docs reflect correct concepts
- [ ] Tutorial demonstrates proper usage
- [ ] API docs accurate and complete
- [ ] Mathematical relationships correct

## Conclusion

These issues represent fundamental architectural problems that affect the entire system's conceptual foundation. The fixes are extensive but necessary to ensure:

1. **Correct Conceptual Model**: Proper understanding of tensor chunking
2. **Runtime Flexibility**: Components work with any model/tensor size
3. **FINN Integration**: Proper separation of generation-time vs runtime concerns
4. **Developer Experience**: Clear, correct documentation and examples

The lazy DataflowModel instantiation system was designed to handle this correctly but has been subverted by static dimension assignments. Fixing these issues will restore the intended architecture and provide a solid foundation for future development.