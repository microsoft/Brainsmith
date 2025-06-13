# QONNX Datatype Integration Implementation Checklist

## Phase 1: Create Core Components

### Step 1.1: Create DatatypeConstraintGroup class
- [x] Create `brainsmith/dataflow/core/qonnx_types.py` file
- [x] Implement `DatatypeConstraintGroup` dataclass with validation
- [x] Add proper imports and type hints
- [x] Test basic instantiation

### Step 1.2: Implement validation functions
- [x] Implement `validate_datatype_against_constraints()` function
- [x] Implement `_matches_constraint_group()` helper function
- [x] Add proper error handling for invalid QONNX datatypes
- [x] Test validation logic with example datatypes

### Step 1.3: Write unit tests for core components
- [x] Test `DatatypeConstraintGroup` validation
- [x] Test constraint matching for all datatype families (INT, UINT, FIXED, etc.)
- [x] Test edge cases (empty constraints, invalid datatypes)
- [x] Test bitwidth range validation

## Phase 2: Update InterfaceMetadata

### Step 2.1: Modify InterfaceMetadata class
- [x] Update `brainsmith/dataflow/core/interface_metadata.py`
- [x] Add `datatype_constraints` field
- [x] Remove any default datatype fields
- [x] Add `validates_datatype()` method
- [x] Add `get_constraint_description()` method

### Step 2.2: Update InterfaceMetadata imports
- [x] Add import for `DatatypeConstraintGroup`
- [x] Add import for `validate_datatype_against_constraints`
- [x] Update any existing type hints

### Step 2.3: Test InterfaceMetadata changes
- [x] Test constraint validation
- [x] Test constraint description generation
- [x] Test empty constraints behavior

## Phase 3: Update DataflowInterface

### Step 3.1: Add factory method to DataflowInterface
- [x] Add `from_metadata_and_runtime_datatype()` classmethod
- [x] Implement QONNX datatype resolution
- [x] Implement constraint validation with clear error messages
- [x] Test factory method with valid and invalid datatypes

### Step 3.2: Update DataflowInterface to require QONNX dtype
- [x] Ensure `dtype` field is `BaseDataType` type
- [x] Update any existing datatype handling to use QONNX
- [x] Remove any fallback datatype logic

### Step 3.3: Test DataflowInterface changes
- [x] Test successful interface creation with valid datatypes
- [x] Test error handling with invalid datatypes
- [x] Test error messages are clear and helpful

## Phase 4: Update Template Generation

### Step 4.1: Update RTL parser for constraint groups
- [x] Modify pragma parsing to generate `DatatypeConstraintGroup` objects
- [x] Update template context generation
- [x] Remove default datatype generation
- [x] Test pragma parsing with example RTL

### Step 4.2: Update Jinja2 templates
- [x] Update `hw_custom_op_phase2.py.j2` template
- [x] Add proper imports for constraint groups
- [x] Remove default datatype template code
- [x] Test template generation

## Phase 5: Update AutoHWCustomOp Integration

### Step 5.1: Update AutoHWCustomOp datatype methods
- [x] Modify `get_input_datatype()` to require user-specified datatypes
- [x] Update `get_output_datatype()` similarly
- [x] Remove any default datatype fallback logic
- [x] Add clear error messages when datatypes not specified

### Step 5.2: Update dataflow model building
- [x] Update `_build_dataflow_model_with_defaults()` to use new factory method
- [x] Update `_convert_metadata_datatype()` for constraint groups
- [x] Test model building with new constraint validation

### Step 5.3: Test AutoHWCustomOp integration
- [x] Test successful operation with valid datatypes
- [x] Test error handling when datatypes not specified
- [x] Test constraint validation during model building

## Phase 6: Integration Testing

### Step 6.1: End-to-end testing
- [x] Test full RTL → template → instantiation flow
- [x] Test with vector_add example
- [x] Test error scenarios throughout pipeline
- [x] Verify error messages are helpful

### Step 6.2: Backward compatibility testing
- [x] Test existing code still works where possible
- [x] Document breaking changes
- [x] Update any example code or documentation

### Step 6.3: Performance testing
- [ ] Verify constraint validation performance
- [ ] Test with multiple constraint groups
- [ ] Ensure no significant performance regression

## Phase 7: Documentation and Cleanup

### Step 7.1: Update documentation
- [ ] Update any relevant docstrings
- [ ] Create simple usage examples
- [ ] Document the new constraint group format

### Step 7.2: Code cleanup
- [ ] Remove any unused imports
- [ ] Clean up temporary debug code
- [ ] Ensure consistent code style

### Step 7.3: Final validation
- [ ] Run full test suite
- [ ] Verify no regressions in existing functionality
- [ ] Test with BERT demo if possible

## Implementation Status

**Current Status**: Phase 6 Steps 6.1-6.2 Complete - End-to-end integration and backward compatibility verified
**Next Step**: Step 6.3 - Performance testing (optional) or Phase 7 - Documentation and Cleanup  
**Estimated Completion**: Steps 6.1-6.2 complete, Phase 6 mostly done

## Notes

- Focus on simplicity and clear error messages
- No default datatypes - user must always specify
- Use existing QONNX functions where possible
- Maintain minimal API surface