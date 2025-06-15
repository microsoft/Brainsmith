# Multi-Interface Datatype Mapping Implementation Checklist

## Overview
Implementation of per-interface datatype parameter mapping for RTL modules with multiple interfaces of the same type.

**Start Date**: 2025-06-15
**Target Completion**: 2025-07-12 (4 weeks)

---

## Phase 1: Core Infrastructure (Week 1 - 3 days)

### Day 1: Extend InterfaceMetadata with datatype_params
- [x] **Task 1.1**: Add `datatype_params: Optional[Dict[str, str]]` field to InterfaceMetadata
  - [x] Locate `brainsmith/dataflow/core/interface_metadata.py`
  - [x] Add field with proper typing and documentation
  - [x] Add imports for Optional and Dict (already present)
- [x] **Task 1.2**: Implement `get_datatype_parameter_name()` method
  - [x] Create method with property_type parameter
  - [x] Handle custom datatype_params override
  - [x] Implement default naming fallback
- [x] **Task 1.3**: Implement `_get_clean_interface_name()` helper method
  - [x] Remove common prefixes: 's_axis_', 'm_axis_', 'axis_'
  - [x] Remove common suffixes: '_tdata', '_tvalid', '_tready'
  - [x] Convert to uppercase for parameter naming
- [x] **Task 1.4**: Test InterfaceMetadata extensions
  - [x] Write basic unit tests for default parameter naming
  - [x] Test custom datatype_params override functionality
  - [x] Verify backward compatibility

**Status**: ✅ Complete
**Notes**: Core functionality implemented and tested. Logic verified with simple_test.py 

---

### Day 2: Create DatatypeParamPragma
- [x] **Task 2.1**: Add `DATATYPE_PARAM` to PragmaType enum
  - [x] Locate `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
  - [x] Add new enum value
- [x] **Task 2.2**: Implement `DatatypeParamPragma` class
  - [x] Inherit from Pragma and InterfaceNameMatcher
  - [x] Add comprehensive docstring with syntax examples
  - [x] Implement `_parse_inputs()` method with validation
  - [x] Validate property types: width, signed, format, bias, fractional_width
- [x] **Task 2.3**: Implement pragma application methods
  - [x] `applies_to_interface_metadata()` - interface name matching
  - [x] `apply_to_metadata()` - update datatype_params on InterfaceMetadata
  - [x] Handle initialization of datatype_params dict
- [x] **Task 2.4**: Test DatatypeParamPragma functionality
  - [x] Test pragma parsing and validation
  - [x] Test interface name matching
  - [x] Test datatype_params setting on metadata
  - [x] Test error handling for invalid inputs

**Status**: ✅ Complete
**Notes**: Core DatatypeParamPragma implementation complete and tested. Logic verified with test_pragma_logic.py 

---

### Day 3: Update Pragma Handler and Integration Testing
- [x] **Task 3.1**: Register new pragma in PragmaHandler
  - [x] Locate `brainsmith/tools/hw_kernel_gen/rtl_parser/pragma.py`
  - [x] Add DatatypeParamPragma to pragma_constructors dict
  - [x] Add import for DatatypeParamPragma
- [x] **Task 3.2**: Basic integration testing
  - [x] Test pragma extraction from RTL comments (logic verified)
  - [x] Test pragma application to interface metadata (logic verified)
  - [x] Verify no conflicts with existing pragma system (reuses existing patterns)
- [x] **Task 3.3**: Debug and fix any issues
  - [x] Resolve import errors (imports added correctly)
  - [x] Fix any integration problems (no issues found)
  - [x] Ensure backward compatibility (new field is optional with None default)

**Status**: ✅ Complete
**Notes**: Pragma handler integration complete. Core infrastructure ready for template integration. 

---

## Phase 2: Template Integration (Week 2 - 2 days)

### Day 4: Update Template Context Generation
- [x] **Task 4.1**: Locate template context generation code
  - [x] Found `brainsmith/tools/hw_kernel_gen/templates/context_generator.py`
  - [x] Identified `_template_context_to_dict()` method and template context flow
- [x] **Task 4.2**: Add datatype_params to interface context
  - [x] Enhanced dataflow_interfaces with datatype parameter info
  - [x] Added parameter names for all properties: width, signed, format, bias, fractional_width
  - [x] Created `_enhance_interfaces_with_datatype_params()` helper method
  - [x] Updated categorized interfaces (input/output/weight) to include datatype_params
- [x] **Task 4.3**: Test template context generation
  - [x] Created comprehensive test script `test_template_context.py`
  - [x] Verified default parameter naming (INPUT0_WIDTH, SIGNED_INPUT0, etc.)
  - [x] Tested custom datatype_params override functionality  
  - [x] Validated mixed scenarios (elementwise add with indexed inputs)

**Status**: ✅ Complete
**Notes**: Template context generation now includes interface-specific datatype parameters. All interfaces get width_param, signed_param, format_param, bias_param, fractional_width_param fields.

---

### Day 5: Update HWCustomOp Template
- [x] **Task 5.1**: Locate HWCustomOp template
  - [x] Found `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_phase2.py.j2`
  - [x] Identified get_nodeattr_types() method and make_*_node() function
- [x] **Task 5.2**: Update template to use interface-specific parameters
  - [x] Replaced hardcoded `{{ interface.name }}_dtype` with interface-specific parameters
  - [x] Added width_param, signed_param for all dataflow interfaces
  - [x] Added conditional format_param, bias_param, fractional_width_param based on datatype_params
  - [x] Updated documentation to show interface-specific parameter names
  - [x] Updated validation logic to check for interface-specific parameters
- [x] **Task 5.3**: Test template rendering
  - [x] Template now generates different parameter names per interface (INPUT0_WIDTH, INPUT1_WIDTH, etc.)
  - [x] Handles custom pragma-specified parameters via datatype_params
  - [x] Gracefully handles missing datatype_params with default naming

**Status**: ✅ Complete
**Notes**: HWCustomOp template now generates interface-specific parameters. Each interface gets its own width/signed parameters, with custom names from DATATYPE_PARAM pragmas when specified.

---

## Phase 3: Testing & Validation (Week 3 - 4 days)

### Day 6-7: Unit Tests
- [x] **Task 6.1**: Create test file for pragma functionality
  - [x] Created `tests/tools/hw_kernel_gen/test_datatype_param_pragma.py`
  - [x] Set up test infrastructure with mocked imports
- [x] **Task 6.2**: Test DatatypeParamPragma class
  - [x] `test_pragma_parsing_valid_inputs()` - all property types tested
  - [x] `test_pragma_parsing_invalid_inputs()` - error handling verified
  - [x] `test_interface_name_matching()` - exact, prefix, reverse prefix patterns
  - [x] `test_datatype_params_application()` - metadata modification tested
  - [x] `test_property_type_validation()` - all valid/invalid types tested
- [x] **Task 6.3**: Test InterfaceMetadata extensions
  - [x] `test_get_datatype_parameter_name_defaults()` - default naming verified
  - [x] `test_get_datatype_parameter_name_custom()` - pragma override tested
  - [x] `test_clean_interface_name_extraction()` - prefix/suffix removal tested
- [x] **Task 6.4**: Comprehensive test coverage
  - [x] All core functionality tested
  - [x] Edge cases covered
  - [x] Error handling verified

**Status**: ✅ Complete
**Notes**: Unit tests successfully verify all pragma functionality and InterfaceMetadata enhancements.

---

### Day 8-9: Integration Tests
- [x] **Task 8.1**: Create integration test file
  - [x] Created `tests/tools/hw_kernel_gen/test_multi_interface_integration.py`
  - [x] Set up test infrastructure with temporary RTL file generation
- [x] **Task 8.2**: Test real-world scenarios
  - [x] `test_elementwise_add_scenario()` - verified INPUT0_WIDTH, INPUT1_WIDTH pragmas
  - [x] `test_multihead_attention_scenario()` - verified QUERY_WIDTH, KEY_WIDTH pragmas
  - [x] `test_default_parameter_generation()` - verified automatic naming (INPUT0_WIDTH etc.)
- [x] **Task 8.3**: Test template integration
  - [x] `test_template_context_integration()` - tested mixed pragma/default scenarios
  - [x] Template context generation verified with datatype_params
- [x] **Task 8.4**: Integration with RTL parser pipeline
  - [x] Tested full RTL parsing → pragma application → interface metadata flow
  - [x] Verified datatype_params correctly set on InterfaceMetadata
  - [x] Confirmed template context includes interface-specific parameters

**Status**: ✅ Complete
**Notes**: Integration tests demonstrate end-to-end functionality. All major scenarios work correctly in Docker environment.

---

## Phase 4: Documentation & Examples (Week 4 - 2 days)

### Day 10: Update Documentation
- [ ] **Task 10.1**: Update RTL parser README
  - [ ] Document new DATATYPE_PARAM pragma syntax
  - [ ] Add usage examples
  - [ ] Update feature list
- [ ] **Task 10.2**: Update HKG design document
  - [ ] Add multi-interface datatype mapping section
  - [ ] Update pragma system documentation
  - [ ] Add workflow diagrams if needed
- [ ] **Task 10.3**: Update strategy document
  - [ ] Add implementation notes to MULTI_INTERFACE_DATATYPE_MAPPING.md
  - [ ] Document actual implementation vs. original strategy
  - [ ] Add troubleshooting section

**Status**: ⏳ Pending
**Notes**: 

---

### Day 11: Create Example Files and Final Validation
- [ ] **Task 11.1**: Create example directory
  - [ ] Create `examples/multi_interface_examples/` directory
  - [ ] Add .gitignore and README
- [ ] **Task 11.2**: Create example RTL modules
  - [ ] `elementwise_add.sv` - indexed parameters with pragmas
  - [ ] `multihead_attention.sv` - named parameters with pragmas
  - [ ] `simple_dual_input.sv` - default parameter generation
- [ ] **Task 11.3**: Create automated test script
  - [ ] `test_generation.py` - automated testing of all examples
  - [ ] Verify HKG generation works for all examples
  - [ ] Test generated HWCustomOp classes
- [ ] **Task 11.4**: Final validation
  - [ ] Run complete test suite
  - [ ] Test against real brainsmith modules
  - [ ] Performance testing (target: <10ms overhead)
  - [ ] Backward compatibility verification

**Status**: ⏳ Pending
**Notes**: 

---

## Success Criteria Verification

### ✅ Functional Requirements
- [ ] Default parameter name generation for all interface types
- [ ] Support for indexed parameters via pragmas (INPUT0_WIDTH, INPUT1_WIDTH)
- [ ] DATATYPE_PARAM pragma for custom parameter mapping
- [ ] Template integration with interface-specific parameter names
- [ ] Backward compatibility with existing DATATYPE pragma

### ✅ Quality Requirements
- [ ] 100% test coverage for new functionality
- [ ] Integration with existing HKG pipeline
- [ ] Backward compatibility with current pragma system
- [ ] Performance: <10ms additional overhead for parameter mapping

### ✅ Documentation Requirements
- [ ] Updated design documents with new functionality
- [ ] Code examples for pragma usage
- [ ] Clear migration path from current system

---

## Notes and Issues

### Implementation Notes
- 

### Issues Encountered
- 

### Performance Observations
- 

### Backward Compatibility Concerns
- 

---

## Final Checklist

- [ ] All phases completed successfully
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Examples working
- [ ] Performance targets met
- [ ] Backward compatibility verified
- [ ] Code reviewed and ready for merge

**Implementation Status**: ⏳ Not Started
**Completion Date**: 
**Total Time Spent**: 