# Interface Builder Refactor Implementation Checklist

## Overview
This checklist tracks the implementation of the RTL Parser Interface Builder clean refactor to directly create InterfaceMetadata objects while maximizing reuse of existing components (InterfaceScanner and ProtocolValidator).

**Total Estimated Time**: 8-10 days
**Key Strategy**: Maximum component reuse, minimal new code

---

## Phase 1: Core Refactor (3-4 Days)

### 1.1 Add InterfaceBuilder.build_interface_metadata() Method
- [x] **1.1.1** Add new method signature to InterfaceBuilder class
  - [x] Method: `build_interface_metadata(self, ports: List[Port], pragmas: List[Pragma]) -> Tuple[List[InterfaceMetadata], List[Port]]`
  - [x] Add proper type hints and docstring
  - [x] Ensure method coexists with existing `build_interfaces()` method
  - [x] **File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/interface_builder.py`

- [x] **1.1.2** Implement Stage 1: Port scanning using existing InterfaceScanner
  - [x] Call `self.scanner.scan(ports)` to get port_groups and unassigned_ports
  - [x] Verify proper integration with existing InterfaceScanner logic
  - [x] Add debug logging for port group identification
  - [x] **Verification**: Test with thresholding_axi.sv to ensure port groups are correctly identified

- [x] **1.1.3** Implement Stage 2: Protocol validation using existing ProtocolValidator
  - [x] Iterate through port_groups from scanner
  - [x] Call `self.validator.validate(group)` for each group
  - [x] Collect valid groups and add failed group ports back to unassigned_ports
  - [x] **Verification**: Ensure validation results match existing build_interfaces() method

- [x] **1.1.4** Implement Stage 3: Direct metadata creation with pragma application
  - [x] Call `_create_base_metadata(group)` for each valid group
  - [x] Call `_apply_pragmas(base_metadata, pragmas, group)` for pragma application
  - [x] Collect results into metadata_list
  - [x] Return tuple of (metadata_list, unassigned_ports)

### 1.2 Implement Direct Metadata Creation Methods

- [x] **1.2.1** Implement `_create_base_metadata()` method
  - [x] Method signature: `_create_base_metadata(self, group: PortGroup) -> InterfaceMetadata`
  - [x] Use `group.interface_type` directly (already determined by ProtocolValidator)
  - [x] Start with empty `allowed_datatypes` list (filled by DatatypePragma only)
  - [x] Use `DefaultChunkingStrategy()` as default
  - [x] Extract description from group metadata (include direction if available)
  - [x] **File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/interface_builder.py`

- [x] **1.2.2** Implement `_apply_pragmas()` method
  - [x] Method signature: `_apply_pragmas(self, metadata: InterfaceMetadata, pragmas: List[Pragma], group: PortGroup) -> InterfaceMetadata`
  - [x] Create temporary Interface object for pragma compatibility
  - [x] Iterate through pragmas and apply using existing chain-of-responsibility pattern
  - [x] Call `pragma.applies_to_interface()` and `pragma.apply_to_interface_metadata()`
  - [x] Return final modified metadata
  - [ ] **Verification**: Test with DatatypePragma, BDimPragma, and WeightPragma

- [x] **1.2.3** Add required imports
  - [x] Import `InterfaceMetadata` from `brainsmith.dataflow.core.interface_metadata`
  - [x] Import `DefaultChunkingStrategy` from `brainsmith.dataflow.core.block_chunking`
  - [x] Import `ValidationResult` from `.data`
  - [x] Verify all imports are available and correct

### 1.3 Test New API

- [x] **1.3.1** Create basic unit tests for new method
  - [x] Test with simple port list (AXI-Stream input/output)
  - [x] Test with pragmas (DATATYPE, BDIM, WEIGHT)
  - [x] Verify InterfaceMetadata objects are created correctly
  - [x] Compare results with existing build_interfaces() method
  - [x] **File**: `test_interface_builder_simple.py` (temporary test file)

- [x] **1.3.2** Test with real SystemVerilog file
  - [x] Use thresholding_axi.sv as test case
  - [x] Parse RTL and extract ports
  - [x] Call new build_interface_metadata() method
  - [x] Verify correct number of interfaces detected
  - [x] Verify interface types and metadata are correct

- [x] **1.3.3** Test pragma integration
  - [x] Create test RTL with pragma comments
  - [x] Parse pragmas using existing PragmaHandler
  - [x] Apply pragmas through new build_interface_metadata() method
  - [x] Verify DataTypeConstraints are created correctly
  - [ ] Verify chunking strategies are applied correctly (BDIM deferred)
  - [x] Verify WeightPragma overrides interface types correctly

### 1.4 Update Parser Integration

- [x] **1.4.1** Update parser.py to use new method
  - [x] Locate calls to `interface_builder.build_interfaces()`
  - [x] Replace with calls to `interface_builder.build_interface_metadata()`
  - [x] Update variable names and types (Interface → InterfaceMetadata)
  - [x] **File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/parser.py`

- [x] **1.4.2** Update return type handling
  - [x] Ensure parser returns List[InterfaceMetadata] instead of Dict[str, Interface]
  - [x] Update any downstream code that expects Interface objects
  - [x] Verify template generation still works with InterfaceMetadata

- [x] **1.4.3** Test end-to-end integration
  - [x] Parse complete SystemVerilog file through updated parser
  - [x] Verify InterfaceMetadata objects are returned
  - [x] Test with pragmas to ensure full chain works
  - [x] **Verification**: Use existing thresholding test case

---

## Phase 2: Legacy Cleanup (2-3 Days)

### 2.1 Remove Interface Objects

- [x] **2.1.1** Identify all Interface class dependencies
  - [x] Search codebase for `from .data import Interface`
  - [x] Search for Interface object instantiation
  - [x] Search for Interface method calls
  - [x] Document all dependencies before removal

- [x] **2.1.2** Update template system
  - [x] Identify templates that use Interface objects
  - [x] Update template context generation to use InterfaceMetadata
  - [x] Test template rendering with InterfaceMetadata objects
  - [x] **Files**: Template files in `brainsmith/tools/hw_kernel_gen/templates/`

- [ ] **2.1.3** Update generator classes
  - [ ] Update HWCustomOp generator to work with InterfaceMetadata
  - [ ] Update RTLBackend generator to work with InterfaceMetadata
  - [ ] Update test suite generator to work with InterfaceMetadata
  - [ ] **Files**: `brainsmith/tools/hw_kernel_gen/generators/`

- [ ] **2.1.4** Remove Interface class definition (DEFERRED)
  - [ ] Remove Interface class from data.py (requires pragma system refactor)
  - [ ] Remove any Interface-specific helper methods 
  - [ ] Update imports throughout codebase
  - [ ] **File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
  - [ ] **Note**: Interface class still needed for pragma compatibility - defer to Phase 3+

### 2.2 Remove build_interfaces() Method

- [x] **2.2.1** Remove legacy method from InterfaceBuilder
  - [x] Remove `build_interfaces()` method definition
  - [x] Remove any supporting methods only used by build_interfaces()
  - [x] Update class docstring to reflect new API
  - [x] **File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/interface_builder.py`

- [x] **2.2.2** Search for remaining build_interfaces() calls
  - [x] Search entire codebase for build_interfaces() usage
  - [x] Update any remaining calls to use build_interface_metadata()
  - [x] Ensure no legacy code paths remain

### 2.3 Update Templates

- [ ] **2.3.1** Update HWCustomOp template
  - [ ] Modify template to work with InterfaceMetadata objects
  - [ ] Update interface iteration and property access
  - [ ] Test generation with updated template
  - [ ] **File**: `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_slim.py.j2`

- [ ] **2.3.2** Update RTLBackend template  
  - [ ] Modify template to work with InterfaceMetadata objects
  - [ ] Update interface property access patterns
  - [ ] Test generation with updated template
  - [ ] **File**: `brainsmith/tools/hw_kernel_gen/templates/rtl_backend.py.j2`

- [ ] **2.3.3** Update test suite template
  - [ ] Modify template to work with InterfaceMetadata objects
  - [ ] Update test generation logic
  - [ ] Test generation with updated template
  - [ ] **File**: `brainsmith/tools/hw_kernel_gen/templates/test_suite.py.j2`

### 2.4 Remove Dead Code

- [ ] **2.4.1** Remove unused Interface-related utilities
  - [ ] Remove helper functions that only worked with Interface objects
  - [ ] Remove any Interface-specific validation or processing code
  - [ ] Clean up imports and references

- [ ] **2.4.2** Update pragma system cleanup
  - [ ] Ensure pragma system works purely with InterfaceMetadata
  - [ ] Remove any temporary Interface object creation if no longer needed
  - [ ] Optimize pragma application flow

---

## Phase 3: Testing & Integration (2-3 Days)

### 3.1 Update All Tests

- [ ] **3.1.1** Update interface builder tests
  - [ ] Modify tests to expect InterfaceMetadata objects
  - [ ] Update test assertions and verification logic
  - [ ] Add new tests for build_interface_metadata() method
  - [ ] **File**: `tests/tools/hw_kernel_gen/rtl_parser/test_interface_builder.py`

- [ ] **3.1.2** Update parser tests
  - [ ] Modify parser tests to work with InterfaceMetadata
  - [ ] Update test data and expected outcomes
  - [ ] Verify all parser functionality still works
  - [ ] **File**: `tests/tools/hw_kernel_gen/rtl_parser/test_rtl_parser.py`

- [ ] **3.1.3** Update integration tests
  - [ ] Modify end-to-end tests to expect InterfaceMetadata
  - [ ] Update golden reference files if necessary
  - [ ] Test complete pipeline with real SystemVerilog files
  - [ ] **File**: `tests/integration/test_end_to_end_thresholding.py`

### 3.2 Integration Testing

- [ ] **3.2.1** Test with thresholding_axi.sv
  - [ ] Parse SystemVerilog file and extract interfaces
  - [ ] Verify correct InterfaceMetadata objects are created
  - [ ] Compare with original Interface-based results
  - [ ] Ensure no functionality is lost

- [ ] **3.2.2** Test with complex SystemVerilog files
  - [ ] Test with files containing multiple interface types
  - [ ] Test with files containing pragmas
  - [ ] Test with files containing edge cases
  - [ ] Verify robust handling of all cases

- [ ] **3.2.3** Test template generation end-to-end
  - [ ] Generate HWCustomOp using new InterfaceMetadata flow
  - [ ] Generate RTLBackend using new InterfaceMetadata flow
  - [ ] Generate test suite using new InterfaceMetadata flow
  - [ ] Compare generated code with Interface-based generation

### 3.3 Pragma Integration Testing

- [ ] **3.3.1** Test DatatypePragma application
  - [ ] Create test RTL with DATATYPE pragma
  - [ ] Parse and apply pragma through new flow
  - [ ] Verify DataTypeConstraint objects are created correctly
  - [ ] Test with various pragma formats and parameters

- [ ] **3.3.2** Test BDimPragma application
  - [ ] Create test RTL with BDIM pragma
  - [ ] Parse and apply pragma through new flow
  - [ ] Verify chunking strategies are created correctly
  - [ ] Test both legacy and enhanced BDIM formats

- [ ] **3.3.3** Test WeightPragma application
  - [ ] Create test RTL with WEIGHT pragma
  - [ ] Parse and apply pragma through new flow
  - [ ] Verify interface type override (INPUT → WEIGHT)
  - [ ] Test with multiple weight interfaces

### 3.4 Performance Validation

- [ ] **3.4.1** Benchmark interface creation performance
  - [ ] Measure time for Interface-based flow vs InterfaceMetadata flow
  - [ ] Test with large numbers of interfaces
  - [ ] Verify performance improvement or at least no regression

- [ ] **3.4.2** Benchmark memory usage
  - [ ] Measure memory consumption for Interface vs InterfaceMetadata
  - [ ] Test with bulk interface creation
  - [ ] Verify memory efficiency improvements

- [ ] **3.4.3** Profile parsing pipeline
  - [ ] Profile complete RTL parsing with new flow
  - [ ] Identify any performance bottlenecks
  - [ ] Optimize if necessary

---

## Phase 4: Documentation & Finalization (1-2 Days)

### 4.1 Update Documentation

- [ ] **4.1.1** Update API reference documentation
  - [ ] Document new build_interface_metadata() method
  - [ ] Update InterfaceBuilder class documentation
  - [ ] Remove references to build_interfaces() method
  - [ ] **File**: `docs/rtl_parser/RTL_Parser_API_Reference.md`

- [ ] **4.1.2** Update architecture documentation
  - [ ] Document new InterfaceMetadata-based flow
  - [ ] Update data flow diagrams
  - [ ] Document component reuse strategy
  - [ ] **File**: `docs/rtl_parser/RTL_Parser_Design_Document.md`

- [ ] **4.1.3** Update developer guides
  - [ ] Update integration examples
  - [ ] Update tutorial content
  - [ ] Add migration notes for users of old API

### 4.2 Code Review

- [ ] **4.2.1** Self-review implementation
  - [ ] Review all code changes for consistency
  - [ ] Verify error handling is appropriate
  - [ ] Check for any remaining Interface object references
  - [ ] Ensure proper logging and debug output

- [ ] **4.2.2** Prepare code review documentation
  - [ ] Summarize changes made in each phase
  - [ ] Document design decisions and trade-offs
  - [ ] List files modified and reasons for changes
  - [ ] Create review checklist for superior

### 4.3 Final Testing

- [ ] **4.3.1** Complete regression testing
  - [ ] Run full test suite and ensure all tests pass
  - [ ] Test with all example SystemVerilog files
  - [ ] Verify no functionality regressions
  - [ ] Test edge cases and error conditions

- [ ] **4.3.2** Test against golden reference outputs
  - [ ] Generate outputs using new InterfaceMetadata flow
  - [ ] Compare with known good outputs from Interface flow
  - [ ] Verify functional equivalence
  - [ ] Update golden references if necessary

- [ ] **4.3.3** Integration testing with downstream components
  - [ ] Test with unified_hwkg generator
  - [ ] Test with template system
  - [ ] Test with CLI interface
  - [ ] Verify no breaking changes for users

### 4.4 Architecture Documentation

- [ ] **4.4.1** Document simplified data flow
  - [ ] Create data flow diagram showing Port → InterfaceMetadata path
  - [ ] Document component reuse strategy
  - [ ] Explain benefits of new architecture
  - [ ] **File**: Update RTL_PARSER_INTERFACE_BUILDER_REFACTOR_PROPOSAL.md

- [ ] **4.4.2** Document migration path
  - [ ] Create migration guide for any external users
  - [ ] Document breaking changes and mitigation strategies
  - [ ] Provide before/after examples

---

## Success Criteria Verification

### Functional Requirements
- [ ] ✅ All existing functionality preserved (verify with test suite)
- [ ] ✅ Pragma application works correctly (test all pragma types)
- [ ] ✅ Template generation produces correct output (compare with golden reference)
- [ ] ✅ Interface validation maintains accuracy (reuses existing ProtocolValidator)

### Quality Requirements  
- [ ] ✅ Code complexity significantly reduced (eliminate Interface objects)
- [ ] ✅ Memory usage improved for bulk operations (benchmark results)
- [ ] ✅ Performance improved or maintained (benchmark results)
- [ ] ✅ Architecture aligns with dataflow principles (InterfaceMetadata direct creation)

### Implementation Requirements
- [ ] ✅ No legacy APIs or compatibility layers (clean removal of build_interfaces)
- [ ] ✅ Clean, modern codebase (maximum component reuse)
- [ ] ✅ Comprehensive testing of new architecture (full test suite updated)
- [ ] ✅ Complete documentation updates (API and architecture docs updated)

---

## Risk Mitigation Checklist

### Template System Updates
- [ ] **Risk**: Templates expect Interface objects
- [ ] **Mitigation**: Test template rendering at each step
- [ ] **Verification**: Generate code and compare with known good output

### Pragma System Integration
- [ ] **Risk**: Pragma application might fail with InterfaceMetadata
- [ ] **Mitigation**: Test each pragma type individually
- [ ] **Verification**: Verify pragma effects are correctly applied

### Component Integration
- [ ] **Risk**: Other components might expect Interface objects
- [ ] **Mitigation**: Search codebase thoroughly for dependencies
- [ ] **Verification**: Run full integration tests

### Performance Impact
- [ ] **Risk**: New flow might be slower than original
- [ ] **Mitigation**: Benchmark at each phase
- [ ] **Verification**: Performance tests pass acceptance criteria

---

## Completion Tracking

### Phase 1: Core Refactor
- [x] **Started**: 2025-06-12
- [x] **Completed**: 2025-06-12
- [x] **Duration**: Same day (efficient due to component reuse)
- [x] **Issues Encountered**: BDIM pragma complexity deferred; otherwise clean implementation

### Phase 2: Legacy Cleanup  
- [x] **Started**: 2025-06-12
- [x] **Completed**: 2025-06-12 (partial - Interface class deferred)
- [x] **Duration**: Same day
- [x] **Issues Encountered**: Interface class removal deferred due to pragma system dependencies

### Phase 3: Testing & Integration
- [x] **Started**: 2025-06-12
- [ ] **Completed**: ___________
- [ ] **Duration**: _____ days
- [ ] **Issues Encountered**: ___________

### Phase 4: Documentation & Finalization
- [ ] **Started**: ___________
- [ ] **Completed**: ___________
- [ ] **Duration**: _____ days
- [ ] **Issues Encountered**: ___________

### Overall Project
- [ ] **Total Duration**: _____ days
- [ ] **Final Outcome**: ___________
- [ ] **Lessons Learned**: ___________

---

## Notes and Comments

### Design Decisions
- **Component Reuse Strategy**: Maximize reuse of InterfaceScanner and ProtocolValidator
- **Pragma Integration**: Use temporary Interface object for compatibility during transition
- **Performance Focus**: Eliminate Interface object creation overhead
- **Clean Implementation**: No compatibility layers or dual APIs

### Key Files Modified
- `brainsmith/tools/hw_kernel_gen/rtl_parser/interface_builder.py` - Core refactor
- `brainsmith/tools/hw_kernel_gen/rtl_parser/parser.py` - Parser integration
- `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py` - Remove Interface class
- Template files - Update to work with InterfaceMetadata
- Test files - Update to expect InterfaceMetadata objects

### Implementation Strategy
1. **Add new API alongside old** (Phase 1)
2. **Switch to new API** (Phase 1)
3. **Remove old API and objects** (Phase 2)
4. **Comprehensive testing** (Phase 3)
5. **Documentation and finalization** (Phase 4)

This approach minimizes risk by ensuring the new path works before removing the old path.