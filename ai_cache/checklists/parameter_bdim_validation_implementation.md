# Parameter-BDIM Validation Implementation Checklist

**Date:** January 6, 2025  
**Scope:** Implement parameter validation in BDimPragma and template generation updates  
**Priority:** High

## Overview

Implement the corrected understanding where:
1. **BDimPragma validates** parameter existence during pragma parsing
2. **Template generation** applies whitelisted defaults (not RTL parser)  
3. **FINN extracts** and sets node attributes automatically
4. **Generated subclass** collects parameters using `get_nodeattr()`

## Phase 1: BDimPragma Parameter Validation ✅ **COMPLETED**

### ✅ Task 1.1: Add Module Parameter Access to BDimPragma
- [x] **File:** `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
- [x] **Location:** `BDimPragma` class
- [x] **Action:** Add method to access module parameters during pragma parsing
- [x] **Implementation:** Added class variable `_module_parameters` and methods `set_module_parameters()` and `_get_module_parameters()`
- [x] **Dependencies:** Updated parser to call `BDimPragma.set_module_parameters()` after parameter extraction
- [x] **Fix Applied:** Resolved dataclass mutable default error by removing type annotation from class variable

### ✅ Task 1.2: Implement Parameter Validation in BDimPragma._parse_inputs()
- [x] **File:** `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py` 
- [x] **Location:** `BDimPragma._parse_inputs()` method around line 514
- [x] **Action:** Add parameter existence validation
- [x] **Implementation:** Deferred validation to `apply_to_metadata()` phase using `_validate_parameters()` method
- [x] **Test:** ✅ **COMPLETED** - Validation working correctly for valid parameters, unknown parameters, and magic numbers

### ✅ Task 1.3: Update Pragma Parser Integration
- [x] **File:** `brainsmith/tools/hw_kernel_gen/rtl_parser/parser.py`
- [x] **Action:** Pass module parameters to pragma instantiation
- [x] **Implementation:** Updated parameter extraction to call `BDimPragma.set_module_parameters()` after parameters are parsed
- [x] **Test:** ✅ **COMPLETED** - Integration complete and tested manually

### ✅ Task 1.4: Test Parameter Validation Functionality
- [x] **Test Cases Validated:**
  - ✅ Valid parameter references (e.g., `[PE]` where PE exists)
  - ✅ Unknown parameter rejection (e.g., `[UNKNOWN_PARAM]` with clear error message)
  - ✅ Magic number rejection (e.g., `[16]` properly blocked)
  - ✅ Mixed shapes (e.g., `[PE,:]` working correctly)
- [x] **Error Messages:** Clear and informative, listing available parameters
- [x] **End-to-End Integration:** ✅ **COMPLETED** - Full RTL parsing with parameter validation working
- [x] **Error Propagation:** ✅ **COMPLETED** - PragmaError properly re-raised to fail parse

### ✅ Task 1.5: Update Error Handling and Test Fixes
- [x] **File:** `brainsmith/tools/hw_kernel_gen/rtl_parser/parser.py`
- [x] **Action:** Updated `_apply_pragmas_to_metadata()` to re-raise PragmaError for validation failures
- [x] **Implementation:** Added specific PragmaError checking to ensure validation errors fail the parse
- [x] **Test Updates:** Fixed 4 integration tests that had invalid parameter references (caught by validation)
- [x] **Test Results:** ✅ **ALL 16 INTEGRATION TESTS PASSING** - Validation working correctly in real test scenarios

### 🎉 **Phase 1 Summary: SUCCESSFULLY COMPLETED**

**What was implemented:**
- ✅ BDimPragma now validates that all parameter names in `[PE]`, `[SIMD,PE]` block shapes exist in the module
- ✅ Magic numbers like `[16]` are properly rejected with clear error messages
- ✅ Validation happens during pragma application phase when parameters are available
- ✅ Validation errors properly fail the RTL parsing process (no silent failures)
- ✅ All existing tests updated and passing, demonstrating validation catches real errors

**Key technical achievements:**
- **Deferred Validation Architecture**: Validation happens during `apply_to_metadata()` instead of `_parse_inputs()` to ensure parameters are available
- **Class Variable Parameter Storage**: Used untyped class variable to avoid dataclass mutable default issues
- **Error Propagation**: PragmaError specifically re-raised in parser to ensure validation failures stop parsing
- **Comprehensive Testing**: End-to-end validation from RTL string to final parsed metadata

**Impact:**
- ✅ **Eliminates Parameter-BDIM Mismatches**: No more `[UNKNOWN_PARAM]` errors at runtime
- ✅ **Enforces Design Goals**: Magic numbers blocked, only parameter names allowed
- ✅ **Clear Error Messages**: Developers see exactly which parameters are available
- ✅ **Robust Integration**: Works seamlessly with existing RTL parser and pragma system

## Phase 2: Template Generation Updates

### ✅ Task 2.1: Define Parameter Whitelist Configuration
- [ ] **File:** `brainsmith/tools/hw_kernel_gen/template_generation.py` (or new config file)
- [ ] **Action:** Define whitelisted default parameters
- [ ] **Implementation:**
  ```python
  # Configuration for parameters that can have defaults
  WHITELISTED_DEFAULTS = {
      "PE": 1,
      "SIMD": 1, 
      "PARALLEL": 1,
      "WIDTH": 8,
      # Add more as needed, but keep list small
  }
  ```
- [ ] **Documentation:** Document why these specific parameters are whitelisted

### ✅ Task 2.2: Update Template Context Generation
- [ ] **File:** Template context generation code
- [ ] **Action:** Apply defaults during template generation, not RTL parsing
- [ ] **Implementation:**
  ```python
  def build_template_context(kernel_metadata: KernelMetadata) -> TemplateContext:
      # Apply whitelisted defaults
      applied_defaults = {}
      for param in kernel_metadata.parameters:
          if param.name in WHITELISTED_DEFAULTS:
              # Use RTL default if available, otherwise whitelist default
              applied_defaults[param.name] = param.default_value or WHITELISTED_DEFAULTS[param.name]
      
      return TemplateContext(
          parameter_definitions=kernel_metadata.parameters,
          whitelisted_defaults=applied_defaults,
          required_attributes=[p.name for p in kernel_metadata.parameters if p.name not in applied_defaults],
          # ... other context
      )
  ```

### ✅ Task 2.3: Update Generated Subclass Template
- [ ] **File:** HWCustomOp template (Jinja2 template file)
- [ ] **Action:** Update `__init__` method to use `get_nodeattr()` directly
- [ ] **Implementation:**
  ```python
  def __init__(self, onnx_node, **kwargs):
      # FINN extracts and sets attributes - we just collect them
      runtime_parameters = {}
      {% for param in parameter_definitions %}
      runtime_parameters["{{ param.name }}"] = self.get_nodeattr("{{ param.name }}")
      {% endfor %}
      
      super().__init__(
          onnx_node=onnx_node,
          interface_metadata=self.get_interface_metadata(),
          runtime_parameters=runtime_parameters,
          **kwargs
      )
  ```
- [ ] **Remove:** `_extract_runtime_parameters_from_onnx()` method

### ✅ Task 2.4: Update Node Attribute Definitions
- [ ] **File:** HWCustomOp template
- [ ] **Action:** Ensure all module parameters become ONNX node attributes
- [ ] **Implementation:**
  ```python
  def get_nodeattr_types(self) -> Dict[str, Tuple[str, bool, Any]]:
      attrs = {}
      {% for param in parameter_definitions %}
      {% if param.name in whitelisted_defaults %}
      attrs["{{ param.name }}"] = ("i", False, {{ whitelisted_defaults[param.name] }})
      {% else %}
      attrs["{{ param.name }}"] = ("i", True, None)  # Required
      {% endif %}
      {% endfor %}
      
      attrs.update(super().get_enhanced_nodeattr_types())
      return attrs
  ```

## Phase 3: Testing and Validation

### ✅ Task 3.1: Add BDimPragma Parameter Validation Tests
- [ ] **File:** `tests/tools/hw_kernel_gen/rtl_parser/test_new_bdim_pragma.py`
- [ ] **Action:** Add tests for parameter validation
- [ ] **Test Cases:**
  ```python
  def test_bdim_parameter_validation_error():
      """Test error when BDIM references unknown parameter."""
      # RTL with parameter PE, pragma uses UNKNOWN_PARAM
      with pytest.raises(PragmaError) as exc_info:
          # Parse RTL with invalid BDIM pragma
          pass
      assert "references unknown parameter 'UNKNOWN_PARAM'" in str(exc_info.value)
  
  def test_bdim_parameter_validation_success():
      """Test successful validation when parameter exists."""
      # RTL with parameter PE, pragma uses PE
      # Should parse without error
      pass
  ```

### ✅ Task 3.2: Add Template Generation Tests
- [ ] **File:** `tests/tools/hw_kernel_gen/template_generation/test_parameter_defaults.py`
- [ ] **Action:** Test default application during template generation
- [ ] **Test Cases:**
  ```python
  def test_whitelisted_defaults_applied():
      """Test that whitelisted parameters get defaults in template context."""
      pass
  
  def test_non_whitelisted_no_defaults():
      """Test that non-whitelisted parameters become required attributes."""
      pass
  
  def test_generated_subclass_nodeattr_types():
      """Test that generated code has correct node attribute definitions."""
      pass
  ```

### ✅ Task 3.3: Integration Testing
- [ ] **File:** `tests/tools/hw_kernel_gen/integration/test_parameter_integration.py`
- [ ] **Action:** End-to-end testing of parameter flow
- [ ] **Test Cases:**
  ```python
  def test_rtl_to_template_parameter_flow():
      """Test complete flow from RTL with parameters to generated subclass."""
      # 1. Parse RTL with parameters and BDIM pragmas
      # 2. Generate template context
      # 3. Verify template context has correct defaults and requirements
      # 4. Generate subclass code
      # 5. Verify subclass has correct node attributes
      pass
  ```

### ✅ Task 3.4: Existing Test Updates
- [ ] **Files:** All existing BDIM and template generation tests
- [ ] **Action:** Update tests to work with new parameter validation
- [ ] **Verify:** All 44+ existing tests still pass

## Phase 4: Documentation and Configuration

### ✅ Task 4.1: Update Design Documentation
- [ ] **File:** `ai_cache/designs/RTL_TO_TEMPLATE_FLOW_DESIGN_V2.md`
- [ ] **Status:** ✅ **COMPLETED** - Updated with corrected understanding
- [ ] **Action:** Document parameter whitelist and validation rules

### ✅ Task 4.2: Add Configuration Documentation
- [ ] **File:** Project documentation or config file
- [ ] **Action:** Document whitelisted parameters and rationale
- [ ] **Content:**
  ```markdown
  ## Parameter Defaults Whitelist
  
  Only these parameters can have default values:
  - PE: Processing Element count (default: 1)
  - SIMD: SIMD parallelism (default: 1)  
  - PARALLEL: General parallelism (default: 1)
  - WIDTH: Data width (default: 8)
  
  All other parameters must be provided by FINN as ONNX node attributes.
  ```

### ✅ Task 4.3: CLI Help Updates
- [ ] **File:** CLI help text and documentation
- [ ] **Action:** Update to reflect new parameter handling
- [ ] **Content:** Explain that parameter values are set by FINN, not during template generation

## Implementation Notes

### Critical Dependencies
1. **Parser Context Access:** BDimPragma needs access to module parameters during parsing
2. **Template Engine:** Ensure Jinja2 templates can access parameter lists and defaults
3. **FINN Integration:** Verify FINN's `get_nodeattr()` behavior matches expectations

### Potential Issues
1. **Circular Dependencies:** BDimPragma validating parameters may create dependency cycles
2. **Parser Refactoring:** May need to modify parser to pass parameters to pragmas
3. **Template Complexity:** Generated code may become more complex with parameter handling

### Success Criteria
- [ ] **Validation Works:** BDimPragma rejects unknown parameters immediately
- [ ] **Defaults Applied:** Template generation correctly applies whitelisted defaults
- [ ] **FINN Compatible:** Generated subclasses work with FINN's node creation workflow
- [ ] **Case Sensitive:** Parameter names preserved exactly from RTL to ONNX attributes
- [ ] **All Tests Pass:** No regressions in existing functionality

## Rollback Plan

If implementation faces blockers:
1. **Revert to Current:** Keep existing parameter resolution bridge
2. **Defer Validation:** Implement BDimPragma validation later
3. **Simplify Defaults:** Use fixed defaults instead of whitelisted approach

## Definition of Done

- [ ] BDimPragma validates parameter existence during parsing
- [ ] Template generation applies whitelisted defaults only
- [ ] Generated subclass code uses `get_nodeattr()` for parameter extraction
- [ ] All module parameters become ONNX node attributes
- [ ] Comprehensive test coverage for new functionality
- [ ] Documentation updated with new parameter handling approach
- [ ] No regressions in existing tests (44+ tests still pass)