# ✅ **BrainSmith API Simplification - Implementation Checklist**
## Detailed Task List for Code Review Response

---

## 📋 **Phase 1: Core API Simplification (CRITICAL)**

### **File: `brainsmith/core/api.py`**

#### **❌ Remove Legacy Functions**
- [ ] Delete `brainsmith_explore()` function (lines 16-83)
- [ ] Delete `brainsmith_roofline()` function (lines 85-103)  
- [ ] Delete `brainsmith_dataflow_analysis()` function (lines 105-123)
- [ ] Delete `brainsmith_generate()` function (lines 125-143)
- [ ] Delete `brainsmith_workflow()` function (lines 440-464)
- [ ] Delete `explore_design_space()` legacy wrapper (lines 146-190)
- [ ] Delete all `_route_to_existing_legacy_system()` related code (lines 385-394)
- [ ] Delete `_convert_to_legacy_format()` function (lines 396-403)
- [ ] Delete `_create_legacy_error_result()` function (lines 405-413)
- [ ] Delete `_create_mock_blueprint()` function (lines 423-437)

#### **✅ Add New Core Function**
- [ ] Implement `forge()` function with signature:
  ```python
  def forge(
      model_path: str,
      blueprint_path: str,
      objectives: Dict[str, Any] = None,
      constraints: Dict[str, Any] = None,
      target_device: str = None,
      is_hw_graph: bool = False,
      build_core: bool = True,
      output_dir: Optional[str] = None
  ) -> Dict[str, Any]:
  ```

#### **🔧 Update Blueprint Loading**
- [ ] Modify `_load_and_validate_blueprint()` to throw hard error (no mock fallback)
- [ ] Remove ImportError handling that creates mock blueprints
- [ ] Add comprehensive error messages for blueprint validation failures

#### **🛠️ Add New Helper Functions**
- [ ] Implement `_validate_inputs()` function
- [ ] Implement `_setup_dse_configuration()` function  
- [ ] Implement `_run_full_dse()` function
- [ ] Implement `_run_hw_optimization_dse()` function
- [ ] Implement `_generate_dataflow_core()` function
- [ ] Implement `_assemble_results()` function
- [ ] Implement `_save_forge_results()` function

### **File: `brainsmith/__init__.py`**

#### **❌ Remove Legacy Imports**
- [ ] Remove lines 39-43:
  ```python
  from .core.api import (
      brainsmith_explore, brainsmith_roofline, brainsmith_dataflow_analysis,
      brainsmith_generate, brainsmith_workflow, validate_blueprint,
      explore_design_space  # Legacy compatibility wrapper
  )
  ```

#### **✅ Add New Core Import**
- [ ] Add new import:
  ```python
  from .core.api import forge, validate_blueprint
  ```

#### **❌ Remove Legacy Functions**
- [ ] Delete `build_model()` function (lines 88-122)
- [ ] Delete `optimize_model()` function (lines 125-142)
- [ ] Remove these from `__all__` list (lines 343-396)

#### **✅ Update Exports**
- [ ] Update `__all__` list to include only:
  ```python
  __all__ = [
      # Core toolchain
      'forge',
      'validate_blueprint',
      
      # Core data structures  
      'DesignSpace',
      'DesignPoint',
      'ParameterDefinition',
      
      # Blueprint system
      'Blueprint',
      'load_blueprint', 
      'list_blueprints',
      
      # DSE system
      'DSEInterface',
      'DSEAnalyzer',
      'ParetoAnalyzer',
      
      # Supplementary tools
      'roofline_analysis',
      'RooflineProfiler'
  ]
  ```

#### **🔄 Update Version**
- [ ] Change `__version__ = "0.4.0"` to `"0.5.0"`
- [ ] Update `__description__` to reflect simplified API

---

## 📋 **Phase 2: Tools Interface Migration (HIGH)**

### **File: `brainsmith/tools/__init__.py`**

#### **✅ Create Tools Interface**
- [ ] Create new file if it doesn't exist
- [ ] Add imports:
  ```python
  from .profiling import roofline_analysis, RooflineProfiler
  from .hw_kernel_gen import generate_hw_kernel
  
  __all__ = [
      'roofline_analysis',
      'RooflineProfiler',
      'generate_hw_kernel'
  ]
  ```

### **File: `brainsmith/tools/profiling/__init__.py`**

#### **✅ Create Profiling Interface**  
- [ ] Create new file
- [ ] Implement `RooflineProfiler` class
- [ ] Implement `roofline_analysis()` wrapper function
- [ ] Import existing roofline functionality

### **File: `brainsmith/tools/profiling/roofline.py`**

#### **🔧 Update Existing Roofline**
- [ ] No changes needed to existing implementation
- [ ] Verify import paths work correctly

### **File: `brainsmith/tools/profiling/model_profiling.py`**

#### **🔧 Update Model Profiling**
- [ ] No changes needed to existing implementation
- [ ] Verify import paths work correctly

---

## 📋 **Phase 3: Documentation Updates (HIGH)**

### **File: `README.md`**

#### **✅ Update Main Documentation**
- [ ] Replace old API examples with `forge()` usage
- [ ] Add section on supplementary tools
- [ ] Update installation and quick start guides
- [ ] Add migration section for existing users

### **File: `docs/MIGRATION_GUIDE.md`**

#### **✅ Create Migration Guide**
- [ ] Create new migration guide document
- [ ] Map old functions to new equivalents
- [ ] Provide code examples for each migration case
- [ ] List breaking changes and workarounds

### **Update Example Files**

#### **File: `demos/bert/end2end_bert.py`**
- [ ] Replace API calls with `forge()` function
- [ ] Update import statements
- [ ] Test functionality with new API

#### **File: `examples/` directory**
- [ ] Update all example scripts to use new API
- [ ] Replace roofline imports with tools module
- [ ] Add new examples showing `forge()` usage patterns

---

## 📋 **Phase 4: Testing Updates (CRITICAL)**

### **Create New Test Files**

#### **File: `tests/test_forge_api.py`**
- [ ] Create comprehensive tests for `forge()` function
- [ ] Test all parameter combinations
- [ ] Test error conditions and validation
- [ ] Test both execution paths (`is_hw_graph=True/False`)
- [ ] Test output directory handling

#### **File: `tests/test_tools_interface.py`**
- [ ] Create tests for tools module
- [ ] Test `RooflineProfiler` class
- [ ] Test `roofline_analysis()` function
- [ ] Verify tools are separate from core toolchain

### **Update Existing Test Files**

#### **File: `tests/functional/api/test_highlevel_api.py`**
- [ ] Remove tests for deleted functions
- [ ] Update remaining tests to use `forge()`
- [ ] Add tests for new input validation
- [ ] Add tests for blueprint hard error validation

#### **File: `tests/test_api.py`**
- [ ] Remove legacy API tests
- [ ] Add `forge()` function tests
- [ ] Update import tests

---

## 📋 **Phase 5: Cleanup & Validation (MEDIUM)**

### **Remove Unused Files**

#### **Check for Orphaned Code**
- [ ] Search for references to deleted functions across codebase
- [ ] Remove any unused helper functions
- [ ] Clean up imports that are no longer needed
- [ ] Remove legacy compatibility modules if they exist

### **File: `brainsmith/core/legacy_support.py`**
- [ ] Review if this file is still needed
- [ ] Remove if no longer referenced
- [ ] Update if it serves other purposes

### **Update Type Hints**
- [ ] Ensure all new functions have proper type hints
- [ ] Update existing type hints where needed
- [ ] Add imports for typing modules

---

## 📋 **Phase 6: Integration Testing (CRITICAL)**

### **End-to-End Testing**
- [ ] Test complete workflow: `forge()` with ONNX model and blueprint
- [ ] Test checkpoint mode: `build_core=False`
- [ ] Test hardware graph mode: `is_hw_graph=True`
- [ ] Test output directory functionality
- [ ] Test error handling for invalid inputs

### **Tools Testing**
- [ ] Test roofline analysis import from tools module
- [ ] Verify tools work independently of core API
- [ ] Test backwards compatibility of roofline functionality

### **Blueprint Validation Testing**
- [ ] Test hard error on invalid blueprint
- [ ] Test hard error on missing blueprint
- [ ] Verify no fallback to mock blueprints
- [ ] Test comprehensive error messages

---

## 📋 **Phase 7: Performance Validation (MEDIUM)**

### **Benchmark New API**
- [ ] Compare performance of `forge()` vs old API
- [ ] Measure memory usage
- [ ] Test with various model sizes
- [ ] Verify no performance regressions

### **Load Testing**
- [ ] Test with multiple concurrent `forge()` calls
- [ ] Test memory cleanup between calls
- [ ] Verify resource management

---

## 🎯 **Success Criteria Checklist**

- [ ] **Single Core Function**: Only `forge()` remains in core API
- [ ] **Tools Separation**: Roofline analysis moved to `brainsmith.tools`
- [ ] **Hard Blueprint Errors**: No mock blueprint fallbacks
- [ ] **No Legacy Code**: All unused legacy interfaces removed
- [ ] **Preserved Functionality**: Core DSE and generation capabilities work
- [ ] **Updated Documentation**: Complete docs and migration guide
- [ ] **Comprehensive Testing**: All functionality tested with new API
- [ ] **Performance Validation**: No regressions in performance
- [ ] **Clean Codebase**: No orphaned code or broken references

---

## 📊 **Implementation Order Priority**

### **Week 1 (CRITICAL)**
1. Remove legacy functions from `api.py`
2. Implement basic `forge()` function
3. Update blueprint validation (hard error)
4. Update `__init__.py` imports and exports

### **Week 2 (HIGH)**  
1. Create tools interface structure
2. Move roofline to tools module
3. Create comprehensive `forge()` implementation
4. Basic testing of new API

### **Week 3 (MEDIUM)**
1. Update all documentation
2. Create migration guide
3. Update examples and demos
4. Comprehensive testing

### **Week 4 (LOW)**
1. Performance validation
2. Final cleanup
3. Integration testing
4. Release preparation

This checklist provides a complete task-by-task breakdown for implementing the code review feedback and can be used to track progress through the implementation process.