# Test Cleanup Patterns Analysis

## Summary of Issues Found

### 1. Outdated Import Patterns

The following test files are using outdated imports:

#### `test_kernel_model_validation.py`
- Line 16: `from brainsmith.core.dataflow.interface_model import InterfaceModel` - InterfaceModel no longer exists
- Line 32, 37: Missing `InterfaceDirection` import - this enum has been removed
- Line 33, 37: Using `DataType.from_string()` - should use `datatype_from_string()` from qonnx_types
- Line 35, 37, 126, 157, 307: Using `InputDefinition` with `direction=InterfaceDirection.OUTPUT` - should use `OutputDefinition` instead

#### `test_sdim_migration.py`
- Contains correct imports but has commented code showing migration patterns
- No immediate issues but could be cleaned up

### 2. Missing Values/Syntax Errors

#### `test_tiling_functions.py`
- Line 261: `direction=,` - missing value after equals
- Line 278: `direction=,` - missing value after equals

### 3. Files Using InterfaceDirection (No Longer Exists)

The following files reference `InterfaceDirection` which has been removed:
- `test_function_based_examples.py` (lines 43, 51, 110, 117)
- `test_integration_block_stream_dims.py` (lines 40, 47, 111, 119, 399)
- `test_interface_definition_dims.py` (line 226)
- `test_kernel_model_validation.py` (lines 37, 126, 157, 307)

### 4. Pattern Changes Required

1. **InterfaceDefinition → InputDefinition/OutputDefinition**
   - Already correctly named in most places

2. **InterfaceModel → InputInterface/OutputInterface**
   - Need to update imports in `test_kernel_model_validation.py`

3. **InterfaceDirection.INPUT → Use InputDefinition**
   - InterfaceDirection.WEIGHT → Use InputDefinition with is_weight=True
   - InterfaceDirection.OUTPUT → Use OutputDefinition

4. **DataType.from_string() → datatype_from_string()**
   - Import from qonnx_types module

### 5. Recommended Fixes

1. Remove all references to `InterfaceDirection`
2. Replace `InterfaceModel` imports with `InputInterface`/`OutputInterface`
3. Update `DataType.from_string()` calls to use `datatype_from_string()`
4. Fix missing values in `test_tiling_functions.py`
5. Update interface creation patterns:
   ```python
   # OLD
   InputDefinition(name="out", direction=InterfaceDirection.OUTPUT, ...)
   
   # NEW
   OutputDefinition(name="out", ...)
   ```

### 6. Files That Need Updates

Priority files to fix:
1. `test_kernel_model_validation.py` - Multiple issues
2. `test_tiling_functions.py` - Syntax errors
3. `test_function_based_examples.py` - InterfaceDirection usage
4. `test_integration_block_stream_dims.py` - InterfaceDirection usage
5. `test_interface_definition_dims.py` - InterfaceDirection usage

## Next Steps

These test files need to be updated to match the new architecture where:
- Direction is implicit in the class name (InputDefinition vs OutputDefinition)
- InterfaceModel is split into InputInterface and OutputInterface
- DataType handling uses the qonnx_types module functions