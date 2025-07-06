# Phase 1 Simplification Migration Guide

This guide helps users migrate from the previous validation-heavy Phase 1 to the new simplified version.

## Philosophy Change

**Before**: Comprehensive validation with helpful error messages  
**After**: Trust engineers, let Python errors speak for themselves

## What Was Removed

### 1. Validation Methods (All Removed)
- `_validate_field_type()` - Type checking with helpful messages
- `_validate_required_field()` - Required field validation  
- `_get_field_with_validation()` - Validated field extraction
- `_validate_transforms()` - Transform existence checking
- `_validate_kernel_exists()` - Kernel plugin validation
- `_validate_backends_for_kernel()` - Backend compatibility checks

### 2. Error Context (All Removed)
- Line/column tracking in errors
- Available plugin suggestions
- Helpful error messages
- Custom error formatting

### 3. Features Removed
- `forge_optimized()` method
- Plugin optimization/caching
- Validation cache
- Warning system
- Deprecated field warnings
- Logging and summaries

## What Errors You'll See Now

### Invalid Blueprint Fields
**Before**: `BlueprintParseError: hw_compiler.kernels must be a list, got str`  
**After**: `KeyError: 'kernels'` or `AttributeError`

### Missing Required Fields
**Before**: `BlueprintParseError: search.constraints[0].metric is required`  
**After**: `KeyError: 'metric'`

### Invalid Enum Values
**Before**: `BlueprintParseError: Unknown search.strategy: random. Supported: ['exhaustive', 'adaptive']`  
**After**: `ValueError: 'random' is not a valid SearchStrategy`

### Missing Plugins
**Before**: `PluginNotFoundError: Kernel 'MyKernel' not found. Available: ['LayerNorm', 'MatMul', ...]`  
**After**: Empty list returned from registry, may fail in Phase 2

### Invalid Types
**Before**: `BlueprintParseError: global.timeout_minutes must be an integer, got str`  
**After**: `TypeError` during processing

## Minimal Validation Remaining

Only 2 checks remain in the validator:
1. **Model file exists** - `ValidationError: Model file not found: path/to/model.onnx`
2. **Total combinations < limit** - `ValidationError: Too many combinations: 2000000 > 1000000`

## Migration Steps

### 1. Remove Error Handling
If your code catches specific blueprint errors:
```python
# Before
try:
    design_space = forge(model_path, blueprint_path)
except PluginNotFoundError as e:
    print(f"Plugin issue: {e}")
    # Show available plugins from error

# After  
try:
    design_space = forge(model_path, blueprint_path)
except KeyError as e:
    print(f"Missing field: {e}")
except ValueError as e:
    print(f"Invalid value: {e}")
```

### 2. Update API Usage
```python
# Before
forge_api = ForgeAPI(verbose=True)
design_space = forge_api.forge_optimized(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml",
    optimize_plugins=True
)

# After
design_space = forge("model.onnx", "blueprint.yaml")
```

### 3. Debug Raw Errors
When you get Python errors, check:
- `KeyError` → Missing field in blueprint
- `ValueError` → Invalid enum or value  
- `AttributeError` → Wrong structure/type
- `TypeError` → Type mismatch

## Benefits

1. **Faster** - No validation overhead
2. **Simpler** - 40% less code to maintain
3. **Clearer** - Direct Python errors are unambiguous
4. **Fearless** - No hand-holding for engineers

## Philosophy

> "We're building for engineers, not end users. Perfect Code is fearless code."

If your blueprint is invalid, Python will tell you exactly what's wrong through standard exceptions. This is intentional and good.