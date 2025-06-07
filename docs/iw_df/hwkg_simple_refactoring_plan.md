# HWKG Simple Refactoring Plan

## Current Problem

The HWKG class bypasses the new Phase 3 `HWCustomOpGenerator` and duplicates generation logic inline. We need to integrate the enhanced generators directly without over-engineering.

## Simple Solution

### 1. Rename and Integrate Generators

**Rename HWCustomOpGenerator to HWCustomOpGenerator** - it's now the primary (and only) HWCustomOp generator.

**Direct Integration Approach:**
- HWKG methods call generator classes directly
- No registry, no complex configuration
- Simple, clean integration with existing generators

### 2. Refactored Generator Structure

```
brainsmith/tools/hw_kernel_gen/generators/
├── rtl_template_generator.py     # Existing - RTL wrapper generation
├── hw_custom_op_generator.py     # Renamed from HWCustomOpGenerator 
├── rtl_backend_generator.py      # Future - RTL backend generation
├── test_suite_generator.py       # Future - Test generation
└── documentation_generator.py    # Future - Documentation generation
```

### 3. Updated HWKG Implementation

Replace inline generation methods with direct generator calls:

```python
class HardwareKernelGenerator:
    def __init__(self, ...):
        # ... existing initialization ...
        
        # Initialize generators
        self.rtl_template_generator = None  # Lazy init
        self.hw_custom_op_generator = None  # Lazy init
    
    def _get_rtl_template_generator(self):
        """Get RTL template generator instance."""
        if self.rtl_template_generator is None:
            # Import here to avoid circular imports
            from .generators.rtl_template_generator import generate_rtl_template
            self.rtl_template_generator = generate_rtl_template
        return self.rtl_template_generator
    
    def _get_hw_custom_op_generator(self):
        """Get HWCustomOp generator instance."""
        if self.hw_custom_op_generator is None:
            from .generators.hw_custom_op_generator import HWCustomOpGenerator
            self.hw_custom_op_generator = HWCustomOpGenerator()
        return self.hw_custom_op_generator
    
    def _generate_rtl_template(self):
        """Generate RTL wrapper template using dedicated generator."""
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot generate RTL template: RTL data not parsed.")
        
        print("--- Generating RTL Template ---")
        generator = self._get_rtl_template_generator()
        output_path = generator(self.hw_kernel_data, self.output_dir)
        self.generated_files["rtl_template"] = output_path
        print(f"RTL Template generation complete. Output: {output_path}")
    
    def _generate_hw_custom_op(self):
        """Generate HWCustomOp using Phase 3 enhanced generator."""
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot generate HWCustomOp: RTL data not parsed.")
        
        print("--- Generating HWCustomOp Instance ---")
        
        # Get generator instance
        generator = self._get_hw_custom_op_generator()
        
        # Prepare output path
        class_name = generate_class_name(self.hw_kernel_data.name)
        output_file = self.output_dir / f"{class_name.lower()}.py"
        
        # Generate using Phase 3 enhanced generator
        generated_code = generator.generate_hwcustomop(
            hw_kernel=self.hw_kernel_data,
            output_path=output_file,
            class_name=class_name,
            source_file=str(self.rtl_file_path.name)
        )
        
        self.generated_files["hw_custom_op"] = output_file
        print(f"HWCustomOp generation complete. Output: {output_file}")
        return output_file
```

### 4. Remove Inline Generation Methods

**Delete these methods from HWKG:**
- `_generate_auto_hwcustomop_with_dataflow()`
- `_build_enhanced_template_context()`
- All inline Jinja2 template handling in HWKG

**Keep these simplified:**
- `_generate_hw_custom_op()` - calls HWCustomOpGenerator
- `_generate_rtl_template()` - calls existing function
- Future methods for other generators

### 5. Enhanced HWCustomOpGenerator Integration

The renamed `HWCustomOpGenerator` (formerly HWCustomOpGenerator) needs minimal changes:

```python
# In hw_custom_op_generator.py
class HWCustomOpGenerator:  # Renamed from HWCustomOpGenerator
    """
    Phase 3 HWCustomOp generator with enhanced TDIM pragma integration.
    
    This is the primary HWCustomOp generator, replacing all previous implementations.
    Features:
    - Enhanced TDIM pragma support with parameter validation
    - Slim template generation (68% code reduction)
    - Automatic chunking strategy generation
    - AXI interface type classification
    """
    
    def generate_hwcustomop(self, hw_kernel: HWKernel, output_path: Path, 
                           class_name: Optional[str] = None, source_file: str = "unknown.sv") -> str:
        """
        Generate enhanced HWCustomOp class from parsed RTL data.
        
        This method integrates:
        - Phase 3 enhanced TDIM pragma parsing
        - Automatic interface classification (AXI_STREAM -> INPUT/OUTPUT)
        - Slim template generation with embedded chunking strategies
        - Parameter validation and constraint enforcement
        """
        # ... existing implementation ...
```

### 6. Future Generator Integration Pattern

For future generators, follow the same simple pattern:

```python
def _generate_rtl_backend(self):
    """Generate RTL backend using dedicated generator."""
    if not self.hw_kernel_data:
        raise HardwareKernelGeneratorError("Cannot generate RTL backend: RTL data not parsed.")
    
    print("--- Generating RTL Backend ---")
    
    from .generators.rtl_backend_generator import RTLBackendGenerator
    generator = RTLBackendGenerator()
    
    class_name = generate_class_name(self.hw_kernel_data.name)
    output_file = self.output_dir / f"{class_name.lower()}_rtlbackend.py"
    
    output_path = generator.generate_backend(
        hw_kernel=self.hw_kernel_data,
        output_path=output_file,
        dataflow_model=self.dataflow_model
    )
    
    self.generated_files["rtl_backend"] = output_path
    print(f"RTL Backend generation complete. Output: {output_path}")
```

## Implementation Steps

### Step 1: Rename and Clean Up (30 minutes)
1. **Rename HWCustomOpGenerator to HWCustomOpGenerator**
   ```bash
   mv brainsmith/tools/hw_kernel_gen/generators/hw_custom_op_generator.py \
      brainsmith/tools/hw_kernel_gen/generators/hw_custom_op_generator_old.py
   # Update class name and references
   ```

2. **Update all imports and references**
   - Test files
   - Documentation
   - Examples

### Step 2: Integrate into HWKG (45 minutes)
1. **Replace `_generate_hw_custom_op()` method**
   - Remove inline template handling
   - Add direct generator call
   - Maintain same interface

2. **Add generator getter method**
   - Lazy initialization
   - Clean error handling

3. **Remove deprecated methods**
   - `_generate_auto_hwcustomop_with_dataflow()`
   - `_build_enhanced_template_context()`

### Step 3: Test Integration (15 minutes)
1. **Run existing tests**
   - Update test imports
   - Verify same outputs
   - Check Phase 3 features

2. **Test end-to-end pipeline**
   - Verify HWKG works with new generator
   - Check generated files match expectations

## Benefits of Simple Approach

### 1. **Minimal Changes**
- No complex registry or configuration system
- Direct integration with existing HWKG structure
- Maintains existing API compatibility

### 2. **Clear Separation**
- Generation logic in dedicated generator classes
- HWKG focuses on orchestration
- Easy to test generators independently

### 3. **Easy Extension**
- Simple pattern for adding new generators
- No framework overhead
- Clear file organization

### 4. **Maintainable**
- Fewer moving parts
- Clear dependencies
- Straightforward debugging

## Files to Modify

### 1. Rename Generator
```
brainsmith/tools/hw_kernel_gen/generators/hw_custom_op_generator.py
# - Rename HWCustomOpGenerator -> HWCustomOpGenerator
# - Update docstrings and comments
```

### 2. Update HWKG
```
brainsmith/tools/hw_kernel_gen/hkg.py
# - Replace _generate_hw_custom_op() method
# - Add _get_hw_custom_op_generator() method
# - Remove deprecated inline methods
```

### 3. Update Tests
```
tests/integration/test_end_to_end_thresholding.py
# - Update imports to use new class name
# - Verify Phase 3 functionality
```

### 4. Update Documentation
```
docs/iw_df/phase3_end_to_end_architecture.md
# - Update class names and integration flow
```

This simple refactoring achieves the goal of integrating Phase 3 enhanced generators into HWKG without over-engineering. The result is a clean, maintainable architecture that's easy to extend.