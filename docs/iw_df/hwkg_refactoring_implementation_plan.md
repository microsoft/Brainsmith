# HWKG Refactoring Implementation Plan

## Overview
Integrate Phase 3 enhanced generators into HWKG by replacing inline generation methods with direct generator calls. Simple, clean integration without over-engineering.

## Step-by-Step Implementation

### Step 1: Rename HWCustomOpGenerator to HWCustomOpGenerator

#### 1.1 Update the Generator Class
**File:** `brainsmith/tools/hw_kernel_gen/generators/hw_custom_op_generator.py`

**Changes:**
```python
# Line 57: Update class name
class HWCustomOpGenerator:  # Changed from HWCustomOpGenerator
    """
    Phase 3 HWCustomOp generator with enhanced TDIM pragma integration.
    
    This is the primary HWCustomOp generator, replacing all previous implementations.
    Features:
    - Enhanced TDIM pragma support with parameter validation
    - Slim template generation (68% code reduction)
    - Automatic chunking strategy generation
    - AXI interface type classification
    """
```

#### 1.2 Update Function Name in Same File
**File:** `brainsmith/tools/hw_kernel_gen/generators/hw_custom_op_generator.py`

**Changes:**
```python
# Line 285: Update function name
def create_hwcustomop(rtl_file: Path, output_dir: Path, class_name: Optional[str] = None) -> Path:
    # Changed from create_slim_hwcustomop
    
    # Line 304: Update generator instantiation
    generator = HWCustomOpGenerator()  # Changed from HWCustomOpGenerator()
```

#### 1.3 Update Test Imports
**File:** `tests/integration/test_end_to_end_thresholding.py`

**Changes:**
```python
# Line 878: Update import
from brainsmith.tools.hw_kernel_gen.generators.hw_custom_op_generator import HWCustomOpGenerator

# Line 970: Update instantiation
generator = HWCustomOpGenerator()  # Changed from HWCustomOpGenerator()
```

### Step 2: Update HWKG to Use the Generator

#### 2.1 Add Generator Getter Method
**File:** `brainsmith/tools/hw_kernel_gen/hkg.py`

**Add after line 131 (after dataflow initialization):**
```python
# Generator instances (lazy initialization)
self._hw_custom_op_generator = None
```

**Add new method after line 257 (_build_dataflow_model method):**
```python
def _get_hw_custom_op_generator(self):
    """Get HWCustomOp generator instance with lazy initialization."""
    if self._hw_custom_op_generator is None:
        try:
            from .generators.hw_custom_op_generator import HWCustomOpGenerator
            self._hw_custom_op_generator = HWCustomOpGenerator()
        except ImportError as e:
            raise HardwareKernelGeneratorError(f"Could not import HWCustomOpGenerator: {e}")
    return self._hw_custom_op_generator
```

#### 2.2 Replace _generate_hw_custom_op Method
**File:** `brainsmith/tools/hw_kernel_gen/hkg.py`

**Replace the entire `_generate_hw_custom_op` method (lines 258-286):**
```python
def _generate_hw_custom_op(self):
    """
    Generate HWCustomOp using Phase 3 enhanced generator.
    
    Replaces inline generation with direct call to HWCustomOpGenerator.
    """
    if not self.hw_kernel_data:
        raise HardwareKernelGeneratorError("Cannot generate HWCustomOp: RTL data not parsed.")
        
    if not self.dataflow_enabled:
        raise HardwareKernelGeneratorError(
            "HWCustomOp generation requires dataflow framework. "
            "Please ensure brainsmith.dataflow is available."
        )
        
    print("--- Generating HWCustomOp Instance ---")
    
    # Get generator instance
    generator = self._get_hw_custom_op_generator()
    
    # Prepare output path
    class_name = generate_class_name(self.hw_kernel_data.name)
    output_file = self.output_dir / f"{class_name.lower()}.py"
    
    # Generate using Phase 3 enhanced generator
    try:
        generated_code = generator.generate_hwcustomop(
            hw_kernel=self.hw_kernel_data,
            output_path=output_file,
            class_name=class_name,
            source_file=str(self.rtl_file_path.name)
        )
        
        self.generated_files["hw_custom_op"] = output_file
        print(f"HWCustomOp generation complete. Output: {output_file}")
        return output_file
        
    except Exception as e:
        raise HardwareKernelGeneratorError(f"HWCustomOp generation failed: {e}")
```

#### 2.3 Remove Deprecated Methods
**File:** `brainsmith/tools/hw_kernel_gen/hkg.py`

**Delete these entire methods:**
1. `_generate_auto_hwcustomop_with_dataflow` (lines 288-328)
2. `_build_enhanced_template_context` (lines 330-377)

#### 2.4 Update generate_auto_hwcustomop Method
**File:** `brainsmith/tools/hw_kernel_gen/hkg.py`

**Replace `generate_auto_hwcustomop` method (lines 379-430):**
```python
def generate_auto_hwcustomop(self, template_path: str, output_path: str) -> str:
    """
    Public method for generating AutoHWCustomOp with Phase 3 enhancements.
    
    Args:
        template_path: Path to Jinja2 template file (for compatibility)
        output_path: Output file path for generated class
        
    Returns:
        Path to generated file
        
    Raises:
        HardwareKernelGeneratorError: If generation fails
    """
    if not self.dataflow_enabled:
        raise HardwareKernelGeneratorError("AutoHWCustomOp generation requires dataflow framework")
        
    if not self.hw_kernel_data:
        self._parse_rtl()
        
    if not self.dataflow_model:
        self._build_dataflow_model()
        
    try:
        # Get generator and generate
        generator = self._get_hw_custom_op_generator()
        
        # Extract class name from output path
        output_file = Path(output_path)
        class_name = output_file.stem.replace('_', '').title() + 'HWCustomOp'
        
        generated_code = generator.generate_hwcustomop(
            hw_kernel=self.hw_kernel_data,
            output_path=output_file,
            class_name=class_name,
            source_file=str(self.rtl_file_path.name)
        )
        
        print(f"AutoHWCustomOp generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        raise HardwareKernelGeneratorError(f"AutoHWCustomOp generation failed: {e}")
```

### Step 3: Update Other Generator Methods (Future Pattern)

#### 3.1 Update RTL Backend Method
**File:** `brainsmith/tools/hw_kernel_gen/hkg.py`

**Replace `_generate_rtl_backend` method (lines 433-455) to follow the same pattern:**
```python
def _generate_rtl_backend(self):
    """
    Generate RTL backend using dedicated generator (future implementation).
    
    Currently generates placeholder backend with dataflow support.
    """
    if not self.hw_kernel_data:
        raise HardwareKernelGeneratorError("Cannot generate RTL backend: RTL data not parsed.")
        
    print("--- Generating RTL Backend ---")
    
    # TODO: Replace with dedicated RTLBackendGenerator when implemented
    # For now, use the existing inline implementation
    output_path = self._generate_auto_rtlbackend_with_dataflow()
        
    self.generated_files["rtl_backend"] = output_path
    print(f"RTL Backend generation complete. Output: {output_path}")
```

#### 3.2 Update Imports
**File:** `brainsmith/tools/hw_kernel_gen/hkg.py`

**Update the commented imports (lines 22-24):**
```python
from .generators.rtl_template_generator import generate_rtl_template
# NOTE: HWCustomOp generation now uses HWCustomOpGenerator class
# from .generators.rtl_backend_generator import RTLBackendGenerator  # Future
# from .generators.doc_generator import DocumentationGenerator  # Future
```

### Step 4: Update Documentation and Examples

#### 4.1 Update Phase 3 Architecture Documentation
**File:** `docs/iw_df/phase3_end_to_end_architecture.md`

**Update section starting around line 105:**
```markdown
### 3. Template System and Code Generation

#### Slim Template Generation
- **HWCustomOpGenerator** (renamed from HWCustomOpGenerator)
- Integrated directly into HWKG orchestrator
- Phase 3 enhanced TDIM pragma support
- 68% code reduction vs traditional templates
```

#### 4.2 Update Examples
**File:** `examples/phase3_enhanced_tdim_demo.py`

**Update import (if exists):**
```python
from brainsmith.tools.hw_kernel_gen.generators.hw_custom_op_generator import HWCustomOpGenerator
```

### Step 5: Testing and Validation

#### 5.1 Update Test Validation
**File:** `tests/integration/test_end_to_end_thresholding.py`

**Update assertions in test methods to use new class name:**
```python
# In test methods, ensure assertions reference HWCustomOpGenerator
assert "HWCustomOpGenerator" in str(type(generator))
```

#### 5.2 Run Tests
Execute the following tests to verify integration:

```bash
# Test Phase 3 functionality
python -m pytest tests/integration/test_end_to_end_thresholding.py::TestEndToEndThresholding::test_generate_phase3_enhanced_hwcustomop_subclass -v

# Test slim template generation
python -m pytest tests/integration/test_end_to_end_thresholding.py::TestEndToEndThresholding::test_generate_phase3_slim_template_hwcustomop -v

# Test full end-to-end pipeline
python -m pytest tests/integration/test_end_to_end_thresholding.py -k "phase3" -v
```

## Implementation Checklist

### Phase 1: Rename and Update Generator ✓
- [ ] Rename `HWCustomOpGenerator` to `HWCustomOpGenerator` in class definition
- [ ] Update function name `create_slim_hwcustomop` to `create_hwcustomop`
- [ ] Update test imports and instantiations
- [ ] Update docstrings and comments

### Phase 2: Integrate into HWKG ✓
- [ ] Add `_hw_custom_op_generator` instance variable
- [ ] Add `_get_hw_custom_op_generator()` method
- [ ] Replace `_generate_hw_custom_op()` method
- [ ] Delete deprecated methods (`_generate_auto_hwcustomop_with_dataflow`, `_build_enhanced_template_context`)
- [ ] Update `generate_auto_hwcustomop()` method
- [ ] Update imports and comments

### Phase 3: Clean Up and Document ✓
- [ ] Update documentation files
- [ ] Update examples and demos
- [ ] Run and verify tests
- [ ] Update architecture diagrams

## Expected Results

After implementation:

1. **HWKG Integration**: HWKG will call HWCustomOpGenerator directly instead of duplicating logic
2. **Phase 3 Features**: All Phase 3 enhanced TDIM pragma features work through HWKG
3. **Clean Architecture**: Clear separation between orchestration (HWKG) and generation (HWCustomOpGenerator)
4. **Maintainability**: Easier to extend with future generators following the same pattern
5. **Test Compatibility**: All existing tests pass with updated imports

## Files Modified Summary

1. `brainsmith/tools/hw_kernel_gen/generators/hw_custom_op_generator.py` - Rename class
2. `brainsmith/tools/hw_kernel_gen/hkg.py` - Major refactoring of generation methods
3. `tests/integration/test_end_to_end_thresholding.py` - Update imports
4. `docs/iw_df/phase3_end_to_end_architecture.md` - Update documentation
5. `examples/phase3_enhanced_tdim_demo.py` - Update imports (if exists)

Total estimated time: **90 minutes**