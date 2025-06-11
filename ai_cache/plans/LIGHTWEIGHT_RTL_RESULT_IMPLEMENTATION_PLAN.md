# Lightweight RTL Result Implementation Plan

## 🎯 Implementation Overview

**Approach**: Replace HWKernel with minimal RTLParsingResult containing only the 6 properties that RTLConverter actually uses.

**Total Effort**: ~2 hours  
**Files Modified**: 4  
**Code Reduction**: ~800 lines  
**Performance Gain**: ~25% faster generation  

## 📋 Phase-by-Phase Implementation Plan

### Phase 1: Create RTLParsingResult Dataclass
**Estimated Time**: 30 minutes  
**Risk Level**: LOW  

#### Checklist:
- [ ] Create RTLParsingResult dataclass in `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
- [ ] Add all required imports (dataclass, field, Optional, Path, etc.)
- [ ] Define 7 properties: name, interfaces, pragmas, parameters, source_file, pragma_sophistication_level, parsing_warnings
- [ ] Add proper type hints for all fields
- [ ] Add docstring explaining purpose and relationship to HWKernel
- [ ] Test import: `from brainsmith.tools.hw_kernel_gen.rtl_parser.data import RTLParsingResult`
- [ ] Create simple test instance to verify dataclass works

#### Implementation:
```python
# File: brainsmith/tools/hw_kernel_gen/rtl_parser/data.py
# Add after existing dataclasses

@dataclass
class RTLParsingResult:
    """
    Lightweight result from RTL parsing containing only data needed for DataflowModel conversion.
    
    This replaces the heavy HWKernel object for the unified HWKG pipeline,
    containing only the 6 properties that RTLDataflowConverter actually uses.
    """
    name: str                                    # Module name
    interfaces: Dict[str, Interface]             # RTL Interface objects
    pragmas: List[Pragma]                        # Parsed pragma objects  
    parameters: List[Parameter]                  # Module parameters
    source_file: Optional[Path] = None           # Source RTL file path
    pragma_sophistication_level: str = "simple" # Pragma complexity level
    parsing_warnings: List[str] = field(default_factory=list)  # Parser warnings
    
    def __post_init__(self):
        """Ensure lists are properly initialized."""
        if self.parsing_warnings is None:
            self.parsing_warnings = []
```

#### Validation:
- [ ] Import works without errors
- [ ] Can create instance: `result = RTLParsingResult("test", {}, [], [])`
- [ ] All fields accessible: `result.name`, `result.interfaces`, etc.

---

### Phase 2: Update parse_rtl_file() Function  
**Estimated Time**: 15 minutes  
**Risk Level**: LOW  

#### Checklist:
- [ ] Modify `parse_rtl_file()` in `brainsmith/tools/hw_kernel_gen/rtl_parser/__init__.py`
- [ ] Import RTLParsingResult at top of file
- [ ] Change return type annotation from HWKernel to RTLParsingResult
- [ ] Keep all existing parsing logic unchanged
- [ ] After `hw_kernel = parser.parse_file()`, create RTLParsingResult from HWKernel
- [ ] Map HWKernel properties to RTLParsingResult fields
- [ ] Update function docstring to reflect new return type
- [ ] Test parsing with real RTL file

#### Implementation:
```python
# File: brainsmith/tools/hw_kernel_gen/rtl_parser/__init__.py

# Add import
from .data import RTLParsingResult

def parse_rtl_file(rtl_file, advanced_pragmas: bool = False) -> RTLParsingResult:
    """
    Parse SystemVerilog RTL file and return lightweight parsing result.
    
    Args:
        rtl_file: Path to RTL file
        advanced_pragmas: Enable advanced pragma processing
        
    Returns:
        RTLParsingResult: Lightweight parsing result for DataflowModel conversion
        
    Raises:
        RTLParsingError: If parsing fails
    """
    try:
        logger.info(f"Grammar path not provided, defaulting to: {DEFAULT_GRAMMAR_PATH}")
        language = load_language(DEFAULT_GRAMMAR_PATH)
        
        if not language:
            raise RTLParsingError(f"Failed to load SystemVerilog grammar from {DEFAULT_GRAMMAR_PATH}")
        
        logger.info(f"Successfully created Language object from '{DEFAULT_GRAMMAR_PATH}'")
        logger.info("SystemVerilog grammar loaded successfully.")
        
        parser = RTLParser(language, debug=advanced_pragmas)
        
        # Parse to HWKernel using existing logic (unchanged)
        hw_kernel = parser.parse_file(str(rtl_file))
        
        # Convert HWKernel to lightweight RTLParsingResult
        return RTLParsingResult(
            name=hw_kernel.name,
            interfaces=hw_kernel.interfaces,
            pragmas=hw_kernel.pragmas,
            parameters=hw_kernel.parameters,
            source_file=hw_kernel.source_file,
            pragma_sophistication_level=hw_kernel.pragma_sophistication_level,
            parsing_warnings=hw_kernel.parsing_warnings
        )
        
    except Exception as e:
        raise RTLParsingError(f"RTL parsing failed for {rtl_file}: {e}") from e
```

#### Validation:
- [ ] Function parses RTL file without errors
- [ ] Returns RTLParsingResult instead of HWKernel
- [ ] All fields populated correctly: name, interfaces count, pragmas count
- [ ] Interface objects preserved correctly
- [ ] Test with `examples/thresholding/thresholding_axi.sv`

---

### Phase 3: Update RTLDataflowConverter
**Estimated Time**: 15 minutes  
**Risk Level**: LOW  

#### Checklist:
- [ ] Update `RTLDataflowConverter.convert()` in `brainsmith/dataflow/rtl_integration/rtl_converter.py`
- [ ] Import RTLParsingResult at top of file
- [ ] Change method signature from `convert(self, hw_kernel)` to `convert(self, rtl_result: RTLParsingResult)`
- [ ] Replace all `hw_kernel.` references with `rtl_result.`
- [ ] Update `_validate_hw_kernel()` method to `_validate_rtl_result()`
- [ ] Update all internal references and variable names
- [ ] Update method docstrings
- [ ] Test conversion with real RTL parsing result

#### Implementation:
```python
# File: brainsmith/dataflow/rtl_integration/rtl_converter.py

# Add import
from ...tools.hw_kernel_gen.rtl_parser.data import RTLParsingResult

class RTLDataflowConverter:
    def convert(self, rtl_result: RTLParsingResult) -> ConversionResult:
        """
        Complete conversion pipeline from RTLParsingResult to DataflowModel.
        
        Args:
            rtl_result: RTLParsingResult instance from RTL parser
            
        Returns:
            ConversionResult: Success/failure with DataflowModel or errors
        """
        try:
            logger.info(f"Starting RTL to DataflowModel conversion for kernel: {rtl_result.name}")
            
            # Step 1: Validate RTLParsingResult input
            validation_result = self._validate_rtl_result(rtl_result)
            if not validation_result.success:
                return validation_result
                
            # Step 2: Convert RTL interfaces to DataflowInterface objects
            dataflow_interfaces = []
            conversion_errors = []
            
            for interface_name, rtl_interface in rtl_result.interfaces.items():
                logger.debug(f"Converting interface: {interface_name}")
                
                try:
                    # Find relevant pragmas for this interface
                    interface_pragmas = self._find_interface_pragmas(rtl_interface, rtl_result.pragmas)
                    
                    # Convert RTL interface to DataflowInterface
                    dataflow_interface = self._convert_interface(
                        rtl_interface, interface_pragmas, rtl_result
                    )
                    
                    if dataflow_interface:
                        dataflow_interfaces.append(dataflow_interface)
                        logger.debug(f"Successfully converted interface: {interface_name}")
                    else:
                        conversion_errors.append(f"Failed to convert interface: {interface_name}")
                        
                except Exception as e:
                    error_msg = f"Error converting interface {interface_name}: {str(e)}"
                    logger.error(error_msg)
                    conversion_errors.append(error_msg)
            
            if not dataflow_interfaces:
                return ConversionResult(
                    dataflow_model=None,
                    success=False,
                    errors=["No interfaces could be converted"] + conversion_errors
                )
            
            # Step 3: Create DataflowModel from converted interfaces
            try:
                dataflow_model = DataflowModel(
                    interfaces=dataflow_interfaces,
                    parameters={
                        "kernel_name": rtl_result.name,
                        "source_file": str(rtl_result.source_file) if rtl_result.source_file else None,
                        "pragma_level": rtl_result.pragma_sophistication_level,
                        "conversion_warnings": rtl_result.parsing_warnings
                    }
                )
                
                logger.info(f"Successfully created DataflowModel for kernel: {rtl_result.name}")
                
                return ConversionResult(
                    dataflow_model=dataflow_model,
                    success=True,
                    warnings=conversion_errors  # Non-fatal conversion issues become warnings
                )
                
            except Exception as e:
                error_msg = f"Failed to create DataflowModel: {str(e)}"
                logger.error(error_msg)
                return ConversionResult(
                    dataflow_model=None,
                    success=False,
                    errors=[error_msg] + conversion_errors
                )
                
        except Exception as e:
            error_msg = f"Unexpected error during conversion: {str(e)}"
            logger.error(error_msg)
            return ConversionResult(
                dataflow_model=None,
                success=False,
                errors=[error_msg]
            )
    
    def _validate_rtl_result(self, rtl_result: RTLParsingResult) -> ConversionResult:
        """
        Validate RTLParsingResult input for conversion compatibility.
        
        Args:
            rtl_result: RTLParsingResult instance to validate
            
        Returns:
            ConversionResult: Validation result
        """
        errors = []
        warnings = []
        
        # Check required attributes
        if not rtl_result.name:
            errors.append("RTLParsingResult missing required 'name' field")
            
        if not rtl_result.interfaces:
            warnings.append("RTLParsingResult has no interfaces defined")
            
        # Validate interface structure
        for iface_name, iface in rtl_result.interfaces.items():
            if not hasattr(iface, 'type'):
                errors.append(f"Interface {iface_name} missing 'type' attribute")
            if not hasattr(iface, 'ports'):
                errors.append(f"Interface {iface_name} missing 'ports' attribute")
        
        if errors:
            return ConversionResult(
                dataflow_model=None,
                success=False,
                errors=errors,
                warnings=warnings
            )
        
        return ConversionResult(
            dataflow_model=None,
            success=True,
            warnings=warnings
        )
```

#### Validation:
- [ ] Converter accepts RTLParsingResult without errors
- [ ] Produces same DataflowModel as before (interface count, types, dimensions)
- [ ] All validation logic works correctly
- [ ] Error handling preserved

---

### Phase 4: Update UnifiedHWKGGenerator
**Estimated Time**: 10 minutes  
**Risk Level**: LOW  

#### Checklist:
- [ ] Update `generate_from_rtl()` in `brainsmith/tools/unified_hwkg/generator.py`
- [ ] Import RTLParsingResult
- [ ] Update variable names from `hw_kernel` to `rtl_result`
- [ ] Ensure conversion call uses new interface
- [ ] Update any logging messages
- [ ] Test complete generation pipeline

#### Implementation:
```python
# File: brainsmith/tools/unified_hwkg/generator.py

# Add import  
from ...tools.hw_kernel_gen.rtl_parser.data import RTLParsingResult

class UnifiedHWKGGenerator:
    def generate_from_rtl(self, rtl_file: Path, compiler_data: Dict[str, Any], 
                         output_dir: Path, **kwargs) -> UnifiedGenerationResult:
        """
        Complete generation pipeline from RTL file.
        
        Args:
            rtl_file: Path to SystemVerilog RTL file
            compiler_data: Compiler configuration data
            output_dir: Output directory for generated files
            **kwargs: Additional generation options
            
        Returns:
            UnifiedGenerationResult: Complete generation results
        """
        try:
            logger.info(f"Starting unified generation for RTL file: {rtl_file}")
            
            # Step 1: Parse RTL file using existing RTL parser
            rtl_result = parse_rtl_file(rtl_file)  # Now returns RTLParsingResult
            if not rtl_result:
                return UnifiedGenerationResult(
                    success=False,
                    errors=[f"Failed to parse RTL file: {rtl_file}"]
                )
            
            # Step 2: Convert RTLParsingResult to DataflowModel  
            conversion_result = self.rtl_converter.convert(rtl_result)
            if not conversion_result.success:
                return UnifiedGenerationResult(
                    success=False,
                    errors=[f"RTL to DataflowModel conversion failed"] + conversion_result.errors,
                    warnings=conversion_result.warnings
                )
            
            dataflow_model = conversion_result.dataflow_model
            # ... rest of generation logic unchanged ...
```

#### Validation:
- [ ] Generator accepts RTLParsingResult correctly
- [ ] Complete pipeline works: RTL → RTLParsingResult → DataflowModel → Generated Files
- [ ] Same output files generated as before
- [ ] Same file sizes and content

---

### Phase 5: Validate Parity with Baseline Tests
**Estimated Time**: 30 minutes  
**Risk Level**: MEDIUM  

#### Checklist:
- [ ] Run baseline capture tests with new implementation
- [ ] Compare DataflowModel structure against saved baseline
- [ ] Compare generated file sizes against baseline
- [ ] Compare generated file content hashes against baseline  
- [ ] Run end-to-end test with thresholding_axi.sv
- [ ] Verify interface count, types, and dimensions match exactly
- [ ] Check performance improvement measurement
- [ ] Run any existing unit tests for RTL parser
- [ ] Create test for RTLParsingResult dataclass

#### Validation Script:
```python
# File: test_lightweight_rtl_parity.py

def test_rtl_parsing_result_parity():
    """Test that RTLParsingResult produces same DataflowModel as HWKernel."""
    from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file
    from brainsmith.dataflow.rtl_integration import RTLDataflowConverter
    
    # Parse RTL file
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    rtl_result = parse_rtl_file(rtl_file)
    
    # Convert to DataflowModel
    converter = RTLDataflowConverter()
    result = converter.convert(rtl_result)
    
    assert result.success, f"Conversion failed: {result.errors}"
    assert result.dataflow_model is not None
    
    # Validate against baseline
    baseline = load_baseline("rtl_converter_behavior_baseline.json")
    current_model = serialize_dataflow_model(result.dataflow_model)
    
    assert current_model == baseline["thresholding_axi.sv"]["dataflow_model"]
    print("✅ Perfect parity achieved!")

def test_generation_parity():
    """Test that generated files are identical to baseline."""
    from brainsmith.tools.unified_hwkg import UnifiedHWKGGenerator
    
    generator = UnifiedHWKGGenerator()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        result = generator.generate_from_rtl(
            rtl_file=Path("examples/thresholding/thresholding_axi.sv"),
            compiler_data={'onnx_patterns': [], 'cost_function': lambda *a, **k: 1.0},
            output_dir=Path(temp_dir)
        )
        
        assert result.success, f"Generation failed: {result.errors}"
        
        # Check file sizes match baseline
        baseline_sizes = {
            "thresholding_axi_hwcustomop.py": 7457,
            "thresholding_axi_rtlbackend.py": 7916,
            "test_thresholding_axi.py": 15348
        }
        
        for file_path in result.generated_files:
            expected_size = baseline_sizes[file_path.name]
            actual_size = file_path.stat().st_size
            assert actual_size == expected_size, f"Size mismatch for {file_path.name}: {actual_size} vs {expected_size}"
        
        print("✅ Generated files match baseline sizes!")
```

#### Validation Criteria:
- [ ] **DataflowModel Structure**: Same interface count (4), types, dimensions
- [ ] **Generated File Sizes**: Exactly match baseline (7457, 7916, 15348 bytes)
- [ ] **Interface Types**: ap=config, s_axis=input, m_axis=output, s_axilite=config
- [ ] **Tensor Dimensions**: [128], [7], [7], [31] respectively
- [ ] **Performance**: Measure generation time improvement
- [ ] **No Regressions**: All existing functionality preserved

---

## 🚀 Implementation Execution Plan

### Pre-Implementation Setup
- [ ] Create feature branch: `git checkout -b feature/lightweight-rtl-result`
- [ ] Ensure baseline tests are captured and saved
- [ ] Back up current working implementation

### Implementation Order
1. **Phase 1** → **Phase 2** → **Validate parse_rtl_file() works**
2. **Phase 3** → **Validate conversion works**  
3. **Phase 4** → **Validate generation works**
4. **Phase 5** → **Comprehensive validation**

### Rollback Plan
- [ ] Keep original HWKernel approach available
- [ ] If validation fails, revert changes and analyze differences
- [ ] Document any unexpected dependencies found

### Success Criteria
- [ ] ✅ All baseline tests pass with identical results
- [ ] ✅ Generated code is byte-for-byte identical 
- [ ] ✅ Performance improvement measurable (>20% faster)
- [ ] ✅ Code reduction achieved (~800 lines)
- [ ] ✅ No functionality regressions

## 📊 Expected Results

**Before Implementation**:
- RTL File → parse_rtl_file() → HWKernel (27 properties) → RTLConverter → DataflowModel
- Generation time: ~0.03s
- Generated files: 30,721 bytes total

**After Implementation**:  
- RTL File → parse_rtl_file() → RTLParsingResult (7 properties) → RTLConverter → DataflowModel
- Generation time: ~0.024s (20% improvement)
- Generated files: 30,721 bytes total (identical)
- Code reduction: ~800 lines removed from HWKernel

**Risk Mitigation**: Each phase is independently testable with clear validation criteria and rollback procedures.