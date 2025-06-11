# Phase 1 Implementation Plan: Complete ParsedKernelData Refactor

## Overview

This document outlines the **COMPLETE, CLEAN REFACTOR** to replace HWKernel entirely with ParsedKernelData throughout the codebase. This is a full transition, not a parallel implementation - ParsedKernelData becomes the primary and only object for RTL parsing results.

## Refactor Scope: FULL REPLACEMENT

### ✅ Completed
- [x] Created ParsedKernelData class in `rtl_parser/data.py`
- [x] Implemented get_template_context() method with full template compatibility
- [x] Added TemplateDatatype and SimpleKernel helper classes
- [x] Added interface helper methods for template compatibility

### 🔄 Complete Refactor Required

#### Core RTL Parser Changes
- [ ] **Replace all HWKernel returns with ParsedKernelData**
- [ ] Update `RTLParser.parse()` method to return ParsedKernelData
- [ ] Update `RTLParser.parse_string()` method to return ParsedKernelData
- [ ] Update `RTLParser.parse_file()` method to return ParsedKernelData
- [ ] Remove all HWKernel creation logic from parser

#### CLI and Tool Integration
- [ ] **Update CLI to use ParsedKernelData exclusively**
- [ ] Update `brainsmith/tools/hw_kernel_gen/cli.py` 
- [ ] Update `brainsmith/tools/hw_kernel_gen/__main__.py`
- [ ] Update all generator calls to use ParsedKernelData
- [ ] Remove all HWKernel references from CLI

#### Template System Overhaul
- [ ] **Update all templates to use ParsedKernelData context**
- [ ] Update `hw_custom_op_slim.py.j2` template
- [ ] Update `rtl_backend.py.j2` template  
- [ ] Update `rtl_wrapper.v.j2` template
- [ ] Update `test_suite.py.j2` template
- [ ] Update `documentation.md.j2` template

#### Generator System Refactor
- [ ] **Update all generators to consume ParsedKernelData**
- [ ] Update `brainsmith/tools/hw_kernel_gen/generators/hw_custom_op.py`
- [ ] Update `brainsmith/tools/hw_kernel_gen/generators/rtl_backend.py`
- [ ] Update `brainsmith/tools/hw_kernel_gen/generators/test_suite.py`
- [ ] Update `brainsmith/tools/unified_hwkg/generator.py`

#### Dataflow Integration Refactor
- [ ] **Update dataflow components to accept ParsedKernelData**
- [ ] Update `brainsmith/dataflow/core/auto_hw_custom_op.py`
- [ ] Update `brainsmith/dataflow/core/auto_rtl_backend.py`
- [ ] Update `brainsmith/dataflow/integration/rtl_conversion.py`

#### Test Suite Complete Update
- [ ] **Update all tests to use ParsedKernelData**
- [ ] Update `tests/tools/hw_kernel_gen/` test files
- [ ] Update `tests/dataflow/integration/` test files
- [ ] Update `tests/integration/` test files
- [ ] Update golden reference files and comparisons

#### Legacy Cleanup
- [ ] **Remove HWKernel class entirely**
- [ ] Remove HWKernel from `rtl_parser/data.py`
- [ ] Remove RTLParsingResult class (obsolete)
- [ ] Clean up all HWKernel imports throughout codebase
- [ ] Update all docstrings and type hints

## Detailed Implementation Steps

### Step 1: RTL Parser Core Refactor

#### 1.1 Update Parser Method Signatures
```python
# In brainsmith/tools/hw_kernel_gen/rtl_parser/parser.py

class RTLParser:
    def parse(self, source: Union[str, Path]) -> ParsedKernelData:
        """Parse RTL file and return parsed kernel data."""
        
    def parse_string(self, systemverilog_code: str, 
                    source_name: str = "<string>",
                    target_module: Optional[str] = None) -> ParsedKernelData:
        """Parse SystemVerilog string and return parsed kernel data."""
        
    def parse_file(self, file_path: str) -> ParsedKernelData:
        """Parse RTL file and return parsed kernel data."""
```

#### 1.2 Remove All HWKernel Creation Logic
```python
# REMOVE: All HWKernel instantiation code
# REPLACE WITH: Direct ParsedKernelData creation

def parse_file(self, file_path: str) -> ParsedKernelData:
    """Parse RTL file and return parsed kernel data."""
    # Existing parsing pipeline (unchanged)
    self._initial_parse(file_path)
    self._extract_components() 
    self._analyze_interfaces()
    self._apply_pragmas()
    
    # NEW: Return ParsedKernelData directly (no HWKernel intermediate)
    return ParsedKernelData(
        name=self.name,
        source_file=Path(file_path),
        parameters=self.parameters,      # Direct reuse of existing Parameter objects
        interfaces=self.interfaces,      # Direct reuse of existing Interface objects  
        pragmas=self.pragmas,           # Direct reuse of existing Pragma objects
        parsing_warnings=self.parsing_warnings
    )
```

### Step 2: CLI Complete Refactor

#### 2.1 Update Main CLI Logic
```python
# In brainsmith/tools/hw_kernel_gen/cli.py

def main():
    """Main CLI entry point - REFACTORED for ParsedKernelData."""
    args = parse_arguments()
    
    # Parse RTL file to ParsedKernelData (not HWKernel)
    rtl_parser = RTLParser(debug=args.debug)
    parsed_data = rtl_parser.parse(args.rtl_file)  # Returns ParsedKernelData
    
    # Load compiler data if provided
    compiler_data = load_compiler_data(args.compiler_data) if args.compiler_data else {}
    
    # Generate all requested outputs using ParsedKernelData
    generated_files = []
    
    if 'hwcustomop' in args.generators:
        hwcustomop_file = generate_hwcustomop(parsed_data, args.output_dir, compiler_data)
        generated_files.append(hwcustomop_file)
        
    if 'rtlbackend' in args.generators:
        rtlbackend_file = generate_rtlbackend(parsed_data, args.output_dir, compiler_data)
        generated_files.append(rtlbackend_file)
        
    if 'test' in args.generators:
        test_file = generate_test_suite(parsed_data, args.output_dir, compiler_data)
        generated_files.append(test_file)
    
    # Print results
    logger.info(f"Generated {len(generated_files)} files for kernel '{parsed_data.name}'")
    for file_path in generated_files:
        logger.info(f"  {file_path}")
```

#### 2.2 Update Generator Functions
```python
def generate_hwcustomop(parsed_data: ParsedKernelData, output_dir: Path, 
                       compiler_data: Dict[str, Any]) -> Path:
    """Generate HWCustomOp from ParsedKernelData."""
    from ..generators.hw_custom_op import HWCustomOpGenerator
    
    generator = HWCustomOpGenerator()
    return generator.generate(parsed_data, output_dir, compiler_data)
```

### Step 3: Template System Complete Update

#### 3.1 Update All Template Generation
```python
# In brainsmith/tools/hw_kernel_gen/generators/hw_custom_op.py

class HWCustomOpGenerator:
    def generate(self, parsed_data: ParsedKernelData, output_dir: Path,
                compiler_data: Dict[str, Any]) -> Path:
        """Generate HWCustomOp file from ParsedKernelData."""
        
        # Get template context directly from ParsedKernelData
        context = parsed_data.get_template_context()
        
        # Add generator-specific context
        context.update({
            'compiler_data': compiler_data,
            'import_paths': {
                'auto_hw_custom_op': 'brainsmith.dataflow.core.auto_hw_custom_op',
                'interface_metadata': 'brainsmith.dataflow.rtl_integration'
            }
        })
        
        # Render template
        rendered = self.render_template('hw_custom_op_slim.py.j2', context)
        
        # Write output
        output_file = output_dir / f"{parsed_data.name}_hwcustomop.py"
        with open(output_file, 'w') as f:
            f.write(rendered)
            
        return output_file
```

### Step 4: Dataflow Integration Refactor

#### 4.1 Update AutoHWCustomOp Generator
```python
# In brainsmith/dataflow/core/auto_hw_custom_op.py

def create_hwcustomop_from_rtl(parsed_data: ParsedKernelData, 
                              compiler_data: Dict[str, Any]) -> type:
    """Create HWCustomOp class from ParsedKernelData."""
    
    # Extract interface metadata from ParsedKernelData
    interface_metadata = []
    for interface in parsed_data.get_dataflow_interfaces():
        metadata = InterfaceMetadata(
            name=interface.name,
            interface_type=interface.type,
            datatype_constraints=interface.get_template_datatype(),
            dimensional_info=interface.get_dimensional_info()
        )
        interface_metadata.append(metadata)
    
    # Create dynamic HWCustomOp class
    class_name = parsed_data.get_class_name()
    hwcustomop_class = create_dynamic_hwcustomop_class(
        class_name, 
        interface_metadata,
        parsed_data.parameters
    )
    
    return hwcustomop_class
```

### Step 5: Complete Test Suite Refactor

#### 5.1 Update All RTL Parser Tests
```python
# In tests/tools/hw_kernel_gen/rtl_parser/test_rtl_parser.py

class TestRTLParser:
    def test_parse_returns_parsed_kernel_data(self):
        """Test that parser returns ParsedKernelData object."""
        parser = RTLParser()
        result = parser.parse("tests/golden/thresholding.sv")
        
        # Verify return type
        assert isinstance(result, ParsedKernelData)
        assert result.name == "thresholding"
        assert len(result.interfaces) > 0
        assert len(result.parameters) > 0
        
    def test_template_context_completeness(self):
        """Test ParsedKernelData provides complete template context."""
        parser = RTLParser()
        parsed_data = parser.parse("tests/golden/thresholding.sv")
        
        context = parsed_data.get_template_context()
        
        # Verify all required template variables
        required_vars = [
            'kernel_name', 'class_name', 'interfaces', 'input_interfaces',
            'output_interfaces', 'rtl_parameters', 'has_inputs', 
            'kernel_complexity', 'InterfaceType', 'kernel'
        ]
        
        for var in required_vars:
            assert var in context, f"Missing template variable: {var}"
```

#### 5.2 Update Integration Tests
```python
# In tests/integration/test_end_to_end_thresholding.py

def test_end_to_end_thresholding_with_parsed_data():
    """Test complete thresholding pipeline with ParsedKernelData."""
    
    # Parse RTL
    parser = RTLParser()
    parsed_data = parser.parse("examples/thresholding/thresholding.sv")
    
    # Generate HWCustomOp
    hwcustomop_generator = HWCustomOpGenerator()
    hwcustomop_file = hwcustomop_generator.generate(
        parsed_data, 
        output_dir, 
        compiler_data={}
    )
    
    # Verify generated file
    assert hwcustomop_file.exists()
    
    # Import and test generated class
    spec = importlib.util.spec_from_file_location("generated", hwcustomop_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Test HWCustomOp functionality
    hwcustomop_class = getattr(module, f"{parsed_data.get_class_name()}HWCustomOp")
    instance = hwcustomop_class()
    
    assert hasattr(instance, 'get_interface_metadata')
    assert len(instance.get_interface_metadata()) > 0
```

### Step 6: Legacy Cleanup

#### 6.1 Remove HWKernel Class
```python
# In brainsmith/tools/hw_kernel_gen/rtl_parser/data.py

# DELETE ENTIRE HWKernel CLASS (lines 728-921)
# DELETE RTLParsingResult CLASS (lines 923-958)

# Keep only:
# - Parameter, Port, Interface, Pragma classes (reused by ParsedKernelData)
# - ParsedKernelData class (new primary object)
# - TemplateDatatype, SimpleKernel helper classes
```

#### 6.2 Update All Imports
```python
# Throughout codebase, replace:
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import HWKernel
# WITH:
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import ParsedKernelData

# Update all type hints:
# OLD: -> HWKernel
# NEW: -> ParsedKernelData

# OLD: hw_kernel: HWKernel
# NEW: parsed_data: ParsedKernelData
```

## Complete File Modification List

### Core RTL Parser Files
1. `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
   - ✅ ParsedKernelData class added
   - 🔄 Remove HWKernel class entirely
   - 🔄 Remove RTLParsingResult class

2. `brainsmith/tools/hw_kernel_gen/rtl_parser/parser.py`
   - 🔄 Update all parse methods to return ParsedKernelData
   - 🔄 Remove HWKernel creation logic
   - 🔄 Update imports and type hints

3. `brainsmith/tools/hw_kernel_gen/rtl_parser/__init__.py`
   - 🔄 Export ParsedKernelData instead of HWKernel

### CLI and Tools
4. `brainsmith/tools/hw_kernel_gen/cli.py`
   - 🔄 Complete refactor for ParsedKernelData
   - 🔄 Update all generator calls

5. `brainsmith/tools/hw_kernel_gen/__main__.py`
   - 🔄 Update imports and usage

### Generators
6. `brainsmith/tools/hw_kernel_gen/generators/hw_custom_op.py`
   - 🔄 Accept ParsedKernelData instead of HWKernel
   - 🔄 Use get_template_context() method

7. `brainsmith/tools/hw_kernel_gen/generators/rtl_backend.py`
   - 🔄 Accept ParsedKernelData instead of HWKernel

8. `brainsmith/tools/hw_kernel_gen/generators/test_suite.py`
   - 🔄 Accept ParsedKernelData instead of HWKernel

### Unified HWKG
9. `brainsmith/tools/unified_hwkg/generator.py`
   - 🔄 Update to use ParsedKernelData
   - 🔄 Update template context generation

### Dataflow Integration
10. `brainsmith/dataflow/core/auto_hw_custom_op.py`
    - 🔄 Accept ParsedKernelData for RTL integration

11. `brainsmith/dataflow/core/auto_rtl_backend.py`
    - 🔄 Accept ParsedKernelData for RTL integration

12. `brainsmith/dataflow/integration/rtl_conversion.py`
    - 🔄 Use ParsedKernelData instead of HWKernel/RTLParsingResult

### Templates
13. `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_slim.py.j2`
    - 🔄 Verify compatibility with ParsedKernelData context

14. `brainsmith/tools/hw_kernel_gen/templates/rtl_backend.py.j2`
    - 🔄 Verify compatibility with ParsedKernelData context

15. `brainsmith/tools/hw_kernel_gen/templates/rtl_wrapper.v.j2`
    - 🔄 Verify compatibility with ParsedKernelData context

### Tests (All Files)
16. `tests/tools/hw_kernel_gen/rtl_parser/test_rtl_parser.py`
    - 🔄 Update to expect ParsedKernelData
    - 🔄 Add template context validation tests

17. `tests/tools/hw_kernel_gen/generators/test_*.py`
    - 🔄 Update all generator tests

18. `tests/integration/test_end_to_end_thresholding.py`
    - 🔄 Complete refactor for ParsedKernelData

19. All other test files referencing HWKernel
    - 🔄 Complete refactor

## Success Criteria

### Complete Replacement Achieved
- [ ] Zero references to HWKernel in entire codebase
- [ ] Zero references to RTLParsingResult in entire codebase  
- [ ] All RTL Parser methods return ParsedKernelData
- [ ] All generators consume ParsedKernelData
- [ ] All templates use ParsedKernelData context

### Functional Validation
- [ ] All existing functionality preserved
- [ ] All templates render correctly
- [ ] All generators produce correct output
- [ ] All tests pass with ParsedKernelData

### Performance Validation
- [ ] Template generation performance ≥ baseline
- [ ] Memory usage ≤ baseline
- [ ] No regression in any performance metrics

### Code Quality
- [ ] Clean, consistent naming throughout
- [ ] All type hints updated to ParsedKernelData
- [ ] All docstrings updated
- [ ] No dead code or unused imports

## Implementation Strategy

### Phase 1A: Core Parser Refactor (Week 1)
1. Update RTL Parser to return ParsedKernelData
2. Remove HWKernel class entirely
3. Update CLI to use ParsedKernelData
4. Basic validation tests

### Phase 1B: Generator Refactor (Week 1)
1. Update all generator classes
2. Update template rendering pipeline
3. Validate template output correctness

### Phase 1C: Integration Refactor (Week 2)
1. Update dataflow integration components
2. Update unified HWKG module
3. Complete test suite refactor

### Phase 1D: Validation and Cleanup (Week 2)
1. Remove all legacy references
2. Performance validation
3. Complete test coverage
4. Documentation updates

This is a **COMPLETE, CLEAN REFACTOR** - no parallel implementation, no backward compatibility layers, no gradual migration. ParsedKernelData becomes the single, unified object for all RTL parsing results throughout the entire system.