# HWKG-Dataflow Synthesis Plan: Complete Integration Strategy

**Objective**: Full transition from dual architectures to unified Interface-Wise Dataflow Modeling with RTL integration.

**Strategy**: Clean break with comprehensive migration path, no legacy compatibility burden.

## 1. Target Architecture Overview

### New Unified Pipeline
```
SystemVerilog RTL → RTL Parser → DataflowModel → AutoClasses → FINN Integration
                         ↓
                   Pragma Processing → ChunkingStrategies
                         ↓  
                   Template Generation → Instantiation Code
```

### Core Components (Post-Synthesis)
1. **RTLDataflowConverter**: RTL → DataflowModel pipeline
2. **PragmaProcessor**: Enhanced @brainsmith → ChunkingStrategy
3. **UnifiedGenerator**: DataflowModel → AutoHWCustomOp/AutoRTLBackend instantiation
4. **MathematicalTemplates**: Thin templates for instantiation, not implementation
5. **ValidationFramework**: Complete axiom compliance checking

## 2. Synthesis Architecture Design

### 2.1 Core Module: `brainsmith.tools.unified_hwkg`

**Location**: `/home/tafk/dev/brainsmith-2/brainsmith/tools/unified_hwkg/`

**Structure**:
```
unified_hwkg/
├── __init__.py                 # Main API exports
├── cli.py                      # Unified CLI interface
├── converter.py                # RTL → DataflowModel conversion
├── pragma_processor.py         # Enhanced pragma → strategy conversion
├── generator.py                # DataflowModel → code generation
├── templates/                  # Minimal instantiation templates
│   ├── hwcustomop_instantiation.py.j2
│   ├── rtlbackend_instantiation.py.j2
│   └── test_suite.py.j2
└── validation/                 # Complete validation framework
    ├── axiom_validator.py
    └── integration_validator.py
```

### 2.2 Enhanced Dataflow Core

**Enhancements to**: `/home/tafk/dev/brainsmith-2/brainsmith/dataflow/`

**New modules**:
```
dataflow/
├── rtl_integration/            # NEW MODULE
│   ├── __init__.py
│   ├── rtl_converter.py        # HWKernel → DataflowModel
│   ├── pragma_converter.py     # Pragma → ChunkingStrategy
│   └── interface_mapper.py     # RTL Interface → DataflowInterface
├── enhanced_chunking/          # ENHANCED MODULE  
│   ├── pragma_strategies.py    # BDIM-derived strategies
│   └── layout_detection.py     # Advanced ONNX layout analysis
└── validation/                 # ENHANCED MODULE
    └── rtl_validation.py       # RTL-specific validation
```

## 3. Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)

#### 1.1 Create RTL → DataflowModel Converter
**File**: `brainsmith/dataflow/rtl_integration/rtl_converter.py`

```python
class RTLDataflowConverter:
    """Convert HWKernel to DataflowModel with complete pragma processing."""
    
    def convert(self, hw_kernel: HWKernel) -> DataflowModel:
        """
        Complete conversion pipeline:
        1. Extract interfaces from HWKernel
        2. Apply pragma-based chunking strategies
        3. Build DataflowInterface objects
        4. Create unified DataflowModel
        """
        
    def _convert_interface(self, rtl_interface: Interface, 
                          pragmas: List[Pragma]) -> DataflowInterface:
        """Convert single RTL interface to DataflowInterface."""
        
    def _apply_pragma_strategies(self, interface: DataflowInterface, 
                                pragmas: List[Pragma]) -> DataflowInterface:
        """Apply BDIM/DATATYPE pragmas to interface."""
```

#### 1.2 Create Enhanced Pragma Processor
**File**: `brainsmith/dataflow/rtl_integration/pragma_converter.py`

```python
class PragmaToStrategyConverter:
    """Convert @brainsmith pragmas to ChunkingStrategy instances."""
    
    def convert_bdim_pragma(self, pragma: BDimPragma) -> ChunkingStrategy:
        """Convert BDIM pragma to appropriate chunking strategy."""
        
    def convert_datatype_pragma(self, pragma: DatatypePragma) -> DataTypeConstraint:
        """Convert DATATYPE pragma to constraint."""
        
    def convert_weight_pragma(self, pragma: WeightPragma) -> Dict[str, Any]:
        """Convert WEIGHT pragma to interface metadata."""
```

#### 1.3 Create Unified Generator
**File**: `brainsmith/tools/unified_hwkg/generator.py`

```python
class UnifiedHWKGGenerator:
    """Generate AutoHWCustomOp/AutoRTLBackend instances from DataflowModel."""
    
    def generate_hwcustomop(self, dataflow_model: DataflowModel, 
                           kernel_name: str) -> Path:
        """Generate HWCustomOp instantiation file."""
        
    def generate_rtlbackend(self, dataflow_model: DataflowModel, 
                           kernel_name: str) -> Path:
        """Generate RTLBackend instantiation file."""
        
    def generate_test_suite(self, dataflow_model: DataflowModel, 
                           kernel_name: str) -> Path:
        """Generate test suite for the kernel."""
```

### Phase 2: Template Replacement (Weeks 3-4)

#### 2.1 Minimal Instantiation Templates

**Template**: `hwcustomop_instantiation.py.j2`
```python
# AUTO-GENERATED: Do not edit manually
"""{{ kernel_name }} HWCustomOp implementation using Interface-Wise Dataflow Modeling."""

from brainsmith.dataflow import AutoHWCustomOp, DataflowModel, DataflowInterface
from brainsmith.dataflow.rtl_integration import create_interface_metadata

class {{ class_name }}HWCustomOp(AutoHWCustomOp):
    """Auto-generated HWCustomOp for {{ kernel_name }} kernel."""
    
    def __init__(self, onnx_node, **kwargs):
        # Create interface metadata from generated configuration
        interface_metadata = [
            {% for interface in interfaces %}
            create_interface_metadata(
                name="{{ interface.name }}",
                interface_type="{{ interface.interface_type }}",
                chunking_strategy={{ interface.chunking_strategy }},
                dtype_constraints={{ interface.dtype_constraints }},
                axi_metadata={{ interface.axi_metadata }}
            ),
            {% endfor %}
        ]
        
        super().__init__(onnx_node, interface_metadata, **kwargs)
    
    def get_nodeattr_types(self):
        """Get node attribute types with dataflow enhancements."""
        attrs = super().get_enhanced_nodeattr_types()
        
        # Add kernel-specific attributes if needed
        {% if custom_attributes %}
        attrs.update({
            {% for attr_name, attr_config in custom_attributes.items() %}
            "{{ attr_name }}": {{ attr_config }},
            {% endfor %}
        })
        {% endif %}
        
        return attrs
```

**Template**: `rtlbackend_instantiation.py.j2`
```python
# AUTO-GENERATED: Do not edit manually  
"""{{ kernel_name }} RTLBackend implementation using Interface-Wise Dataflow Modeling."""

from brainsmith.dataflow import AutoRTLBackend

class {{ class_name }}RTLBackend(AutoRTLBackend):
    """Auto-generated RTLBackend for {{ kernel_name }} kernel."""
    
    def __init__(self):
        super().__init__()
        
        # Set dataflow interfaces configuration
        self.dataflow_interfaces = {
            {% for interface in interfaces %}
            "{{ interface.name }}": {
                "interface_type": "{{ interface.interface_type }}",
                "dtype": {{ interface.dtype_config }},
                "tensor_dims": {{ interface.tensor_dims }},
                "block_dims": {{ interface.block_dims }},
                "stream_dims": {{ interface.stream_dims }},
                "axi_metadata": {{ interface.axi_metadata }}
            },
            {% endfor %}
        }
    
    def get_nodeattr_types(self):
        """Get node attribute types with RTL enhancements."""
        return super().get_enhanced_nodeattr_types()
    
    def code_generation_dict(self):
        """Generate RTL code generation dictionary."""
        return super().generate_enhanced_code_dict()
```

#### 2.2 Remove Old Template System

**Deprecated Components** (to be removed):
- `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_slim.py.j2`
- `brainsmith/tools/hw_kernel_gen/templates/rtl_backend.py.j2`
- `brainsmith/tools/hw_kernel_gen/generators/hw_custom_op.py` (complex template logic)
- `brainsmith/tools/hw_kernel_gen/generators/rtl_backend.py` (complex template logic)

### Phase 3: CLI Integration (Week 5)

#### 3.1 Unified CLI Interface
**File**: `brainsmith/tools/unified_hwkg/cli.py`

```python
def main():
    """Unified HWKG CLI with complete dataflow integration."""
    parser = argparse.ArgumentParser(
        description="Unified Hardware Kernel Generator with Interface-Wise Dataflow Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (replaces old HWKG)
  python -m brainsmith.tools.unified_hwkg thresholding.sv compiler_data.py -o output/
  
  # Advanced validation
  python -m brainsmith.tools.unified_hwkg thresholding.sv compiler_data.py -o output/ --validate-axioms
  
  # Performance analysis
  python -m brainsmith.tools.unified_hwkg thresholding.sv compiler_data.py -o output/ --analyze-performance
        """
    )
    
    # Core arguments (compatible with old HWKG)
    parser.add_argument('rtl_file', help='SystemVerilog RTL file')
    parser.add_argument('compiler_data', help='Python compiler data file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    
    # Enhanced arguments
    parser.add_argument('--validate-axioms', action='store_true', 
                       help='Validate Interface-Wise Dataflow axiom compliance')
    parser.add_argument('--analyze-performance', action='store_true',
                       help='Generate performance analysis report')
    parser.add_argument('--optimize-parallelism', action='store_true',
                       help='Auto-optimize iPar/wPar parameters')
    parser.add_argument('--debug-dataflow', action='store_true',
                       help='Debug dataflow model construction')
```

#### 3.2 CLI Implementation
```python
def unified_generation_pipeline(args):
    """Complete unified generation pipeline."""
    
    # 1. Parse RTL using existing RTL parser
    hw_kernel = parse_rtl_file(args.rtl_file)
    
    # 2. Convert to DataflowModel
    converter = RTLDataflowConverter()
    dataflow_model = converter.convert(hw_kernel)
    
    # 3. Validate if requested
    if args.validate_axioms:
        validation_result = validate_dataflow_model(dataflow_model)
        if not validation_result.is_valid:
            print("❌ Axiom validation failed:")
            for error in validation_result.errors:
                print(f"   - {error}")
            return False
    
    # 4. Generate code
    generator = UnifiedHWKGGenerator()
    hwcustomop_file = generator.generate_hwcustomop(dataflow_model, hw_kernel.name)
    rtlbackend_file = generator.generate_rtlbackend(dataflow_model, hw_kernel.name)
    test_file = generator.generate_test_suite(dataflow_model, hw_kernel.name)
    
    # 5. Performance analysis if requested
    if args.analyze_performance:
        performance_report = generate_performance_analysis(dataflow_model)
        performance_file = args.output / "performance_analysis.json"
        performance_file.write_text(json.dumps(performance_report, indent=2))
    
    return True
```

### Phase 4: Migration & Cleanup (Week 6)

#### 4.1 Deprecation Strategy

**Step 1**: Mark old HWKG as deprecated
```python
# brainsmith/tools/hw_kernel_gen/__init__.py
import warnings

warnings.warn(
    "hw_kernel_gen is deprecated. Use brainsmith.tools.unified_hwkg instead.",
    DeprecationWarning,
    stacklevel=2
)
```

**Step 2**: Create compatibility shim
```python
# brainsmith/tools/hw_kernel_gen/cli.py (updated)
def main():
    """Compatibility shim - redirects to unified HWKG."""
    print("⚠️  hw_kernel_gen is deprecated.")
    print("🔄 Redirecting to unified HWKG...")
    
    from brainsmith.tools.unified_hwkg.cli import main as unified_main
    unified_main()
```

**Step 3**: Update documentation and examples

#### 4.2 File Removal Plan

**Phase 4.1**: Mark as deprecated (keep files)
**Phase 4.2**: Remove after 1 release cycle

**Files to remove**:
```
brainsmith/tools/hw_kernel_gen/
├── generators/
│   ├── hw_custom_op.py         # ❌ REMOVE - replaced by AutoHWCustomOp
│   └── rtl_backend.py          # ❌ REMOVE - replaced by AutoRTLBackend
├── templates/
│   ├── hw_custom_op_slim.py.j2 # ❌ REMOVE - replaced by instantiation template
│   └── rtl_backend.py.j2       # ❌ REMOVE - replaced by instantiation template
└── pragma_integration/        # ❌ REMOVE - moved to dataflow/rtl_integration/
```

**Files to keep**:
```
brainsmith/tools/hw_kernel_gen/
├── rtl_parser/                 # ✅ KEEP - core RTL parsing (enhance)
├── config.py                   # ✅ KEEP - configuration system 
├── errors.py                   # ✅ KEEP - error handling
└── data.py                     # ✅ KEEP - basic data structures
```

## 4. Enhanced Feature Implementation

### 4.1 Complete Axiom Validation

**File**: `brainsmith/dataflow/validation/axiom_validator.py`

```python
class AxiomValidator:
    """Complete validation of Interface-Wise Dataflow axioms."""
    
    def validate_hwkg_axioms(self, hw_kernel: HWKernel, 
                            dataflow_model: DataflowModel) -> ValidationResult:
        """Validate HWKG axioms compliance."""
        
    def validate_dataflow_axioms(self, dataflow_model: DataflowModel) -> ValidationResult:
        """Validate Interface-Wise Dataflow axioms."""
        
    def validate_rtl_parser_axioms(self, hw_kernel: HWKernel) -> ValidationResult:
        """Validate RTL Parser axioms."""
```

### 4.2 Performance Analysis Framework

**File**: `brainsmith/tools/unified_hwkg/performance_analyzer.py`

```python
class PerformanceAnalyzer:
    """Complete performance analysis using DataflowModel."""
    
    def analyze(self, dataflow_model: DataflowModel) -> PerformanceReport:
        """Generate comprehensive performance analysis."""
        
    def optimize_parallelism(self, dataflow_model: DataflowModel, 
                           constraints: Dict[str, Any]) -> OptimizationResult:
        """Find optimal iPar/wPar configuration."""
        
    def generate_resource_estimates(self, dataflow_model: DataflowModel) -> ResourceReport:
        """Generate detailed resource usage estimates."""
```

### 4.3 Enhanced ONNX Integration

**File**: `brainsmith/dataflow/enhanced_chunking/layout_detection.py`

```python
class AdvancedLayoutDetector:
    """Advanced ONNX layout detection and chunking optimization."""
    
    def detect_layout_from_onnx(self, onnx_node, model_wrapper) -> str:
        """Detect tensor layout from ONNX graph patterns."""
        
    def suggest_optimal_chunking(self, layout: str, 
                               tensor_shape: List[int]) -> ChunkingStrategy:
        """Suggest optimal chunking strategy for detected layout."""
```

## 5. Migration Guide for Users

### 5.1 Command Line Migration

**Old HWKG**:
```bash
python -m brainsmith.tools.hw_kernel_gen thresholding.sv compiler_data.py -o output/
```

**New Unified HWKG**:
```bash
python -m brainsmith.tools.unified_hwkg thresholding.sv compiler_data.py -o output/
```

**Enhanced Usage**:
```bash
# With validation
python -m brainsmith.tools.unified_hwkg thresholding.sv compiler_data.py -o output/ --validate-axioms

# With performance analysis  
python -m brainsmith.tools.unified_hwkg thresholding.sv compiler_data.py -o output/ --analyze-performance

# With optimization
python -m brainsmith.tools.unified_hwkg thresholding.sv compiler_data.py -o output/ --optimize-parallelism
```

### 5.2 Code Migration

**Old Generated Code**:
```python
# Old: Complex generated implementation
class ThresholdingHWCustomOp(HWCustomOp):
    def get_exp_cycles(self):
        # Complex generated calculation code
        return cycles
        
    def get_instream_width(self):
        # Generated width calculation
        return width
```

**New Generated Code**:
```python
# New: Simple instantiation
class ThresholdingHWCustomOp(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        interface_metadata = [
            create_interface_metadata(
                name="in0", interface_type="INPUT",
                chunking_strategy=index_chunking(-1, [16]),
                dtype_constraints=DataTypeConstraint(["INT", "UINT"], 1, 32)
            )
        ]
        super().__init__(onnx_node, interface_metadata, **kwargs)
    
    # All methods inherited from AutoHWCustomOp with DataflowModel
```

### 5.3 Pragma Migration

**Pragmas remain the same** - no user changes needed:
```systemverilog
// @brainsmith BDIM in0_V_data_V -1 [16]
// @brainsmith DATATYPE in0_V_data_V INT,UINT 1 16  
// @brainsmith WEIGHT weights_V_data_V
```

## 6. Testing Strategy

### 6.1 Validation Tests

**Test**: Old HWKG vs New Unified HWKG output equivalence
```python
def test_output_equivalence():
    # Generate with old HWKG
    old_result = old_hwkg.generate(rtl_file, compiler_data)
    
    # Generate with unified HWKG  
    new_result = unified_hwkg.generate(rtl_file, compiler_data)
    
    # Compare functional equivalence
    assert functionally_equivalent(old_result, new_result)
```

**Test**: Mathematical correctness
```python
def test_mathematical_correctness():
    dataflow_model = create_test_model()
    
    # Test all axiom relationships
    assert validate_axiom_1(dataflow_model)  # Data hierarchy
    assert validate_axiom_2(dataflow_model)  # Core relationship
    # ... all 10 axioms
```

### 6.2 Performance Tests

**Test**: Generation performance
```python
def test_generation_performance():
    # Unified HWKG should be faster (less template complexity)
    old_time = benchmark_old_hwkg()
    new_time = benchmark_unified_hwkg()
    
    assert new_time < old_time * 1.5  # Allow some overhead for dataflow model
```

**Test**: Mathematical accuracy
```python
def test_mathematical_accuracy():
    # Test against known solutions
    dataflow_model = create_bert_model()
    intervals = dataflow_model.calculate_initiation_intervals(iPar, wPar)
    
    assert intervals.cII["in0"] == expected_cII
    assert intervals.L == expected_total_cycles
```

## 7. Documentation Updates

### 7.1 Updated Architecture Documentation

**File**: `docs/unified_architecture.md`
- Complete architectural overview
- Integration benefits
- Migration guide
- Performance improvements

### 7.2 API Documentation

**File**: `docs/api/unified_hwkg.md`
- Complete API reference
- Examples for all features
- Best practices guide

### 7.3 Tutorial Updates

**File**: `docs/examples/unified_tutorial/`
- Updated BERT demo using unified HWKG
- Performance analysis examples
- Advanced optimization examples

## 8. Success Metrics

### 8.1 Technical Metrics

**Elimination of Technical Debt**:
- ✅ Zero placeholders in generated code
- ✅ Zero mocks in production code  
- ✅ Single unified architecture
- ✅ Complete axiom compliance

**Performance Improvements**:
- 🎯 50% reduction in generated code size (templates → instantiation)
- 🎯 90% reduction in template complexity
- 🎯 100% mathematical accuracy (vs placeholder calculations)

### 8.2 User Experience Metrics

**API Simplicity**:
- ✅ Same CLI interface (backward compatible)
- ✅ Enhanced features available via flags
- ✅ Automatic optimization options

**Error Handling**:
- ✅ Unified error framework
- ✅ Mathematical validation errors
- ✅ Actionable error messages

## 9. Risk Mitigation

### 9.1 Compatibility Risks

**Risk**: Breaking existing workflows
**Mitigation**: 
- Compatibility shim for old CLI
- Gradual deprecation process
- Extensive testing against existing examples

### 9.2 Performance Risks  

**Risk**: DataflowModel overhead
**Mitigation**:
- Performance benchmarking at each phase
- Optimize DataflowModel construction
- Cache computed results

### 9.3 Complexity Risks

**Risk**: Over-engineering the integration
**Mitigation**:
- Incremental implementation
- Continuous validation against axioms
- Focus on eliminating placeholders/mocks

## 10. Timeline & Resource Allocation

### Detailed Schedule

**Week 1**: Core Infrastructure
- Day 1-2: RTLDataflowConverter
- Day 3-4: PragmaToStrategyConverter  
- Day 5: UnifiedHWKGGenerator skeleton

**Week 2**: Template Replacement
- Day 1-2: Minimal instantiation templates
- Day 3-4: Remove old template logic
- Day 5: Integration testing

**Week 3**: CLI Integration
- Day 1-2: Unified CLI interface
- Day 3-4: Enhanced features (validation, analysis)
- Day 5: Performance optimization

**Week 4**: Testing & Validation
- Day 1-2: Comprehensive test suite
- Day 3-4: Performance benchmarking
- Day 5: Documentation updates

**Week 5**: Migration & Cleanup
- Day 1-2: Deprecation strategy
- Day 3-4: File removal
- Day 5: Final integration testing

**Week 6**: Release Preparation
- Day 1-2: Documentation completion
- Day 3-4: Tutorial updates
- Day 5: Release validation

### Resource Requirements

**Development**: 1 senior engineer (full-time, 6 weeks)
**Testing**: 0.5 engineer (weeks 3-6)
**Documentation**: 0.25 engineer (weeks 4-6)

## 11. Conclusion

This synthesis plan provides a **complete transition path** from the current dual architecture to a unified Interface-Wise Dataflow Modeling system with RTL integration.

**Key Benefits**:
1. **Eliminates all placeholders and mocks**
2. **Provides mathematical foundation throughout**
3. **Maintains backward compatibility during transition**
4. **Significantly reduces code complexity**
5. **Enables advanced optimization features**

**Success Criteria**:
- Zero technical debt remaining
- Complete axiom compliance
- Enhanced user experience
- Performance improvements
- Clean, maintainable architecture

The plan ensures a **clean break** from the legacy template-heavy approach while providing a smooth migration path for existing users.