# Hardware Kernel Generator (HWKG) Code Review Guide

## Overview for Code Reviewers

This guide provides a structured approach for reviewing the Hardware Kernel Generator (HWKG) system, which consists of two main components:
- **Dataflow Framework** (`brainsmith/dataflow/`) - Interface-wise modeling and abstraction
- **Hardware Kernel Generator** (`brainsmith/tools/hw_kernel_gen/`) - RTL integration and code generation

## Executive Summary

**What is HWKG?**
The Hardware Kernel Generator automates the integration of custom RTL (SystemVerilog) hardware kernels into the FINN neural network compiler. It eliminates manual template creation by:

1. **Parsing SystemVerilog** to extract interface definitions and pragmas
2. **Converting to standardized abstractions** via the Dataflow Framework
3. **Generating minimal code** using base classes instead of verbose templates
4. **Validating correctness** through comprehensive constraint checking

**Key Achievement**: Reduced generated code by ~90% through interface-driven base classes.

## Architecture Overview

```
RTL File (SystemVerilog) 
    ↓
RTL Parser → Interface Discovery → Pragma Extraction
    ↓
Dataflow Framework → Interface Modeling → Validation
    ↓
Code Generation → FINN Integration Files → Testing
```

## Part 1: Dataflow Framework (`brainsmith/dataflow/`)

### 🎯 Purpose
Provides a unified abstraction layer for hardware interfaces, eliminating the complexity of manual FINN integration through standardized modeling.

### 📁 Directory Structure Review Checklist

#### Core Components (`core/`)
| File | Purpose | Review Focus |
|------|---------|--------------|
| `dataflow_interface.py` | **CRITICAL** - Core interface abstraction | Mathematical relationships, validation logic |
| `dataflow_model.py` | **CRITICAL** - Computational model | Performance calculations, parallelism optimization |
| `auto_hw_custom_op.py` | **HIGH** - Generated HWCustomOp base class | Runtime shape extraction, lazy initialization |
| `auto_rtl_backend.py` | **HIGH** - Generated RTLBackend base class | Resource estimation, synthesis integration |
| `validation.py` | **MEDIUM** - Constraint validation framework | Error handling, constraint checking |
| `tensor_chunking.py` | **MEDIUM** - Tensor shape utilities | Shape inference, chunking strategies |
| `interface_metadata.py` | **MEDIUM** - Metadata containers | Data structure validation |
| `class_naming.py` | **LOW** - Naming conventions | Style consistency |

#### Integration Layer (`integration/`)
| File | Purpose | Review Focus |
|------|---------|--------------|
| `rtl_conversion.py` | **HIGH** - RTL Parser to Dataflow conversion | Interface mapping, metadata extraction |

### 🔍 Key Review Areas

#### 1. Mathematical Precision (`dataflow_interface.py:142-214`)
**CRITICAL**: The three-tier dimension system must be mathematically correct:

```python
# Review this relationship carefully:
qDim = [768]        # Original tensor shape (BERT hidden size)
tDim = [96]         # Processing chunk size  
num_tensors = qDim ÷ tDim = [8]  # Number of chunks (computed via get_num_tensors())

# Validation constraints:
assert qDim[i] % tDim[i] == 0  # Must be evenly divisible
assert tDim[i] % sDim[i] == 0  # Streaming constraint
```

**Review Questions:**
- Are dimension relationships mathematically sound?
- Is the distinction between qDim (original) and num_tensors (computed) clear?
- Do validation constraints prevent invalid configurations?

#### 2. Runtime Shape Extraction (`auto_hw_custom_op.py:147-308`)
**HIGH IMPORTANCE**: The system extracts actual tensor shapes from FINN at runtime instead of using hardcoded values.

```python
# Review this pattern:
def _extract_runtime_tensor_shape(self, interface_name: str) -> List[int]:
    """Extract actual shapes from ModelWrapper - eliminates static configuration"""
```

**Review Questions:**
- Is the fallback hierarchy robust (ModelWrapper → chunker → default)?
- Are error conditions handled gracefully?
- Does lazy initialization work correctly?

#### 3. Unified Computational Model (`dataflow_model.py:103-165`)
**CRITICAL**: Single method replaces multiple scattered calculations:

```python
def calculate_initiation_intervals(self, iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals:
    """Unified calculation replacing manual per-operation math"""
```

**Review Questions:**
- Are the performance calculations correct?
- Is parallelism optimization mathematically sound?
- Does bottleneck analysis identify correct constraints?

## Part 2: Hardware Kernel Generator (`brainsmith/tools/hw_kernel_gen/`)

### 🎯 Purpose
Automates RTL integration pipeline from SystemVerilog parsing to FINN code generation.

### 📁 Directory Structure Review Checklist

#### Core Pipeline Components
| File | Purpose | Review Focus |
|------|---------|--------------|
| `hkg.py` | **CRITICAL** - Main entry point and orchestration | Command-line interface, workflow coordination |
| `rtl_parser/parser.py` | **CRITICAL** - SystemVerilog parsing engine | Grammar correctness, interface extraction |
| `generators/enhanced_*.py` | **HIGH** - Code generation engines | Template logic, base class usage |

#### RTL Parser (`rtl_parser/`)
| File | Purpose | Review Focus |
|------|---------|--------------|
| `parser.py` | **CRITICAL** - Main parsing logic | AST processing, error handling |
| `interface_scanner.py` | **HIGH** - Interface discovery | Signal detection, naming conventions |
| `interface_builder.py` | **HIGH** - Interface construction | Metadata creation, validation |
| `protocol_validator.py` | **MEDIUM** - AXI protocol validation | Standard compliance |
| `pragma.py` | **MEDIUM** - Pragma extraction | Constraint parsing |
| `grammar.py` | **LOW** - Tree-sitter grammar | SystemVerilog syntax rules |

#### Code Generation (`generators/`)
| File | Purpose | Review Focus |
|------|---------|--------------|
| `enhanced_hw_custom_op_generator.py` | **HIGH** - HWCustomOp generation | Base class inheritance, minimal templates |
| `enhanced_rtl_backend_generator.py` | **HIGH** - RTLBackend generation | Resource estimation, synthesis flow |
| `rtl_template_generator.py` | **MEDIUM** - Verilog wrapper generation | Signal mapping, protocol compliance |

#### Support Systems
| File | Purpose | Review Focus |
|------|---------|--------------|
| `errors.py` | **MEDIUM** - Error handling framework | Error hierarchy, context management |
| `enhanced_template_manager.py` | **MEDIUM** - Template processing | Jinja2 integration, context building |
| `analysis/enhanced_*.py` | **LOW** - Analysis utilities | Interface analysis, pragma processing |

### 🔍 Key Review Areas

#### 1. RTL Parser Robustness (`rtl_parser/parser.py:45-180`)
**CRITICAL**: Must handle real-world SystemVerilog correctly:

```systemverilog
// Review: Can the parser handle these patterns?
module thresholding_axi #(
    parameter SIMD = 8,
    parameter THRESHOLD = 50
)(
    input  logic ap_clk,
    input  logic ap_rst_n,
    // AXI-Stream interfaces...
);
```

**Review Questions:**
- Does the parser handle parameterized modules correctly?
- Are interface naming conventions (ap_clk, ap_rst_n) detected properly?
- Is error recovery robust for malformed RTL?

#### 2. Code Generation Quality (`generators/enhanced_hw_custom_op_generator.py:67-145`)
**HIGH IMPORTANCE**: Generated code should be minimal and inherit from base classes:

```python
# Review: Generated template should look like this (minimal):
class Auto{{ class_name }}HWCustomOp(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        interface_metadata = create_{{ kernel_name }}_metadata()
        super().__init__(onnx_node, interface_metadata, **kwargs)
    
    # Most methods inherited from AutoHWCustomOp base class
```

**Review Questions:**
- Is the generated code minimal (not verbose)?
- Are base class methods properly inherited?
- Is the template context correctly built?

#### 3. Error Handling and Recovery (`errors.py:45-180`)
**MEDIUM IMPORTANCE**: Must provide actionable error messages:

```python
# Review: Error context should be helpful
class RTLParsingError(BrainsmithError):
    """RTL parsing failed with specific module/line information"""
```

**Review Questions:**
- Are error messages actionable for developers?
- Is the error hierarchy logical?
- Are contexts preserved through the call stack?

## Critical Integration Points

### 🔗 RTL Parser → Dataflow Conversion (`integration/rtl_conversion.py`)
**This is where the two systems connect - review carefully:**

```python
def convert_interfaces(self, rtl_interfaces: List[RTLInterface]) -> List[DataflowInterface]:
    """Convert RTL parser output to Dataflow interfaces"""
```

**Review Questions:**
- Is the conversion logic correct?
- Are all RTL interface types mapped properly?
- Is metadata preserved correctly?

### 🔗 Template Generation Pipeline (`orchestration/pipeline_orchestrator.py`)
**Review the end-to-end workflow:**

```python
def execute_generation_pipeline(self, rtl_file: Path, output_dir: Path) -> GenerationResult:
    """Full pipeline: Parse → Convert → Generate → Validate"""
```

**Review Questions:**
- Is the pipeline robust to failures?
- Are intermediate results validated?
- Is rollback/cleanup handled properly?

## Performance and Resource Considerations

### 📊 What to Look For

1. **Memory Usage**: Large RTL files and complex templates
2. **Parse Time**: SystemVerilog parsing can be expensive
3. **Template Rendering**: Jinja2 performance with complex contexts
4. **Validation Overhead**: Comprehensive constraint checking

### 🎯 Optimization Areas

1. **Caching**: Template compilation, parse results
2. **Lazy Loading**: Dataflow model building, shape extraction
3. **Parallel Processing**: Multiple file generation
4. **Resource Estimation**: Mathematical model accuracy

## Testing Strategy Review

### 🧪 Test Categories to Verify

1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: End-to-end pipeline
3. **Golden Reference Tests**: Generated code comparison
4. **Real-world Examples**: Actual RTL from users

### 🔍 Test Coverage Focus Areas

| Component | Critical Tests | Coverage Target |
|-----------|---------------|-----------------|
| RTL Parser | Malformed input, edge cases | >90% |
| Dataflow Interface | Mathematical validation | >95% |
| Code Generation | Template correctness | >85% |
| Error Handling | All error paths | >90% |

## Common Issues and Anti-patterns

### ❌ Red Flags During Review

1. **Hardcoded Dimensions**: Should use runtime extraction
2. **Verbose Generated Code**: Should inherit from base classes
3. **Missing Validation**: All inputs must be validated
4. **Poor Error Messages**: Must be actionable
5. **Tight Coupling**: Components should be modular

### ✅ Good Patterns to Verify

1. **Lazy Initialization**: DataflowModel building on demand
2. **Interface-Driven Design**: Metadata controls everything
3. **Comprehensive Validation**: Multiple validation layers
4. **Base Class Inheritance**: Minimal generated code
5. **Robust Error Handling**: Context-rich error reporting

## Questions for Code Review Discussion

### Architecture Questions
1. Is the separation between Dataflow Framework and HWKG clear and logical?
2. Are the interfaces between components well-defined?
3. Is the system extensible for future RTL patterns?

### Implementation Questions
1. Are the mathematical relationships in the dimension system correct?
2. Is the runtime shape extraction robust and reliable?
3. Are error conditions handled comprehensively?

### Quality Questions
1. Is the generated code maintainable and debuggable?
2. Are the templates readable and extensible?
3. Is the test coverage sufficient for production use?

## Conclusion

The HWKG system represents a significant architectural achievement in automating RTL integration. The key innovation is the combination of:

1. **Runtime dimension extraction** (vs. static configuration)
2. **Interface-driven base classes** (vs. verbose code generation)
3. **Unified computational modeling** (vs. scattered calculations)
4. **Comprehensive validation** (vs. manual error checking)

Focus your review on the mathematical correctness, error handling robustness, and architectural clarity of these core innovations.

---

**For questions during review, focus on:**
- Mathematical relationships in `dataflow_interface.py`
- Runtime extraction logic in `auto_hw_custom_op.py`
- Parser robustness in `rtl_parser/`
- Generated code quality in `generators/`