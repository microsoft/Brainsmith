# RTL Parser Test Coverage Design Document

## Overview

This document outlines the comprehensive test coverage strategy for the RTL Parser module in the Brainsmith Hardware Kernel Generator. The RTL Parser is a critical component that extracts metadata from SystemVerilog files to generate FINN-compatible hardware kernels.

## Test Coverage Goals

- **Line Coverage**: ≥95% for all modules
- **Branch Coverage**: ≥90% for critical paths
- **Error Handling**: 100% coverage
- **Regression Tests**: For all previously identified bugs

## Component Structure

The RTL Parser consists of the following components that require testing:

```
rtl_parser/
├── parser.py              # Main facade (237 lines)
├── ast_parser.py          # Tree-sitter operations (237 lines)
├── component_extractor.py # Parameter/port extraction (581 lines)
├── workflow_orchestrator.py # Workflow coordination (327 lines)
├── interface_builder.py   # Interface detection
├── parameter_linker.py    # Auto-linking logic
├── pragma.py              # Pragma handling
├── protocol_validator.py  # Protocol validation
├── data.py               # Data structures (140 lines)
└── pragmas/              # Pragma implementations
    ├── base.py           # Base classes (209 lines)
    ├── module.py         # TOP_MODULE pragma
    ├── interface.py      # DATATYPE, WEIGHT, DATATYPE_PARAM
    ├── parameter.py      # ALIAS, DERIVED_PARAMETER
    └── dimension.py      # BDIM, SDIM pragmas
```

## Test Categories

### 1. Unit Tests

#### 1.1 ASTParser Tests

**Test Cases:**
- Grammar loading
  - Default grammar path resolution
  - Custom grammar path
  - Missing grammar file error
  - Invalid grammar file error
- Source parsing
  - Valid SystemVerilog syntax
  - Empty files
  - Files with only comments
  - Unicode content handling
- Syntax error detection
  - Single syntax error
  - Multiple syntax errors
  - Error location reporting
- Module finding
  - Single module
  - Multiple modules
  - Nested modules (should handle correctly)
  - No modules

**Edge Cases:**
- Very large files (>10K lines)
- Deeply nested AST structures
- Malformed but parseable syntax

#### 1.2 ComponentExtractor Tests

**Module Name Extraction:**
- Simple identifiers (`module foo`)
- Escaped identifiers (`module \foo-bar`)
- With parameters (`module foo #(...)`)
- With complex headers

**Parameter Extraction:**
- Basic parameters
  ```verilog
  parameter WIDTH = 32;
  parameter logic [7:0] DEPTH = 16;
  parameter type T = logic [31:0];
  ```
- Type parameters
  ```verilog
  parameter type DATA_T = logic [31:0];
  ```
- Complex default values
  ```verilog
  parameter WIDTH = $clog2(DEPTH);
  parameter SIZE = WIDTH * 8;
  ```
- Localparam filtering (should be excluded)
- Mixed parameter/localparam declarations

**Port Extraction:**
- ANSI-style ports (required)
  ```verilog
  input logic [31:0] data,
  output logic valid,
  input wire clk
  ```
- Non-ANSI ports (should error)
- Complex datatypes
  ```verilog
  input logic [WIDTH-1:0] data [0:DEPTH-1]
  ```
- Interface ports (should handle ERROR nodes gracefully)
- Parameterized widths

**Error Scenarios:**
- Malformed parameter declarations
- Invalid port syntax
- Mixed ANSI/non-ANSI (should error)

#### 1.3 WorkflowOrchestrator Tests

**Module Selection:**
- Single module (no pragma needed)
- Multiple modules with TOP_MODULE pragma
- Multiple modules without pragma (error)
- Explicit target_module parameter
- TOP_MODULE pragma mismatch
- Multiple TOP_MODULE pragmas (error)

**Parsing Coordination:**
- Success path with all components
- Component extraction failures
- Pragma extraction failures
- Error propagation

**Auto-linking Control:**
- Enable/disable auto-linking
- Auto-linking with existing pragma links
- Parameter exclusion logic

#### 1.4 Pragma Tests

**For each pragma type (8 total):**

**TOP_MODULE Pragma:**
- Valid: `@brainsmith top my_module`
- Invalid: Missing module name
- Invalid: Extra arguments

**DATATYPE Pragma:**
- Valid: `@brainsmith datatype in0 t_input`
- Valid: `@brainsmith datatype weight0 t_weight s_weight`
- Invalid: Missing interface name
- Invalid: No datatype specified

**WEIGHT Pragma:**
- Valid: `@brainsmith weight w0`
- Invalid: Missing interface name
- Interaction with interface type detection

**DATATYPE_PARAM Pragma:**
- Valid: `@brainsmith datatype_param in0 width IN_WIDTH`
- Valid: `@brainsmith datatype_param accumulator signed ACC_SIGNED`
- Invalid: Unknown property type
- Invalid: Missing arguments
- Internal vs interface targets

**ALIAS Pragma:**
- Valid: `@brainsmith alias NUM_CHANNELS PE`
- Invalid: Missing target name
- Invalid: Missing nodeattr name

**DERIVED_PARAMETER Pragma:**
- Valid: `@brainsmith derived N_BITS "$clog2(N_LEVELS)"`
- Invalid: Missing expression
- Invalid: Python syntax error in expression

**BDIM Pragma:**
- Valid: `@brainsmith bdim out0 OUT_BDIM`
- Valid: Advanced expressions
- Invalid: Missing arguments

**SDIM Pragma:**
- Valid: `@brainsmith sdim in0 IN_SDIM`
- Invalid: Missing interface name

**Common Tests for All Pragmas:**
- Line number tracking
- Error message quality
- `applies_to_interface_metadata()` method
- `apply_to_kernel()` method

#### 1.5 InterfaceBuilder Tests

**Port Grouping:**
- AXI-Stream detection (TDATA, TVALID, TREADY)
- AXI-Lite detection (AWADDR, WDATA, etc.)
- Clock/reset as Global Control
- Custom interface patterns
- Orphan ports

**Interface Type Assignment:**
- Input streams
- Output streams
- Weight interfaces (via pragma)
- Control interfaces

**InterfaceMetadata Creation:**
- Correct port mapping
- Datatype metadata assignment
- Chunking parameter assignment
- Protocol validation

#### 1.6 ParameterLinker Tests

**Interface Parameter Linking:**
- Pattern detection (PREFIX_WIDTH, PREFIX_SIGNED)
- Case sensitivity
- Multiple matching parameters
- No matching parameters
- Already linked parameters (skip)

**Internal Parameter Linking:**
- Common patterns (ACC_WIDTH, THRESH_WIDTH)
- Exclusion of interface prefixes
- Exclusion of already-linked parameters
- Custom pattern groups

### 2. Integration Tests

#### 2.1 Pragma Integration Scenarios

**Multiple Pragmas Same Interface:**
```verilog
// @brainsmith datatype in0 input_t
// @brainsmith bdim in0 IN_BDIM
// @brainsmith sdim in0 IN_SDIM
```

**Pragma Conflicts:**
- Multiple DATATYPE pragmas for same interface
- Conflicting BDIM/SDIM values
- WEIGHT pragma on output interface

**Pragma Dependencies:**
- DATATYPE_PARAM requires matching interface
- BDIM/SDIM require streaming interface

**Internal Datatype Collection:**
- DATATYPE_PARAM for non-interface targets
- Grouping by target name
- Multiple properties for same target

#### 2.2 Parameter Linking Integration

**Auto-linking with Pragmas:**
- Pragma overrides auto-detection
- Partial pragma specification
- Complex expressions in pragmas

**Exposed Parameter Management:**
- Parameters linked by pragmas (not exposed)
- Parameters linked by auto-detection (not exposed)
- Remaining parameters (exposed)

#### 2.3 End-to-End Parsing Scenarios

**Complete RTL Examples:**
1. Simple streaming kernel
2. Multi-interface kernel
3. Kernel with weights
4. Kernel with complex parameters
5. Kernel with all pragma types

**Real-world Examples:**
- Thresholding kernel
- Matrix multiply kernel
- Pooling kernel
- Custom protocol kernel

### 3. Error and Edge Case Tests

#### 3.1 Error Scenarios

**File Errors:**
- File not found
- Permission denied
- Empty file
- Binary file

**Syntax Errors:**
- Invalid SystemVerilog
- Partial/truncated files
- Encoding issues

**Validation Errors:**
- No input interfaces
- No output interfaces
- Missing Global Control
- Invalid pragma combinations

#### 3.2 Edge Cases

**Large Files:**
- Performance with 10K+ lines
- Memory usage monitoring
- Timeout handling

**Complex Structures:**
- Deeply nested parameters
- Recursive width expressions
- Generate blocks (unsupported)
- SystemVerilog interfaces (unsupported)

**Unicode Handling:**
- Unicode in comments
- Unicode in identifiers
- Mixed encodings

#### 3.3 Validation Tests

**KernelMetadata Validation:**
- Required interfaces
- Parameter references
- Datatype completeness
- Warning collection

**Interface Validation:**
- Protocol compliance
- Port direction consistency
- Required signals

### 4. Performance and Stress Tests

#### 4.1 Performance Benchmarks

- Parse time vs file size
- Memory usage vs complexity
- Pragma processing overhead
- Tree-sitter efficiency

#### 4.2 Stress Scenarios

- Maximum file size handling
- Maximum module complexity
- Maximum pragma count
- Concurrent parsing (if applicable)

### 5. Regression Tests

#### 5.1 Known Issues

- Non-ANSI port declaration detection
- Interface port ERROR node handling
- Localparam filtering
- Parameter type extraction

#### 5.2 Backward Compatibility

- Legacy pragma formats
- Old parsing behavior
- Migration warnings

## Test Implementation Strategy

### Directory Structure

```
tests/
├── unit/
│   ├── test_ast_parser.py
│   ├── test_component_extractor.py
│   ├── test_workflow_orchestrator.py
│   ├── test_pragma_*.py  # One per pragma type
│   ├── test_interface_builder.py
│   └── test_parameter_linker.py
├── integration/
│   ├── test_pragma_integration.py
│   ├── test_parameter_linking.py
│   └── test_end_to_end.py
├── fixtures/
│   ├── valid_rtl/
│   ├── invalid_rtl/
│   └── edge_cases/
└── utils/
    ├── rtl_generator.py
    └── test_helpers.py
```

### Test Utilities

**RTL Generator:**
- Programmatic SystemVerilog generation
- Pragma injection helpers
- Module templates

**Test Helpers:**
- AST comparison utilities
- Metadata validation helpers
- Error assertion helpers

### Coverage Reporting

- Use pytest-cov for coverage
- Generate HTML reports
- Track coverage trends
- Enforce minimum coverage in CI

## Continuous Integration

### Test Execution

1. Unit tests first (fast feedback)
2. Integration tests
3. Performance tests (nightly)
4. Regression suite

### Quality Gates

- All tests must pass
- Coverage thresholds enforced
- Performance benchmarks tracked
- Memory leak detection

## Future Considerations

### Extensibility Testing

- New pragma types
- New interface protocols
- Custom validation rules

### Tool Integration

- VS Code extension testing
- CLI integration tests
- Error message quality

## Conclusion

This comprehensive test coverage design ensures the RTL Parser is robust, maintainable, and reliable. The test suite should be implemented incrementally, starting with critical unit tests and expanding to integration scenarios. Regular coverage reporting and CI integration will maintain quality over time.