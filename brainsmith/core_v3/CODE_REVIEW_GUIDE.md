# Code Review Guide - DSE V3 Phase 1

## Overview

This guide helps reviewers understand the DSE V3 Phase 1 implementation and highlights key areas for review. The implementation creates a Design Space Constructor that parses Blueprint YAML files and constructs validated design spaces for FPGA hardware exploration.

## Architecture Summary

```
User Input (ONNX + Blueprint YAML)
    ↓
Forge API (forge.py)
    ↓
Blueprint Parser (parser.py)
    ↓
Design Space Construction (data_structures.py)
    ↓
Validation (validator.py)
    ↓
Validated DesignSpace Object
```

## Key Review Areas

### 1. Data Structures (`phase1/data_structures.py`)

**Purpose**: Define the core data model for design spaces.

**Review Focus**:
- [ ] Are the dataclasses well-structured and intuitive?
- [ ] Is the kernel/transform parsing logic in `HWCompilerSpace` clear?
- [ ] Does the combination counting in `get_total_combinations()` handle all cases?
- [ ] Are the string representations (`__str__`) helpful for debugging?

**Key Design Decisions**:
- Kernels can be strings, lists with backends, or mutually exclusive groups
- Optional elements use `~` prefix or None in lists
- Transforms support both flat lists and phase-based organization

**Example to Verify**:
```python
# This should parse correctly:
kernels = [
    "MatMul",                        # Auto-import all backends
    ["Softmax", ["hls", "rtl"]],    # Specific backends
    [["LayerNorm", ["standard"]], "RMSNorm"],  # Mutually exclusive
    [None, "Transpose"]              # Optional
]
```

### 2. Blueprint Parser (`phase1/parser.py`)

**Purpose**: Parse Blueprint YAML files into DesignSpace objects.

**Review Focus**:
- [ ] Does the parser handle all YAML formats correctly?
- [ ] Are error messages helpful when parsing fails?
- [ ] Is the version checking appropriate?
- [ ] Does the parser preserve the order of kernels/transforms?

**Key Areas**:
- `_parse_hw_compiler()` - Handles the complex kernel/transform formats
- `_parse_processing()` - Parses pre/post-processing steps
- `load_blueprint()` - File loading with proper error handling

**Edge Cases to Check**:
- Empty kernel/transform lists
- Invalid YAML syntax
- Missing required fields
- Unsupported versions

### 3. Validator (`phase1/validator.py`)

**Purpose**: Validate design spaces for correctness and feasibility.

**Review Focus**:
- [ ] Are validation rules appropriate?
- [ ] Do warnings help users avoid common mistakes?
- [ ] Are the thresholds reasonable (1000 combos warning, 10000 error)?
- [ ] Is the validation comprehensive but not overly restrictive?

**Key Validations**:
- Model file existence
- Kernel/transform format validity
- Search constraint operators
- Combination count limits
- Common build steps presence

**Consider**:
- Should we validate kernel/transform names against a registry?
- Are the warning thresholds appropriate for real use cases?
- Should we add more semantic validations?

### 4. Forge API (`phase1/forge.py`)

**Purpose**: Main entry point that orchestrates the entire process.

**Review Focus**:
- [ ] Is the API intuitive and easy to use?
- [ ] Does logging provide enough information?
- [ ] Is error handling comprehensive?
- [ ] Does the summary output help users understand their design space?

**Key Features**:
- Simple interface: `forge(model_path, blueprint_path)`
- Detailed logging with verbose mode
- Comprehensive error messages
- Design space summary generation

### 5. Exception Handling (`phase1/exceptions.py`)

**Purpose**: Custom exceptions for clear error reporting.

**Review Focus**:
- [ ] Are exception types appropriate?
- [ ] Do they carry enough context for debugging?
- [ ] Is the hierarchy logical?

### 6. Tests (`tests/`)

**Purpose**: Comprehensive test coverage for all components.

**Review Focus**:
- [ ] Do tests cover all code paths?
- [ ] Are edge cases tested?
- [ ] Are test fixtures representative of real use?
- [ ] Do integration tests verify the complete flow?

**Test Structure**:
- Unit tests for each component
- Integration tests for end-to-end flows
- Fixtures with simple and complex blueprints
- Error scenario testing

## Code Quality Checklist

### Style and Conventions
- [ ] Code follows Python PEP 8 conventions
- [ ] Type hints used throughout
- [ ] Docstrings for all public functions/classes
- [ ] Clear variable and function names
- [ ] No excessive complexity in functions

### Design Patterns
- [ ] Single responsibility principle followed
- [ ] Clean separation between parsing, validation, and construction
- [ ] Appropriate use of dataclasses
- [ ] Extensible design for future phases

### Error Handling
- [ ] All file operations have proper error handling
- [ ] Clear error messages that help users fix issues
- [ ] Appropriate use of custom exceptions
- [ ] No silent failures

### Performance Considerations
- [ ] Combination counting is efficient
- [ ] No unnecessary iterations or copies
- [ ] Memory usage reasonable for large design spaces

## Specific Review Questions

1. **Kernel Format Flexibility**: Is the support for multiple kernel formats (string, list with backends, mutually exclusive) clear and necessary?

2. **Transform Organization**: Should we support both flat and phase-based transform organization, or standardize on one?

3. **Validation Strictness**: Are we validating enough? Too much? Should validation be pluggable?

4. **YAML Format**: Is the YAML format intuitive for users? Should we use lists instead of tuples throughout?

5. **Extension Points**: Are there enough hooks for Phase 2 and Phase 3 integration?

6. **Logging**: Is the logging level and detail appropriate? Should we use structured logging?

7. **Testing**: Are the tests maintainable? Do they test behavior rather than implementation?

## Integration Points to Verify

1. **Phase 2 Integration**: Will the DesignSpace object provide everything needed for exploration?

2. **FINN Integration**: Does the HWCompilerSpace structure map well to FINN's requirements?

3. **Backend Flexibility**: Can we easily add new backends beyond FINN?

## Potential Improvements to Discuss

1. **Blueprint Inheritance**: Should we support including/extending other blueprints?

2. **Validation Plugins**: Should validation be extensible with custom rules?

3. **Caching**: Should parsed blueprints be cached for performance?

4. **Schema Validation**: Should we use JSON Schema or similar for blueprint validation?

5. **Async Support**: Should the API support async operations for parallel parsing?

## Security Considerations

- [ ] No arbitrary code execution from blueprints
- [ ] Path traversal prevention in file operations
- [ ] Input size limits to prevent DoS
- [ ] Safe YAML loading (using safe_load)

## Documentation Review

- [ ] Is the README clear and helpful?
- [ ] Are code examples correct and runnable?
- [ ] Is the blueprint format well-documented?
- [ ] Are error messages actionable?

## Testing the Implementation

To manually test the implementation:

```bash
# Run all tests
./smithy exec "cd brainsmith/core_v3 && python -m pytest tests/ -v"

# Try the simple example
./smithy exec "cd brainsmith/core_v3 && python -c \"
from phase1 import forge
ds = forge('tests/fixtures/simple_model.onnx', 'tests/fixtures/simple_blueprint.yaml')
print(ds)
\""

# Try with a complex blueprint
./smithy exec "cd brainsmith/core_v3 && python -c \"
from phase1 import forge
ds = forge('tests/fixtures/simple_model.onnx', 'tests/fixtures/complex_blueprint.yaml')
print(f'Total combinations: {ds.get_total_combinations()}')
\""
```

## Review Outcome

After review, please provide feedback on:

1. **Must Fix**: Critical issues that block merging
2. **Should Fix**: Important improvements for code quality
3. **Consider**: Suggestions for future improvements
4. **Positive**: Things done well that should be preserved

## Notes for Reviewers

- This is a complete reimplementation (V3) with no backward compatibility requirements
- The design prioritizes clarity and extensibility over performance optimization
- Phase 2 and 3 will build on this foundation, so the interfaces are important
- The implementation follows the Prime Directives, especially "Break Fearlessly"