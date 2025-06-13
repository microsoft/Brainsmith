# RTL Parser Method Refactoring Implementation Plan

## Overview
Refactor RTL Parser to have a clean architecture where `parse()` is the core string parser and `parse_file()` reads files then calls `parse()`. This eliminates code duplication and creates a more logical API.

## Current Architecture Issues
- `parse_string()` and `parse_file()` both implement the 3-stage parsing pipeline independently
- `parse()` is a confusing dispatcher method that checks if input is file vs string
- Code duplication in error handling and metadata creation
- Inconsistent implementations between string and file parsing

## Target Architecture
```
parse(systemverilog_code: str, source_name: str = "<string>") -> KernelMetadata
├─ Core 3-stage parsing pipeline
├─ Single source of truth for parsing logic
└─ Clean string-based parsing API

parse_file(file_path: str) -> KernelMetadata  
├─ Reads file content
├─ Calls parse(file_content, file_path)
└─ Handles file-specific errors
```

## Implementation Steps

### Phase 1: Refactor `parse` to be Core String Parser
- [ ] **1.1** Replace current `parse` method with core string parsing logic
  - [ ] Change signature to `parse(self, systemverilog_code: str, source_name: str = "<string>") -> KernelMetadata`
  - [ ] Move 3-stage pipeline from `parse_string` to `parse`
  - [ ] Use `_initial_parse_string` (or create unified `_initial_parse`)
  - [ ] Include proper error handling and KernelMetadata creation

- [ ] **1.2** Update `_initial_parse` method to handle both file and string inputs
  - [ ] Make `_initial_parse` work with either file_path OR (code, source_name)
  - [ ] Keep file-based logic for `parse_file`
  - [ ] Or create unified method that handles both

### Phase 2: Refactor `parse_file` to Use `parse`
- [ ] **2.1** Update `parse_file` to read file and delegate to `parse`
  - [ ] Read file content at start of method
  - [ ] Call `self.parse(file_content, file_path)`
  - [ ] Remove duplicated 3-stage pipeline
  - [ ] Keep file-specific error handling (FileNotFoundError, etc.)

### Phase 3: Remove Redundant Methods
- [ ] **3.1** Remove `parse_string` method entirely
  - [ ] Delete method definition
  - [ ] Update any imports or references

- [ ] **3.2** Clean up `_initial_parse_string` method
  - [ ] Either remove if no longer needed
  - [ ] Or integrate into unified `_initial_parse` method

### Phase 4: Testing and Verification
- [ ] **4.1** Update tests that use old method names
  - [ ] Find tests calling `parse_string()`
  - [ ] Update to call `parse()` instead
  - [ ] Verify test functionality

- [ ] **4.2** Run comprehensive testing
  - [ ] Test `parse()` with string input
  - [ ] Test `parse_file()` with file input
  - [ ] Verify end-to-end functionality works
  - [ ] Check error handling works correctly

## Benefits
1. **Single Source of Truth**: Core parsing logic in one place
2. **Clean API**: Intuitive method names and responsibilities
3. **Reduced Code Duplication**: Eliminate redundant implementations
4. **Better Maintainability**: Changes only need to be made in one place
5. **Clearer Architecture**: parse() for strings, parse_file() for files

## Risk Mitigation
- Keep file-specific error handling in `parse_file()`
- Ensure all tests continue to pass
- Maintain backward compatibility by updating calling code
- Test with both simple and complex SystemVerilog files