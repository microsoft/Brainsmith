# Thresholding Directory Cleanup Proposal

**Date**: 2025-07-07  
**Directory**: `/brainsmith/kernels/thresholding/`  
**Purpose**: Remove outdated test files and documentation from earlier development iterations

## Summary

The thresholding directory contains many test files and reports from earlier AutoHWCustomOp development phases. With the successful implementation of the generic AutoHWCustomOp base class and working KI pipeline, most of these files are now obsolete and should be removed to avoid confusion.

## Files to DELETE

### 1. Outdated Test Reports (3 files)
These reports document failed tests from earlier development iterations before the architecture was finalized:

- **`AutoHWCustomOp_Test_Report_20250702_201821.md`**
  - Old test report showing 0/5 tests passing
  - Pre-dates the working KI implementation
  - No longer relevant
  
- **`DEBUGGING_FINDINGS.md`**
  - Documents issues that have been resolved in current implementation
  - Contains outdated architectural decisions (e.g., bandaid SIMD/PE logic)
  - Historical interest only
  
- **`TEST_ANALYSIS_REPORT.md`**
  - Analysis of flawed original tests
  - Documents issues with FIXED datatype that no longer apply
  - Superseded by current working implementation

### 2. Obsolete Test Files (8 files)
These tests were written for earlier iterations and don't work with the current architecture:

- **`test_auto_hw_simple.py`** - Early prototype test
- **`test_auto_finn_integration.py`** - Tests outdated integration approach
- **`test_behavioral_execution.py`** - Tests non-existent behavioral execution
- **`test_cppsim.py`** - Tests incomplete cppsim implementation
- **`test_finn_pipeline.py`** - Tests outdated pipeline approach
- **`test_simple_auto_instantiation.py`** - Redundant simple instantiation test
- **`test_thresholding_comparison_v2.py`** - Flawed comparison test (see TEST_ANALYSIS_REPORT.md)
- **`generate_full_test_report.py`** - Generates the obsolete test report

### 3. Outdated Generated Files (1 directory)
- **`bsmith/`** directory
  - Contains auto-generated files from an earlier KI run
  - Should use the official test output directory instead
  - Includes: `generation_metadata.json`, `generation_summary.txt`, and generated Python/Verilog files

## Files to KEEP

### 1. Core RTL Files (2 files)
- **`thresholding_axi.sv`** - Original RTL implementation
- **`thresholding_axi_bw.sv`** - Bitwidth-parameterized RTL (used for testing)

### 2. FINN Integration (1 directory)
- **`finn/`** directory
  - Contains manual FINN implementation for comparison
  - Useful reference for understanding FINN's approach
  - Files: `thresholding.py`, `thresholding_rtl.py`, `thresholding_template_wrapper.v`

### 3. Working Test (1 file)
- **`test_rtl_generation.py`**
  - Current working test that validates RTL generation
  - Successfully demonstrates KI functionality
  - Should be moved to main test suite

## Recommended Actions

1. **Delete all files marked for deletion** (12 files + 1 directory)
2. **Move `test_rtl_generation.py`** to `/brainsmith/tools/hw_kernel_gen/tests/`
3. **Keep RTL source files** as examples of hardware kernels
4. **Keep `finn/` directory** as reference implementation

## Migration Path

For historical reference, the deleted files can be accessed through git history:
```bash
git log --follow -- brainsmith/kernels/thresholding/DEBUGGING_FINDINGS.md
```

## Benefits of Cleanup

1. **Clarity**: Removes confusion about which tests are current
2. **Maintenance**: Reduces false positives when searching/grepping
3. **Documentation**: Working code serves as better documentation than failed tests
4. **Focus**: Highlights the successful KI implementation

## Alternative Approach

If preserving history is important, consider:
1. Create an `archive/` subdirectory
2. Move obsolete files there with a README explaining their historical context
3. Add `archive/` to `.gitignore` to prevent accidental usage

## Conclusion

The thresholding directory has served its purpose as a development testbed. Now that KI is working, it's time to clean up the experimental artifacts and maintain only the essential files that demonstrate the working system.