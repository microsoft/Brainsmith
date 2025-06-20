# Hardware Kernel Generator End-to-End Tests

This directory contains comprehensive end-to-end tests for the Hardware Kernel Generator (HKG).

## Overview

The end-to-end test suite verifies that HKG correctly:
1. Parses SystemVerilog RTL files with all pragma types
2. Generates all expected output files
3. Produces correct Python integration code
4. Creates valid RTL wrappers

## Test Structure

```
tests/
├── test_e2e_generation.py   # Main test script with golden reference support
├── test_kernel_e2e.sv       # Comprehensive test RTL exercising all features
├── run_e2e_test.sh         # Convenient test runner script
├── golden/                 # Golden reference outputs (git-tracked)
│   └── test_kernel_e2e/    # Golden outputs for test kernel
└── output/                 # Test run outputs (git-ignored)
```

## Running Tests

### First Time Setup - Generate Golden Reference

```bash
# Generate golden reference outputs
./run_e2e_test.sh --golden

# Or directly:
./smithy exec python test_e2e_generation.py --generate-golden
```

After generating golden outputs, **manually verify their correctness** before committing.

### Regular Testing - Verify Against Golden

```bash
# Run test and compare against golden reference
./run_e2e_test.sh

# Or directly:
./smithy exec python test_e2e_generation.py
```

## Test Coverage

The `test_kernel_e2e.sv` module exercises:

### Pragma Types
- `TOP_MODULE` - Module selection with decoy module
- `DATATYPE` - Interface datatype constraints
- `DATATYPE_PARAM` - Datatype parameter mapping (interface & internal)
- `BDIM`/`SDIM` - Block and stream dimension configuration
- `WEIGHT` - Weight interface designation
- `ALIAS` - Parameter aliasing for user-friendly names
- `DERIVED_PARAMETER` - Computed parameters

### Interface Types
- Global Control (ap_clk, ap_rst_n)
- AXI-Stream inputs
- AXI-Stream outputs
- AXI-Stream weights
- AXI-Lite control

### Features
- Parameter auto-linking
- Internal datatype bindings
- Multi-parameter interfaces
- Complex parameter expressions

## Golden Reference Management

### Updating Golden Reference

When HKG generation logic intentionally changes:

1. Review the changes carefully
2. Regenerate golden reference:
   ```bash
   ./run_e2e_test.sh --golden
   ```
3. Manually verify new outputs are correct
4. Commit updated golden files

### What's Compared

The test performs intelligent comparison:
- **Python files**: AST comparison (ignores formatting)
- **Verilog files**: Content comparison (ignores comments/whitespace)
- **JSON files**: Deep comparison (ignores timestamps)
- **Text files**: Key content patterns (ignores dynamic data)

## Troubleshooting

### Test Failures

If tests fail, the output will show:
1. Which step failed (parsing, generation, comparison, test execution)
2. Specific differences from golden reference
3. Generated test output if applicable

### Common Issues

1. **Missing golden reference**: Run with `--golden` first
2. **Intentional changes**: Regenerate golden after verifying changes
3. **Environment issues**: Ensure running within `./smithy exec`

## Adding New Tests

To add new test cases:

1. Create new test RTL file with desired pragmas
2. Update `test_e2e_generation.py` to include new test
3. Generate golden reference for new test
4. Verify outputs manually
5. Commit both test and golden files