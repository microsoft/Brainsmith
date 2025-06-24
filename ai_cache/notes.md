# CLAUDE.md Verification Notes - 2025-06-23

## Key Findings

### Directory Structure Changes
- No `brainsmith/dataflow/` directory exists - this appears outdated
- No `brainsmith/custom_op/` directory exists - this appears outdated
- The codebase has been reorganized under `brainsmith/libraries/` with subdirectories:
  - `analysis/` - Contains hw_kernel_gen tool
  - `kernels/` - Hardware kernel implementations
  - `transforms/` - DSE transformations
  - `blueprints_v2/` - Blueprint v2 system
  - `automation/` - Automation tools
  - `operators/` - Operator implementations

### Testing Structure
- Main test directory exists at `tests/`
- No `tests/dataflow/` subdirectory - appears outdated
- No `tests/end2end/bert/` directory - this appears to be incorrect
- Test files are in root of `tests/` directory focusing on blueprint v2, DSE, etc.

### Demo Structure
- `demos/bert/` exists with correct files
- `demos/bert_new/` and `demos/bert_direct/` also exist
- quicktest.sh exists and matches documented commands

### Commands Verification
- pytest commands should be updated to reflect actual test structure
- BERT demo commands are mostly correct
- Hardware kernel generator is at different path: `brainsmith/libraries/analysis/tools/hw_kernel_gen/`

### Configuration Files
- `.editorconfig` exists and matches documented settings
- No pytest.ini in project root (one exists in deps/brevitas)

### Docker/Smithy
- `./smithy` script exists and is executable
- Docker workflow documented correctly

### Architecture Updates Needed
- Interface type system location incorrect
- Dataflow modeling references are outdated
- Custom operations structure has changed
- Need to update paths and component descriptions