# Development Notes

## 2024-12-24: Shuffle Helper Migration
- Moved `shuffle_perfect_loopnest_coeffs()` and `innerloop_moves()` from `brainsmith/libraries/transforms/operations/shuffle_helpers.py` into `InferShuffle` class as static methods
- Deleted the original `shuffle_helpers.py` file
- All tests pass, functionality preserved

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

## 2025-06-27: FINN Transform Registration Progress

### Completed
- ✅ Created FINN plugin infrastructure (registry, adapters)
- ✅ Registered all 39 streamline transforms
  - collapse_repeated.py: 3 transforms
  - sign_to_thres.py: 1 transform
  - reorder.py: 23 transforms
  - absorb.py: 11 transforms (7 new, 4 existing)
  - __init__.py: 1 meta-transform (Streamline)
- ✅ Verified registration with comprehensive test

### Key Achievement
All streamline transforms now use consistent @transform decorator pattern and are available via the FINN plugin system. This represents ~42% of the 92 total FINN transforms.

### Next Priority
Register fpgadataflow transforms (Phase 2 of the plan) - approximately 50 transforms including layer inference, infrastructure insertion, and compilation transforms.